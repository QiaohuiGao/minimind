
from torch.utils.data import Dataset,DataLoader
from datasets import load_dataset
import torch.nn.functional as F
import torch
import random
from transformers import AutoTokenizer
from Mymodel import Config,MyModel
import wandb
import os
import copy

def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    # apply_chat_template 渲染 reasoning 数据时，没有思考内容的 assistant 回答会带上空 <think> 标签
    # 80% 概率移除，避免模型学到"空思考"这种无意义行为；保留 20% 让模型知道可以不思考
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content

class DPODataset(Dataset):
    def __init__(self,file_path, tokenizer, max_len):
        super().__init__()
        self.tokenizer=tokenizer
        self.max_len=max_len
        self.padding=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id=tokenizer(f"{tokenizer.bos_token}assitant\n", add_special_tokens=False).input_ids
        self.eos_id=tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids
        self.datasets=load_dataset("json", data_files=file_path, split="train")
    
    def __len__(self):
        return len(self.datasets)
    
    def __getitem__(self, index):
        sample=self.datasets[index]
        chosen = sample['chosen']  
        rejected = sample['rejected']  
        chosen_prompt=self.tokenizer.apply_chat_template(chosen, tokenize=False, add_generation_prompt=False) #hasn't been embedding, just add specal marks
        chosen_prompt=post_processing_chat(chosen_prompt)
        rejected_prompt=self.tokenizer.apply_chat_template(rejected, tokenize=False, add_generation_prompt=False)
        rejected_prompt=post_processing_chat(rejected_prompt)
        
        chosen_encoding=self.tokenizer(chosen_prompt,truncation=True,max_length=self.max_len, padding="max_length")
        rejected_encoding=self.tokenizer(rejected_prompt,truncation=True,max_length=self.max_len, padding="max_length")
        
        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)
        # input_ids:  <bos>user 你好<eos> <bos>assistant 你好呀<eos>  <pad> <pad>
        # loss_mask:    0    0   0   0      0      0       1 1 1 1      0    0
        #                                                 └─assistant 回答─┘
        
        rejected_input_ids=rejected_encoding["input_ids"]
        rejected_loss_mask=self.generate_loss_mask(chosen_input_ids)
        
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)
        
        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }
        
    def generate_loss_mask(self,input_ids):
        loss_mask=[0]*len(input_ids)
        i=0
        while i<len(input_ids):
            if input_ids[i:i+len(self.bos_id)]==self.bos_id:
                start=i+len(self.bos_id)
                end=start
                while end<len(input_ids):
                    if input_ids[end:end+len(self.eos_id)]==self.eos_id:
                        break
                    # loss_mask[end]=1
                    end+=1
                for j in range(start, min(end+self.max_len),self.max_len):
                    loss_mask[j]=1
                i=end+len(self.eos_id) if end<len(input_ids) else len(input_ids)
            else:
                i+=1
        return loss_mask    

def get_seq_logprob(model, x, y, mask):
    """
    返回：每个样本在它自己回答 token 上的 log 概率之和  → shape [batch]
    x:    [batch, seq]   输入
    y:    [batch, seq]   目标（错位1的下一个token）
    mask: [batch, seq]   只有 assistant 回答处=1
    """
    logits, _, _ = model(x)                   # [B, seq, vocab]
    log_probs = F.log_softmax(logits, dim=-1) # [B, seq, vocab] 转成每个 token 的 log 概率（在 vocab 维归一化再取 log）
    token_logp = torch.gather(log_probs, dim=2, index=y.unsqueeze(-1)).squeeze(-1) # [B, seq] 用 y 当索引，挑出"真实下一个 token"那一列的 log 概率
    seq_logp = (token_logp * mask).sum(dim=-1)  # [B]只保留回答部分（mask 把 prompt/padding 清零），沿 seq 求和 → 每条回答一个总分
    return seq_logp

def dpo_loss(ref_logp, policy_logp, beta=0.1):
    """
    ref_logp / policy_logp: [batch]，上半 = chosen，下半 = rejected
    （训练时 x = torch.cat([x_chosen, x_rejected])，所以前一半是好回答、后一半是坏回答）
    返回：一个标量 loss
    """
    # 1. batch 的上半是 chosen、下半是 rejected，切开
    bs = policy_logp.shape[0] // 2
    chosen_policy, reject_policy = policy_logp[:bs], policy_logp[bs:]
    chosen_ref,    reject_ref    = ref_logp[:bs],    ref_logp[bs:]

    # 2. 各自算"好 − 坏"的 log 概率差
    pi_logratios  = chosen_policy - reject_policy    # policy 觉得"好比坏"高多少
    ref_logratios = chosen_ref    - reject_ref       # ref   觉得"好比坏"高多少

    # 3. 训练目标：让 policy 比 ref 更会区分好坏 → logits 越大越好
    logits = pi_logratios - ref_logratios

    # 4. -logsigmoid：logits 越大 loss 越小；.mean() 对整个 batch 平均
    loss = -F.logsigmoid(beta * logits).mean()
    return loss


def train_dpo():
    config=Config()
    wandb.init(
        project="mymodel_dpo",
        name=f"dpo-lr{config.learning_rate}-epochs{config.epochs}",
        config=vars(config) 
    )
    
    tokenizer=AutoTokenizer.from_pretrained(os.path.dirname(__file__))
    
    data_path=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","dataset","dpo_dataset_mini.jsonl"))
    dpo_dataset=DPODataset(data_path,tokenizer, max_len=config.max_seq_len)
    dpo_dataloader=DataLoader(dpo_dataset,batch_size=config.batch_size, shuffle=True )
    
    model=MyModel(config).to(config.device)
    weight_path=os.path.abspath(os.path.join(os.path.dirname(__file__),"my_model.pt"))
    model.load_state_dict(torch.load(weight_path,map_location=config.device))#load weight to device
    
    # ref_model=model.clone()
    ref_model=copy.deepcopy(model)
    ref_model.eval()
    ref_model.requires_grad_(False)
    
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate)
    
    model.train() 
    for epoch in range(config.epochs):
        for step, batch in enumerate(dpo_dataloader):
            x_chosen=batch["x_chosen"].to(config.device)
            x_rejected=batch["x_rejected"].to(config.device)
            y_chosen = batch['y_chosen'].to(config.device)
            y_rejected = batch['y_rejected'].to(config.device)
            mask_chosen = batch['mask_chosen'].to(config.device)
            mask_rejected = batch['mask_rejected'].to(config.device)
            
            x=torch.cat([batch["x_chosen"],batch["x_rejected"]]).to(config.device)
            y=torch.cat([batch["y_chosen"],batch["y_rejected"]]).to(config.device)
            mask=torch.cat([batch["mask_chosen"],batch["mask_rejected"]]).to(config.device)
            
            with torch.no_grad():                       # ref 不要梯度
                ref_logp = get_seq_logprob(ref_model, x, y, mask)
            policy_logp = get_seq_logprob(model, x, y, mask)
            
            loss = dpo_loss(ref_logp, policy_logp, beta=config.beta)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            
if __name__=="__main__":
    train_dpo()