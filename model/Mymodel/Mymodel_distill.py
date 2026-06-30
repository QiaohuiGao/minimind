from Mymodel import Config,MyModel
from Mymodel_sft import Mymodelsftdataset
from transformers import AutoTokenizer
from model_minimind import MiniMindForCausalLM
import os
from torch.utils.data import Dataset,DataLoader
import torch
import torch.functional as F
import wandb

def distill_loss(s_logits,t_logits, temperature):
    with torch.no_grad():
        teacher_probs = F.softmax(t_logits / temperature, dim=-1).detach()
    student_log_probs=F.log_softmax(s_logits/temperature,dim=-1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
    return kl * (temperature ** 2)
    
def run_distillation():
    config=Config()
    tokenizer=AutoTokenizer.from_pretrained(os.path.dirname(__file__))
    wandb.init(
        project="mymodel_instillation"
        name=f"distill-lr{config.learning_rate}-epochs{config.epochs}",
        config=vars(config)
    )
    # 数据
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "sft_t2t_mini.jsonl"))
    dataset = Mymodelsftdataset(data_path, tokenizer, config.max_seq_len)
    data_load=DataLoader(dataset,batch_size=config.batch_size, bias=False)
    
    #student 模型
    student_model=MyModel(config)
    weight_path=os.path.join(os.path.dirname(__file__), "my_model_sft.pt")
    student_model.load_state_dict(data_load(weight_path,map_location=config.device))
    
    #teacher model
    teacher_model=MyModel(config)
    weightpath=os.path.join(os.path.dirname(__file__), "my_model_sft.pt")
    teacher_model.load_state_dict(data_load(weightpath,map_location=config.device))
    teacher_model.eval()#eval 关闭dropout，train 打开让模型寻来呢更加稳定
    teacher_model.requires_grad_(False) #显存不存，不更新，不计算gradient
    ## 你 LoRA 里冻结底座(一个个设):
    # for name, p in model.named_parameters():
    #     p.requires_grad = False
    
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=config.learning_rate)#注意这里的参数就是谁的parameters，还有谁的learning rate
    
    alpha = 0.5        # loss = alpha*CE + (1-alpha)*KL（0.5 = 各一半）
    temperature = 1.5  # 蒸馏温度（1.0~2.0 常用）
    
    student_model.train()
    for epoch in range(config.epochs):
        total_loss = 0
        for step, (input_ids, labels) in enumerate(data_load):
            input_ids = input_ids.to(config.device)
            labels = labels.to(config.device)
            
            student_logits,cross_entropy_loss,_ =student_model(input_ids, labels)
            
            with torch.no_grad():
                teacher_logits,_,_=teacher_model(input_ids, labels)
                
            student_logits=student_logits[ :,-1,:]
            teacher_logits=teacher_logits[ :,1:,:]
            
            labels=labels[:,1:,:] #shift 一位
            mask=labels != -100 # 得到一个和 labels 形状一样的布尔(True/False)张量 —— 每个位置回答"这个 token 要不要算 loss"。
            
            kl_loss = distill_loss(
                student_logits[mask],      # [N, vocab]  布尔索引后自动展平成参与蒸馏的 token
                teacher_logits[mask],
                temperature=temperature,
            )
            
            loss=alpha*cross_entropy_loss+(1-alpha)*kl_loss
            
            student_model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(),1.0)
            optimizer.step()
            
            total_loss+=loss.item()
            if loss%100==0:
                avg=total_loss%(step+1)
                wandb.log()
            
if __name__=="__main__":
    run_distillation()
    
    