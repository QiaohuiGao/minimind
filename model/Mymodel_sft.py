import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import torch
import wandb
import random
from transformers import AutoTokenizer
from torch.utils.data import DataLoader,Dataset
from datasets import Features, Value, load_dataset
# MyModelSFTDataset 已在 Mymodel.py 中定义并完善，直接导入使用
# 包含：create_chat_prompt / generate_labels / __getitem__
from model.Mymodel import Config, MyModel, MyModelSFTDataset

def pre_processing_chat(conversation, add_system_ratio=0.2):
    # if conversation.get("role")=="system" or isinstance(conversation["tools"],str):
    #     return conversation
    if any(conv.get('tools') for conv in conversation): 
        return conversation
    
    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    
    if conversation[0].get("role")!="system":
        if random.random()<add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversation#直接加最简洁
    return conversation    

def post_processing_chat(prompt,empty_think_ratio=0.2):
    if '<think></think>' in prompt and random.random() >empty_think_ratio:
        prompt=prompt.replace("<think></think>","")
    return prompt

    
class Mymodelsftdataset(Dataset):
    def __init__(self, sftdata_path,tokenizer, max_len):
        self.max_len=max_len
        self.tokenizer=tokenizer
        #Features = 告诉库原始数据格式是什么（防止推断出错）defines the internal structure of a dataset. It is used to specify the underlying serialization format.
        feature=Features({'coversations':[{"role":Value('string'),'content': Value('string'), 'reasoning_content': Value('string'), 'tools': Value('string'), 'tool_calls': Value('string')}]})
        #conversation:
            #role:
            #content:
            #tools
            #reasoning_content
            #tool_calls
        self.datasets=load_dataset("json", data_file=sftdata_path, split="train", features=feature)
        #load_dataset(path, data_files=None, split=None, features=None, ...)
        self.bos_token=self.tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_token=self.tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
    
    def __len__(self):
        return len(self.datasets)
    
    def create_chat_prompt(self,conversations):
        # [{"role":"user","content":"你好"}, {"role":"assistant","content":"你好啊"}]
        #             ↓ create_chat_prompt
        # "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n你好啊<|im_end|>\n"
        messages=[]
        tools=None
        for message in conversations:
            message=dict(message)
            if message.get("role")=="system" and message.get("tools"):
                tools=json.loads(messages["tools"])if isinstance(message["tools"],str) else message["tools"]
            if message.get("tool_calls") and isinstance(message["tool_calls",str]):
                message["tool_calls"]=json.loads(message["tool_calls"])
            message.append(message)
        return self.tokenizer.apply_chat_prompt(messages, self.tokenizer, add_genetation_prompt=False, tools=tools)
        
    def generate_labels(self, input_ids):
        labels=[-100]*len(input_ids)
        
        i=0
        while i<len(input_ids):
            if input_ids[i:i+len(self.bos_token)]==self.bos_token:
                start=i+len(self.bos_token)
                end=start 
                while end<len(input_ids):
                    if input_ids[end:end+len(self.eos_token)]==self.eos_token:
                        break
                    end+=1#找本轮的结束标志/n
                    
                for j in range(start,min(len(self.eos_token+end),self.max_len)):
                    labels[i]=input_ids[j]#在本轮有效回答里给label
                    
                i=end+len(self.eos_token) if end<len(input_ids) else len(input_ids)
            else:
                i+=1
        return labels

    def __getitem__(self, index):
        sample=self.datasets[index]
        
        conversation=pre_processing_chat(sample["conversation"])
        
        prompt = post_processing_chat(prompt)#delete empty thinking process
              
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        
              
                
                
                
                
def train_mymodel_sft():
    # SFT 用更小的 learning rate，避免破坏 pretrain 学到的语言能力
    config = Config(learning_rate=1e-5)

    wandb.init(
        project="mymodel-sft",
        name=f"sft-lr{config.learning_rate}-epochs{config.epochs}",
        config=vars(config)
    )

    # tokenizer 在 model/ 目录下（tokenizer.json + tokenizer_config.json）
    tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(__file__))

    # Dataset 是纯 CPU 数据对象，不能 .to(device)；模型才需要 .to(device)
    dataset = MyModelSFTDataset("../dataset/sft_t2t_mini.jsonl", tokenizer, config.max_seq_len)

    val_size = int(len(dataset) * 0.05)
    train_size = len(dataset) - val_size
    train_data, eval_data = torch.utils.data.random_split(dataset, [train_size, val_size])

    dataloader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(eval_data, batch_size=config.batch_size, shuffle=False, num_workers=8)

    model = MyModel(config).to(config.device)

    # SFT 从 pretrain checkpoint 开始，不是随机初始化（关键步骤）
    pretrain_path = os.path.join(os.path.dirname(__file__), "my_model.pt")
    model.load_state_dict(torch.load(pretrain_path, map_location=config.device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)  # optim 不是 optimal

    model.train()
    for epoch in range(config.epochs):
        total_loss = 0  # 在 step 循环外初始化，才能累积整个 epoch 的 loss
        for step, (input_ids, labels) in enumerate(dataloader):
            input_ids = input_ids.to(config.device)
            labels = labels.to(config.device)

            # SFT 训练不用 KV cache（完整序列一次过），不传 past_key_values
            _, loss, _ = model(input_ids, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # nn.utils 不是 utils；带下划线的 in-place 版本
            optimizer.step()  # step 是方法调用，必须加 ()

            total_loss += loss.item()

            if step % 100 == 0:
                avg_loss = total_loss / (step + 1)
                print(f"Epoch {epoch+1}/{config.epochs} | Step {step}/{len(dataloader)} | Loss: {loss.item():.4f} | Avg: {avg_loss:.4f}")
                wandb.log({"train_loss": loss.item(), "avg_loss": avg_loss}, step=step)

                if step % 500 == 0:
                    model.eval()
                    val_total = 0
                    with torch.no_grad():
                        for eval_ids, eval_labels in val_dataloader:
                            eval_ids = eval_ids.to(config.device)
                            eval_labels = eval_labels.to(config.device)
                            _, eval_loss, _ = model(eval_ids, eval_labels)
                            val_total += eval_loss.item()  # .item() 把 tensor 转成 float，否则 tensor 累加会保留计算图

                    val_avg = val_total / len(val_dataloader)
                    model.train()
                    print(f">> val loss: {val_avg:.4f}")
                    wandb.log({"val_loss": val_avg}, step=step)

        print(f"Epoch {epoch+1} 完成，平均 Loss: {total_loss / len(dataloader):.4f}")

    # state_dict() 是参数字典（可序列化）；parameters() 是迭代器，不能直接 save
    save_path = os.path.join(os.path.dirname(__file__), "my_model_sft.pt")
    torch.save(model.state_dict(), save_path)
    print(f"SFT 权重已保存至 {save_path}")

    wandb.finish()  # 训练全部结束后调用，不能放在 epoch 循环里


if __name__ == "__main__":
    train_mymodel_sft()
