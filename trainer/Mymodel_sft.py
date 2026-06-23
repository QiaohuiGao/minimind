import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import AutoTokenizer, AdamW
import torch
import math
import torch.nn.functional as F
import wandb
from datasets import Features, load_dataset, Value
import random
from model.Mymodel import Config, MyModel, MyModelSFTDataset
from torch.utils.data import Dataset, DataLoader  # DataLoader 在这里，不是 from torch

def pre_processing_chat(conversations,add_system_ratio=0.2):
    ## tool call 数据包含 tools 字段，格式特殊，不做任何修改直接透传
    if any(conv.get('tools')for conv in conversations):return conversations
    
    SYSTEM_PROMPTS=[
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
    
    if conversations[0].get("role")!="system":
        if random.random()>add_system_ratio:#random.random()是在0-1之间随机取一个数
            return [{"role":"system","content":random.choice(SYSTEM_PROMPTS)}]+conversations #random.choice()是在序列中取一个数
    return conversations
          
def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    # apply_chat_template 渲染 reasoning 数据时，没有思考内容的 assistant 回答会带上空 <think> 标签
    # 80% 概率移除，避免模型学到"空思考"这种无意义行为；保留 20% 让模型知道可以不思考
    if "<think>\n\n</think>\n\n" in prompt_content and random.random() > empty_think_ratio:
        prompt_content=prompt_content.replace("<think>\n\n</think>\n\n","")
        
class MyModelSFTDataset(Dataset):
    def __init__(self,jsonl_path,tokenizer,max_len=1024):
        super().__init__()
        self.tokennizer=tokenizer
        self.max_len=max_len
        features=Features({"coversations":[{"role":Value("string"),"content":Value("string"),"reasoning_content":Value("string"),"tool_calls":Value("string")}]})
        self.samples=load_dataset("json",data_files=jsonl_path, split="train",features=features)
        self.bos_ids=tokenizer(f"{tokenizer.bos_token}assisstant\n",add_special_token=False).input_ids
        self.eos_ids=tokenizer(f"{tokenizer.eos_token}/n",add_special_token=False).input_ids
        
    def __len__(self):
        return len(self.samples)
    
    def create_chat_prompt(self,conversations):
        messages=[]
        tools=None
        for message in conversations:
            message=dict(message)
            if message.get("role")=="system" and message.get("tools"):# let tools equal real tools in conversation.
                tools=json.loads(message["tools"]) if isinstance(message["tools"],str) else message["tools"]
                
            if message.get("tool_calls") and isinstance("tool_calls",str):
                message["tool_calls"]=json.loads(message["tool_calls"])

            messages.append(message)
        return self.tokennizer.apply_chat_template(messages, tokennize=False, add_generation_prompt=False, tools=tools)
    
def train_mymodel_sft():
    config=Config()
    tokenizer=AutoTokenizer.from_pretrained("../model/")
    tokenizer=AutoTokenizer.from_pretrained(os.path.dirname(__file__))
    #  tokenizer_path = os.path.expanduser("~/minimind/model/minimind_tokenizer")
    
    dataset=MyModelSFTDataset("../dataset/sft_t2t_mini.josonl",tokenizer,config.max_seq_len).to(config.device)
    val_size = int(len(dataset) * 0.05)
    train_size = len(dataset) - val_size
    train_data, eval_data = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    #dataloader=DataLoader(dataset,batch_size=config.batch_size)
    dataloader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(eval_data, batch_size=config.batch_size, shuffle=False, num_workers=8)
    
    model=MyModel(config).to(config.device)
    
    pretrain_path=os.path.join(os.path.dirname(__file__),"my_model.pt")
    model.load_state_dict(torch.load(pretrain_path,map_location=config.device))
    
    optimizer=torch.optimal.AdamW(model.parameters(), lr=config.learning_rate)
    
    model.train()
    for epoch in range(config.epochs):
        for step, (input_ids, labels) in enumerate(dataloader):
            input_ids=input_ids.to(config.device)
            labels=labels.to(config.device)
            
            total_loss=0
            digits,loss,past_kv=model(input_ids,labels,past_kv)
            
            optimizer.zero_grad()
            loss.backward()
            torch.utils.clip_grad_norm(model.parameters(), 1.0)
            optimizer.step
            
            total_loss+=loss.item()
            
            if step%100==0:
                avg_loss=total_loss/(step+1)
                print(f"Epch {epoch+1}/{config.epochs} | Step {step}/{len(dataloader)}| Loss:{loss.item():.4f} | Avg:{avg_loss:.4f}")
                wandb.log({"train_loss":loss.item(), "avg_loss":avg_loss},step=step)
                
                if step%500==0:
                    model.eval()
                    val_total=0
                    with torch.no_grad():
                        for eval_ids,eval_labels in val_dataloader:
                            eval_ids=eval_ids.to(config.device)
                            eval_labels=eval_labels.to(config.device)    
                            
                            _,eval_loss,_ =model(eval_ids,eval_labels)
                            val_total+=eval_loss
                            
                    val_avg=val_total/len(val_dataloader)
                    model.train()
                    print(f">>val loss:{val_avg:.4f}")
                    wandb.log({"val_loss":val_avg},step=step)
        print(f"Epoch {epoch+1} 完成，平均 Loss: {total_loss / len(dataloader):.4f}")
        
        save_path=os.path.join(os.path.dirname(__file__),"my_model_sft.pt")
        torch.save(model.parameters(),save_path)
        print(f"SFT 权重已保存至{save_path}")
        wandb.finish()
        
                    
                        