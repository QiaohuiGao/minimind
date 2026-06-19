import json
from torch.utils.data import Dataset
import torch.nn as nn
from transformers import AutoTokenizer
import torch
from torch import DataLoader 
import os
import math
import torch.nn.functional as F

# load data and prepare data as input dataset:
# raw joson file->dataset_long_list
class Config(nn.Module):
    def __init__(self,  **kwargs):
        super().__init__()
        vocab_size = 6400
        hidden_size = 256
        num_layers = 4
        num_heads = 4
        head_dim = 64          # hidden_size // num_heads
        intermediate_size = 512
        max_seq_len = 256
        dropout = 0.1

        # 训练
        batch_size = 32
        learning_rate = 5e-4
        epochs = 1
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
#what the Dataset for? load data from path, prepare data for dataloader
#there three important function:
#__init__
    #datasets=
    #max_len
    #tokennizer
#__len__
#__getitem__(idx)
    # text=self.dataset[idx]["text"]
    # token=self.tokennizer(text,dd_special_tokens=False, max_length=self.max_length - 2, truncation=True).input_ids
    # add_token=[self.tokennizer.bos_token_id]+token+[self.tokennizer.eos_token_id]
    # pad_len=self.max_len-len(add_token)
    # tokens=add_token+pad_len*[self.tokennizer.pad]
    # transfer to tenor: torch.tensor(tokens,type=torch.long)
    # labels=input_ids.clone()
    # pad=-100
    # labels[input_ids==self.tokenizer.pad_token_id] = -100
    # return input_ids,labels 
class MyModelDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len):
        self.dataset=[]
        self.data_path=data_path
        self.tokenizer=tokenizer
        self.max_len=max_len
        with open(data_path, 'r') as f:
            for line in f:
                self.dataset.append(json.loads(line))
        # the other way:
        # self.samples = load_dataset('json', data_files=data_path, split='train')
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,idx):
        #this part have problem: i aready have data_path, why need unpack it here?
        # data_file=open(path,"r") if open(path,"r") else KeyError
        # for l in data_file:
        #     self.dataset
        cur_text=self.dataset[idx]["text"]
        cur_token=self.tokenizer(cur_text,add_special_tokens=False,
                                max_length=self.max_length - 2, truncation=True).input_ids
        cur_token= [self.tokenizer.bos_token_id] + cur_token + [self.tokenizer.eos_token_id]
        pad_len=self.max_len-len(cur_token)
        padded_token=cur_token+pad_len*[self.tokenizer.pad_token]
        
        #transfer to ternsor
        input_ids=torch.tensor(padded_token,dtype=torch.long)
        labels = input_ids.clone()
        
        # labels: pad 位置设为 -100（不计算 loss）
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels
    
def repeat_kv(x,n_rep):
    if n_rep==1:
        return x
    B, kv_heads, S, D=x.shape
    return x[: ,: ,None,: ,: ].expand(B, kv_heads, n_rep, S, D).reshape(B, kv_heads*n_rep, S, D)

def precompute_rope_freqs(head_dim, max_seq_len, base=10000.0):
    """预计算 RoPE 的频率，只算一次，所有层共享"""
    # 每对维度一个频率：freq_i = 1 / (base ^ (2i / dim))
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    # 每个位置 × 每个频率 → [max_seq_len, head_dim/2]
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)
    # 转成复数形式：e^(i*theta) = cos(theta) + i*sin(theta)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # [max_seq_len, head_dim/2] 复数
    return freqs_cis

def apply_rope(x, freqs_cis):
    """对 Q 或 K 施加 RoPE"""
    # x: [B, heads, S, head_dim] → 把相邻两个维度配对，变成复数
    B, H, S, D = x.shape
    x_complex = torch.view_as_complex(x.float().reshape(B, H, S, D // 2, 2))  # [B, H, S, D/2] 复数
    # freqs_cis: [S, D/2] → [1, 1, S, D/2] 广播
    freqs = freqs_cis[:S].unsqueeze(0).unsqueeze(0).to(x.device)
    # 复数乘法 = 旋转
    x_rotated = x_complex * freqs
    # 复数 → 实数，恢复原 shape
    return torch.view_as_real(x_rotated).reshape(B, H, S, D).type_as(x)

class RMSNorm(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.q_proj=nn.Linear(config.hidden_size,config.hidden_size, bias=False)
        self.k_proj=nn.Linear(config.hidden_size,config.hidden_size, bias=False)
        self.v_proj=nn.Linear(config.hidden_size,config.hidden_size, bias=False)
        self.o_proj=nn.Linear(config.hidden_size,config.hidden_size, bias=False)
    
#what attention class need to do:
#generate q,k,v,o matrix
#q,k,v->proj(x).view.transpose(1,2)
#apply positional embeddings to q,k
#if save_kv: cached_k,cached_v=past_key_values
#new_k,new_v=torch.cat((cached_k,k),dim=2)
#repeat_kv heads if GQA: k,v= repeat_kv(k/v,self.n_rep)
#calculate scores=(q@k.transpose(1,2)/math.sqrt(head_dim))*new_v for each head
#mask:total_s=k.size(2)->casual_mask=torch.triu(torch.full(s,total_s), float("-inf"),device=self.config.device),
#                                                   diagonal= total_s-s+1                                              
#scores+=causual_mask
#attn=F.softmax(scores,dim=-1)
#
class MyModelAttention(nn.Module):
    def __init__(self,config:Config):
        super().__init__()
        self.config=config
        self.n_rep=config.num_heads//config.num_keyvalue_heads #几个头共享一个kv cache
        self.q_proj=nn.Linear(config.hidden_size, config.num_heads*config.head_dim)
        self.k_proj=nn.Linear(config.hidden_size, config.num_keyvalue_heads*config.head_dim)
        self.v_proj=nn.Linear(config.hidden_size, config.num_keyvalue_heads*config.head_dim)
        self.o_proj=nn.Linear(config.num_keyvalue_heads*config.head_dim, config.hidden_size)
        self.attn_dropout = nn.Dropout(config.dropout)     
        
    def forward(self,input_ids,freqs_cis, past_kv=None):
        B,S,_=input_ids
        q=self.q_proj(input_ids).view(B,S,self.config.num_heads,self.config.head_dim).transpose(1,2)
        k=self.q_proj(input_ids).view(B,S,self.config.num_keyvalue_heads,self.config.head_dim).transpose(1,2)
        v=self.q_proj(input_ids).view(B,S,self.config.num_keyvalue_heads,self.config.head_dim).transpose(1,2)
        
        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        if past_kv is not None:
            past_k,past_v=past_kv
            k=torch.cat((past_k,k),dim=2)#[b,heads_num, total_S,head_dim]->cat previous k information
            v=torch.cat((past_v,v),dim=2)
        new_kv = (k, v)
        
        k = repeat_kv(k, self.n_rep)  # [B, num_heads, total_S, head_dim]
        v = repeat_kv(v, self.n_rep) #（[B, num_kv_heads, S, head_dim]， [B, num_kv_heads, S, head_dim]）
        
        total_S=k.size(2) #total_s
        scores=(q @ k.transpose(-2,-1)/math.sqrt(self.comfig.num_dim))*v #[B,heads,s,total_s]
        
        casual_mask=torch.triu(torch.full((S,total_S),float("-inf"),device=self.config.device),diagonal=total_S-S+1)
        scores=scores+casual_mask  #[s,total_s]
        
        attn_weight=F.softmax(scores,dim=-1)# [S, total_S] 广播到 [B, heads, S, total_S
        attn_weight=self.attn_dropout(attn_weight)
        output=(attn_weight@v).transpose(1,2).reshape(B,S,-1)
        return self.o_proj(output),new_kv
               

class MyModelFFN(nn.Module):
    def __init__(self, config:Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
    def forward(self,x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
                
class MyModelBlocks(nn.Module):
    def __init__(self, config:Config):
        super().__init__()
        self.norm=RMSNorm(config)
        self.attn=MyModelAttention(config)
        self.ffn=MyModelFFN(config)
        self.dropout=nn.Dropout(config.dropout)
        
    def forward(self,x, freqs_cis, past_kv=None):#after positonal embedding
        attn_out,new_kv=self.attn(self.norm(x),freqs_cis, past_kv)
        x = x + self.dropout(attn_out)#一次残差链接
        x = x +self.dropout(self.ffn(self.norm(2)))
        return x,new_kv
        
        
#model->multiple blocks and also compine the blocks with
#embedding 
#ModuleList->how many blocks or layers
#Last Norm
#lm_head
class MyModel(nn.Module):
    def __init__(self, config:Config):
        super().__init__()
        self.config=config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks=nn.ModuleList([MyModelBlocks for _ in range(config.num_layers)])
        self.norm=RMSNorm(config)
        self.lm_head=nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight=self.embed.weights
        # 预计算 RoPE 频率（注册为 buffer：跟着模型走，但不参与训练）
        freqs_cis = precompute_rope_freqs(config.head_dim, config.max_seq_len)
        self.register_buffer("freqs_cis", torch.view_as_real(freqs_cis))  # 存实数形式，兼容 save/load       
        self._init_weights()
        
    def forward(self, input_ids, labels=None, past_key_values=None):
        batch,seq_len,_=input_ids
        x = self.embed(input_ids) # [B, S, hidden]
        #calculate the RoPE need switch logits
        start_pos=past_key_values[0][0].size(2) if past_key_values else 0
        freqs_cis = self._get_freqs_cis()[start_pos: start_pos + batch]
        new_key_values=[]
        for i, block in enumerate(self.blocks):
            past_kv=past_key_values[i] if past_key_values else None
            x,new_kv=block(x,freqs_cis,past_kv)
            new_key_values.append(new_kv)
        x=self.norm(x)
        logits=self.lm_head(x)
        loss=None
        if labels is not None:
            loss=F.cross_entropy(
                logits[:,:-1, :].reshape(-1,self.onfig.vocab_size),
                logits[:,1:].reshape(-1),
                ignore_index=-100
            )
        return logits, loss, new_key_values
        
    
    def _init_weight(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform(module.weight)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight,mean=0.0, std=0.0)
                
    
#train model key steps:
#tokennizer->i skipped this one
#opmitimizer
#datasets->dateloader
#model->move to device
#model train
#define epoch 
#for each epoch:
#   for step, (input_ids, labels) in enumerate(DataLoader(config)):
#   input_ids,labels -> move to devices
#   loss=0
#   logits, loss= model(input_ids,labels) what is logits look like here?
#   optmizer.zero_to_grad() clear the optimizer
#   loss.backward() backword
#   optimizer.step()
#   total_loss+=loss.item()
#   avg_loss=total_loss/(step+1)
#   print avg_loss

def TrainMyModel():
    config=Config()
    tokenizer=AutoTokenizer.from_pretrained("./minimind_tokenizer")
    dataset=Dataset("../dataset/pretain_t2t_mini.jsonl",max_len=2048)
    datasets=DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=8)
    model=MyModel(config).to(config.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate)
    model.train()
    for epoch in range(config.epoch):
        total_loss=0
        for step, (input_ids, labels) in enumerate(datasets):
            input_ids=input_ids.to(config.device)
            labels=labels.to(config.device)
            
            logits, loss=model(input_ids,labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.utils.clip_grad_norm(model.parameters(),1.0) # 梯度裁剪：梯度总范数超过 1.0 时等比缩小，防止梯度爆炸
            optimizer.step()
            
            total_loss+=loss.item()
            
            if step%100==0:
                avg_loss=total_loss/(step+1)
                print(f" Epoch {epoch+1}/{config.epochs}| Step{step}/{len(datasets)}|Loss:{loss.item():.4f}|Avg: {avg_loss:.4f}")
                
            avg_loss=total_loss/(step+1)
            print(f"Epoch {epoch+1}完成，平均 Loss:{avg_loss:.4f}")
            
    #save
    save_path=os.path.join(os.path.direname(__file__), "my_model.pt")
    torch.save(model.state_dict(),save_path)
    
    #eval
    model.eval()
    prompt="how are you"
    input_ids=tokenizer(prompt, return_tensor="pt").input_ids.to(config.device)
    generated_ids=input_ids[0].tolist()
    past_key_value=None
    with torch.no_grad():
        for _ in range():
                 
            
            
            
            
    
    
    
    
    
