import json
from torch.utils.data import Dataset, DataLoader  # DataLoader 在这里，不是 from torch
import torch.nn as nn
from transformers import AutoTokenizer
import torch
import os
import math
import torch.nn.functional as F
import wandb
from datasets import Features,load_dataset,Value
import random

class Config:  # 不能继承 nn.Module，Config 只是普通数据类
    def __init__(self, **kwargs):
        self.vocab_size = 6400
        self.hidden_size = 512
        self.num_layers = 6
        self.num_heads = 4
        self.num_keyvalue_heads = 2   # GQA：2个KV头共享给4个Q头
        self.head_dim = 64
        self.intermediate_size = 1024
        self.max_seq_len = 1024
        self.dropout = 0.1

        self.batch_size = 32
        self.learning_rate = 5e-4
        self.epochs = 1
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        for k, v in kwargs.items():  # 支持 Config(hidden_size=512) 覆盖默认值
            setattr(self, k, v)


class MyModelDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len):
        self.dataset = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        # with open(data_path, 'r') as f:
        #     for line in f:
        #         self.dataset.append(json.loads(line))
        self.samples = load_dataset('json', data_files=data_path, split='train')
            

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        cur_text = self.dataset[idx]["text"]
        cur_token = self.tokenizer(
            cur_text,
            add_special_tokens=False,
            max_length=self.max_len - 2,  # self.max_len，不是 self.max_length
            truncation=True
        ).input_ids
        cur_token = [self.tokenizer.bos_token_id] + cur_token + [self.tokenizer.eos_token_id]
        pad_len = self.max_len - len(cur_token)
        padded_token = cur_token + pad_len * [self.tokenizer.pad_token_id]  # pad_token_id，不是 pad_token

        input_ids = torch.tensor(padded_token, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels


def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    B, kv_heads, S, D = x.shape
    return x[:, :, None, :, :].expand(B, kv_heads, n_rep, S, D).reshape(B, kv_heads * n_rep, S, D)


# 预计算 RoPE 的"角度表"：旋转角度 θ = 角频率 ω × 位置(把 position 当成 time)
# 角度只依赖 position 和第几对维度，与 token 内容无关，所以算一次反复用
def precompute_rope_freqs(head_dim, max_seq_len, base=10000.0):
    # 角频率 ω：每对(pair)维度一个，几何递减(秒针→分针→时针)
    # arange(0,head_dim,2) 每隔2取一个 → 共 head_dim//2 个 pair；shape [head_dim//2]
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))#（0，head_dim,2）每两个取一值求角频率
    t = torch.arange(max_seq_len).float()#求时间/也就是位置   # shape [max_seq_len]
    # outer：每个位置 × 每个 ω = 角度 θ；变量被复用，这里 freqs 已是"角度"不是"角频率"
    freqs = torch.outer(t, freqs)                    # shape [max_seq_len, head_dim//2]
    # polar(模长=1, 角度=θ) = e^{iθ} = cosθ + i·sinθ；这才是真正预计算并返回的东西
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64, [max_seq_len, head_dim//2]
    return freqs_cis

#sftdataset: conversations->rule out tools/tool calls,add system prompt to make it be sensitive to system requirements, and sperate the assistant answer

# 运行时把预计算的旋转真正作用到 Q/K 上：向量两两配对成复数 → 乘 e^{iθ} 完成旋转 → 拆回实数
def apply_rope(x, freqs_cis):
    B, H, S, D = x.shape  # B=batch, H=heads, S=seq_len, D=head_dim
    # 把向量最后一维 D 拆成 [D//2, 2] 即两两配对，再看成复数(实部,虚部)；
    # .float() 是为了在 fp32 下算旋转，避免 bf16/fp16 精度丢失
    x_complex = torch.view_as_complex(x.float().reshape(B, H, S, D // 2, 2))  # [B,H,S,D//2] complex
    # 取前 S 个位置的角度表；unsqueeze 两次 → [1,1,S,D//2] 以便对 B、H 做 broadcasting
    freqs = freqs_cis[:S].unsqueeze(0).unsqueeze(0).to(x.device)
    # 核心：复数乘单位复数 e^{iθ} = 把向量旋转 θ 角度(长度不变)，位置信息刻进方向里
    x_rotated = x_complex * freqs
    # 拆回实数 [B,H,S,D//2,2] → reshape 回 [B,H,S,D]，type_as 还原 dtype 给后续 attention 用
    return torch.view_as_real(x_rotated).reshape(B, H, S, D).type_as(x)


class RMSNorm(nn.Module):
    # 原来这个类的 __init__ 里写的是 Attention 的投影矩阵，完全错了
    # RMSNorm 只需要一个可学习的 scale weight
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return (x.float() / rms * self.weight).type_as(x)


class MyModelAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.n_rep = config.num_heads // config.num_keyvalue_heads
        self.q_proj = nn.Linear(config.hidden_size, config.num_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_keyvalue_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_keyvalue_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_heads * config.head_dim, config.hidden_size, bias=False)  # 输入是 num_heads*head_dim
        self.attn_dropout = nn.Dropout(config.dropout)

    def forward(self, x, freqs_cis, past_kv=None):
        B, S, _ = x.shape  # 参数名改为 x，加 .shape
        q = self.q_proj(x).view(B, S, self.config.num_heads, self.config.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.config.num_keyvalue_heads, self.config.head_dim).transpose(1, 2)  # k_proj，不是 q_proj
        v = self.v_proj(x).view(B, S, self.config.num_keyvalue_heads, self.config.head_dim).transpose(1, 2)  # v_proj，不是 q_proj

        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
        new_kv = (k, v)
        
        #  [B, num_keyvalue_heads, past_seq_len, head_dim]
        # └批量┘ └─KV 头数（GQA）─┘ └已缓存的长度┘ └每头维度┘
        # dim0        dim1            dim2         dim3
        
        # past_k: [B, kv_heads, past_seq_len, head_dim]
        #    k:   [B, kv_heads,      S,       head_dim]
        # ─────────────────────────────────────────── cat(dim=2)
        #    k:   [B, kv_heads, past_seq_len + S, head_dim]

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        total_S = k.size(2)
        # scores 只是 q@k，不乘 v；属性是 head_dim 不是 num_dim；self.config 不是 self.comfig
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.config.head_dim)

        causal_mask = torch.triu(
            torch.full((S, total_S), float("-inf"), device=self.config.device), 
            diagonal=total_S - S + 1 #保留 列号 - 行号 ≥ d 的位置，其余清零。
        )
        #已经缓存了 2 个 token（past=2），这步来 3 个新 query（S=3），所以 total_S = 5，d = 5 - 3 + 1 = 3。
        #             key位置→  0    1    2    3    4
        # query2(真实位置2)      ·    ·    ·   -inf -inf
        # query3(真实位置3)      ·    ·    ·    ·   -inf
        # query4(真实位置4)      ·    ·    ·    ·    ·
        scores = scores + causal_mask

        attn_weight = F.softmax(scores, dim=-1)
        # scores 形状 [B, num_heads, S, total_S]，dim=-1 是最后一维 = total_S（所有 key 那一维）。
        # 所以 softmax 是对每个 query 那一行的所有 key 做归一化：
        #                 key0   key1   key2   key3
        # query_i 这一行:  2.0    1.0    0.5    -inf     ← scores（含 mask）
        #                   │softmax(dim=-1)
        #                   ▼
        # attn_weight:    0.60   0.22   0.13    0.00     ← 加起来=1
        # mask 阶段:  把未来位置设成 -inf
        # softmax:   -inf → e^(-inf) → 0      （未来权重归零）
        #         其余正常分配，剩下的加起来还是 1
        attn_weight = self.attn_dropout(attn_weight)
        output = (attn_weight @ v).transpose(1, 2).reshape(B, S, -1)
        # 用这组占比去加权平均所有 value 向量：60% 的 key0 的 v + 22% 的 key1 的 v + …。
        # 所以 softmax 的输出本质是"这个 query 该从哪些 token 那里、各取多少信息"的配方。
        return self.o_proj(output), new_kv


class MyModelFFN(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MyModelBlocks(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size)  # Attention 前的 Norm
        self.ffn_norm = RMSNorm(config.hidden_size)   # FFN 前的 Norm，必须是两个独立实例
        self.attn = MyModelAttention(config)
        self.ffn = MyModelFFN(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, freqs_cis, past_kv=None):
        attn_out, new_kv = self.attn(self.attn_norm(x), freqs_cis, past_kv)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))  # self.norm(2) → self.ffn_norm(x)
        return x, new_kv


class MyModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([MyModelBlocks(config) for _ in range(config.num_layers)])  # MyModelBlocks(config)，不是 MyModelBlocks
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # embed.weight，不是 embed.weights

        freqs_cis = precompute_rope_freqs(config.head_dim, config.max_seq_len)
        self.register_buffer("freqs_cis", freqs_cis)  # 直接存复数 tensor

        self._init_weights()

    def forward(self, input_ids, labels=None, past_key_values=None):
        B, S = input_ids.shape  # input_ids 是 2D [B, S]，不是 3D
        x = self.embed(input_ids)

        start_pos = past_key_values[0][0].size(2) if past_key_values else 0
        freqs_cis = self.freqs_cis[start_pos: start_pos + S]  # +S 不是 +B

        new_key_values = []
        for i, block in enumerate(self.blocks):
            past_kv = past_key_values[i] if past_key_values else None
            x, new_kv = block(x, freqs_cis, past_kv)
            new_key_values.append(new_kv)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, self.config.vocab_size),  # self.config，不是 self.onfig
                labels[:, 1:].reshape(-1),   # labels 不是 logits
                ignore_index=-100
            )
        return logits, loss, new_key_values

    def _init_weights(self):  # 原来写的是 _init_weight，少了 s，和调用处不匹配
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)  # 要用带下划线的 in-place 版本
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)  # std=0.02，不是 0.0（0.0 等于不初始化）


def TrainMyModel():
    config = Config()
    #add wandb
    wandb.init(
        project="mymodel",
        name=f"hs{config.hidden_size}-layers{config.num_layers}-inter{config.intermediate_size}-lr{config.learning_rate}-maxlen{config.max_seq_len}",
        config=vars(config)  # 把所有超参记进去
    )
    
    tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(__file__)) #from_pretrained 接受的是目录路径,不管叫什么都无所谓
    full_dataset = MyModelDataset(
        "../dataset/pretrain_t2t_mini.jsonl",
        tokenizer=tokenizer,
        max_len=config.max_seq_len
    )
    #把数据分为train and eval
    val_size = int(len(full_dataset) * 0.05)
    train_size = len(full_dataset) - val_size

    train_data,eval_data= torch.utils.data.random_split(full_dataset, [train_size, val_size])

    dataloader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=8)
    val_dataloader=DataLoader(eval_data,batch_size=config.batch_size, shuffle=False, num_workers=8)
    model = MyModel(config).to(config.device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    model.train()
    for epoch in range(config.epochs):
        total_loss = 0
        for step, (input_ids, labels) in enumerate(dataloader):
            input_ids = input_ids.to(config.device)
            labels = labels.to(config.device)

            logits, loss, _ = model(input_ids, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

            if step % 100 == 0:
                avg_loss = total_loss / (step + 1)
                print(f"Epoch {epoch+1}/{config.epochs} | Step {step}/{len(dataloader)} | Loss: {loss.item():.4f} | Avg: {avg_loss:.4f}")
                wandb.log({"loss": loss.item(), "avg_loss": avg_loss}, step=step)
                if step%500==0:
                    model.eval()
                    val_total=0
                    with torch.no_grad(): #告诉 PyTorch 这段代码里不需要计算梯度。
                        for eval_id, eval_labels in val_dataloader:
                            eval_id=eval_id.to(config.device)
                            eval_labels=eval_labels.to(config.device)
                            _,val_loss, _=model(eval_id,eval_labels)
                            val_total+=val_loss.item()
                    val_avg=val_total/len(val_dataloader)                      
                    model.train()
                    print(f"Epoch {epoch +1}/{config.epochs} |step {step}/{len(dataloader)} | Loss: {loss.item():.4f} | Avg: {avg_loss:.4f} | Val: {val_avg:.4f}")
                    wandb.log({"loss": loss.item(),"avg_loss":avg_loss, "val_loss":val_loss}, step=step)
                else:
                    print(f"Epoch {epoch+1}/{config.epochs} | Step {step}/{len(dataloader)} | Loss: {loss.item():.4f} | Avg: {avg_loss:.4f}")
                    wandb.log({"loss": loss.item(), "avg_loss": avg_loss}, step=step)
                            
                    
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} 完成，平均 Loss: {avg_loss:.4f}")

    save_path = os.path.join(os.path.dirname(__file__), "my_model.pt")  # dirname，不是 direname
    #__file__ — Python 内置变量，表示当前脚本的路径，比如 /home/qiaohui/Project/minimind/model/Mymodel.py
    #os.path.dirname(...) — 取路径的"目录部分"，去掉文件名，得到 /home/qiaohui/Project/minimind/model
    #os.path.join(..., "my_model.pt") — 拼接路径，得到 /home/qiaohui/Project/minimind/model/my_model.pt
    
    torch.save(model.state_dict(), save_path)
    #model.state_dict() — 把模型所有的 weights 打包成一个字典，key 是层的名字，value 是对应的 tensor
    #torch.save(..., save_path) — 把这个字典序列化写入磁盘
    

    # eval：greedy 贪心生成
    model.eval()
    prompt = "how are you"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(config.device)  # return_tensors，不是 return_tensor
    generated_ids = input_ids[0].tolist()
    past_key_values = None
    with torch.no_grad():
        for _ in range(50):  # 最多生成 50 个 token
            logits, _, past_key_values = model(input_ids, past_key_values=past_key_values)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            if next_token.item() == tokenizer.eos_token_id:
                break
            generated_ids.append(next_token.item())
            input_ids = next_token  # 下一步只输入新生成的 token

    print(tokenizer.decode(generated_ids, skip_special_tokens=True))
    wandb.finish()


def evalPrompt():
    config = Config()
    model = MyModel(config).to(config.device)
    #加入pretrain的参数，先找绝对路径，然后再放weights

    save_path = os.path.join(os.path.dirname(__file__), "my_model.pt")
    model.load_state_dict(torch.load(save_path, map_location=config.device))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(__file__))
    prompt = ["你叫什么名字"]
    
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(config.device)
    generated_ids = prompt_ids[0].tolist()
    
    past_kv = None
    with torch.no_grad():
        for _ in range(100):
            logits, _, past_kv = model(prompt_ids, past_key_values=past_kv)
            logits_last = logits[:, -1, :].clone()  # shape: [B, vocab_size]
            # repetition penalty：对已经生成过的 token 降低 logits，防止无限重复
            # logits > 0 的 token 除以 penalty（降低），< 0 的乘以 penalty（压得更低）
            for token_id in set(generated_ids):
                if logits_last[0, token_id] > 0:
                    logits_last[0, token_id] /= 1.3
                else:
                    logits_last[0, token_id] *= 1.3
            # temperature=1.0：保持原始分布，越大越随机，越小越保守
            probs = torch.softmax(logits_last / 1.0, dim=-1)
            # multinomial 按概率随机采样，不是固定取最高（argmax）
            next_token = torch.multinomial(probs, num_samples=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
            generated_ids.append(next_token.item())
            prompt_ids = next_token
    print(tokenizer.decode(generated_ids, skip_special_tokens=True))
          

if __name__ == "__main__":
    # TrainMyModel()
    evalPrompt()
