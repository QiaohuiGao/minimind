"""
现代 Decoder-Only Transformer 预训练模板
包含现代 LLM 的核心组件：RoPE + GQA + KV Cache + SwiGLU + RMSNorm
可作为 LLaMA / Qwen / Mistral 等主流模型的学习范式

GQA 统一覆盖三种 attention：
  num_kv_heads = num_heads     → MHA（每个 Q head 有自己的 KV）
  num_kv_heads = 1             → MQA（所有 Q head 共享 1 组 KV）
  num_kv_heads = num_heads / 2 → GQA（每几个 Q head 共享 1 组 KV）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import os
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


# ==================== 1. 配置 ====================

class SimpleConfig:
    vocab_size = 6400
    hidden_size = 256
    num_layers = 4
    num_heads = 4              # Query heads
    num_kv_heads = 2           # KV heads（GQA：每 2 个 Q head 共享 1 组 KV）
    head_dim = 64              # hidden_size // num_heads
    intermediate_size = 512
    max_seq_len = 256
    dropout = 0.1

    # 训练
    batch_size = 32
    learning_rate = 5e-4
    epochs = 1
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    data_path = os.path.expanduser("~/.cache/modelscope/hub/datasets/gongjy/minimind_dataset/pretrain_t2t_mini.jsonl")
    tokenizer_path = os.path.expanduser("~/minimind/model/minimind_tokenizer")


# ==================== 2. 数据集 ====================

class SimpleDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        with open(data_path, 'r') as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]['text']
        # tokenizer() 返回 BatchEncoding 对象（含 input_ids, attention_mask 等），用 .input_ids 取出 token ID 列表
        # truncation=True: 超过 max_length 的部分截断；max_length - 2: 留位置给 BOS/EOS
        tokens = self.tokenizer(text, add_special_tokens=False,
                                max_length=self.max_length - 2, truncation=True).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        # padding：将序列补齐到 max_length，使 batch 内所有样本等长
        pad_len = self.max_length - len(tokens)
        input_ids = tokens + [self.tokenizer.pad_token_id] * pad_len
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        # labels: pad 位置设为 -100（不计算 loss）
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels


# ==================== 3. RoPE（旋转位置编码）====================

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


# ==================== 4. repeat_kv（GQA 核心操作）====================

def repeat_kv(x, n_rep):
    """把 KV heads 复制 n_rep 次，匹配 Q heads 的数量
    x: [B, kv_heads, S, head_dim] → [B, kv_heads * n_rep, S, head_dim]
    MHA 时 n_rep=1（不复制），GQA 时 n_rep=num_heads/num_kv_heads"""
    if n_rep == 1:
        return x  # MHA：KV heads 数量已经等于 Q heads，不需要复制
    B, kv_heads, S, D = x.shape
    # 先扩展一个维度 → [B, kv_heads, n_rep, S, D] → 合并回 [B, kv_heads*n_rep, S, D]
    return x[:, :, None, :, :].expand(B, kv_heads, n_rep, S, D).reshape(B, kv_heads * n_rep, S, D)


# ==================== 5. 模型组件 ====================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).type_as(x) * self.weight


class Attention(nn.Module):
    """现代多头注意力：支持 GQA (MHA/GQA/MQA) + RoPE + KV Cache"""
    def __init__(self, config: SimpleConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = config.num_heads // config.num_kv_heads  # 每组 KV 被几个 Q head 共享

        # Q 投影：num_heads 个 head；KV 投影：num_kv_heads 个 head（GQA 的关键）
        self.q_proj = nn.Linear(config.hidden_size, config.num_heads * config.head_dim, bias=False) #[out,in]->[config.num_heads * config.head_dim,hidden_size]
        self.k_proj = nn.Linear(config.hidden_size, config.num_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_heads * config.head_dim, config.hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)

    def forward(self, x, freqs_cis, past_kv=None):
        """
        x: [B, S, hidden]
        freqs_cis: RoPE 频率
        past_kv: (cached_k, cached_v) 或 None（训练时为 None）
        返回: output, (new_k, new_v)
        """
        B, S, _ = x.shape
        # 投影 + reshape 成多头
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)     # [B, num_heads, S, head_dim]
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)  # [B, num_kv_heads, S, head_dim]
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)  # [B, num_kv_heads, S, head_dim]

        # RoPE：只对 Q 和 K 加位置编码（V 不加）
        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        # KV Cache：推理时拼接历史 KV，避免重复计算
        if past_kv is not None:
            cached_k, cached_v = past_kv
            k = torch.cat([cached_k, k], dim=2)  # [B, kv_heads, cached_len + S, head_dim]
            v = torch.cat([cached_v, v], dim=2)
        new_kv = (k, v)  # 保存当前 KV 供下一步使用

        # GQA：把 KV heads 复制到和 Q heads 一样多
        k = repeat_kv(k, self.n_rep)  # [B, num_heads, total_S, head_dim]
        v = repeat_kv(v, self.n_rep)  # [B, num_heads, total_S, head_dim]（只复制 heads 维度，seq_len 不变）

        # 注意力分数
        total_S = k.size(2) # [B, num_heads, total_S, head_dim] 第二维上所有token的总长度,取一个数字，是一个标量
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  #[B, num_heads, S, total_S]

        # causal mask：上三角填 -inf，下三角填 0，shape [S, total_S]
        causal_mask = torch.triu(torch.full((S, total_S), float('-inf'), device=x.device),
                                 diagonal=total_S - S + 1)
        scores = scores + causal_mask # [S, total_S] 广播到 [B, heads, S, total_S]，-inf 位置经 softmax 后精确变 0 → 看不到未来 token

        attn_weights = F.softmax(scores, dim=-1) #[B, heads, S, total_S]
        attn_weights = self.attn_dropout(attn_weights)#[B, num_heads, S, total_S]

        # 加权求和 + 合并多头
        output = (attn_weights @ v).transpose(1, 2).reshape(B, S, -1)
        return self.o_proj(output), new_kv


class FeedForward(nn.Module):
    """SwiGLU FFN"""

    def __init__(self, config: SimpleConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """一个 Transformer Block: RMSNorm → Attention → 残差 → RMSNorm → FFN → 残差"""
    def __init__(self, config: SimpleConfig):
        super().__init__()
        self.attn = Attention(config)
        self.ffn = FeedForward(config)
        self.norm1 = RMSNorm(config.hidden_size)
        self.norm2 = RMSNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, freqs_cis, past_kv=None):
        attn_out, new_kv = self.attn(self.norm1(x), freqs_cis, past_kv)
        x = x + self.dropout(attn_out)                 # Pre-Norm + 残差
        x = x + self.dropout(self.ffn(self.norm2(x)))   # Pre-Norm + 残差
        return x, new_kv


# ==================== 6. 完整模型 ====================

class SimpleTransformer(nn.Module):
    def __init__(self, config: SimpleConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # weight tying: embedding 和 lm_head 共享权重
        self.lm_head.weight = self.embed.weight

        # 预计算 RoPE 频率（注册为 buffer：跟着模型走，但不参与训练）
        freqs_cis = precompute_rope_freqs(config.head_dim, config.max_seq_len)
        self.register_buffer("freqs_cis", torch.view_as_real(freqs_cis))  # 存实数形式，兼容 save/load
        self._init_weights()

    def _get_freqs_cis(self):
        """从 buffer 恢复复数形式的 RoPE 频率"""
        return torch.view_as_complex(self.freqs_cis)

    def _init_weights(self):
        # 遍历所有子模块，给权重赋合理的初始值（让训练起步更快更稳定）
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)          # Linear: Xavier 均匀分布，根据输入输出维度自动算范围
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)  # Embedding: 正态分布，GPT 系列经验值

    def forward(self, input_ids, labels=None, past_key_values=None):
        """
        input_ids: [B, S]
        labels: [B, S] 或 None
        past_key_values: list of (k, v) per layer，推理用；训练时为 None
        """
        B, S = input_ids.shape
        x = self.embed(input_ids)  # [B, S, hidden]

        # 计算 RoPE 需要的位置偏移（有 KV Cache 时从 cached_len 开始）
        # past_key_values[0][0] shape: [B, kv_heads, cached_seq_len, head_dim]，.size(2) 取 cached_seq_len
        start_pos = past_key_values[0][0].size(2) if past_key_values else 0
        freqs_cis = self._get_freqs_cis()[start_pos: start_pos + S]  # 只取当前 token 对应位置的 RoPE 频率

        # 逐层通过 Transformer Block
        new_key_values = []
        for i, block in enumerate(self.blocks):
            past_kv = past_key_values[i] if past_key_values else None
            x, new_kv = block(x, freqs_cis, past_kv)
            new_key_values.append(new_kv) #shape[B, num_kv_heads, seq_len, head_dim]

        x = self.norm(x)                    # [B, S, hidden]
        logits = self.lm_head(x)            # [B, S, vocab]

        loss = None
        if labels is not None:
            # next token prediction: logits[:-1] 预测 labels[1:]（错开一位对齐）
            # reshape(-1, vocab): [B, S-1, vocab] → [B*(S-1), vocab]，拉平给 cross_entropy
            # reshape(-1):        [B, S-1] → [B*(S-1)]，-1 表示让 PyTorch 自动算维度大小
            # ignore_index=-100:  labels 里 -100（PAD 位）不计算 loss
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, self.config.vocab_size),
                labels[:, 1:].reshape(-1),
                ignore_index=-100
            )
        return logits, loss, new_key_values

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ==================== 7. 训练 ====================

def train():
    config = SimpleConfig()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)

    # 数据
    print(f"加载数据: {config.data_path}")
    dataset = SimpleDataset(config.data_path, tokenizer, config.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    print(f"数据条数: {len(dataset)}, batch 数: {len(dataloader)}")

    # 模型
    model = SimpleTransformer(config).to(config.device)
    print(f"模型参数量: {model.count_parameters():,} ({model.count_parameters() / 1e6:.1f}M)")
    print(f"设备: {config.device}")
    print(f"Attention 类型: num_heads={config.num_heads}, num_kv_heads={config.num_kv_heads} "
          f"→ {'MHA' if config.num_kv_heads == config.num_heads else 'MQA' if config.num_kv_heads == 1 else 'GQA'}")

    # 优化器：告诉 optimizer "你负责更新 model.parameters() 这些权重，学习率用 5e-4"
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # 训练循环
    model.train()  # 切换到训练模式（启用 Dropout 等），不是执行训练
    for epoch in range(config.epochs):
        total_loss = 0
        for step, (input_ids, labels) in enumerate(dataloader):  # 每次取一个 batch（32 条）
            # 把数据从 CPU RAM 搬到 GPU HBM（显存），和模型在同一设备上才能计算
            input_ids = input_ids.to(config.device)
            labels = labels.to(config.device)

            # 训练时不用 KV Cache（past_key_values=None）
            logits, loss, _ = model(input_ids, labels)

            optimizer.zero_grad()   # 清空上一轮的梯度
            loss.backward()         # 反向传播：算每个权重的梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪：防止梯度爆炸
            optimizer.step()        # 用梯度更新权重

            total_loss += loss.item()  # .item() 取出纯数字，不保留计算图

            if step % 100 == 0:
                avg_loss = total_loss / (step + 1)
                print(f"Epoch {epoch+1}/{config.epochs} | Step {step}/{len(dataloader)} | Loss: {loss.item():.4f} | Avg: {avg_loss:.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} 完成, 平均 Loss: {avg_loss:.4f}")

    # 保存
    save_path = os.path.join(os.path.dirname(__file__), "simple_model.pt")
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存到: {save_path}")

    # 简单测试生成（使用 KV Cache 加速推理）
    model.eval()
    prompt = "你好"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(config.device)
    generated_ids = input_ids[0].tolist()  # 收集所有 token
    past_key_values = None
    with torch.no_grad():
        for _ in range(50):
            logits, _, past_key_values = model(input_ids, past_key_values=past_key_values)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids.append(next_token.item())
            if next_token.item() == tokenizer.eos_token_id:
                break
            input_ids = next_token  # 有 KV Cache，只需要喂最新的 1 个 token
    print(f"生成: {tokenizer.decode(generated_ids, skip_special_tokens=True)}")


if __name__ == "__main__":
    train()
