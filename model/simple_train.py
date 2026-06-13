"""
最简单的 Decoder-Only Transformer 预训练
- 没有 GQA、没有 MoE、没有 Flash Attention、没有 KV Cache
- 只有最核心的: Embedding → N × (Attention + FFN) → LM Head
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
        tokens = self.tokenizer(text, add_special_tokens=False,
                                max_length=self.max_length - 2, truncation=True).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        # pad
        pad_len = self.max_length - len(tokens)
        input_ids = tokens + [self.tokenizer.pad_token_id] * pad_len
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        # labels: pad 位置设为 -100（不计算 loss）
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels


# ==================== 3. 模型组件 ====================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).type_as(x) * self.weight


class SimpleAttention(nn.Module):
    """最简单的多头注意力，没有 GQA、没有 RoPE、没有 KV Cache"""

    def __init__(self, config: SimpleConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, S, _ = x.shape

        # 投影 + reshape 成多头
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, S, dim]
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # 注意力分数
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, heads, S, S]

        # causal mask: 上三角设为 -inf，防止看到未来 token
        causal_mask = torch.triu(torch.full((S, S), float('-inf'), device=x.device), diagonal=1)
        scores = scores + causal_mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # 加权求和 + 合并多头
        output = (attn_weights @ v).transpose(1, 2).reshape(B, S, -1)  # [B, S, hidden]
        return self.o_proj(output)


class SimpleFeedForward(nn.Module):
    """SwiGLU FFN"""

    def __init__(self, config: SimpleConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SimpleBlock(nn.Module):
    """一个 Transformer Block: Norm → Attention → 残差 → Norm → FFN → 残差"""

    def __init__(self, config: SimpleConfig):
        super().__init__()
        self.attn = SimpleAttention(config)
        self.ffn = SimpleFeedForward(config)
        self.norm1 = RMSNorm(config.hidden_size)
        self.norm2 = RMSNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))   # Pre-Norm + 残差
        x = x + self.dropout(self.ffn(self.norm2(x)))     # Pre-Norm + 残差
        return x


# ==================== 4. 完整模型 ====================

class SimpleTransformer(nn.Module):
    def __init__(self, config: SimpleConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([SimpleBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # weight tying: embedding 和 lm_head 共享权重
        self.lm_head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None):
        # input_ids: [B, S]
        x = self.embed(input_ids)           # [B, S, hidden]

        for block in self.blocks:
            x = block(x)                    # [B, S, hidden]

        x = self.norm(x)                    # [B, S, hidden]
        logits = self.lm_head(x)            # [B, S, vocab]

        loss = None
        if labels is not None:
            # logits[:-1] 预测 labels[1:]（next token prediction）
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, self.config.vocab_size),
                labels[:, 1:].reshape(-1),
                ignore_index=-100
            )
        return logits, loss

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ==================== 5. 训练 ====================

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

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # 训练循环
    model.train()
    for epoch in range(config.epochs):
        total_loss = 0
        for step, (input_ids, labels) in enumerate(dataloader):
            input_ids = input_ids.to(config.device)
            labels = labels.to(config.device)

            logits, loss = model(input_ids, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

            if step % 100 == 0:
                avg_loss = total_loss / (step + 1)
                print(f"Epoch {epoch+1}/{config.epochs} | Step {step}/{len(dataloader)} | Loss: {loss.item():.4f} | Avg: {avg_loss:.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} 完成, 平均 Loss: {avg_loss:.4f}")

    # 保存
    save_path = os.path.join(os.path.dirname(__file__), "simple_model.pt")
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存到: {save_path}")

    # 简单测试生成
    model.eval()
    prompt = "你好"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(config.device)
    with torch.no_grad():
        for _ in range(50):
            logits, _ = model(input_ids)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    print(f"生成: {tokenizer.decode(input_ids[0], skip_special_tokens=True)}")


if __name__ == "__main__":
    train()
