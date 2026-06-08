# Day 1 Deep Dive: MiniMind 模型结构拆解

## 深挖问题 1：MiniMind 的最小 forward 路径是什么？
打开 `model/model_minimind.py`：
1. `MiniMindForCausalLM.forward`
2. `MiniMindModel.forward`
3. `MiniMindBlock.forward`
4. `Attention.forward`
5. `FeedForward.forward`
model(input_ids)                          # 你调用的入口
  → MiniMindForCausalLM.forward()         # 最外层：管 lm_head 和 loss
    → MiniMindModel.forward()             # 中间层：管 embedding + 所有 block
      → embed_tokens(input_ids)           # token id → 向量
      → 8× MiniMindBlock.forward()        # 每个 block：
        → RMSNorm → Attention.forward()   #   注意力（Q/K/V/RoPE/mask）
        → RMSNorm → FeedForward.forward() #   FFN（SwiGLU）
      → RMSNorm                           # 最终 norm
    → lm_head → logits                    # 向量 → vocab 得分
    → cross_entropy loss                  # 算 loss（训练时）


你要能追踪：

```text
input_ids
-> embed_tokens
-> dropout
-> N x MiniMindBlock
-> RMSNorm
-> lm_head
-> logits
-> cross_entropy loss
```

重点观察：

- `input_ids.shape == [batch, seq_len]`
- `hidden_states.shape == [batch, seq_len, hidden_size]`
- `logits.shape == [batch, seq_len, vocab_size]`

## 深挖问题 2：为什么 forward 里是这样算 loss？

代码位置：`MiniMindForCausalLM.forward`

核心逻辑：

```python
x = logits[..., :-1, :]
y = labels[..., 1:]
loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
```

理解方式：

- 位置 0 的 logits 预测 token 1。
- 位置 1 的 logits 预测 token 2。
- 最后一个位置没有下一个 token，所以丢掉。
- label 的第一个 token 没有被任何上文预测，所以丢掉。

`ignore_index=-100` 是训练中最重要的 mask 约定之一。pretrain 用它忽略 padding；SFT 用它忽略 user/system/prompt 部分。

> **笔记：loss 计算代码逐行拆解**
>
> `x = logits[..., :-1, :]` — `...`=保留batch维, `:-1`=丢最后一个位置(没有next token), `:`=词表维全保留 → shape: [B, T-1, 6400]
>
> `y = labels[..., 1:]` — `1:`=丢第一个位置(没有上文预测它) → shape: [B, T-1]
>
> `x.view(-1, x.size(-1))` — view=reshape(改变形状,数据不变)。把[B, T-1, 6400]压扁成[B*(T-1), 6400]。`-1`让PyTorch自动算该维大小。因为 `F.cross_entropy` 要求输入是 [N, C] (N个样本, C个类别)，不认识三维。
>
> `y.view(-1)` — 把[B, T-1]压扁成[B*(T-1)]，和x一一对应。
>
> `ignore_index=-100` — 标签为-100的位置跳过不算loss。Pretrain中PAD位置设-100；SFT中user/system部分设-100(只对assistant回答算loss)。

## 深挖问题 3：Attention 里 GQA 是怎么实现的？

看 `Attention.__init__`：

- `num_attention_heads`
- `num_key_value_heads`
- `n_rep = q_heads // kv_heads`

看 `repeat_kv`：

- K/V heads 先少算。
- attention 前把 K/V 复制到和 Q heads 对齐。

工业理解：

- 训练时省一部分参数和计算。
- 推理时 KV cache 更小。
- 对小模型通常是合理取舍。

> **笔记：GQA 实现细节**
> MiniMind 8Q:4KV，K/V 投影 768→384（省一半参数）。计算 attention 前用 `repeat_kv` 把 4 个 KV 头复制成 8 个对齐 Q 头，复制操作几乎零成本。
>
> **Q: KV Cache 是什么？为什么只在推理时用？**
> 推理时逐 token 生成，算第 N 个 token 需要前 N-1 个的 K/V。如果每次重算，生成第 100 个 token 要重算前 99 个的 K/V。KV Cache 把算过的 K/V 缓存在 GPU 显存里，每步只算新 token 的 K/V 拼到缓存后面。训练时所有 token 一次性输入，一个 forward 算完，不需要缓存。GQA 减少 KV 头数 = 缓存更小 = 同样显存能处理更长序列。
>
> **Q: 数据从哪里到哪里？**
> 硬盘(jsonl) → CPU 内存(DataLoader 读取拼 batch) → GPU 显存(.to(device)) → GPU 计算。GPU 擅长并行计算，CPU 负责数据准备。
>
> **Q: repeat_kv 后为什么要 transpose(1,2)？**
> 两步操作：① `repeat_kv` 把 4 个 KV 头复制成 8 个对齐 Q；② `transpose(1,2)` 交换 seq_len 和 heads 维度。
> 转置前 `[B, S, 8, 96]` → 转置后 `[B, 8, S, 96]`。heads 在前让 PyTorch 把 8 个 head 当 batch 维度并行计算：`Q[B,8,S,96] × K^T[B,8,96,S] = scores[B,8,S,S]`，一次矩阵乘法搞定全部 head。
>
> **Q: KV Cache 拼接过程？**
> `xk = torch.cat([past_key_value[0], xk], dim=1)` 在 seq_len 维度拼接历史 K 和新 K。
> 例：第 3 步生成 "AI"，缓存 `[B,2,4,96]`（"我""爱"）+ 新 `[B,1,4,96]`（"AI"）→ `[B,3,4,96]`。
> Q 只有 1 个新 token，但可以和所有历史 K 算注意力，因果性天然满足（历史都在前面）。
>
> **Q: 因果 Mask 的 `scores[:,:,:,-seq_len:]` 是什么？**
> `triu(1)` 生成上三角全 -inf 矩阵，加到 scores 上，softmax 后未来位置变 0。
> `-seq_len:` 是为了代码通用性：训练时 Q=K 等长，取整个 K 维度；KV Cache 推理时实际走不到这行。
>
> **Q: 残差连接在 MiniMind 哪里？**
> `MiniMindBlock.forward` 里的两个 `x = x + ...`：
> `x = x + self.attention(self.attention_norm(x))` — 残差1
> `x = x + self.ffn(self.ffn_norm(x))` — 残差2
> 8 个 block × 2 = 16 次残差，保证梯度直通。这也是 Pre-Norm 结构：Norm 在里面，残差在外面。

## 深挖问题 4：MoE 为什么有 aux loss？

看 `MOEFeedForward.forward`：

- gate 给每个 token 分配 expert。
- `topk_idx` 决定 token 去哪个 expert。
- `aux_loss` 鼓励 expert 负载更均衡。

工业风险：

- 没有负载均衡，部分 expert 可能被过度使用。
- 原生 PyTorch MoE 训练不一定快，因为 expert 分派有额外调度开销。

> **笔记：MoE aux loss 原理**
> Gate 网络给每个 token 分配 expert，容易"赢者通吃"——某个 expert 表现好就获得越多 token，其他 expert 越来越少被训练。aux_loss = (实际负载分布 · 路由概率分布) × num_experts × coef，惩罚不均匀分配，和主 loss 一起反向传播：`total_loss = CE_loss + aux_loss`。

## 今日小实验

写一个临时 Python 片段，不修改项目文件：

```bash
python - <<'PY'
import torch
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

config = MiniMindConfig(hidden_size=128, num_hidden_layers=2)
model = MiniMindForCausalLM(config)
input_ids = torch.randint(0, config.vocab_size, (2, 16))
labels = input_ids.clone()
out = model(input_ids, labels=labels)
print(out.logits.shape)
print(out.loss.item())
PY
```

你要解释：

- 为什么随机模型也有 loss？
- loss 大概应接近 `log(vocab_size)`，为什么？

> **笔记：**
> 随机权重也会产生 logits，CrossEntropy 照样能算。随机权重 → 预测概率近似均匀 1/6400 → loss = -log(1/6400) = log(6400) ≈ 8.76。这是纯瞎猜的理论最差水平，训练就是让 loss 从这里不断下降。
>
> **Q: Q/K/V 投影矩阵一开始是什么？**
> `nn.Linear` 在 `__init__` 时自动随机初始化权重，此时 xq/xk/xv 无意义。经过大量训练后，q_proj 学会投影成"我在找什么"(Query)，k_proj 学会"我有什么可被找到"(Key)，v_proj 学会"我要传递的内容"(Value)。加载预训练模型时，`load_state_dict` 覆盖随机值。
