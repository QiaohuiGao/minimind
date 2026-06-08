# Day 1 Exam

## A. 代码阅读题

1. `MiniMindForCausalLM.forward` 中 `logits[..., :-1, :]` 的含义是什么？
   > **答：** 丢掉序列维度的最后一个位置。`...`=保留batch维, `:-1`=去掉最后位置, `:`=词表维全保留。因为最后一个位置没有下一个 token 可预测，所以丢掉。结果 shape 从 [B, T, 6400] 变为 [B, T-1, 6400]。

2. `labels[..., 1:]` 为什么要右移？
   > **答：** 用位置 t 的 logits 预测位置 t+1 的 token，所以 labels 从第 1 个位置开始取（第 0 个 token 没有上文来预测它）。`logits[:-1]` 和 `labels[1:]` 对齐后，位置 0 预测 token 1，位置 1 预测 token 2，以此类推。

3. `ignore_index=-100` 在 pretrain 和 SFT 中分别解决什么问题？
   > **答：** 告诉 CrossEntropy 跳过该位置不算 loss。Pretrain：忽略 PAD 位置（短序列补齐的填充，不是真实 token）。SFT：忽略 system/user 的 prompt 部分（只对 assistant 的回答算 loss，不想让模型学"怎么提问"）。

4. `Attention.forward` 中 `past_key_value` 的作用是什么？
   > **答：** 推理时的 KV Cache。缓存之前 token 的 K/V 向量，新 token 只需算自己的 K/V 拼到缓存后面，避免重复计算。只在推理时使用，训练时所有 token 一次输入不需要缓存。

5. `repeat_kv` 为什么只重复 K/V，不重复 Q？
   > **答：** GQA 的设计就是"Q 多 KV 少"（8Q:4KV）。Q 本来就有 8 个头不需要复制。K/V 只有 4 个头，要复制成 8 个才能和 Q 做矩阵乘法。KV 头少 = KV 投影参数省一半 + 推理时 KV Cache 省一半显存。Q 保持多头是为了保证表达力。

## B. 手写题

给定：

```text
input_ids = [BOS, 我, 爱, AI, EOS, PAD, PAD]
```
请写出 causal LM 中每个位置预测的目标 token。
output_ids=[我, 爱, AI, EOS, PAD, PAD]
再写出 padding 后的 labels，其中 PAD 对应 `-100`。
output_ids=[我, 爱, AI, EOS, -100, -100]


## C. 面试题

1. RoPE 为什么能支持比训练长度更长的外推？
2. RMSNorm 为什么比 LayerNorm 更轻？
3. 小模型中为什么 vocab size 不能无限大？
4. GQA 相比 MHA 的主要收益是什么？

## D. 实验题

运行 Day 1 deep dive 的随机模型 forward 实验，记录：

- logits shape[]
- loss 数值[]
- `log(6400)` 的值
回答：随机初始化模型的 loss 为什么和 `log(vocab_size)` 接近？

## E. 通过标准

你能不看代码讲出：

```text
input_ids -> embeddings -> blocks -> lm_head -> logits -> shifted CE loss
```

并能解释 `-100`、shift、RoPE、GQA。
