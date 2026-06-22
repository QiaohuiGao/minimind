# MyModel 超参数实验记录

> 原则：每次只改一个变量，其他保持不变，记录结果后再改下一个。
> Baseline 是当前默认配置，每次实验都和它对比。

---

## Baseline 配置

```python
hidden_size       = 256
num_layers        = 4
num_heads         = 4
num_keyvalue_heads= 2
head_dim          = 64
intermediate_size = 512
max_seq_len       = 256
dropout           = 0.1
batch_size        = 32
learning_rate     = 5e-4
epochs            = 1
```

---

## 实验记录表

| # | 改动 | 改动值 | 参数量 | Step 500 Loss | Step 2000 Loss | 最终 Avg Loss | Val Loss | 现象/备注 | 结论 |
|---|------|--------|--------|--------------|----------------|--------------|---------|---------|------|
| 0 | Baseline | — | ~8M | | 5.08 | ~4.19@5900 | | loss 稳定下降，无 LR schedule | 基准 |
| 1 | LR schedule | cosine decay | ~8M | | | | | | |
| 2 | max_seq_len | 256→512 | ~8M | | | | | | |
| 3 | intermediate_size | 512→1024 | | | | | | | |
| 4 | hidden_size | 256→512 | | | | | | | |
| 5 | num_layers | 4→6 | | | | | | | |
| 6 | dropout | 0.1→0.0 | | | | | | | |
| 7 | learning_rate | 5e-4→1e-3 | ~8M | | | | | | |
| 8 | batch_size | 32→128 | ~8M | | | | | | |
| 9 | num_keyvalue_heads | 2→4（MHA）| ~8M | | | | | | |
| 10 | | | | | | | | | |

---

## 参数量估算公式（填表前先算）

```
Embedding：   vocab_size × hidden_size
每个 Block：  4 × hidden_size²（Attention）
            + 3 × hidden_size × intermediate_size（FFN）
共 num_layers 个 Block

Baseline 约：
  Embedding：  6400 × 256 = 1.6M
  每 Block：   4×256² + 3×256×512 = 0.26M + 0.39M = 0.65M
  4 Blocks：   2.6M
  总计：       ≈ 4.2M（加上 norm、lm_head）
```

---

## 各参数改大的预期影响（实验前先填猜测）

| 参数 | 改大后预期 | 实测是否符合 |
|------|-----------|-------------|
| learning_rate | 前期快，后期可能 overshoot | |
| max_seq_len | loss 可能略升（更难预测长距离），但泛化更好 | |
| intermediate_size | loss 更低，但慢 | |
| hidden_size | loss 更低，参数量大幅增加，慢 | |
| num_layers | loss 更低，但梯度更难传 | |
| dropout | loss 略升，过拟合风险降低 | |
| batch_size | 曲线更平滑，速度不一定变 | |

---

## 实验结论汇总（跑完后填）

| 结论 | 证据（实验编号）|
|------|--------------|
| | |
| | |
| | |
