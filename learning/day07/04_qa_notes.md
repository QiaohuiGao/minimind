# Day 7 问答笔记：GPU 基础与参数量估算

---

## Q1. nvidia-smi 里 `7807MiB / 97887MiB` 是什么意思？

**已用显存 / 总显存**，单位 MiB（Mebibyte）。

```
97887 MiB ÷ 1024 ≈ 95.6 GiB ≈ 96 GB（厂商标称）
7807 MiB ≈ 7.6 GB（当前占用）
```

MiB 是二进制单位（1 MiB = 1024 × 1024 bytes），和 MB（1000 × 1000）略有区别，所以看起来比 96 GB 略小，但对的上。

---

## Q2. FP32 全参数训练为什么每个参数需要 16 bytes？

两件事叠在一起：

**第一：FP32 = 32 bit = 4 bytes**，每存一个数占 4 bytes。

**第二：AdamW 训练时需要同时维护 4 项数据：**

| 项 | 谁产生 | 用途 |
|----|--------|------|
| weights | 模型本身 | 参数值，推理只需要这个 |
| gradients | backward pass | 告诉 weights 往哪更新 |
| m（一阶矩） | Adam 维护 | gradient 的滑动平均，记"方向" |
| v（二阶矩） | Adam 维护 | gradient 平方的滑动平均，记"波动大小" |

```
FP32 训练 + AdamW = 4 bytes × 4 项 = 16 bytes / 参数
```

SGD 没有 m 和 v，只需要 8 bytes/参数。

---

## Q3. Optimizer states 里的 m 和 v 具体是什么？

Adam 给每个参数维护两个滑动平均值，用来自适应调整步长：

**m（一阶矩，first moment）** — 记"方向"

```
m = β₁ × m_prev + (1 - β₁) × gradient
```

β₁ 默认 0.9，即 90% 继承上一步方向 + 10% 吸收新 gradient。平滑掉单步噪声。

**v（二阶矩，second moment）** — 记"波动大小"

```
v = β₂ × v_prev + (1 - β₂) × gradient²
```

β₂ 默认 0.999。v 大说明这个参数 gradient 一直剧烈，步长被压小；v 小说明平稳，步长放大。这是 Adam 能自适应 lr 的原因。

**实际更新：**

```
w = w - lr × m / (√v + ε)
```

**为什么 optimizer states 必须 FP32：** m 和 v 是累积量，每步做很小的增量更新，BF16 精度不够会截断丢失，积累误差后 optimizer 失效。

---

## Q4. 混合精度训练（BF16）和量化（Quantization）是一回事吗？

不是，目的和时机都不同：

| | 量化 | 混合精度训练 |
|--|------|------------|
| 目的 | 省推理显存 | 加速训练计算 |
| 发生时机 | 训练完之后 | 训练过程中 |
| 精度损失 | 有（INT4/INT8 会掉点）| 几乎没有 |
| 典型工具 | GPTQ、AWQ、llama.cpp | PyTorch AMP、`bf16=True` |

混合精度训练产出的还是正常精度的模型权重，没有"压缩"。量化是训练完之后对权重做压缩，用于部署推理。

混合精度的完整内存布局（每参数）：

| 存什么 | 精度 | bytes |
|--------|------|-------|
| weights 主副本 | FP32 | 4 |
| weights 计算副本 | BF16 | 2 |
| gradients | BF16 | 2 |
| m | FP32 | 4 |
| v | FP32 | 4 |
| **合计** | | **16** |

省的主要是 activations（∝ batch_size，数量大），optimizer states 还是 FP32，省不了多少。

---

## Q5. BF16 全称是什么？

**Brain Float 16**，Google Brain 团队设计的神经网络训练专用格式。

---

## Q6. GPU 里的 SRAM 是什么？

SRAM 是 GPU 芯片内部的缓存，类比 CPU 的 L1/L2 Cache。

存储层级从快到慢：

```
寄存器（Registers）      ← 最快，极小，每个线程私有
    ↓
SRAM（Shared Memory）    ← 很快（~19 TB/s），几百 KB，一组线程共享
    ↓
HBM / GDDR（显存）       ← 慢（~2-4 TB/s），几十 GB，全 GPU 共享
    ↓
系统 RAM
    ↓
硬盘
```

nvidia-smi 显示的、OOM 报的、`torch.cuda.memory_allocated()` 查的——全都是 HBM/GDDR 这层。SRAM 是 GPU 内部自动管理的，用户看不到也不用手动管。

Flash Attention 快的原因：手动控制分块，把数据尽量留在 SRAM 里算完，减少来回读写 HBM 的次数。

---

## Q7. 为什么说大模型推理的瓶颈是显存带宽？KV Cache 不能解决吗？

**decode 阶段每次只生成一个 token，但要把整个模型 weights 从显存读一遍。**

```
7B 模型 BF16 weights ≈ 14 GB
每生成一个 token → 搬 14 GB 数据 → 做极少量计算
```

计算单元（Tensor Core）还没"热身"，数据就读完了，算力严重闲置。类比：1000 台机器，每次只来 1 个零件的订单，传送带速度才是瓶颈。

**KV Cache 解决的是不同的问题：**

```
KV Cache 缓存：历史 token 的 K/V  ← 省了重新计算
模型 weights：                    ← 每步还是要全读一遍，省不掉
```

KV Cache 让你不用重算历史 token 的 K/V，但模型本身的 weights 每步都要从显存搬到计算单元，这个无法缓存。而且上下文越长，KV Cache 自身也越大，反而会额外加重带宽压力。

训练时 batch_size=128，相当于同时处理 128 个样本，矩阵够大，Tensor Core 才能满载。decode 只有 1 个 token，矩阵太小，带宽成了唯一瓶颈。

---

## Q8. minimind 63.91M 参数量是怎么算出来的？

模型配置：

```
hidden_size = 768, num_hidden_layers = 8
num_attention_heads = 8（Q头）, num_key_value_heads = 4（KV头，GQA）
head_dim = 96, intermediate_size = 2432
vocab_size = 6400, tie_word_embeddings = True
```

### Embedding 层

```
vocab_size × hidden_size = 6400 × 768 = 4,915,200 ≈ 4.9M
```

### 每个 Transformer Block

**Attention（Q/K/V/O）：**

```
Q: 768 × (8×96) = 768 × 768 = 589,824
K: 768 × (4×96) = 768 × 384 = 294,912
V: 768 × 384               = 294,912
O: 768 × 768               = 589,824
Attention 合计              = 1,769,472
```

**FFN（SwiGLU 三矩阵）：**

```
gate_proj: 768 × 2432 = 1,867,776
up_proj:   768 × 2432 = 1,867,776
down_proj: 2432 × 768 = 1,867,776
FFN 合计               = 5,603,328
```

**2 × RMSNorm：** `2 × 768 = 1,536`（可忽略）

**每 Block 合计：** `1,769,472 + 5,603,328 + 1,536 = 7,374,336 ≈ 7.4M`

### 汇总

```
Embedding：          4,915,200
8 × Block：         58,994,688
Final RMSNorm：            768
LM Head：    tied，不额外计算

总计 ≈ 63,910,656 ≈ 63.91M ✅
```

### 规律

```
Embedding：  vocab_size × hidden_size
每个 Block： ≈ 4 × hidden_size²          （Attention，GQA 时略小）
           + 3 × hidden_size × intermediate_size  （FFN，占 Block 的 75%+）
```

- `hidden_size` 翻倍 → 参数量约变 4 倍（矩阵两个维度都变）
- `intermediate_size` 对参数量影响最大（FFN 是主体）
- `vocab_size` 只影响 Embedding，一般不是主要贡献

---

## Q9. `with torch.no_grad()` 是什么意思？

告诉 PyTorch 这段代码里不需要计算梯度。

训练时每次 forward 都会记录中间结果，为 backward 做准备，这会额外占显存和时间。val 只是"看一下 loss 是多少"，不需要 backward，所以用 `no_grad` 跳过这些记录，更快更省显存。

```python
with torch.no_grad():
    _, val_loss, _ = model(val_ids, val_labels)  # 不记录梯度，只看结果
```

---

## Q10. val_loader 为什么 `shuffle=False`？

train 需要 shuffle 是为了让每个 batch 数据随机，避免模型按顺序"背"数据。

val 只是测量，不训练，数据顺序不影响结果，没必要 shuffle，保持原顺序就行。

---

## Q11. "每 500 步跑 val"具体是什么逻辑？

不是跑一个 batch，是跑完**整个 val_dataloader 所有 batch**，算出一个平均 val loss。

```
step=500 时：
  暂停训练
  把 val 集全部数据跑一遍（所有 batch）
  算平均 val loss
  恢复训练
step=501 继续...
```

**Train loss vs Val loss 的含义：**

| | 含义 |
|--|--|
| Train loss | 模型在见过、学过的数据上表现如何 |
| Val loss | 模型在从没见过的数据上表现如何 |

类比学生：train loss = 做过很多遍的作业题得分；val loss = 突击测验新题得分。

**情况 A — 真的在学习：**
```
Step 500:  train 5.0,  val 5.1   ← 差距小
Step 1000: train 4.5,  val 4.6   ← 一起往下降
Step 2000: train 3.8,  val 3.9   ← 还是紧跟着
```

**情况 B — overfit（在背答案）：**
```
Step 500:  train 5.0,  val 5.1   ← 开始正常
Step 2000: train 2.5,  val 4.5   ← gap 出现
Step 3000: train 1.8,  val 5.2   ← val 反而在升
```

---

## Q12. val 时用的是最新的 weights 吗？怎么传过去的？

不需要"传"，`model.eval()` 和 `model.train()` 用的是**同一个 model 对象**，weights 没有复制。

```python
model.train()   # 训练模式，weights 一直在更新
# ... 训练 500 步

model.eval()    # 切换模式，weights 没变，还是同一份最新的
# ... 跑 val，测当前 weights 在新数据上的表现

model.train()   # 切回训练模式，继续更新同一份 weights
```

`model.eval()` 只做两件事：关掉 dropout、关掉 BatchNorm 统计更新。weights 本身一直是同一份。

---

## Q13. 什么时候可以停止训练？模型训练好的标志是什么？

没有固定标志，但有几个实用判断依据：

**1. Val loss 不再下降（最重要）**

```
Step 2000: val 4.5
Step 2500: val 4.4
Step 3000: val 4.4
Step 3500: val 4.45  ← 横盘或微升
```

连续几个 checkpoint val loss 不动，说明这个模型容量 + 这份数据已经学到头了，继续跑意义不大。

**2. Train/Val gap 开始变大（overfit 信号）**

```
Step 1000: train 4.5, val 4.6  ← gap=0.1，正常
Step 3000: train 3.0, val 4.4  ← gap=1.4，overfit 开始
```

gap 明显拉大说明模型开始背训练数据，继续跑质量反而下降。

**3. 生成结果可读（直观标准）**

```
坏的结果：今今今今天天天气气气气  ← 重复，没学会
好的结果：今天天气晴朗，适合出门  ← 连贯，有效果
```

**对当前小模型（~8M）的实际期望：**

| 模型规模 | 合理的最终 val loss | 对应 PPL |
|---------|-------------------|---------|
| MyModel（~8M）| ~3.5–4.0 | ~33–55 |
| minimind 63M | ~2.0 | ~7.4 |

模型容量小，不可能降到和 63M 一样低，降到 3.5 左右差不多就到头了。

**停下的时机总结：**

```
优先看：val loss 连续 3–5 个 checkpoint 不再下降
其次看：train/val gap 明显拉大
最后看：生成结果是否基本可读
```

不需要等完全收敛，val loss 进入平台期就可以停，继续跑只是浪费时间和电。
