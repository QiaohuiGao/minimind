# Day 1 实验记录：超参数对训练的影响（控制变量法）

> **目标**：通过"每次只改一个变量"的对照实验，亲手感受各超参数对 loss 和训练的影响。
> **方法**：控制变量法——固定其他所有参数，只变一个，对比 loss 曲线。

---

## 第 0 步：准备小数据子集（让每个实验只跑几分钟）

跑满整份数据太慢，先切一个小子集用于快速实验：

```bash
cd ~/Project/minimind
head -n 50000 dataset/pretrain_t2t_mini.jsonl > dataset/pretrain_tiny.jsonl
```
> 5 万条样本，batch=128 时约 390 步/epoch，单次实验约 3~5 分钟。

之后所有实验都用 `--data_path ../dataset/pretrain_tiny.jsonl`。

**基准配置（不变的部分）**：`epochs=1, batch_size=128, hidden_size=768, num_hidden_layers=8`

---

## 实验指令清单

> 跑前 `cd ~/Project/minimind/trainer` 并确认 `(.venv)`。加 `--use_wandb` 可在网页看曲线；不加则看终端 loss。

### 组 A：学习率 learning_rate（最重要，先做这组）
```bash
python train_pretrain.py --data_path ../dataset/pretrain_tiny.jsonl --epochs 1 --learning_rate 1e-5   # A1 极小
python train_pretrain.py --data_path ../dataset/pretrain_tiny.jsonl --epochs 1 --learning_rate 5e-4   # A2 默认
python train_pretrain.py --data_path ../dataset/pretrain_tiny.jsonl --epochs 1 --learning_rate 5e-3   # A3 偏大
python train_pretrain.py --data_path ../dataset/pretrain_tiny.jsonl --epochs 1 --learning_rate 5e-2   # A4 极大(可能发散)
```

### 组 B：批大小 batch_size（固定 lr=5e-4）
```bash
python train_pretrain.py --data_path ../dataset/pretrain_tiny.jsonl --epochs 1 --batch_size 16    # B1
python train_pretrain.py --data_path ../dataset/pretrain_tiny.jsonl --epochs 1 --batch_size 128   # B2
python train_pretrain.py --data_path ../dataset/pretrain_tiny.jsonl --epochs 1 --batch_size 512   # B3
```

### 组 C：模型大小（固定 lr=5e-4, batch=128）
```bash
python train_pretrain.py --data_path ../dataset/pretrain_tiny.jsonl --epochs 1 --hidden_size 256 --num_hidden_layers 4   # C1 小模型
python train_pretrain.py --data_path ../dataset/pretrain_tiny.jsonl --epochs 1 --hidden_size 768 --num_hidden_layers 8   # C2 默认
python train_pretrain.py --data_path ../dataset/pretrain_tiny.jsonl --epochs 1 --hidden_size 1024 --num_hidden_layers 12 # C3 大模型
```

> 💡 看每个实验启动时打印的 `Model Params: xxM` 记下参数量；训练中用另一个终端 `nvidia-smi` 看显存。

---

## 📝 观察表（理论预期值 — GPU driver 故障日补充）

> 以下为基于 ML 理论的预期值，非实测数据。数据条件：50k 条样本，batch=128，约 390 步/epoch，max_seq_len=340。
> loss 初始值约为 ln(6400)≈8.76（完全随机），模型经初始化后通常从 3.5～4.5 开始下降。

| 编号 | 改变的变量 | 理论预期 | step100 loss | step300 loss | 末尾 loss | 现象 | 参数量 | 核心规律 |
|------|-----------|---------|-------------|-------------|---------|------|-------|---------|
| A1 | lr=1e-5 | 步子太小，几乎不动 | ~3.8 | ~3.5 | ~3.3 | 极平稳但几乎不降 | — | lr 太小：gradient 方向正确但步长太小，几百步内看不出明显收敛 |
| A2 | lr=5e-4 | 默认，收敛最好 | ~3.2 | ~2.6 | ~2.1 | 平稳稳降 | — | 甜点：够快又不震荡 |
| A3 | lr=5e-3 | 前期快但会震荡 | ~3.0 | ~2.6 | ~2.3 | 轻微震荡 | — | 偏大：初期收敛快，后期在 loss valley 附近来回跳 |
| A4 | lr=5e-2 | 发散 | >4.0 或 NaN | NaN | NaN | 发散/NaN | — | 极大：gradient 太猛，权重直接冲出 loss valley，loss 上升或变 NaN |
| B1 | batch=16 | 曲线极抖，但3125步后可能最低 | ~3.3（抖） | ~2.8（抖） | ~2.0（3125步末） | 高度震荡 | 更少 | 小 batch：gradient 噪声大→曲线抖，但 update 次数多（3125步 vs 390步） |
| B2 | batch=128 | 默认，平滑稳定 | ~3.2 | ~2.6 | ~2.1（390步末） | 平稳 | 中 | 甜点：噪声和 update 次数平衡 |
| B3 | batch=512 | 极平滑但97步就结束 | ~2.8（97步末） | —（已结束） | ~2.8（97步末） | 极平稳 | 最多 | 大 batch：每步梯度精准→曲线平滑，但 epoch 内 update 次数少（只有97步） |
| C1-h256 | hidden=256, L=4 | 快收敛但有容量上限 | ~3.0 | ~2.5 | ~2.3 | 平稳，后段趋于平 | 8.33M / ~2G | 小模型：收敛快，但参数少→容量上限低，loss 降到某处就停了 |
| C(基准) | hidden=768, L=8 | 默认，平衡 | ~3.2 | ~2.6 | ~2.1 | 平稳 | ~63M / ~8G | 数据量和模型大小匹配的甜点 |
| C2-h512 | hidden=512, L=6 | 介于C1和默认之间 | ~3.1 | ~2.5 | ~2.2 | 平稳 | ~30M / ~5G | 中等模型：容量和收敛速度折中 |
| C3-h1024 | hidden=1024, L=12 | 50k数据喂不饱，欠拟合 | ~3.4 | ~3.0 | ~2.8 | 平稳但降得慢 | ~112M / ~14G | 大模型欠拟合：参数多但数据少，大部分参数没被有效训练到 |
| D1-nah4 | heads=4, kv=4 | head_dim=192，每头表达深 | ~3.1 | ~2.5 | ~2.2 | 平稳 | ~63M | 头少但深：前期收敛快（每头信息丰富），但多样性不足（只4个视角） |
| D(基准) | heads=8, kv=4 | head_dim=96，平衡 | ~3.2 | ~2.6 | ~2.1 | 平稳 | ~63M | 甜点：表达深度和多样性平衡 |
| D2-nah16 | heads=16, kv=4 | head_dim=48，每头太浅 | ~3.3 | ~2.8 | ~2.3 | 轻微抖动 | ~63M | 头多但浅：Q/K 点积精度下降，注意力分数不准 |
| D3-nah32 | heads=32, kv=4 | head_dim=24，极浅 | ~3.5 | ~3.0 | ~2.6 | 抖动 | ~63M | 过多头：head_dim=24 太小，无法有效计算相似度，性能明显下降 |
| E1-MQA | nkv=1, Q=8 | KV参数最少，质量略降 | ~3.3 | ~2.7 | ~2.3 | 平稳 | 略小 | MQA：所有Q头共用1套KV，推理省显存，但KV投影子空间多样性损失大 |
| E2-nkv2 | nkv=2, Q=8 | 略好于MQA | ~3.2 | ~2.65 | ~2.2 | 平稳 | 略小 | 每4个Q头共用1套KV，折中 |
| E(基准) | nkv=4, Q=8 | 默认GQA | ~3.2 | ~2.6 | ~2.1 | 平稳 | ~63M | LLaMA/minimind的经验甜点：显存省一半，质量几乎无损 |
| E3-MHA | nkv=8, Q=8 | 标准MHA，参数最多 | ~3.1 | ~2.55 | ~2.0 | 平稳 | 略大 | 每Q头有独立KV：质量最高，但KV cache是GQA的2倍（推理贵） |
| F1-hd48 | head_dim=48 | 低于自然值96，注意力精度低 | ~3.4 | ~2.9 | ~2.5 | 轻微抖动 | 略小 | head_dim太小：Q·K^T 计算精度差，attention score 不准，曲线抖 |
| F2-hd64 | head_dim=64 | 介于48和96之间 | ~3.3 | ~2.75 | ~2.3 | 轻微抖动 | 略小 | 接近但不如默认 |
| F(基准) | head_dim=96 | 自然值（768÷8），最稳 | ~3.2 | ~2.6 | ~2.1 | 平稳 | ~63M | 与hidden_size自然匹配的甜点 |
| F3-hd128 | head_dim=128 | 参数增多，50k或欠拟合 | ~3.2 | ~2.6 | ~2.1 | 平稳 | 略大 | head_dim大：每头更深，短实验效果相近，长训练可能更好 |
| F4-hd256 | head_dim=256 | 注意力层大幅增参，欠拟合 | ~3.3 | ~2.7 | ~2.3 | 平稳 | 明显更大 | head_dim极大：投影矩阵巨大，50k数据训不动这么多参数 |
| G1-ffn1024 | inter=1024 | FFN太窄，容量不足 | ~3.4 | ~2.9 | ~2.5 | 平稳 | 较小 | FFN是"记忆库"，太窄则模型存不下语言规律，loss存在上限 |
| G2-ffn1536 | inter=1536 | 容量略不足 | ~3.3 | ~2.75 | ~2.3 | 平稳 | 较小 | 接近但不如默认 |
| G(基准) | inter=2432 | 默认（768×π≈2432） | ~3.2 | ~2.6 | ~2.1 | 平稳 | ~63M | π倍 hidden_size 是经验公式，FFN宽度与Attention的信息处理量匹配 |
| G3-ffn3072 | inter=3072 | FFN更宽，容量增 | ~3.1 | ~2.55 | ~2.0 | 平稳 | 较大 | 容量更大，50k数据下还在合理范围，loss略好 |
| G4-ffn4096 | inter=4096 | 参数增多，50k或刚够 | ~3.1 | ~2.55 | ~2.0 | 平稳 | 较大 | FFN参数多，50k可能轻微欠拟合，但与G3差异不显著 |
| G5-ffn6144 | inter=6144 | 参数太多，50k明显欠拟合 | ~3.2 | ~2.6 | ~2.1 | 平稳 | 很大 | FFN过宽：参数数量远超50k数据能喂养的量，收益递减甚至退步 |
| H(基准) | dropout=0.0 | loss最低，预训练不需要正则 | ~3.2 | ~2.6 | ~2.1 | 平稳 | — | 预训练大数据不会过拟合，dropout只是浪费信息 |
| H1-drop0.05 | dropout=0.05 | 影响很小 | ~3.25 | ~2.65 | ~2.15 | 平稳 | — | 轻微dropout：每步随机丢弃5%激活值，稍微减慢学习 |
| H2-drop0.1 | dropout=0.1 | 明显慢于基准 | ~3.3 | ~2.7 | ~2.25 | 平稳 | — | 每步丢10%信息，等效于每步"看少了10%的数据" |
| H3-drop0.2 | dropout=0.2 | 显著变慢 | ~3.5 | ~2.9 | ~2.5 | 平稳但高 | — | dropout 0.2在预训练阶段是过度正则化 |
| H4-drop0.3 | dropout=0.3 | 严重拖慢收敛 | ~3.7 | ~3.1 | ~2.7 | 平稳但很高 | — | 每步丢弃30%信息，梯度信号严重不足，相当于用更小的有效batch训练 |
| I1-theta1e4 | rope_theta=1e4 | 短序列下影响有限 | ~3.25 | ~2.65 | ~2.15 | 平稳 | — | theta小→高频旋转分量衰减快→对长距离位置编码精度差，但seq=340时影响小 |
| I2-theta1e5 | rope_theta=1e5 | 接近基准 | ~3.2 | ~2.62 | ~2.1 | 平稳 | — | theta略小，在340长度内几乎无差别 |
| I(基准) | rope_theta=1e6 | 默认，seq=340下的甜点 | ~3.2 | ~2.6 | ~2.1 | 平稳 | — | 1e6是LLaMA 3的设计值，覆盖很长上下文 |
| I3-theta1e7 | rope_theta=1e7 | 与基准几乎无差 | ~3.2 | ~2.6 | ~2.1 | 平稳 | — | theta更大→位置编码变化更慢→有利于更长序列，但340长度内看不出差异 |
| J(基准) | Dense（无MoE） | 50k小数据下Dense最稳 | ~3.2 | ~2.6 | ~2.1 | 平稳 | ~63M | 所有参数每步都被训练，梯度信号充分 |
| J1-e4k1 | MoE 4专家 top1 | 总参数更大但每步只激活1/4 | ~3.3 | ~2.7 | ~2.3 | 轻微抖动 | ~63M+FFN×3 | 路由（router）需要学习把token分配给合适专家，初期抖动 |
| J2-e8k1 | MoE 8专家 top1 | 8专家只激活1个，利用率极低 | ~3.4 | ~2.8 | ~2.4 | 抖动 | 更大 | 每步只有1/8的FFN参数被更新，50k数据下大部分专家欠训练 |
| J3-e8k2 | MoE 8专家 top2 | 激活2个，好于k1 | ~3.3 | ~2.75 | ~2.3 | 轻微抖动 | 更大 | top-2比top-1更充分利用专家，但仍不如Dense（50k数据少） |
| J4-e16k2 | MoE 16专家 top2 | 16专家极大欠拟合 | ~3.5 | ~3.0 | ~2.7 | 抖动 | 很大 | 16专家中每次只激活2个（12.5%），50k数据根本不够喂饱，欠拟合最严重 |

---

## 🤔 实验后问题——理论解答

### 组 A（学习率）

**1. 哪个 lr 的 loss 降得最快又稳？**
A2（lr=5e-4）。这是默认值，在 minimind 这个规模（63M参数）和这个数据量（50k条）下经过调试的甜点。既能在 390 步内充分收敛，又不会震荡。

**2. lr 太小（A1）会怎样？为什么？**
Loss 曲线几乎是平的，400 步内只降了不到 0.5（从~3.8降到~3.3）。原因：AdamW 每步更新量 = lr × gradient方向，lr=1e-5 时步子小到几乎不动，gradient 方向正确但走不了多远。需要几十倍的步数才能达到 A2 的效果。

**3. lr 太大（A4）会怎样？**
很快发散，loss 上升然后变 NaN。原因：gradient × lr 的步长太大，权重直接冲出 loss valley（loss 曲面的低谷区域），数值爆炸。AdamW 的梯度裁剪（grad_clip=1.0）有一定保护但抵不住 lr=5e-2。

**4. 学习率过大 / 过小 / 合适 的表现总结：**

| 情况 | 现象 | 直觉 |
|------|------|------|
| lr 过小 | 曲线极平，几乎不降 | 迈步太小，走不到低谷 |
| lr 合适 | 稳定下降，最终 loss 最低 | 步子合适，能走到低谷底部 |
| lr 偏大 | 下降但震荡，最终 loss 较高 | 步子大→来回跳过低谷 |
| lr 极大 | 发散/NaN | 直接冲出 loss 曲面，崩掉 |

---

### 组 B（批大小）

**1. batch 越大，loss 曲线更平滑还是更抖？**
越大越平滑。原因：gradient = 对 batch 内所有样本的平均，batch 越大→平均越精准→噪声越小→曲线越平滑。batch=16 时每步只看 16 个样本，gradient 里包含大量随机噪声，曲线剧烈抖动。

**2. batch 改变时，单步速度、显存怎么变？**
- 单步速度（samples/sec）：batch 从 16 → 128 会显著提升（GPU 并行利用率提高）；从 128 → 512 提升越来越小（GPU 已饱和）
- 显存：activations 大小正比于 batch_size，batch 翻 4 倍 → activations 用的显存翻 4 倍

**3. batch 变大时，lr 是否也该跟着变？**
是的，理论上 batch 扩大 k 倍→ lr 乘以 √k（√k scaling rule）。batch=128 → 512（×4），lr 应从 5e-4 → 1e-3。但这只是起点，实际跑会发现 batch=512 的 epoch 内只有 97 步，update 次数太少，即使 lr 放大效果也有限。

---

### 组 C（模型大小）

**1. 模型越大，参数量、显存、每步速度怎么变？**
- 参数量：8.33M → 30M → 63M → 112M，几乎按 hidden_size² 增长（因为 attention 和 FFN 的矩阵维度都是 hidden_size 的倍数）
- 显存：包括 weights + gradients + optimizer states（AdamW 存 m/v 两份）≈ weights × 4，加上 activations
- 每步速度：慢得多。矩阵乘法量 ∝ hidden_size²，模型从 C1 到 C3 每步时间大约慢 10-15 倍

**2. 在相同步数下，大模型的 loss 一定更低吗？**
不一定，50k 数据 + 390 步的条件下，C3（112M）的 loss 反而可能高于 C1（8.33M）。原因：大模型参数多，需要更多数据和步数才能充分训练。这就是"underfitting"（欠拟合）——不是模型太弱，是训练量不够，大部分参数几乎没被有效更新。

**3. 小模型(C1)的 loss 会不会很快就降不动了？**
会。C1 hidden_size=256 → 容量上限低，loss 降到约 2.3 附近就趋于平台（capacity bottleneck）。即使再训 10 个 epoch 也很难降到 2.0 以下。相比之下 C2（63M）还有下降空间。

---

### 组 D（num_attention_heads）

**核心规律：** head_dim = hidden_size ÷ num_heads，num_heads 控制的是**表达深度 vs 多样性**的 trade-off。
- heads 太少（D1=4）：head_dim=192 很深，但只有 4 个视角，多样性不足，最终表达能力有上限
- heads 太多（D3=32）：head_dim=24 太浅，Q/K 点积计算精度低，每个 head "近视"，注意力质量差
- 768 维模型的经验甜点：8 heads（head_dim=96）

---

### 组 E（num_key_value_heads / MQA·GQA·MHA）

**核心规律：** 减少 KV heads 主要影响**推理显存和速度**，对训练 loss 影响有限。
- MQA（nkv=1）：KV cache 最小，推理最快省显存，loss 略高于默认 GQA
- GQA（nkv=4，默认）：LLaMA 3/minimind 的折中方案，loss 几乎无损而推理 KV cache 是 MHA 的 1/2
- MHA（nkv=8）：质量最高，但推理 KV cache 是 GQA 的 2 倍，不值得
- 关键：多样性来自 Q heads 不同，不需要 KV 也各不相同；减少 KV 只是损失了"KV 投影子空间的多样性"，这部分对 loss 影响很小

---

### 组 F（head_dim）

**核心规律：** head_dim 越小 → Q·K^T 点积精度越低 → attention score 越不准 → loss 越高且曲线越抖。
- 默认 head_dim=96（768÷8）是与 hidden_size 自然匹配的值
- head_dim < 64 时明显变差；head_dim > 128 时增加了参数但 50k 数据下可能欠拟合
- 实验中的异常（如 hd=64 短期好于 hd=48）可能是噪声，不要过度解读

---

### 组 G（intermediate_size / FFN 宽度）

**核心规律：** FFN 是 Transformer 的"记忆库"，intermediate_size 决定每层能存储多少知识。
- 太窄（inter=1024）：容量不足，loss 降到某处就停了
- 默认（inter=2432 = 768×π）：经验公式，FFN 宽度与 Attention 信息处理量匹配
- 更宽（inter=3072~4096）：容量增加，50k 数据下轻微收益
- 过宽（inter=6144）：参数量远超数据能喂养的量，收益递减，显存压力增大

---

### 组 H（dropout）

**核心规律：** 预训练阶段不应该用 dropout。
原因：dropout 的目的是防止过拟合（overfitting）。预训练用的是超大数据集（minimind 用百亿 token），每条数据几乎只见一遍，根本来不及过拟合。加 dropout 只是在每步随机丢弃信息，等于浪费了宝贵的训练信号，纯负收益。
SFT/RLHF 阶段数据量小，才可能需要轻微 dropout（如 0.05~0.1）。

---

### 组 I（rope_theta）

**核心规律：** rope_theta 的影响只在**序列长度超过其"有效覆盖范围"时**才明显。
- max_seq_len=340 时，theta 从 1e4 到 1e7 的差异几乎不可见（340 远小于所有 theta 值的有效覆盖长度）
- theta 越大 → 位置编码旋转越慢 → 有效覆盖的上下文长度越长（LLaMA 3 用 5e5 支持 128k context）
- 如果 max_seq_len 增大到 4096 或以上，theta=1e4 会明显变差（远距离 token 的位置编码开始混叠）

---

### 组 J（MoE vs Dense）

**核心规律：** 50k 小数据场景下，Dense 优于 MoE。MoE 的优势要在大规模数据（千亿 token+）下才体现。
- Dense：每步所有参数都被梯度更新，50k 数据能充分训练
- MoE：总参数量翻几倍，但每 token 只激活 top-k 个专家（如 e8k1 → 只有 12.5% 的 FFN 被更新），50k 数据下大部分专家严重欠训练
- 额外负担：router（路由网络）需要学习如何分配 token，早期 routing 不稳定→ loss 抖动 + aux_loss 增加噪声
- MoE 真正的优势：相同**计算量**（FLOPs/step）下，总参数量可以更大（稀疏激活），适合"compute 固定但想要超大模型容量"的大规模训练场景

---

### 综合问题

**如果只给你 5 分钟训练时间，怎么配？**
- 小模型（hidden=256, L=4）：参数少 → 每步快
- 大 batch（256~512）：GPU 利用率高，steps 虽少但每步并行高效
- 合适的 lr（5e-4）：不用调
- 不开 MoE、dropout=0
- 目标：用少量步数快速收敛到模型容量上限

**如果追求最终效果最好（时间不限），怎么配？**
- 大模型（hidden=768 或以上，L=8~12）：容量大，loss 下限低
- 多 epoch（5~10 epoch）+ 更多数据
- 合适 batch（128~256）+ lr 从 5e-4 开始配 cosine decay + warmup
- GQA（nkv=4），head_dim 用自然值（hidden//heads）
- inter_size 用默认（π × hidden）
- dropout=0（预训练阶段）
- 数据量是最关键因素：模型大了必须配上更多数据，否则欠拟合

---

## 实验小贴士

- **不用每个都跑满**：loss 曲线的"形状"在前几百步就能看出差别，看够了可 `Ctrl+C` 停。
- **发散的样子**：loss 不降反升、或变成 `nan`/`inf`，就是 lr 太大了。
- **wandb 对比**：加 `--use_wandb`，多条曲线会自动叠加（运行名含 epochs/batch/lr，改这三个能区分；改 hidden_size 时运行名不区分，靠本表记录）。
- 做完记得把有价值的结论回填到 `04_qa_notes.md`。
