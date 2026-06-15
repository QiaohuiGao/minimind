# Day 1 实验指令集（控制变量 · 自动跑1000步停）

> 配套观察表见 [05_experiment_log.md](05_experiment_log.md)。
> 方案：直接用 mini 数据，每个实验加 `--max_steps 1000` **自动跑到1000步停**（不用手动Ctrl+C）；每次只改一个变量，其余用默认。

---

## 跑之前（每次开新终端都做）

```bash
cd ~/Project/minimind/trainer
source ../.venv/bin/activate          # 确认前缀出现 (.venv)
```

**自动停**：`--max_steps 1000` 会在第1000步自动结束并打印"达到 max_steps"。
**统一规则**：同组所有实验都用 `--max_steps 1000` 才公平。`--log_interval 100` 每100步打印一次loss。

---

## 组 A：学习率 learning_rate（最重要，先做）

> **为什么这样选？** 真实 LLM 预训练（GPT-2: 2.5e-4，GPT-3: 6e-4，LLaMA: 3e-4）
> 都集中在 1e-4 ～ 1e-3，用 2-3x 步长调参。之前的 0.05/0.005/0.0005（10x 跨度）
> 超出实际范围：0.05 必然发散，0.0005 在 1000 步内几乎看不出变化。

> 💡 **强烈建议用 `--run_name` 自己命名**，否则当"被测参数=默认值"时(如 lr=5e-4)自动命名会丢失该参数、甚至和别的实验重名。

```bash
# A1 保守 lr（稳定但收敛慢）
python train_pretrain.py --epochs 1 --max_steps 1000 --log_interval 100 --learning_rate 1e-4 --use_wandb --run_name "A1-lr1e-4"

# A2 主流 lr（大多数论文的选择）
python train_pretrain.py --epochs 1 --max_steps 1000 --log_interval 100 --learning_rate 3e-4 --use_wandb --run_name "A2-lr3e-4"

# A3 默认 lr（MiniMind 默认值）
python train_pretrain.py --epochs 1 --max_steps 1000 --log_interval 100 --learning_rate 5e-4 --use_wandb --run_name "A3-lr5e-4"

# A4 激进 lr（看是否开始抖动，但不会直接发散）
python train_pretrain.py --epochs 1 --max_steps 1000 --log_interval 100 --learning_rate 1e-3 --use_wandb --run_name "A4-lr1e-3"
```

## 组 B：批大小 batch_size（固定 lr=5e-4）

> **为什么这样选？** batch_size 实验允许更大跨度（效果本来就不如 lr 敏感），
> 32/128/512 覆盖"小-中-大"三个典型区间，能清晰看到梯度噪声 vs 训练稳定性的权衡。

```bash
# B1 小 batch（梯度噪声大，更新频繁）
python train_pretrain.py --epochs 1 --max_steps 1000 --log_interval 100 --learning_rate 5e-4 --batch_size 32 --use_wandb

# B2 默认 batch
python train_pretrain.py --epochs 1 --max_steps 1000 --log_interval 100 --learning_rate 5e-4 --batch_size 128 --use_wandb

# B3 大 batch（梯度稳定，但同步耗时）
python train_pretrain.py --epochs 1 --max_steps 1000 --log_interval 100 --learning_rate 5e-4 --batch_size 512 --use_wandb
```

## 组 C：模型宽度 hidden_size（固定 lr=5e-4, batch=128, layers=8）

> **控制变量**：只改 `hidden_size`，`num_hidden_layers` 固定为 8（MiniMind 默认）。
> 基准线（hidden_size=768）已有，不用重跑，只跑下面三条新的。
> 256/512/1024 分别是默认值的 1/3、2/3、4/3 倍，覆盖"窄-中窄-宽"三档。

```bash
# C1 窄模型（hidden_size=256，~4M 参数）
python train_pretrain.py --epochs 1 --max_steps 1000 --log_interval 100 --learning_rate 5e-4 --hidden_size 256 --num_hidden_layers 8 --use_wandb --run_name "C1-h256"

# C2 中窄模型（hidden_size=512，~14M 参数）
python train_pretrain.py --epochs 1 --max_steps 1000 --log_interval 100 --learning_rate 5e-4 --hidden_size 512 --num_hidden_layers 8 --use_wandb --run_name "C2-h512"

# C3 宽模型（hidden_size=1024，~46M 参数）
python train_pretrain.py --epochs 1 --max_steps 1000 --log_interval 100 --learning_rate 5e-4 --hidden_size 1024 --num_hidden_layers 8 --use_wandb --run_name "C3-h1024"
```

> 启动时记下打印的 `Model Params: xxM`（参数量）；另开终端 `nvidia-smi` 看显存。

---

## 架构参数实验（D~J）— 现在全部可从命令行改

> 这些参数原本写死在 MiniMindConfig 里，现已全部暴露成 CLI 参数并传入模型 + 记进 wandb。
> ⚠️ 约束（违反会在启动时报清晰错误）：`num_attention_heads` 必须能被 `num_key_value_heads` 整除；`head_dim` 必须为偶数；`max_position_embeddings ≥ max_seq_len`；MoE 时 `1≤num_experts_per_tok≤num_experts`。

### 组 D：注意力头数 num_attention_heads（默认=8，固定 kv=4）
> **控制变量**：只改 `num_attention_heads`，其余全默认。基准（nah=8）已有，跑下面三条凑成 `4→8→16→32` 翻倍序列。
> 参数量几乎不变，纯粹对比头数影响（头越多每头 head_dim 越小：768/4=192, /8=96, /16=48, /32=24）。
> ⚠️ 取值约束：必须是 `kv=4` 的倍数，且 `768/nah` 为偶数 → 合法值 4/8/12/16/24/32。
```bash
python train_pretrain.py --epochs 1 --max_steps 1000 --log_interval 100 --num_attention_heads 4  --num_key_value_heads 4 --use_wandb --run_name "D1-nah4"
python train_pretrain.py --epochs 1 --max_steps 1000 --log_interval 100 --num_attention_heads 16 --num_key_value_heads 4 --use_wandb --run_name "D2-nah16"
python train_pretrain.py --epochs 1 --max_steps 1000 --log_interval 100 --num_attention_heads 32 --num_key_value_heads 4 --use_wandb --run_name "D3-nah32"
```
> 看：相同总宽度下头数对 loss 的影响（头越多每头维度越小）；总参数量基本不变。

### 组 E：GQA 分组 num_key_value_heads（默认=4，固定 Q头=8）
> **控制变量**：只改 `num_key_value_heads`，其余全默认。基准（nkv=4=GQA 2:1）已有，跑下面三条凑齐 `1→2→4→8` 完整序列。
> ⚠️ 取值约束：必须能整除 `num_attention_heads`(=8) → 合法值只有 1/2/4/8。
> nkv=1=MQA(8:1), nkv=2=GQA(4:1), nkv=4=GQA(2:1,默认), nkv=8=MHA(1:1)。LLaMA-2 用 GQA。
```bash
python train_pretrain.py --epochs 1 --max_steps 1000 --log_interval 100 --num_attention_heads 8 --num_key_value_heads 1 --use_wandb --run_name "E1-MQA"
python train_pretrain.py --epochs 1 --max_steps 1000 --log_interval 100 --num_attention_heads 8 --num_key_value_heads 2 --use_wandb --run_name "E2-nkv2"
python train_pretrain.py --epochs 1 --max_steps 1000 --log_interval 100 --num_attention_heads 8 --num_key_value_heads 8 --use_wandb --run_name "E3-MHA"
```
> 看：KV 头越少越省 KV-cache 显存，质量损失通常很小；MQA(1) → MHA(8) 全梯度对比。

### 组 F：每头维度 head_dim（默认=96，必须偶数）
> **控制变量**：只改 `head_dim`，其余全默认。基准（head_dim=96）已有，跑下面四条凑成 `48/64/96/128/256` 梯度。
> ⚠️ 取值约束：**只需偶数**（RoPE 要求）；head_dim 与 hidden_size 解耦（q_proj/o_proj 自动映射），所以可自由取值。
> 参考：GPT-2 用 64，LLaMA 用 128。256 是大头，看收益是否饱和。注意 head_dim 越大，q/k/v/o 投影参数越多、越慢。
```bash
python train_pretrain.py --epochs 1 --max_steps 1000 --log_interval 100 --head_dim 48  --use_wandb --run_name "F1-hd48"
python train_pretrain.py --epochs 1 --max_steps 1000 --log_interval 100 --head_dim 64  --use_wandb --run_name "F2-hd64"
python train_pretrain.py --epochs 1 --max_steps 1000 --log_interval 100 --head_dim 128 --use_wandb --run_name "F3-hd128"
python train_pretrain.py --epochs 1 --max_steps 1000 --log_interval 100 --head_dim 256 --use_wandb --run_name "F4-hd256"
```
> 看：单头容量对 loss 的影响；q/k/v/o 投影参数随 head_dim 线性变化；attention scaling 1/√head_dim 也随之变。

### 组 G：FFN 中间维度 intermediate_size（默认=2432）
> **控制变量**：只改 `intermediate_size`，其余全默认。基准（2432）已有，跑下面五条凑成 `1024→1536→2432→3072→4096→6144` 梯度。
> ⚠️ 取值约束：**无形状约束**，任意正整数。倍数（相对 hidden=768）：1024≈1.3x, 1536=2x, 2432=3.2x(默认), 3072=4x(GPT风格), 4096≈5.3x, 6144=8x。
```bash
python train_pretrain.py --epochs 1 --max_steps 1000 --log_interval 100 --intermediate_size 1024 --use_wandb --run_name "G1-ffn1024"
python train_pretrain.py --epochs 1 --max_steps 1000 --log_interval 100 --intermediate_size 1536 --use_wandb --run_name "G2-ffn1536"
python train_pretrain.py --epochs 1 --max_steps 1000 --log_interval 100 --intermediate_size 3072 --use_wandb --run_name "G3-ffn3072"
python train_pretrain.py --epochs 1 --max_steps 1000 --log_interval 100 --intermediate_size 4096 --use_wandb --run_name "G4-ffn4096"
python train_pretrain.py --epochs 1 --max_steps 1000 --log_interval 100 --intermediate_size 6144 --use_wandb --run_name "G5-ffn6144"
```
> 看：FFN 容量（仅次于 hidden_size 的容量旋钮）对 loss 的边际收益；参数量随之线性增长，注意是否饱和。

### 组 H：Dropout 正则（默认=0.0）
> **控制变量**：只改 `dropout`，其余全默认。基准（dropout=0.0）已有，跑下面四条凑成 `0.0→0.05→0.1→0.2→0.3` 梯度。
> ⚠️ 取值约束：0~1 之间（实际常用 0~0.3）。预训练大数据几乎不用 dropout（GPT-3/LLaMA 均为 0）。
```bash
python train_pretrain.py --epochs 1 --max_steps 1000 --log_interval 100 --dropout 0.05 --use_wandb --run_name "H1-drop0.05"
python train_pretrain.py --epochs 1 --max_steps 1000 --log_interval 100 --dropout 0.1  --use_wandb --run_name "H2-drop0.1"
python train_pretrain.py --epochs 1 --max_steps 1000 --log_interval 100 --dropout 0.2  --use_wandb --run_name "H3-drop0.2"
python train_pretrain.py --epochs 1 --max_steps 1000 --log_interval 100 --dropout 0.3  --use_wandb --run_name "H4-drop0.3"
```
> 看：dropout 越大训练 loss 越高（正则抑制拟合）；预训练大数据通常 0，这组主要理解正则的作用方向。

### 组 I：RoPE 基础频率 rope_theta（默认=1e6）
> **控制变量**：只改 `rope_theta`，其余全默认。基准（1e6）已有，跑下面三条凑成 `1e4→1e5→1e6→1e7` 梯度。
> ⚠️ 取值约束：无（任意正数）。1e4=LLaMA-1 原始值, 1e6=默认, 1e7=超长上下文风格。
```bash
python train_pretrain.py --epochs 1 --max_steps 1000 --log_interval 100 --rope_theta 1e4 --use_wandb --run_name "I1-theta1e4"
python train_pretrain.py --epochs 1 --max_steps 1000 --log_interval 100 --rope_theta 1e5 --use_wandb --run_name "I2-theta1e5"
python train_pretrain.py --epochs 1 --max_steps 1000 --log_interval 100 --rope_theta 1e7 --use_wandb --run_name "I3-theta1e7"
```
> 看：短序列预训练影响通常很小（loss 几乎重合）；主要影响长上下文外推（短训看不太出来，理解参数意义即可）。

### 组 J：MoE（默认=Dense，use_moe=0）
> **控制变量**：开 MoE 后分别扫"专家数 num_experts"和"激活数 num_experts_per_tok"两个旋钮。基准（Dense）已有。
> ⚠️ 取值约束：`1 ≤ num_experts_per_tok ≤ num_experts`；仅 `use_moe=1` 时生效。
> J1→J2 看"更多专家"(同 top1)；J2→J3 看"更多激活"(同 8 专家)；J4 中规模(与 Mixtral 8专家激活2 同量级)。
```bash
python train_pretrain.py --epochs 1 --max_steps 1000 --log_interval 100 --use_moe 1 --num_experts 4  --num_experts_per_tok 1 --use_wandb --run_name "J1-e4k1"
python train_pretrain.py --epochs 1 --max_steps 1000 --log_interval 100 --use_moe 1 --num_experts 8  --num_experts_per_tok 1 --use_wandb --run_name "J2-e8k1"
python train_pretrain.py --epochs 1 --max_steps 1000 --log_interval 100 --use_moe 1 --num_experts 8  --num_experts_per_tok 2 --use_wandb --run_name "J3-e8k2"
python train_pretrain.py --epochs 1 --max_steps 1000 --log_interval 100 --use_moe 1 --num_experts 16 --num_experts_per_tok 2 --use_wandb --run_name "J4-e16k2"
```
> 看：Dense vs MoE 的 **logits_loss**（纯预测损失，可比）+ aux_loss；总参数量大增但激活算力受 num_experts_per_tok 控制。对比 MoE/Dense 优先看 logits_loss（MoE 的总 loss 含 aux_loss 不可直接比）。

---

## 用 wandb 看叠加曲线对比（推荐）

**任意命令后加 `--use_wandb`** 即可。多次运行自动进同一项目 `MiniMind-Pretrain`，曲线叠加对比。

- **首选 `--run_name "你起的名"`**：自己命名最清晰（如 `--run_name "A3-lr5e-4"`）。⚠️ 否则当"被测参数=默认值"时（如 lr=5e-4、bs=128）自动命名会丢失该参数、甚至不同实验重名 `baseline`。
- 留空 `--run_name` 时**自动从"与默认值不同的参数"生成**（如 `--learning_rate 1e-3` → `lr0.001`；`--num_attention_heads 16` → `nah16`）；已修掉之前到处挂的 `ep1`。
- 无论叫什么名，完整超参都进 wandb config，可在 Runs 表格排序、画"超参→loss"图、平行坐标图（纵向对比的核心）。

**组 A 全开 wandb 的版本（复制即用）：**
```bash
python train_pretrain.py --epochs 1 --max_steps 1000 --log_interval 100 --learning_rate 1e-4 --use_wandb
python train_pretrain.py --epochs 1 --max_steps 1000 --log_interval 100 --learning_rate 3e-4 --use_wandb
python train_pretrain.py --epochs 1 --max_steps 1000 --log_interval 100 --learning_rate 5e-4 --use_wandb
python train_pretrain.py --epochs 1 --max_steps 1000 --log_interval 100 --learning_rate 1e-3 --use_wandb
```

**在网页看对比：**
1. 终端打印的 `View run at https://wandb.ai/.../MiniMind-Pretrain/...` 点进去
2. 点项目名 `MiniMind-Pretrain` 回项目主页
3. Charts 里 loss 曲线自动叠加（每条标运行名）；Runs 表格列出各实验超参可排序

> 公平对比前提：所有实验统一 `--max_steps 1000 --log_interval 100`，曲线点对齐。

## 前提：wandb 已配置（一次性）

```bash
wandb login        # 粘贴 wandb.ai 的 API key（40位），已登录过则跳过
```
> 代码已把 [train_pretrain.py](../../trainer/train_pretrain.py) 的 `import swanlab as wandb` 改为 `import wandb`，用的是 wandb.ai。

---

## 进阶：验证"batch 与 lr 一起放大"

```bash
python train_pretrain.py --epochs 1 --max_steps 1000 --log_interval 100 --batch_size 128 --learning_rate 5e-4   # 基准
python train_pretrain.py --epochs 1 --max_steps 1000 --log_interval 100 --batch_size 512 --learning_rate 5e-4   # 只放大batch
python train_pretrain.py --epochs 1 --max_steps 1000 --log_interval 100 --batch_size 512 --learning_rate 1e-3   # batch+lr一起放大(√4=2倍)
```
对比后两个：同样大 batch，配大 lr 的是不是降得更好？

---

## 关于 --max_steps（新加的参数）

- `--max_steps 1000`：跑到第 1000 步自动停。
- `--max_steps 0`（默认）：不限制，跑满整个 epoch。
- 配合 `--epochs 1` 使用：到 1000 步就停，干净利落。

## 做完别忘

把每个实验到 step 1000 的 loss 填进 [05_experiment_log.md](05_experiment_log.md) 的观察表，并写下结论。
