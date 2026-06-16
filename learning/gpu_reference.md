# GPU 实用参考手册（以 minimind 训练为基础）

> 这不是教科书——是基于真实训练经验整理的速查手册。
> 专业术语保留英文；解释用中文。

---

## 核心三句话（先记这个）

> **显存容量** 决定模型能不能装进去。  
> **显存带宽** 决定数据喂得快不快。  
> **Tensor Core** 决定矩阵算得快不快。

CUDA Core 数量、TFLOPS 数字、RT Core——对 AI 训练帮助有限，选卡时优先级靠后。

---

## 1. nvidia-smi 表盘解读（真实数据）

```
| 59%  83C  P1  300W / 300W |  7807MiB / 97887MiB |  97%  Default |
   ↑     ↑   ↑      ↑              ↑                   ↑
  风扇  温度 Perf  功耗/上限      显存占用/总量        GPU-Util
```

**两个最重要的数，且它们是两回事：**

| 指标 | 含义 | 看什么 |
|------|------|--------|
| `Memory-Usage` | 显存用了多少 | 模型能不能装进去 |
| `GPU-Util` | 计算单元有多忙 | 卡有没有在干活 |

### 关键经验：97% Util ≠ 显存满

minimind 预训练实测：
- 显存：`7807 / 97887 MiB`，只用了约 **8%**
- GPU-Util：**97%**，计算单元已满载

说明：batch_size=32 时，矩阵运算已经把 GPU 算力喂饱了。
这时候加大 batch_size，**不会更快**（算力满了，加工作量只是排队等）。

```bash
watch -n 1 nvidia-smi   # 每秒刷新，Ctrl+C 退出，不影响训练
```

**进程区解读：**
```
PID 42514   C   python   7284MiB   ← 你的训练进程
PID  2898   G   Xorg      175MiB   ← 桌面图形，忽略
```
`C` = 计算进程，`G` = 图形进程。

---

## 2. 显存放了什么（以 minimind 63M 模型为例）

### 显存占用构成

| 内容 | 推理 | 训练（全参数）|
|------|------|--------------|
| 模型 weights | ✅ | ✅ |
| gradients | ❌ | ✅（≈ weights 大小）|
| AdamW optimizer states (m + v) | ❌ | ✅（≈ 2× weights）|
| activations（前向传播中间值）| 小 | 大（∝ batch_size）|
| KV Cache | 推理用 | 训练一般不用 |

### 快速估算（FP32 全参数训练）

```
每个参数需要约 16 bytes：
  4 (weights) + 4 (gradients) + 4 (m) + 4 (v) = 16 bytes

1B 参数 × 16 bytes ≈ 16 GB（还没算 activations）

minimind 63M × 16 bytes ≈ 1 GB weights 部分
加 activations 后实测约 7-8 GB（batch=128）
```

混合精度训练（BF16）可以显著降低显存：weights 和 activations 用 BF16（2 bytes），optimizer states 保持 FP32（4 bytes）。

### 实测（minimind 63M 预训练）

| batch_size | 显存 | GPU-Util |
|-----------|------|---------|
| 32 | ~5 GB | 97% |
| 128 | ~8 GB | 97% |
| 512 | ~22 GB | 97% |

GPU-Util 全程满载，显存随 batch 线性增长（主要是 activations）。

---

## 3. 浮点精度（直接影响速度和显存）

| 精度 | 字节 | 用在哪 | 特点 |
|------|------|--------|------|
| FP32 | 4B | 传统训练、optimizer states | 数值稳定，慢、占显存 |
| BF16 | 2B | **现代 LLM 训练首选** | 动态范围接近 FP32，快，推荐 |
| FP16 | 2B | 混合精度训练 | 动态范围小，容易上溢/下溢 |
| TF32 | — | NVIDIA 矩阵乘法内部格式 | 自动用于 Tensor Core，无需手动 |
| FP8 | 1B | 大模型训练/推理加速 | 需要 scaling，Blackwell 支持 |
| INT8/INT4 | 1B/0.5B | **量化推理** | 省显存，有精度损失 |

**为什么 BF16 比 FP16 更适合训练 LLM：**  
BF16 和 FP32 的指数位数相同（8位），动态范围一致，不容易出现 gradient 上溢/下溢。FP16 指数只有 5 位，训练深层网络时容易数值不稳定。

```python
# 查看 PyTorch 使用的 CUDA 版本
import torch
print(torch.version.cuda)
print(torch.cuda.get_device_name(0))

# 开启 TF32（A100/Blackwell 上默认开启）
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

---

## 4. GPU-Util 低≠没在跑，高≠一定快

### GPU-Util 低的常见原因

| 原因 | 表现 | 解法 |
|------|------|------|
| batch_size 太小 | GPU 等数据，利用率 30-50% | 加大 batch |
| DataLoader 太慢 | GPU 算完了等 CPU 喂数据 | 增加 `num_workers` |
| 频繁 CPU↔GPU 传输 | 每步都在 `.to(device)` | 数据预先 pin memory |
| 模型太小 | 矩阵太小，核心闲置 | 正常现象，小模型就这样 |

### GPU-Util 高 ≠ 一定用到了 Tensor Core

GPU-Util 只说明"GPU 某种单元在工作"，不区分是 CUDA Core 还是 Tensor Core。  
要充分利用 Tensor Core，需要：
- 使用 BF16/FP16（不是 FP32）
- 矩阵维度对齐到 16 的倍数（实际常用 64/128）
- batch_size 足够大

---

## 5. Batch Size 的真相（亲测）

### 大 batch ≠ 更快（GPU 已满载时）

实测 batch_size=32 时 GPU-Util 已 97%，改成 128：
- 每步用时约 4×（处理 4× 数据）
- 每 epoch 步数变成 1/4
- **总时间基本不变**

### 大 batch 真正的收益：训练质量，不是速度

- gradient 更平滑（噪声小，更 stable）
- 可以配更大 LR（√k scaling rule）
- 需要更少 epoch 收敛

### LR 随 batch 的 √k scaling（AdamW 适用）

```
batch_size 扩大 k 倍 → LR 建议乘以 √k

实测：batch 32（默认）→ 128（×4），LR 建议 5e-4 × √4 = 1e-3
```

为什么不是 ×k：AdamW 内部用二阶矩估计已做了部分 normalization，不需要 SGD 那么激进的线性 scaling。

### Batch 太大的代价

1. **OOM（CUDA out of memory）**：activations ∝ batch_size，打爆显存，没有"慢一点跑"，是硬崩溃
2. 超过 GPU 饱和点后，吞吐量（samples/sec）不再增加
3. 容易落入 sharp minima，泛化略差

### 比较 batch 实验要按「见过的数据量」，不是按 step

| batch | loss @ step 1000 | 此时见过的样本数 |
|-------|-----------------|---------------|
| 32 | 5.25 | 32,000 |
| 128 | 4.50 | 128,000 |
| 512 | 4.28 | 512,000 |

step 相同但 batch 不同 = 见过的数据量完全不同，不公平比较。

---

## 6. Flash Attention：为什么快（不是减少计算量）

### 问题所在

标准 Attention 的中间矩阵 `Q·Kᵀ` 大小为 `S×S`，序列长 S=4096 时这个矩阵有 1600 万个元素，要反复读写 HBM（显存，约 2 TB/s）。

### Flash Attention 的解法

```
标准：Q·K → 写 HBM → 读 HBM → softmax → 写 HBM → 读 HBM → ×V
Flash：Q/K/V 分块 → 搬到 SRAM（片上缓存，~19 TB/s，快 10×）
      → 在 SRAM 里完成计算 → 只写最终结果回 HBM
```

**Flash Attention 省的是 HBM 读写次数，不是数学运算量。数学结果完全一样，不是近似。**

### Flash Attention 在 minimind 中的触发条件

代码中四个条件全满足才用 Flash：
1. 硬件支持（PyTorch 2.0+ 且 GPU 支持）
2. `seq_len > 1`（decode 逐 token 时 S=1，没有优化空间）
3. 无 KV Cache（cache 时 Q 和 K 长度不同，causal mask 会算错）
4. 无 PAD mask（Flash 不支持自定义 PAD mask）

实际效果：**训练几乎总走 Flash；推理 prefill 走 Flash；推理 decode 走手动路径。**

---

## 7. KV Cache（推理专用）

### 为什么需要

生成 token 时，每次都要对**全部历史** token 算 Attention：
```
第 100 个 token 生成时，要 Q × K[0:100]
第 101 个 token 生成时，要 Q × K[0:101]
```
历史 token 的 K 和 V 每次重算 = 极大浪费。

### KV Cache 怎么做

把每层每个 token 的 K、V 存下来，下一步直接 cat 拼接：
```
第 1 步: key shape = [B, 1, kv_heads, head_dim]
第 2 步: key shape = [B, 2, kv_heads, head_dim]  （cat 历史 + 新）
第 n 步: key shape = [B, n, kv_heads, head_dim]
```

KV Cache 在 GPU 上（和模型同设备），训练时不用（整序列并行算更快）。

### KV Cache 占多少显存

```
每个 token 的 KV Cache ≈ 2（K+V）× num_layers × num_kv_heads × head_dim × 精度字节数
```

上下文越长，KV Cache 越大。服务多用户时，每个用户都有自己的 KV Cache，显存压力大。

---

## 8. 训练 vs 推理的不同瓶颈

| 阶段 | 主要瓶颈 | 关键指标 |
|------|---------|---------|
| **预训练** | Tensor Core 算力 | BF16 TFLOPS、Tensor Core 效率 |
| **SFT** | 同上，序列较短时带宽也有影响 | 同上 |
| **LLM 推理（prefill）** | 算力（处理整段 prompt） | TFLOPS |
| **LLM 推理（decode）** | **显存带宽**（每步只算 1 token，但要读全部 weights） | GB/s |

> **关键认知：** 大语言模型推理 decode 阶段，瓶颈往往不是算力不够，而是**显存带宽不够**。  
> 卡在等数据从显存读到计算单元，而不是在算。这就是为什么 H200（更高 HBM 带宽）对推理的提升比理论 TFLOPS 比值更大。

---

## 9. 显存不够时的应对方法（按效果排序）

```
OOM 时先看：当前 batch_size → activations 是主要元凶

1. 减小 batch_size（最直接）
2. 用 gradient accumulation（模拟大 batch，但节省显存）
3. 混合精度（BF16/FP16 weights + FP32 optimizer states）
4. gradient checkpointing（重算 activations 换显存，慢约 20-30%）
5. LoRA / QLoRA（冻结 base model，只训练少量参数）
6. 减小 max_seq_len
7. FSDP / ZeRO（多卡分片 optimizer states）
```

PyTorch 查显存：
```python
# 已分配（被 tensor 占用）
print(f"{torch.cuda.memory_allocated()/1024**3:.2f} GB")

# 已缓存（PyTorch 池子，包括暂时未用的）
print(f"{torch.cuda.memory_reserved()/1024**3:.2f} GB")

# 清理缓存池（不会释放还被变量引用的 tensor）
torch.cuda.empty_cache()
```

---

## 10. 常见 GPU 类型对比（实际选型）

| 类别 | 典型型号 | 显存 | 带宽 | 适合 |
|------|---------|------|------|------|
| 消费级 | RTX 4090 / 5090 | 24-32 GB | GDDR7 | 个人研究、小模型 |
| 工作站 | **RTX Pro 6000 Blackwell（你的卡）** | **96 GB** | GDDR7X | 中大规模实验 |
| 数据中心 | H100 / H200 | 80-141 GB | HBM3/3e | 大规模训练、高并发推理 |

**你的 RTX Pro 6000 Blackwell 的位置：**  
96 GB 显存 = 能完整放进去 7B 甚至 13B 模型做全参数训练；  
Blackwell 架构 Tensor Core 支持 FP8；  
GDDR7X 带宽比 RTX 4090 高，但低于 H100 的 HBM3；  
没有 NVLink（消费级/工作站互联靠 PCIe）。

---

## 11. 什么是 CUDA（软件层）

CUDA 不是 GPU，是 NVIDIA 提供的软件平台。

```
你的 Python 代码
    ↓
PyTorch（调用 cuDNN / cuBLAS / NCCL）
    ↓
CUDA Runtime + Driver
    ↓
GPU 硬件
```

常用库：
- `cuDNN`：深度学习算子（conv、attention 等）
- `cuBLAS`：矩阵乘法
- `NCCL`：多 GPU 通信
- `Triton`（OpenAI）：用 Python 写 GPU kernel，Flash Attention 就是用它实现的

**版本兼容性：**  
系统 CUDA 版本 ≠ PyTorch 使用的 CUDA 版本。  
关键是：**NVIDIA driver 要足够新**（driver 向下兼容 CUDA），PyTorch 安装包自带所需的 CUDA runtime。

```bash
# Blackwell 需要 CUDA 12.8+
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

---

## 12. 快速诊断清单

### 训练中遇到问题，先看这几项

| 症状 | 看什么 | 可能原因 |
|------|--------|---------|
| CUDA out of memory | `nvidia-smi` Memory-Usage | batch 太大 / 显存泄漏 |
| GPU-Util 长期 < 50% | CPU 占用、DataLoader | 数据预处理瓶颈 |
| Loss 不下降 / NaN | LR 设置 | LR 太大（发散）或太小 |
| 速度比预期慢 | GPU-Util、是否 BF16 | 精度没开 / batch 太小 |
| 多次运行结果不同 | 随机种子 | 未固定 seed |

```bash
# 固定随机种子（reproducibility）
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

---

## 附：选 GPU 时的优先级

对 AI 研究（语言模型）：

```
1. 显存够不够装模型？（容量）
2. 软件生态兼容吗？（CUDA、PyTorch 版本）
3. 显存带宽够快吗？（推理 decode 瓶颈）
4. BF16/FP16/FP8 Tensor Core 性能？（训练速度）
5. 多卡互联能力？（NVLink > PCIe）
6. 功耗和散热？
7. 价格
---
N. CUDA Core 数量（不重要）
N. TFLOPS 数字（看具体精度，看 Tensor Core，别看 FP32）
N. RT Core（光追用，AI 不关心）
```

> **一句话总结：一张算力稍弱但模型能完整放进去的 GPU，通常比一张算力强但经常 OOM 的更实用。**
