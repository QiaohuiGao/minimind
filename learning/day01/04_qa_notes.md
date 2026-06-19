# Day 1 问答笔记（实操记录）

> 本文件记录第一天实际操作中遇到的、有价值的问题与结论。区别于 `01_prereading`/`02_deep_dive`（课程材料），这里是「踩坑 + 解决」的真实记录。

---

## Q1. 没有写权限装不了包：`EnvironmentNotWritableError`

**现象：** `conda install torch` 报 `current user does not have write permissions to /opt/anaconda3`。

> **笔记：** 系统级 base 环境（`/opt/anaconda3`）通常没有写权限，**不要往 base 里装包**。正确做法是建自己的虚拟环境（装在用户目录，有写权限）。

## Q2. 虚拟环境是什么？`python -m venv .venv` 在干嘛？

> **笔记：** 虚拟环境 = 一个「隔离的小盒子」，里面有独立的 python 和 pip，装的包只属于这个项目，不污染系统、不和别的项目冲突。
> - `python -m venv 名字` → 创建（只需一次）。`.venv` 是惯例名（开头 `.` 会隐藏）。
> - 环境名可自定义，用英文/数字/`-`/`_`，别用空格中文。

## Q3. venv 和 conda 的区别？怎么激活/退出？

> **笔记：** 两种建环境方式，激活命令不同：
> | 方式 | 创建 | 激活 | 退出 |
> |------|------|------|------|
> | venv | `python -m venv minimind-env` | `source minimind-env/bin/activate` | `deactivate` |
> | conda | `conda create -n minimind python=3.10` | `conda activate minimind` | `conda deactivate` |
>
> 退出命令是 `deactivate`（不是 `inactivate`）。conda 新建的环境默认装到 `~/.conda/envs`（用户目录，有写权限），所以能绕开 base 没权限的问题。

## Q4. 怎么判断包装进了虚拟环境，还是误装到 home？

> **笔记：核心判断方法——看路径，不要凭感觉。**
> ```bash
> which python   # 带 .venv/bin/ 才对；显示 /opt/anaconda3/bin 说明没激活
> which pip      # 同上
> pip show 包名   # 看 Location 字段在不在 .venv 里
> ```
> **关键习惯：** 每次新开终端，venv 都不会自动激活，必须先 `source .venv/bin/activate`，看到命令行前面有 `(.venv)` 再操作。没激活就 `pip install`，包会装到 `~/.local`（用户级），不在虚拟环境里。

## Q5. `pip install requirements` 报错？

> **笔记：** 装依赖文件要加 `-r` 和完整文件名：`pip install -r requirements.txt`。
> 不加 `-r` 时，pip 以为你要装一个名叫 `requirements` 的包，报 `No matching distribution`。
> minimind 的 `requirements.txt` 里 `torch`/`torchvision` 被注释掉了，要单独装 GPU 版：
> `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`

## Q6. modelscope 数据集下到哪了？怎么下对地方？

> **笔记：**
> - 默认下到缓存 `~/.cache/modelscope`（和 Python 环境无关，是缓存目录）。
> - 下载中断时，大文件会卡在 `._____temp` 临时目录里（没下完）。
> - **下对地方的技巧：** 用 `--local_dir` 直接下到项目的 dataset 目录：
>   ```bash
>   modelscope download --dataset gongjy/minimind_dataset pretrain_t2t_mini.jsonl --local_dir ./dataset
>   ```
> - 只想快速跑通，下 mini 版即可（`pretrain_t2t_mini` + `sft_t2t_mini`），不用下全量（pretrain_t2t 10GB / sft_t2t 14GB）。

---

## Q7. 项目的数据流是怎么走的？怎么「打点」调试？

**预训练数据流路线图：**
```
jsonl ({"text":...})
  → PretrainDataset.__getitem__   [dataset/lm_dataset.py:47]  文字→编码→加bos/eos→pad
  → DataLoader                    [train_pretrain.py:162]      拼成batch input_ids[B,L]
  → .to(device)                   [train_pretrain.py:28]
  → model.forward(input_ids,labels) [model_minimind.py:443]
    embed → Transformer → hidden[B,L,768] → lm_head → logits[B,L,6400]
    → logits[:-1] 预测 labels[1:] → cross_entropy → loss
  → backward + optimizer.step     [train_pretrain.py:37-49]
  → 日志/保存                      [train_pretrain.py:51-69]
```

> **笔记：打点 = 在关键位置加 print 看数据 shape 和内容。** 推荐顺序：
> 1. **先单独验证数据**（不启动训练）：写个 `debug_data.py`，建 Dataset → 取 `ds[0]` 看单条 → 取一个 batch 看 shape。
> 2. **再在训练里打点**：`__getitem__` 里看编码对不对（解码回去对比原文 + labels 里 -100 是不是 pad 位）；batch 处看 `input_ids.shape`；forward 后看 `logits.shape` 和 `loss`。
> 3. 打点加 `if index < 2` / `if step <= 2` 限制，避免刷屏。
> labels 里 `-100` = 该位置不计算 loss（pad 位置）。

---

## Q8. 现代大模型的数据是怎么处理和清洗的？

> **笔记：分两大阶段，逻辑完全不同。**
>
> **预训练数据**（海量、求量和广，TB~PB 级，全自动清洗）经典流水线：
> 1. **抽取**：从网页 HTML 抽正文（trafilatura），去标签/广告。
> 2. **语种过滤**：fastText 识别语言，留目标语言。
> 3. **质量过滤**（核心）：启发式规则（Gopher/C4：太短、符号比例异常、重复行多→删）+ 模型打分（拿维基当正例训分类器打分）。
> 4. **去重**（关键！）：精确去重（哈希）+ 近似去重——**MinHash+LSH（`datasketch` 库）** 估 Jaccard 相似度删相似文档；**SimHash（`simhash` 库）** 指纹+汉明距离。⚠️ 这两个库就在 minimind 的 requirements.txt 里。去重防止模型死记重复内容、浪费算力。
> 5. **去隐私(PII)/有害内容**：去邮箱手机号、过滤色情暴力。
> 6. **去污染(Decontamination)**：把评测题(MMLU/CEval)从训练集删掉，否则评分虚高=作弊。
> 7. **格式统一**：→ `{"text": "干净文字"}`。
>
> **后训练数据**（少量、求质和准）：
> - **SFT**（`sft_t2t.jsonl`）：构造/筛选高质量问答对，来源含人工标注、公开数据、**模型蒸馏合成**（作者用 qwen3-4b 合成约10w条tool call）。筛选靠 LLM-as-a-judge 打分 + 多样性去重。格式 `{"conversations":[{role,content}]}`。
> - **RL/DPO**（`dpo.jsonl`）：每条是「同一问题的好/坏回答对」(chosen/rejected)，用于偏好对齐。
>
> **一句话总结：** 预训练清洗重在「质量过滤+去重+去污染」，SFT 重在「高质量问答对+格式统一」，DPO 重在「构造好坏对比对」。
>
> minimind 的数据集是作者已清洗好的成品，直接训练即可。

---

## Q9. 怎么看 GPU 利用率？nvidia-smi 表盘完整解读

**命令：** `nvidia-smi`（看一次）；`watch -n 1 nvidia-smi`（每秒刷新实时盯，Ctrl+C 退出，不影响训练）。

**表盘第一区（整卡状态）：**
```
| 59%  83C  P1  300W / 300W |  7807MiB / 97887MiB |  97%  Default |
   ↑     ↑   ↑      ↑              ↑                   ↑
  风扇  温度 性能 功耗/上限      显存用量/总量        算力利用率
```
> **笔记：关键看两个数，且它俩是两回事：**
> - **Memory-Usage（显存用量）**：`7807/97887 MiB` = 用了 7.8G / 共 96G。显存够不够「装得下」。
> - **GPU-Util（算力利用率）**：`97%` = 计算单元忙不忙。卡「算得快不快/有没有闲」。
> - 其他：风扇% / 温度(83℃偏高但安全) / Perf状态(P0最高~P8最低) / 功耗(300W已拉满=Max-Q功耗墙)。
> - ECC、MIG 一般用不到。

**表盘第二区（进程）：**
```
PID 42514   C   python   7284MiB   ← 你的训练进程占了 7.3G
PID  2898   G   Xorg      175MiB   ← 桌面系统占的，忽略
```
> **笔记：** `Type` 列 `C`=计算进程(你的训练)，`G`=图形进程(桌面)。能看出显存被谁吃了。

## Q10. 加大 batch_size 为什么没更快？为什么要同步调大学习率？

> **笔记：踩坑认知——「显存没满 ≠ 可以无脑加 batch 提速」。**
> 判断卡有没有「吃饱」要看 **GPU-Util**，不是显存：
> - 显存 8% 但 GPU-Util 已 97% → **算力已满载**，加大 batch **不会提速**（吞吐量/每秒token数已固定）。
> - 只有 GPU-Util 偏低（卡在等数据/batch太小喂不饱）时，加 batch 才提速。
>
> **实测：** batch 32(39695步/epoch) → 128(9924步/epoch)，总时间基本不变甚至略增。因为每步算4倍活、总步数变1/4，乘起来≈不变。早期 epoch_time(ETA) 偏高是因为启动开销摊在少量步数上，不准，跑久了会降。
>
> **大 batch 的真正收益是「训练质量」不是「速度」：** 梯度噪声更小、更稳，可配更大LR、用更少epoch收敛。
>
> **为什么 batch 和 LR 要一起放大：**
> 1. batch 大 → 梯度估计更准 → 敢迈更大步(大LR)，不怕走偏。
> 2. batch 大 → 总步数变少 → 每步不迈大点会欠训练，需大LR补偿。
>
> **缩放法则**（batch 放大 k 倍）：
> | 法则 | LR 调整 | 适用 |
> |------|--------|------|
> | 线性缩放 | ×k | SGD |
> | 平方根缩放 | ×√k | **Adam/AdamW(minimind用这个)** |
> batch 32→128(×4)：AdamW 推荐 √4=×2，即 `5e-4 → 1e-3`。
>
> **真正提速的手段（不是加batch）：** `--use_compile 1`(torch.compile) / 减少 `--epochs`。

---

## Q11. GPU 内存结构（HBM vs SRAM）与 Flash Attention 的关系

> **笔记：**
> - **HBM（显存）**：就是"24GB显存"，大但慢（~2 TB/s）。存模型权重、Q/K/V 等所有数据。
> - **SRAM（片上缓存）**：每个 SM（计算核心）自带，几十～几百KB，超快（~19 TB/s，快 10 倍）但极小。
> - Flash Attention 本质：把 Q·K·V 分块搬到 SRAM 里算完直接累加，不把 S×S 中间矩阵写回 HBM。减少搬运 = 提速。

## Q12. 两个 Triton（完全不同的东西）

> **笔记：**
> - **OpenAI Triton**：GPU 编程语言，用 Python 语法写 GPU kernel（如 Flash Attention），替代繁琐的 CUDA C++，自动管理线程和内存。
> - **NVIDIA Triton Inference Server**：模型部署服务器，把训练好的模型上线对外提供 HTTP/gRPC API，支持自动 batching、多模型调度。和 Flash Attention 无关。
>
> | | OpenAI Triton | NVIDIA Triton |
> |---|---|---|
> | 是什么 | GPU 编程语言 | 模型部署服务器 |
> | 干什么 | 写高性能 GPU 算子 | 模型上线对外服务 |
> | 阶段 | 训练/算子开发 | 推理/部署 |

## Q13. Flash Attention 原理与触发条件

> **笔记：Flash Attention = 注意力计算的加速算法，数学结果和普通 Attention 完全一样，不是近似。**
>
> **普通 Attention 的问题：**
> ```
> scores = Q @ K^T     # [B, heads, S, S] ← S很大时这个矩阵巨大
> weights = softmax(scores)
> output = weights @ V
> ```
> S=4096 时 S×S=1600万，这个中间矩阵要写到 HBM（显存），读写很慢。
>
> **Flash Attention 怎么解决：** 分块计算，不存中间矩阵。
> ```
> 普通：Q·K → 写回显存 → 读出来 softmax → 写回显存 → 读出来 × V
> Flash：Q/K/V 分小块 → 搬到 SRAM（片上缓存，快10倍）→ 算完直接累加 → 只写最终结果回显存
> ```
> - **省显存**：不需要存 S×S 的中间矩阵
> - **更快**：减少 HBM ↔ 计算单元之间的搬运次数（IO 是瓶颈，不是计算）
>
> **触发条件：** 代码中 `if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1))`，四个条件全满足才用 Flash：
> 1. `self.flash`：硬件支持（PyTorch 2.0+ 且 GPU 支持）
> 2. `seq_len > 1`：推理逐 token 时 S=1，S×S=1，没优化空间
> 3. `not is_causal or 无缓存`：KV Cache 推理时 Q 和 K 长度不同，Flash 的 `is_causal=True` 假设等长会算错
> 4. `无 PAD mask 或全1`：Flash 不支持自定义 PAD mask
>
> 实际效果：训练→几乎总走 Flash；推理 prefill（处理 prompt）→ Flash；推理 decode（逐 token）→ 手动路径。

## Q14. 什么是残差连接（Residual Connection）？

> **笔记：** `output = x + f(x)`，把输入直接加到输出上。
> - 每层只需学"改动量"而非"全部"，任务更简单。
> - 给梯度提供直通公路：`∂output/∂x = 1 + ∂f/∂x`，那个 1 保证梯度不会消失。没有残差，深层网络梯度连乘→消失→训不动。

## Q15. Pre-Norm vs Post-Norm

> **笔记：**
> - Post-Norm（原始 Transformer）：`x = Norm(x + Attention(x))`，残差被包在 Norm 里。
> - Pre-Norm（现代 LLM）：`x = x + Attention(Norm(x))`，残差在 Norm 外面。
>
> Pre-Norm 的优势：残差路径 `x₈ = x₀ + f₁ + f₂ + ... + f₈`，反传时 `∂x₈/∂x₀` 恒有常数 1，梯度直通不衰减。Post-Norm 的梯度要穿过每层 Norm 的缩放，层数多了会衰减。
>
> | | Post-Norm | Pre-Norm |
> |---|---|---|
> | 训练稳定性 | 差，需要 warmup | 好，不容易崩 |
> | 深层网络 | 难训（梯度消失） | 轻松堆几十上百层 |
> | 效果 | 训好了可能略好 | 略差但差距极小 |
>
> 现代 LLM 全用 Pre-Norm：稳定训练比理论最优更重要。

---

## 待深入拓展问题（超出 MiniMind 代码本身，未来研究）

### 多头注意力 & Attention 机制
- [ ] **为什么多头比单个大头好？** 一次 softmax 只能产出一种注意力分布，多头 = 多种模式并行。但"子空间学不同模式"具体是什么？推荐看 Jay Alammar Illustrated Transformer + 3Blue1Brown Attention 视频 + AttentionViz 交互工具实际观察各头的 pattern。
- [ ] **Attention head 实际学到了什么？** 有的头关注语法（主谓），有的关注语义（同义词），有的关注局部窗口。用 BertViz / AttentionViz 可视化工具实操体验。
- [ ] **MHA vs GQA vs MQA 的工业选型依据？** 不只是参数量——要结合推理时 KV Cache 显存、吞吐量、模型质量 tradeoff 一起看。

### 位置编码
- [ ] **RoPE 的数学原理？** 旋转矩阵、复数乘法、为什么 Q·K 只依赖相对距离 |i-j|。当前只知道"旋转 Q/K"，还没理解公式推导。
- [ ] **YaRN 外推的具体做法？** 低频压缩、高频保持、中频过渡——每一步的数学细节和为什么这样分。
- [ ] **RoPE vs ALiBi vs Sinusoidal？** 各种位置编码方案的对比和演进。

### 训练 & 优化
- [ ] **反向传播在 Transformer 里具体怎么走？** 从 loss → lm_head → blocks → attention → q_proj.weight 的梯度链路。理解每层权重怎么更新。
- [ ] **Tensor Parallelism 具体怎么切分 heads？** n_local_heads 在多 GPU 时怎么分配，all-reduce 怎么汇总。
- [ ] **Weight Tying：什么时候该解绑？** 大模型（70B+）解绑 Embedding 和 LM Head 的实验对比和原因。

### GPU & 系统
- [ ] **GPU SM 架构详解？** CUDA core / Tensor core / warp / thread block 的层次关系。
- [ ] **Flash Attention 的分块算法细节？** 在线 softmax（online softmax）怎么做到分块还能算精确 softmax，不是近似。
- [ ] **OpenAI Triton 写一个简单 kernel？** 实操体验用 Python 语法写 GPU 代码，对比 CUDA C++。

### MoE
- [ ] **MoE 的路由策略有哪些？** Top-1 / Top-2 / Expert Choice，各自的 tradeoff。
- [ ] **MoE 训练的负载均衡问题？** aux_loss 之外还有什么方法（如 Switch Transformer 的 capacity factor）。

### 工程实践
- [ ] **HuggingFace 的 AutoModel 注册机制？** model_type → config → model 的工厂模式，trust_remote_code 的安全风险。
- [ ] **LoRA 的变体？** QLoRA（量化+LoRA）、DoRA、AdaLoRA，各自改进了什么。
- [ ] **工业级 FFN intermediate_size 怎么选？** 经验上 ~2.67x hidden_size（对齐到 64/128），SwiGLU 的 3 个矩阵 vs 标准 FFN 的 2 个矩阵对参数量的影响。

---

## Q16. SwiGLU FFN 的结构（gate_proj / up_proj / down_proj）

> **笔记：** 现代 LLM 的 FFN 不再是传统的两层 Linear+ReLU，而是 **SwiGLU**（三个线性层 + 门控机制）。
>
> **代码位置：** `model/model_minimind.py` 的 MiniMindMLP 类。
>
> ```python
> def __init__(self, config, intermediate_size=None):
>     intermediate_size = intermediate_size or config.intermediate_size
>     self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)  # 门控投影
>     self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)  # 升维投影
>     self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)  # 降维投影
>     self.act_fn    = ACT2FN[config.hidden_act]  # SiLU 激活函数
> ```
>
> **数据流（forward）：**
> ```
> x ─→ gate_proj ─→ SiLU ─┐
>                           ├─→ 逐元素相乘 ─→ down_proj ─→ 输出
> x ─→ up_proj ────────────┘
> ```
> 即 `down_proj(SiLU(gate_proj(x)) * up_proj(x))`
>
> **各变量含义：**
> | 变量 | 全称 | 作用 | 维度变化 |
> |------|------|------|----------|
> | `gate_proj` | gate projection | 算门控值，经 SiLU 后决定哪些信息通过 | hidden → intermediate |
> | `up_proj` | up projection | 升维，提取特征 | hidden → intermediate |
> | `down_proj` | down projection | 降回原始维度 | intermediate → hidden |
> | `act_fn` | activation function | SiLU = x × sigmoid(x)，平滑门控 | 不变 |
> | `intermediate_size` | FFN 中间维度 | 一般 ≈ hidden_size × π（对齐到64） | — |
>
> **为什么叫"gate"：** `gate_proj` 经过 SiLU 后值域在 (-0.28, +∞)，接近 0 → "关门"（抑制），大值 → "开门"（放行），与 `up_proj` 相乘 = 选择性保留信息。
>
> **对比传统 FFN：**
> | | 传统 FFN | SwiGLU FFN |
> |---|---|---|
> | 结构 | up → ReLU → down（2个矩阵） | gate+up → SiLU×乘 → down（3个矩阵） |
> | 门控 | 无 | 有（gate_proj 控制信息流） |
> | 效果 | 基线 | 更好（PaLM/LLaMA 验证） |
>
> **`intermediate_size` 为什么不用 `self`：** 它只是 `__init__` 里的临时局部变量，用来创建 `nn.Linear` 后就不再需要——数值已"固化"进权重矩阵的形状里，后续方法不会再读这个数字。
>
> **`ACT2FN` 是什么：** 来自 `from transformers.activations import ACT2FN`，是 HuggingFace 提供的激活函数字典，`ACT2FN['silu']` 返回 `nn.SiLU()`。

---

## Q17. KV Cache 在哪个设备上？训练时用不用？

> **笔记：** KV Cache **跟模型在同一个设备上**（GPU 训练就在 GPU）。
>
> Cache 里存的是 Attention 里 `k_proj(x)` 和 `v_proj(x)` 的输出，这些 tensor 由 GPU 上的 `nn.Linear` 算出，PyTorch tensor 保留在计算它的设备上，不会自动跑去 CPU。下一步推理时 `torch.cat([past_key, new_key])` 要求两者在同一设备，所以 cache 自然一直在 GPU 上。
>
> **训练时一般不用 KV Cache**——整个序列一起算更快（并行）。KV Cache 主要用于**推理时逐 token 生成**，避免重复计算历史 token 的 K/V。
>
> **KV Cache 的具体结构：**
> ```python
> past_key_values = [
>     (key_0, value_0),   # 第 0 层的缓存
>     (key_1, value_1),   # 第 1 层的缓存
>     ...
>     (key_N, value_N),   # 第 N 层的缓存
> ]
> ```
> 每个 key/value 的 shape：`[B, cached_seq_len, kv_heads, head_dim]`
>
> 以 `hidden_size=128, num_kv_heads=2, head_dim=32` 为例，已生成 10 个 token 时：
> - `past_key_values[0]` → 第 0 层的 `(key, value)` 元组
> - `past_key_values[0][0]` → 第 0 层的 key，shape `[B, 10, 2, 32]`
> - `past_key_values[0][0].shape[1]` → `10`（已缓存 token 数，用于算 `start_pos` 给 RoPE 定位）
>
> **推理时逐 token 增长：**
> ```
> 第 1 步: key shape = [1, 1, 2, 32]   # 1 个 token
> 第 2 步: key shape = [1, 2, 2, 32]   # cat 后 2 个
> 第 3 步: key shape = [1, 3, 2, 32]   # cat 后 3 个
> ...每步只算新 token 的 Q/K/V (seq_len=1)，新 K/V cat 到缓存，避免重算历史。
> ```
>
> **KV Cache 完整推理流程（以生成"你好吗"为例）：**
>
> 第 1 步：处理 prompt "你"
> ```
> past_key_values = [None, None, ..., None]   # 每层都是 None
> start_pos = 0
> 每层 Attention：past_key_value=None → 不拼接
>   xk = k_proj(x)  # [1, 1, 2, 32]  (1个token)
>   past_kv = (xk, xv)  ← 存下来返回
> presents = [(k0,v0), (k1,v1), ..., (kN,vN)]
> ```
>
> 第 2 步：生成 "好"
> ```
> past_key_values = 上一步的 presents
> start_pos = 1  ← past_key_values[0][0].shape[1]
> 每层 Attention：只算新 token 的 Q/K/V (seq_len=1)
>   xk = cat([past_key_value[0], new_xk], dim=1)
>        [1,1,2,32] + [1,1,2,32] → [1,2,2,32]  (两个token的K)
>   past_kv = (xk, xv)  ← 更新缓存
> ```
>
> 第 3 步：生成 "吗"
> ```
> start_pos = 2
>   xk = cat([past, new], dim=1)
>        [1,2,2,32] + [1,1,2,32] → [1,3,2,32]  (三个token的K)
> ```
>
> **关键代码对应：**
> - `start_pos = past_key_values[0][0].shape[1]`：从缓存推断已生成多少 token
> - `position_embeddings = freqs_cos[start_pos:start_pos+seq_length]`：RoPE 只取新 token 的位置
> - `xk = torch.cat([past_key_value[0], xk], dim=1)`：历史 K + 新 K 拼接
> - 每步只算新 token 的 Q/K/V，把新 K/V cat 到历史缓存，Attention 能看到所有历史 token 但不重算
>
> **transformers 5.x 兼容性：** transformers 5.x 把 KV Cache 从简单的 list 改成了 `Cache` 对象（有 `.layers` 属性）。minimind 按老格式（list）写的，所以代码里有：
> ```python
> if hasattr(past_key_values, 'layers'): past_key_values = None  # 新格式→丢弃
> past_key_values = past_key_values or [None] * len(self.layers)  # 确保是 list
> ```

---

## Q18. 为什么一个 Transformer Block 需要两个 RMSNorm？

> **笔记：** `input_layernorm` 和 `post_attention_layernorm` 结构完全一样（都是 `RMSNorm(hidden_size)`），但它们是**两个独立实例，各自有独立的可学习参数 `weight`**。
>
> **数据流：**
> ```
> x → input_layernorm → Attention → 残差相加 → post_attention_layernorm → FFN → 残差相加 → 输出
>     ^^^^^^^^^^^^^^^                           ^^^^^^^^^^^^^^^^^^^^^^^^^^
>     归一化 #1（自己的 weight）                    归一化 #2（自己的 weight）
> ```
>
> **为什么不能共用一个：** Attention 之后的数据分布和之前不一样了，所以需要两组独立的缩放参数，分别学习"Attention 前怎么归一化"和"FFN 前怎么归一化"。
>
> 类比：两个人穿同款衣服（结构相同），但尺码不同（参数不同）。

---

## 🔧 对 train_pretrain.py 的改动记录（为做对照实验）

为了方便做超参对照实验 + wandb 可视化，对脚本做了 4 处改动：

1. **可视化换成 wandb**：`import swanlab as wandb` → `import wandb`（用 wandb.ai，国外可直接访问）。需先 `wandb login`（key 40位，从 wandb.ai/authorize 复制）。
2. **新增 `--max_steps` 参数**：跑到第 N 步自动停（默认 0=不限制）。做对照实验时设 `--max_steps 1000`，省得手动 Ctrl+C。实现：循环里 `if args.max_steps and step >= args.max_steps: break`。
3. **暴露全部架构超参为 CLI 参数**：原本只有 hidden_size/num_hidden_layers/use_moe 能从命令行改，其余写死在 MiniMindConfig。现新增 11 个：`num_attention_heads / num_key_value_heads / head_dim / intermediate_size / dropout / rope_theta / max_position_embeddings / num_experts / num_experts_per_tok / moe_intermediate_size / router_aux_loss_coef`，并用 `config_kwargs` 全部传进 MiniMindConfig（head_dim/intermediate_size/moe_intermediate_size 用 0=auto）。同时加了**启动安全校验**（见下方约束）。
4. **wandb 运行名自动反映改动 + config**：运行名改成**自动检测"与默认值不同的参数"**生成（如 `lr0.001`、`nah16-nkv4`，全默认则 `baseline`），并 `wandb.init(..., config=vars(args))` 记录完整超参，可在 wandb Runs 表/平行坐标图分析"任意超参→loss"。

**架构参数约束（经对抗式验证，违反会在启动时 assert 报错）：**
- `num_attention_heads % num_key_value_heads == 0`（GQA 约束）
- `head_dim` 必须为偶数（RoPE 要求）；若 `hidden_size//num_attention_heads` 为奇数，需显式传偶数 `--head_dim`
- `max_position_embeddings >= max_seq_len`
- MoE 时 `1 <= num_experts_per_tok <= num_experts`

> 实验指令集见 [06_experiment_commands.md](06_experiment_commands.md)（组 A~J），观察表见 [05_experiment_log.md](05_experiment_log.md)。
> ⚠️ wandb 踩坑：`wandb`(国外40位key) ≠ `swanlab`(国产86位key)，别把 swanlab 的 key 贴进 `wandb login`。新版 key 格式 `wandb_v1_...` 需较新版 wandb（`pip install -U wandb`）。

### MiniMindConfig 全部超参速查表

| 参数 | 默认 | 类别 | 一句话作用 | 改大影响 |
|------|------|------|-----------|---------|
| hidden_size | 768 | 结构(宽) | 模型宽度 d_model | 容量↑质量↑，参数∝O(h²)、慢 |
| num_hidden_layers | 8 | 结构(深) | Transformer 层数 | 容量↑，参数/算力线性↑ |
| num_attention_heads | 8 | 注意力 | Q 头数 | 子空间更细，每头维度变小 |
| num_key_value_heads | 4 | 注意力 | KV 头数(GQA) | <Q头=GQA省KV显存；=Q头=MHA |
| head_dim | 96(auto) | 注意力 | 每头维度 | 单头容量↑，q/k/v/o 参数线性↑ |
| intermediate_size | 2432(auto) | FFN | FFN 中间维度 | 容量↑(仅次于hidden)，参数线性↑ |
| hidden_act | silu | FFN | 激活函数 | 影响收敛，差异通常小 |
| dropout | 0.0 | 正则 | 丢弃率 | >0 防过拟合，训练loss略升 |
| rope_theta | 1e6 | 位置编码 | RoPE 基础频率 | 影响长上下文外推 |
| max_position_embeddings | 3276 | 位置编码 | 最大位置数 | 决定最长上下文(须≥seq_len) |
| tie_word_embeddings | True | 词表 | embed/lm_head 共享权重 | 省参数、小模型质量↑ |
| vocab_size | 6400 | 词表 | 词表大小 | 须与tokenizer一致，改需重训 |
| use_moe | False | MoE | 是否用专家混合 | 开后总参数↑但稀疏激活 |
| num_experts | 4 | MoE | 专家总数 | 容量↑、参数/路由开销↑ |
| num_experts_per_tok | 1 | MoE | 每token激活专家数 | 激活算力↑、质量↑(须≤num_experts) |
| moe_intermediate_size | =inter(auto) | MoE | 单专家FFN中间维度 | 单专家容量↑ |
| router_aux_loss_coef | 5e-4 | MoE | 负载均衡loss权重 | 强制均衡、防专家坍塌 |
| flash_attn | True | 实现 | Flash Attention 加速 | 提速降显存，结果等价 |
| rms_norm_eps | 1e-6 | 数值 | RMSNorm 防除零 | 影响数值稳定，常规默认 |

---

## 📊 训练记录 #1：第一次完整预训练（2026-06-02）

**模型规模：**
- Model Params: **63.91M**（Trainable 63.912M）—— 稠密模型，未开 MoE

**超参数：**
| 参数 | 值 | 说明 |
|------|-----|------|
| 数据集 | `pretrain_t2t_mini.jsonl` (1.2G) | mini 版预训练数据 |
| hidden_size | 768 | 隐藏层维度 |
| num_hidden_layers | 8 | Transformer 层数 |
| use_moe | 0 | 未用 MoE（所以 aux_loss 恒 0）|
| batch_size | 128 | （从默认 32 改成 128）|
| learning_rate | 5e-4 → 5e-5 | 初始 lr，cosine 衰减到最小 |
| epochs | 2 | 每 epoch 9924 步 |
| 硬件 | RTX PRO 6000 Blackwell (96G) | 显存仅用 ~8G |

**loss 轨迹（关键节点）：**
```
Epoch 1: 7.28(起步,随机初始化) → 4.51(step1000) → 2.97(step2400) → 2.22(epoch末)
Epoch 2: 2.10(起步) → 2.05(中段) → 1.98(最终)
```
> **观察：**
> - 第1轮前期 loss 暴降（7.28→3 只用了约 2400 步），这是模型快速学到基本语言规律；之后下降变缓。
> - 第2轮 loss 在 1.9~2.1 之间小幅波动，已接近该规模+该数据量的收敛区间。
> - 单条 loss 有抖动（如 2.15↔1.93）正常——不同 batch 难易不同。看**整体趋势**而非单点。
> - loss≈2.0 对应 perplexity≈e²≈7.4，比起步的 e^7.3≈1480 大幅下降，模型从"瞎猜"变得相当确定。

**产物：** `out/pretrain_768.pth`（132M，半精度权重）+ `checkpoints/` 续训存档。

**下一步：** 用此 base 权重做 SFT（`train_full_sft.py` + `sft_t2t_mini.jsonl`）→ 得到会对话的 `full_sft_768.pth`。

> **踩坑提醒：** 跑这次前误切到了 `(qiaohui)` conda 环境（不是 `.venv`），导致 `No module named 'datasets'`。注意 `datasets`(HF库) ≠ `dataset`(本地文件夹)，别把 import 路径 `dataset.lm_dataset` 改成 `datasets.lm_dataset`。环境永远确认命令行前缀是 `(.venv)`。

---

## 📈 实验观察规律：learning rate & batch size（亲手实验得出）

### A. learning rate（基础值 = 5e-4）

> **核心规律（用自己的数据验证）：larger lr 前期降得快，但后期"太粗糙"overshoot，最终结果反而差。**

实测——同一组 lr，对比第 100 步（早期）和第 1000 步（后期）的 loss：

| lr | step 100（早期）| step 1000（最终）| 现象 |
|------|---------------|----------------|------|
| 1e-4 | 7.67 | 5.50 | 太小，降得慢 |
| 3e-4 | 7.30 | 4.52 | 好 |
| **5e-4** | 7.28 | **4.49** | **最终最好（=默认值）** |
| 1e-3 | **7.22（早期最低！）** | 5.06 | 前期快、后期被反超 |
| 5e-3 | 7.48 | 6.02 | 偏大，降得慢 |
| 5e-2 | **31.3（爆炸）** | 7.15 | 发散，先飙升再部分恢复 |

> **结论：**
> 1. `1e-3` 第 100 步 loss 最低（早期最快），但第 1000 步被 `5e-4/3e-4` 反超 → 印证"大 lr 前期快、后期 overshoot"。原因：lr 大 = 每步 step 大，靠近 minimum 时会"冲过头"停不准。
> 2. lr **过大**（5e-2）连早期好处都没有，直接 diverge（loss 飙到 31）。
> 3. 所以"最佳 lr"不是固定值，而是**先高后低**——这就是 learning rate schedule 存在的理由。
> 4. scientific notation：`5e-4` = `0.0005`（`e-4` = 小数点左移 4 位）；`e-` 后数字越大、数值越大，故 `1e-3 > 5e-4 > 1e-4`。

### B. learning rate schedule（decaying schedule）

> lr 在训练中**按规则变化**，而非固定。两段组成：**warmup**（lr 从~0 升到 peak，避免开局 overshoot）+ **decay**（peak 逐渐降到很小，精细收敛）。

- **minimind 用的**（[trainer_utils.py:40](../../trainer/trainer_utils.py#L40)）：**cosine decay，无 warmup，floor=peak 的 10%**：
  `lr*(0.1 + 0.45*(1+cos(π*step/total)))` → 从 5e-4 平滑降到 5e-5。
- **modern models**：① 经典 = **warmup + cosine decay**（GPT-3/LLaMA）；② 新趋势 = **WSD**（Warmup-Stable-Decay，MiniCPM/DeepSeek，中间恒定 lr、最后才衰减，便于延长训练）；③ 历史：inverse sqrt（原始 Transformer）、step decay。
- **为什么要 warmup**：开局权重随机、gradient 乱，直接上 peak lr 易 diverge（如 5e-2 那次）；warmup 给缓冲。minimind 模型小、lr 不极端，省了也能跑。

#### 如何实现 linear warmup + cosine decay

minimind 现有 `get_lr`（[trainer_utils.py:40](../../trainer/trainer_utils.py#L40)）只有 cosine decay，加 warmup 只需改这一个函数：

```python
def get_lr(current_step, total_steps, lr, warmup_steps=0):
    if warmup_steps > 0 and current_step < warmup_steps:
        return lr * current_step / warmup_steps      # linear warmup: 0 → peak
    decay_step = current_step - warmup_steps
    decay_total = total_steps - warmup_steps
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * decay_step / decay_total)))
```

调用处透传参数：
```python
lr_now = get_lr(step, total_steps, args.lr, warmup_steps=200)
optimizer.param_groups[0]['lr'] = lr_now
```

**三个关键点：**
- **warmup 步数**：通常取 total steps 的 1~5%，或固定 100~500 步。跑 1000 步对照实验时设 50~100 即可。
- **warmup 起点**：`lr * step / warmup_steps`，step=0 时 lr=0，step=warmup_steps 时恰好到 peak，完全线性。
- **minimind 为何省了也没事**：模型小（63M）、lr 不极端（5e-4），cosine decay 开头本身 lr 就是 peak 且下降极缓。若把 lr 推到 1e-3 以上，加 warmup 会更稳定。

### C. batch size

> **纠正一个 misconception：larger batch ≠ better result。** 每步样本多 ≠ 学得更好，因为同时 **update 次数变少**，两者相抵。

larger batch 真正改变的：
- gradient 更平滑（噪声小，更 stable）
- 每个 epoch 的 weight update 次数变少
- hardware 利用率更高（但**只在 GPU 没吃满前**有效）
- 允许配更大的 lr（√k scaling for AdamW）

  > **√k scaling 是什么：** batch size 扩大 k 倍时，建议同步把 lr 乘以 √k。
  >
  > 原理：gradient 是对一个 batch 里所有样本的平均，batch 越大，平均越精准（噪声越小）。数学上，gradient 的 standard deviation ∝ 1/√B（B = batch size）。batch 大了 → gradient 噪声小了 → 每步"指向"更准确 → 可以迈更大的步（更大的 lr）而不 overshoot。
  >
  > 为什么是 √k 而不是直接 ×k：SGD 理论上可以线性 scale（×k），但 Adam/AdamW 内部已经用二阶矩（gradient 的方差估计）做了 normalization，自动抵消了一部分 batch 大小的影响，所以 scaling 不需要那么激进，经验上 √k 更安全。
  >
  > **实际用法**：minimind 默认 batch=32、lr=5e-4；若把 batch 改成 128（×4），可尝试 lr=5e-4 × √4 = **1e-3**。不是硬规则，只是起点。

> **⚠️ 实测里的陷阱：按 step 比较不公平。**
> | batch | loss @ step 1000 | step1000 时见过的数据量 |
> |-------|------------------|----------------------|
> | 32 | 5.25 | 32,000 samples |
> | 128 | 4.50 | 128,000 |
> | 512 | 4.28 | 512,000 |
>
> 看起来"batch 越大越好"，但 step 1000 时 batch=512 已经见了 **16× 更多数据**！不公平。**正确比较要按 samples-seen 或 wall-clock time，不能按 step。**

> **"over large" 为什么 inefficient（你的疑问）：**
> 1. **OOM（Out of Memory）**：GPU 的 VRAM 是固定的（你的卡 96GB）。训练时显存同时放着：model weights + gradients + optimizer states（AdamW 额外存 m/v 两份，约 2× weights 大小）+ activations（forward pass 的中间结果，大小 ∝ batch size）。batch 太大 → activations 撑爆 VRAM → CUDA 直接报 `RuntimeError: CUDA out of memory` 崩掉，没有"慢一点跑"的余地，是硬限制。
> 2. **过 GPU saturation 后无吞吐增益**：你的 nvidia-smi 在 batch=32 就已 97% util，再大 batch 不提升 samples/sec。
> 3. **diminishing returns**：存在 **critical batch size**，超过它后 batch 翻倍≠收敛快一倍，花 2× compute 换 <2× 进展（浪费）；极大 batch 还可能落入 **sharp minima**，generalization 略差。
>
>    > **Sharp minima vs generalization：**
>    > Loss landscape 是 loss 对所有 weight 的高维"地形图"。
>    > - **Sharp minimum（尖谷）**：像峡谷，周围 loss 壁很陡，weight 稍微一变 loss 就飙升。
>    > - **Flat minimum（宽谷）**：像盆地，周围很平缓，weight 有小扰动 loss 也基本不变。
>    >
>    > **为什么 flat 比 sharp 的 generalization 好：** 训练集 ≠ 测试集，两者有细微的分布差异。Sharp minimum 对这种 shift 敏感——在训练集完美拟合的权重，稍有偏移就在测试集 loss 暴增，即 overfitting。Flat minimum 对扰动不敏感，说明模型学到的是更通用的规律，测试集表现更稳定，这就是 **generalization（泛化）好**。
>    >
>    > **为什么大 batch 容易落入 sharp minima：** 大 batch 的 gradient 噪声小、方向精准，优化路径倾向于直奔最近的 local minimum，而最近的不一定是最宽的。小 batch 有 gradient 噪声，相当于在 landscape 上"随机抖动"，反而更容易从 sharp valley 逃出来、最终落进 flat valley。这是大 batch 训练已知的 tradeoff（不是致命缺点，但要知道）。
>
> 4. 所以 batch size 有 **sweet spot**：够大(gradient 稳、硬件用满) 但别过大(OOM/浪费)，且要和 lr 一起调。

### D. Optimizer：Adam / AdamW

**Adam（Adaptive Moment Estimation）** 是目前最主流的 optimizer。

最简单的 optimizer SGD 每步只做：`w = w - lr × gradient`，所有参数用同一个 lr，但不同参数的 gradient 大小差异极大 → 有的 overshoot，有的走不动。

Adam 给每个参数自动计算"专属 lr"，靠两个额外统计量：
- **m（一阶矩）**：gradient 的滑动平均，记住最近的方向，减少单步噪声（momentum）。
- **v（二阶矩）**：gradient **平方**的滑动平均，衡量这个参数的 gradient 有多大、多剧烈。

实际步长 = `lr × m / √v`。gradient 一直很大的参数（v 大）→ 步长被压小；gradient 一直很小的参数（v 小）→ 步长被放大。自动适应，不需要手动给每个参数调 lr。

**AdamW = Adam + 正确的 Weight Decay**

原版 Adam 把 L2 惩罚混进了 gradient，然后被 Adam 的 v 自适应机制稀释，导致 weight decay 效果失真。AdamW 把 weight decay 从 gradient 里剥离出来，每步单独施加：

```
w = w - lr × (Adam计算出的gradient方向)   ← Adam 负责
w = w × (1 - lr × λ)                      ← weight decay 单独做，直接缩 w
```

**weight decay 作用在 weights 本身，不是作用在 learning rate 上。** 两步完全独立。

> **GPT、LLaMA、minimind 都用 AdamW。**

### E. Regularization vs Normalization（常见混淆）

两个完全不同的概念，名字像但目的和做法都不一样：

| | 作用于 | 解决什么问题 |
|--|--|--|
| **Regularization（正则化）** | **weights（模型参数）** | 防 overfitting，让权重别太大 |
| **Normalization（归一化）** | **activations（中间层的值）** | 稳定训练，让数值分布不爆炸/消失 |

**Regularization** 在 loss 里加惩罚项，迫使权重变小：
- **L2**（Weight Decay）：惩罚项 = `λ × Σ(w²)`，权重被拉小但不为 0，更平滑。AdamW 的 weight decay 就是 L2。
- **L1**：惩罚项 = `λ × Σ|w|`，会把不重要的权重直接推到精确的 0（**稀疏性**），相当于自动删掉无用参数，常用于传统 ML。

**Normalization** 对激活值做缩放，维持数值稳定：
- **LayerNorm**：对单个样本的每层激活值归一化（均值→0，方差→1），Transformer 标配。
- **RMSNorm**：LayerNorm 的简化版，去掉均值那步，只除以 RMS（root mean square）——**minimind 用的就是这个**。
- **BatchNorm**：对一个 batch 内的激活值归一化，CV 常用，LLM 少用。

> 类比：regularization 是"考试前别死背答案"（防过拟合），normalization 是"保持头脑清醒、情绪稳定"（训练稳定）。

### F. hidden_size（模型宽度）与 compute budget

**直觉上 hidden_size 越大越好，但有前提：数据量和训练步数要跟上。**

| hidden_size | 参数量 | 需要多少数据/步才能"发挥出来" |
|---|---|---|
| 256（C1） | 8.33M | 少，50k 条 + 390 步就够学到东西 |
| 768（C2，默认） | 63M | 中，勉强够用 |
| 1024（C3） | 112M | 多，50k 条完全喂不饱 |

大模型不是"学不会"，而是参数太多，需要**更多数据和更多步数**才能充分训练。用 50k 条数据只跑 1 epoch，相当于让一个大容器只装了一点点水——C3 大部分参数几乎没被有效训练到，loss 反而高于 C1。这叫 **underfitting（欠拟合）**——不是模型太弱，而是**训练量不够**。

反过来，hidden_size 太小 → 模型容量不足 → 即使数据再多也学不到复杂语言规律 → loss 降到某个 ceiling 就降不动了，这叫 **capacity bottleneck**。

> **一句话：** hidden_size 越大越好——前提是数据量和训练步数要跟上。数据少、步数少时，小模型反而赢，因为"容量小、容易填满"。这就是为什么小实验用小模型，正式训练才上大模型。

### G. 超参（Hyperparameter）是什么

> **超参 = 训练前你手动设定、训练过程中不被 gradient 更新的参数。**

| | 普通参数（parameters） | 超参（hyperparameters） |
|---|---|---|
| 例子 | model 里的 weights/biases | lr、batch_size、λ、hidden_size |
| 谁来更新 | optimizer 根据 gradient 自动更新 | 你手动设定，训练中不变 |
| 存在哪 | model.state_dict() | argparse / 代码里的固定值 |

lr、batch_size、hidden_size、num_layers、weight_decay、dropout、warmup_steps……全都是超参。

### H. AdamW 的完整参数

```python
torch.optim.AdamW(
    params,             # 要优化的 model weights
    lr=1e-3,            # 学习率（会被 schedule 手动改）
    betas=(0.9, 0.999), # (β1, β2)：m 和 v 的滑动平均衰减系数
    eps=1e-8,           # 防除零的极小值
    weight_decay=0.01   # λ，weight decay 系数，全程固定
)
```

**betas 解释：**
- **β1=0.9**：控制 m（一阶矩，gradient 方向）的"记忆长度"。当前 gradient 占 10%，历史占 90%。
- **β2=0.999**：控制 v（二阶矩，gradient 大小）的"记忆长度"。历史更长，变化更慢。

betas 和 eps 几乎所有人都用默认值，不需要调。**实际需要调的只有 `lr` 和 `weight_decay`**，其余基本不动。

weight decay（λ）全程固定不变；lr 每步通过 `get_lr()` 重新算后手动塞进 `optimizer.param_groups[0]['lr']`。

### I. num_attention_heads 的 trade-off（组 D 实验观察）

head_dim（每个 head 的维度）= hidden_size ÷ num_heads：

| heads | head_dim（768÷n） |
|---|---|
| 4 | 192 |
| 8（默认） | 96 |
| 16 | 48 |
| 32 | 24 |

**为什么 heads 少，前期 loss 降得更快：**
heads 少 → head_dim 大 → 每个 head 能表达更丰富的信息 → 前期收敛快。

**为什么 4 heads 最终被 8 heads 反超：**
multi-head attention 的核心价值是**并行关注不同方面**——一个 head 看语法关系，另一个看语义，另一个看位置距离……heads 越多，模型能同时关注的"视角"越多。4 heads 每个 head 虽然"看得深"，但只有 4 个视角，多样性不够，最终表达能力有上限。

**为什么更多 heads（16、32）更差：**
head_dim 太小（48、24）→ 每个 head 维度太压缩，连基本的 Q/K 相似度都算不准 → 视角多但每个视角"近视"，反而没用。

> **一句话：** heads 的选择是**每个 head 的表达深度（head_dim）** vs **并行视角的多样性（head 数量）**之间的 trade-off。太少 → 多样性不足；太多 → 每个 head 太浅。768 维模型的经验甜点是 8~12 heads。

### J. MQA / GQA / MHA 与 num_key_value_heads（组 E 实验）

标准 attention 计算：`attention_output = softmax(Q × Kᵀ / √d) × V`

- **Q**（Query）：当前 token 在"问"什么
- **K**（Key）：每个位置的 token "是什么"
- **V**（Value）：每个位置的 token "能提供什么信息"

K 和 V 都来自输入序列本身，描述的是"上下文里有什么内容"。三种方案区别只在 KV head 数量：

| 名称 | KV heads | 含义 |
|---|---|---|
| **MHA**（Multi-Head Attention） | = Q heads（8） | 每个 Q head 有自己专属的 KV |
| **GQA**（Grouped Query Attention） | 介于中间（2、4） | 几个 Q head 共享一对 KV |
| **MQA**（Multi-Query Attention） | 1 | 所有 Q heads 共用同一对 KV |

**为什么 Q 不同，KV 却可以共享：**
多个 Q heads 共用同一套 KV，但每个 Q head 的 **Q 向量不同** → attention weights（softmax 分数）不同 → 从同一份 V 里提取出的**加权组合不同**。多样性来自 Q 的不同，不需要 KV 也各不相同。

类比：同一个图书馆（K/V），不同读者（Q heads）带着不同问题来检索——书还是那些书，但每个人关注的章节不一样。

**KV 减少会损失什么：**
MHA 里每个 head 有独立的 KV 投影矩阵，可以把输入映射到不同语义子空间。共享后这部分多样性略有损失——MQA（nkv=1）质量略降，GQA（nkv=4）几乎无损。

**为什么要减少 KV heads：**
KV 在推理时要存进 KV cache（显存），nkv 越少 → cache 越小 → **推理更省显存、更快**。LLaMA 3、minimind 默认都用 GQA（nkv=4）作为折中。

> **一句话：** Q 的多样性保证每个 head 关注不同的东西，KV 只是"被查询的内容"，共享影响不大。损失的只是 KV 投影空间的多样性，这部分对质量影响很小但对显存节省极大。

### K. Head 的真正含义

**Head = 一个独立的 attention 视角**，有自己专属的 Q/K/V 投影矩阵，学会关注输入的某一类关系。

单个 attention head 在做什么：
```
Q = input × W_Q   # "我在找什么"
K = input × W_K   # "我能提供什么索引"
V = input × W_V   # "我能提供什么内容"
```
用 Q 去和所有位置的 K 打分，分数高的位置从 V 里取更多信息。**一个 head 只能学会一种"找法"**——一套 W_Q/W_K/W_V 决定了它只能关注一类关系。

**多个 heads 的意义：**
每个 head 有自己独立的 W_Q/W_K/W_V，通过训练自然分化出不同专长：
- head 1 可能学会关注**语法关系**（主语→动词）
- head 2 可能学会关注**指代关系**（"他"→"张三"）
- head 3 可能学会关注**位置距离**（相邻 token）
- head 4 可能学会关注**语义相似性**

这不是人为设定的，是训练过程中**自动涌现**的分工。所有 heads 的输出最后 concat，再投影回 hidden_size，模型同时获得多个视角的信息。

> **一句话：** head 是 attention 的一个独立"视角单元"，有自己的投影矩阵，学会关注输入中某一类关系。多 heads = 多视角并行，最后汇总。

### L. head_dim 的 trade-off（组 F 实验观察）

head_dim 默认 = `hidden_size ÷ num_heads = 768 ÷ 8 = 96`。

**head_dim 越小的影响：**
每个 head 把信息压缩进更少的维度 → attention score（Q·Kᵀ）计算更"粗糙" → 对不同 batch 的数据更敏感，loss 曲线更抖。

**预期趋势：**

| head_dim | 预期表现 | 原因 |
|---|---|---|
| 48、64（太小） | 偏差且抖动大 | 压缩过度，attention 精度低 |
| 96（默认） | 最稳，甜点 | 与 hidden_size=768、8 heads 自然匹配 |
| 128、256（太大） | 前期看起来好，短实验欠拟合 | 参数多、训练量不够，同 C 组大模型 |

**关于"64 高于 48"这类短实验异常：**
1000 步内两条曲线差距很小时，**很可能是噪声**，不代表真实性能差距。此外，head_dim=48 参数比 64 少，在短实验里可能因"容量小、容易填满"而暂时领先——和 C 组小模型赢大模型是同一个道理。不要过度解读单个数据点，看**整体趋势**：离默认值（96）越远，表现越差。

---

## Q22. loss.backward() 为什么能直接调用？PyTorch autograd 机制

> **笔记：loss 不是普通数字，是一个带"计算历史"的 tensor。**
>
> **loss 的 shape 是标量（scalar）**，就是一个数，比如 `tensor(8.3412)`。因为 `F.cross_entropy` 对所有位置求平均，最终输出一个数。backward 的起点必须是标量——如果 loss 是向量，就没法定义"该往哪个方向调权重"。
>
> **为什么能直接 `.backward()`：**
> PyTorch 的 autograd 在每一步计算时**偷偷记录计算图**：
> ```
> embed(input_ids)                    ← 记录
>     ↓
> blocks(x)                           ← 记录
>     ↓
> lm_head(x)                          ← 记录
>     ↓
> cross_entropy(logits, labels)       ← 记录
>     ↓
> loss                                ← 身上挂着整条计算链路（.grad_fn 属性）
> ```
>
> `loss.grad_fn` 记录了"我是怎么算出来的"：
> ```python
> print(loss.grad_fn)
> # <NllLossBackward0>  ← 知道自己来自 cross_entropy
> #   ← 来自 lm_head (Linear)
> #     ← 来自 blocks
> #       ← 来自 embed
> ```
>
> 调用 `loss.backward()` 时，PyTorch 沿着这条链**从后往前**，用链式法则（chain rule）算出每个权重的梯度，存到 `weight.grad` 里。
>
> **核心理解：** 不是 loss "特殊"——**任何经过计算得到的 scalar tensor 都能调 `.backward()`**。这就是 PyTorch 的核心设计：你只管写前向计算，反向传播自动帮你做。

---

## Q23. Embedding 层的作用、影响性能的因素、SOTA 模型的配置

> **笔记：Embedding = 查表，把离散的 token ID 映射到连续的语义向量空间。**
>
> ```python
> self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
> # token ID 47 ("好") → 查表 → [0.12, -0.53, 0.81, ..., 0.03]  # hidden_size 维向量
> ```
>
> 训练过程中，语义相近的词会被推到向量空间中相近的位置（如"猫"和"狗"的向量距离 < "猫"和"经济"）。
>
> **影响性能的两个维度：**
>
> **1. hidden_size（向量维度）= 模型能编码多少"语义特征"：**
>
> | 模型 | hidden_size | 效果 |
> |---|---|---|
> | simple_train | 256 | 很弱，能学基本模式 |
> | minimind | 768 | 小模型，能生成简单中文 |
> | LLaMA-3 8B | 4096 | 强，能做复杂推理 |
> | LLaMA-3 70B | 8192 | 非常强 |
>
> **2. vocab_size（词表大小）= Embedding 矩阵的行数：**
>
> | 模型 | vocab_size | 特点 |
> |---|---|---|
> | minimind | 6,400 | 极小，中文覆盖有限 |
> | LLaMA 3 | 128,000 | 多语言，中文效率高 |
> | Qwen 2.5 | 151,936 | 中英文优化 |
>
> 词表大 → 常见词更可能是 1 个 token → 同样长度装更多信息 → 效率高。但词表大 → 参数更多。
>
> **所有 SOTA 模型的 Embedding 实现都一样——就是 `nn.Embedding` 查表。** 差异在配置（维度、词表大小），不在算法。常见技巧：
> - **Weight Tying**（LLaMA, Qwen, minimind）：Embedding 和 LM Head 共享权重，省参数
> - **位置信息不在 Embedding 加**：现代模型都用 RoPE（在 Attention 里加），早期模型（BERT, GPT-1）才有 Position Embedding
>
> **SOTA 模型层数与配置对比：**
>
> | 模型 | 参数量 | num_layers | hidden_size | num_heads |
> |---|---|---|---|---|
> | minimind | 26M | 8 | 768 | 16 |
> | LLaMA-3 8B | 8B | 32 | 4096 | 32 |
> | LLaMA-3 70B | 70B | 80 | 8192 | 64 |
> | LLaMA-3 405B | 405B | 126 | 16384 | 128 |
> | Qwen-2.5 7B | 7B | 28 | 3584 | 28 |
> | Qwen-2.5 72B | 72B | 80 | 8192 | 64 |
> | DeepSeek-V3 | 671B (MoE) | 61 | 7168 | 128 |
> | GPT-3 | 175B | 96 | 12288 | 96 |
>
> **规律：**
> - 参数量 ∝ num_layers × hidden_size²，变大靠**加深 + 加宽同时放大**
> - 7-8B 模型一般 28-32 层，70B 一般 80 层，400B+ 约 100-126 层
> - num_heads 通常 = hidden_size / 128（即 head_dim 固定为 128）
> - simple_train.py 的 4 层 256 维是这些模型的"迷你缩影"，结构完全一样——放大配置就能得到 LLaMA

---

## 今日待办状态

- [x] 建立 `.venv` 虚拟环境（`~/Project/minimind/.venv`）
- [x] modelscope 装进 `.venv`
- [x] 清理误装到 home 的包和缓存
- [x] 下载数据集（全量已下到 ~/.cache，再 mv 进 ./dataset/；下载时漏了 `--local_dir` 是坑）
- [x] 安装 PyTorch GPU 版（torch 2.11.0+cu128，Blackwell 必须 cu128）+ requirements.txt
- [x] **成功跑通第一次预训练**（loss 从 ~7.3 正常下降）✅
- [ ] 配置 VSCode 调试（已建 .vscode/launch.json，需装 Python 扩展，用 F5 而非 `python xxx.py`）
- [x] **跑完完整预训练（2轮，loss 7.28→1.98）** ✅ → out/pretrain_768.pth
- [ ] 用 eval_llm.py 测试 base 模型续写效果（`python eval_llm.py --weight pretrain`）
- [ ] 做 SFT 微调（train_full_sft.py + sft_t2t_mini.jsonl）→ 会对话的模型
