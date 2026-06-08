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

## Q13. Flash Attention 的触发条件详解

> **笔记：** 代码中 `if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1))`，四个条件全满足才用 Flash：
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

## 今日待办状态

- [x] 建立 `.venv` 虚拟环境（`~/Project/minimind/.venv`）
- [x] modelscope 装进 `.venv`
- [x] 清理误装到 home 的包和缓存
- [x] 下载数据集（全量已下到 ~/.cache，再 mv 进 ./dataset/；下载时漏了 `--local_dir` 是坑）
- [x] 安装 PyTorch GPU 版（torch 2.11.0+cu128，Blackwell 必须 cu128）+ requirements.txt
- [x] **成功跑通第一次预训练**（loss 从 ~7.3 正常下降）✅
- [ ] 配置 VSCode 调试（已建 .vscode/launch.json，需装 Python 扩展，用 F5 而非 `python xxx.py`）
- [ ] 跑完预训练 → 用 eval_llm.py 测试生成效果
