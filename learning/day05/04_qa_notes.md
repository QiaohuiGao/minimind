# Day 5 问答笔记：LoRA 机制 + Python 工程基础 + 模型架构

> 本篇大多是「手写 LoRA 练习」时冒出来的问题，分三类：
> ① LoRA 原理与实现机制；② Python/工程基础（路径、isinstance、monkey-patch、闭包等）；③ Mymodel.py 架构细节（Embedding/RoPE/KV Cache/causal mask）。

---

## Part 1：LoRA 原理与实现

### Q1. LoRA 里的 `rank` 是什么意思？

**知识锚点：矩阵的秩 (rank)** = 矩阵里有多少个「线性无关的方向」（独立信息维度）。

LoRA 旁路 `ΔW = B @ A`：
```
A: [rank, in]    (nn.Linear(in_features, rank))   降维
B: [out, rank]   (nn.Linear(rank, out_features))  升维
ΔW = B @ A → 形状 [out, in]，但 rank 最多只有 `rank` 这么大
```

`rank` = **人为给 ΔW 设的「信息上限」**，强制让"微调带来的权重改变"只发生在一个低维子空间里。

- **低秩假设**：微调大模型时，权重真正需要改变的部分活在很低维的子空间，不需要全部自由度。
- rank 越大 → 表达力越强但参数越多、越易 overfit、越省不了；rank 越小 → 越省但可能学不动。
- 经验：minimind 这种小模型用 8 就够；一般任务 16~32 最常用；不是越大越好。

**搭档 `alpha`**（本项目 model_lora.py 没用，即系数=1）：完整公式是 `output = W·x + (alpha/rank)·B·A·x`。

---

### Q2. 为什么 LoRA 的 B 矩阵初始化为全零？

训练开始时 `ΔW = B@A = 0` → 输出 == 原模型。这样 LoRA 是「无害接入」，不会一上来就破坏 pretrain/SFT 学到的能力，然后慢慢学。
（A 用高斯随机 `normal_(std=0.02)`，B 用 `zero_()`。如果 A、B 都随机，第一步就把模型搞乱了。）

---

### Q3. LoRA 训练和普通 SFT 训练的区别？

训练五步（forward → zero_grad → backward → clip → step）所有训练都一样。LoRA 只有两处不同：

1. **optimizer 只拿 LoRA 参数**：`AdamW(lora_params, ...)` 而不是 `model.parameters()`。
   → 这是省显存的根本：AdamW 每个参数要存 2 份额外状态，只为几千个 LoRA 参数维护，底座几百万参数不管。
2. **梯度裁剪也只裁 `lora_params`**：底座已 `requires_grad=False` 没梯度。

forward 仍写 `model(input_ids, labels)` 就行 —— 因为 `apply_lora` 已经 monkey-patch 了各层 forward，LoRA 旁路**自动**生效，主训练代码无需感知 LoRA 存在。

---

## Part 2：Python / 工程基础

### Q4. monkey-patch：`forward_with_lora` 到底干了什么？

每个 `nn.Module` 的 `forward` 就是"数据进来怎么算"。`output = layer(x)` 内部就是调 `layer.forward(x)`。

LoRA 想把输出从 `W·x` 改成 `W·x + B·A·x`，但**不改模型源码**，所以**偷偷把层的 forward 换掉**（monkey-patch）：

```python
original_forward = module.forward          # 备份这一层原来的 forward（算 W·x）
def forward_with_lora(x, _orig=original_forward, _lora=lora):
    return _orig(x) + _lora(x)             # W·x + B·A·x
module.forward = forward_with_lora         # 替换（注意：不加括号！存的是函数本身）
```

---

### Q5. 为什么不能直接 `module.forward = original_forward(x) + lora(x)`？

**核心区别：函数 vs 函数调用结果。**

| 写法 | 是 | 
|------|----|
| `original_forward` | 函数本身（"配方"，还没执行）|
| `original_forward(x)` | 执行后的结果（一个具体 tensor，"一盘菜"）|

两个致命问题：
1. `apply_lora` 在训练前调用，那时 `x` 根本不存在 → `NameError`。
2. `module.forward` 必须是个**能被反复调用的函数**，每条新数据来都要重算。存一个"算好的固定结果"，后面所有数据都得到同一个错答案。

**时间线**：`def` 那刻只是"定义配方"不执行；等训练里 `model(x)` 时 PyTorch 才调 `module.forward(真实x)`，这时才真正算 `W·x + B·A·x`。
→ 赋值时**不加括号**（存函数），让"怎么算"留到数据来了再算。

---

### Q6. 闭包大坑：为什么要写 `def f(x, _orig=original_forward, _lora=lora)`？

**错误写法**（直接用外层变量）：
```python
def new_forward(x):
    return original_forward(x) + lora(x)   # ❌
```
问题：函数不立刻执行，等调用时才去查 `original_forward`/`lora`。但它们是 for 循环变量，**每圈都被重新赋值**，循环结束后停在最后一圈的值。
→ 结果**所有层都用了最后一层的 forward 和 lora**，全错。这就是闭包"按引用、延迟取值"。

**正确写法**：用**默认参数**。默认参数在 `def` 那一刻**立即求值并钉死**，每个函数把"属于自己那圈的值"拍成快照，互不干扰。

---

### Q7. `isinstance(对象, 类型)` 怎么用？

判断"对象是不是某类型（或其子类）的实例"，返回 True/False。
- 第 1 参数：被检查的**对象**；第 2 参数：拿来比对的**类型**。
- 比 `type(x)==T` 好，因为 **isinstance 认子类**。
- 第 2 参数可传 tuple 一次查多个：`isinstance(x, (int, float))`。
- 本项目用法：`for ... in model.named_modules(): if isinstance(module, nn.Linear):` 从一堆层里只挑 Linear。

---

### Q8. `setattr(module, "lora", lora)` 是什么？

`setattr(obj, "x", v)` 等价于 `obj.x = v`，区别是**用字符串操作属性**（属性名可动态决定）。
三兄弟（都用字符串）：
- `setattr(obj,"x",v)` = `obj.x=v`（设）
- `getattr(obj,"x")` = `obj.x`（读）
- `hasattr(obj,"x")` → 有没有该属性（判断）

本项目：apply_lora 用 `setattr` 挂上，save/load 用 `hasattr(module,'lora')` 判断再用。

---

### Q9. `device = next(model.parameters()).device` 是什么？

目的：**探测模型当前在哪个设备（cuda/mps/cpu）**。
- `model.parameters()` → 所有参数的**迭代器（generator）**，不能用下标。
- `next(...)` → 取第一个参数 tensor（随便一个就行）。
- `.device` → 这个 tensor 在哪个设备。
用途：apply_lora 里新建的 LoRA 默认在 CPU，要 `.to(device)` 搬到和模型同一设备，否则报"tensor 不在同一设备"。
（也可用 `config.device`；这种写法更"自给自足"，直接看模型真身在哪。）

---

### Q10. 定位项目文件的固定套路

```python
os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
```
- `__file__` → 脚本自己的路径
- `os.path.dirname(p)` → 取目录部分（去掉文件名）
- `os.path.join(a, b)` → 从左到右拼路径（自动加 `/`，跨平台）
- `..` → 往上一层，**接在基准目录后面**（`model/..` 指向其父目录 `minimind`，不是 model 自己）
- `os.path.abspath(p)` → 把带 `..` 的绕路化简成干净绝对路径；也能把相对路径补成绝对路径

组合义：**以脚本为锚点定位到项目根目录**。好处是不依赖运行时 cwd（在哪个目录敲命令都对），换机器也不失效。

数据路径同理：
```python
os.path.join(os.path.dirname(__file__), "..", "dataset", "sft_t2t_mini.jsonl")
# 从 model/ 往上到 minimind/，再进 dataset/，找文件
```

---

### Q11. `sys.path.append(项目根)` 为什么必须有？

`sys.path` = Python 找模块的**目录清单**，`import` 时挨个翻，找不到就 `ModuleNotFoundError`。

`from model.Mymodel import ...` 需要从**项目根 `minimind/`** 出发才能看到 `model` 这个包。但 `python model/xxx.py` 默认只把脚本所在的 `model/` 加进 sys.path，从 model 里看不到"叫 model 的子文件夹"。
→ 所以先 `sys.path.append(项目根)` 把 minimind/ 加进搜索路径，再 import。**必须放在 import 之前**（代码从上往下执行）。

---

## Part 3：Mymodel.py 架构细节

### Q12. `nn.Embedding(vocab_size, hidden_size)` 第一个参数是 in 还是 out？

签名：`nn.Embedding(num_embeddings, embedding_dim)`。

**知识锚点：查表 (lookup table)**，形状 `[vocab_size, hidden_size]`：
- **第 1 参数 `num_embeddings = vocab_size`**：表有多少行 = 词表里多少个不同 token（输入 token id 的取值范围）。
- **第 2 参数 `embedding_dim = hidden_size`**：每行向量多长 = 输出维度（out feature）。

输入 token id（如 42）→ 去第 42 行取出那条向量。
⚠️ 和 `nn.Linear(in_features, out_features)` 参数顺序不同，别混：Linear 先 in 后 out；Embedding 第一个是"表的行数（词表大小）"。

---

### Q13. `precompute_rope_freqs` 的公式（RoPE 预先算角度）

这函数提前算好"第几个位置、第几组数字要转多少角度"，存成复数旋转因子。跟 token 内容无关 → 可一次算好反复用。

**① 每组转速**（只看维度编号，与位置无关）：
$$\theta_i = \frac{1}{\text{base}^{\,2i/d}}, \quad d=\text{head\_dim}$$
```python
freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
# arange(0,head_dim,2) = [0,2,4,...] 就是 2i；得到 d/2 个频率
```
低编号 i 转得快（高频，管近距离），高编号转得慢（低频，管远距离）—— 像钟表秒针/时针。

**② 位置 × 转速 = 角度**：
$$\text{angle}(m,i) = m \times \theta_i$$
```python
t = torch.arange(max_seq_len).float()   # 位置 m = 0,1,2,...
freqs = torch.outer(t, freqs)           # 外积，形状 [max_seq_len, head_dim/2]
```

**③ 角度 → 旋转因子**（欧拉公式，模长=1 表示只转不缩放）：
$$\text{freqs\_cis}(m,i) = e^{\,i\,m\theta_i} = \cos(m\theta_i) + i\sin(m\theta_i)$$
```python
freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # polar(模长, 角度)
```
> 注：$\sin$ 前的 $i$ 是虚数单位，和组编号 $i$ 撞名但无关。

---

### Q13b. `rope_theta`（base）是角频率吗？

不是。⚠️ RoPE 里有两个"theta"撞名，别混：

```python
freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
#              └ base = rope_theta = 10000（频率基数，配置项）┘
#       └─────────── 算出来的 freqs 才是角频率 θ_i ───────────┘
```

| 名字 | 在代码 | 是什么 |
|---|---|---|
| `rope_theta` / `base` | `base=10000` | **频率基数**（固定大常数，原料） |
| 角频率 `θ_i` | 算出的 `freqs` | 每组转速，**由 base 生成**（产物） |

关系：`θ_i = 1.0 / base**(2i/head_dim)`。所以 **rope_theta 是"原料"，角频率是"产物"**；Q13 里学的"每组转速 θ_i"才是真角频率。

**rope_theta 控制什么**：高频→低频的跨度（频率衰减多快）。
- base **大**（1e6）→ 最低频更慢 → 能区分**更远的位置** → 适合**长上下文**。
- base **小** → 频率整体偏快 → 长距离容易"转过头绕回来"分不清。

→ 这就是为什么**长上下文模型把 rope_theta 从 10000 调大到 1e6**：把最低频拉得更慢，分辨更长序列的位置。

```python
# head_dim=4
base=10000 → 第1组 θ = 1/10000^0.5 = 0.01
base=1e6   → 第1组 θ = 1/1e6^0.5   = 0.001  ← 更慢，能分辨更远
```

---

### Q13c. FFN 里 `F.silu(...)` 选的是什么 activation？

```python
def forward(self, x):
    return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

⚠️ 两个名字别混：**SiLU** 是那个激活函数；**SwiGLU** 是用 SiLU 搭起来的整个 FFN 结构。

**拆成三步看**：
```python
gate = F.silu(self.gate_proj(x))   # ① 一条路：gate_proj 后过 SiLU 激活
up   = self.up_proj(x)             # ② 另一条路：up_proj，不激活
out  = self.down_proj(gate * up)   # ③ 两路逐元素相乘，再降回原维度
```
关键：**两条并行 Linear，只有一条过激活，然后相乘** —— "一条当门乘到另一条上"就是 **GLU（门控）**；门用 SiLU → 合称 **SwiGLU**。

**SiLU（= Swish）本身**：
```python
def silu(x):
    return x * torch.sigmoid(x)    # F.silu 就是这个
```
拿 sigmoid（0~1 的软阀门）去缩放 x：x 很大→≈原样放过；x 很负→压到≈0；0 附近平滑过渡。
比 ReLU（`max(0,x)` 负的直接砍 0）好在**平滑可导、负区保留一点梯度**，训练更稳。

**和老式 FFN 对比**：

| | 老式 FFN（GPT-2） | SwiGLU FFN（你这份，LLaMA/Qwen 主流） |
|---|---|---|
| Linear 数 | 2（up+down） | **3（gate+up+down）** |
| 激活 | `down(relu(up(x)))` | `down(silu(gate(x)) * up(x))` |
| 效果 | 基线 | 同参数量下 loss 更低 |

> 多了一条 `gate_proj`，为不让总参数爆，现代模型把中间维度 `intermediate_size` 设得比"4×hidden"小（常见 ≈ `8/3 × hidden`）。看到 intermediate_size 不是 hidden 整 4 倍就是这原因。

---

### Q14. KV Cache 里 `past_kv` 的 shape？

`past_kv` 是 tuple `(past_k, past_v)`，每个 tensor 形状：
```
[B, num_keyvalue_heads, past_seq_len, head_dim]
 └批量┘ └KV头数(GQA, 少)┘ └已缓存长度┘ └每头维度┘
 dim0       dim1            dim2        dim3
```
⚠️ 是 `num_keyvalue_heads` 不是 `num_heads`（GQA，K/V 头更少；`repeat_kv` 之后才扩展开）。

`torch.cat((past_k, k), dim=2)` 在 **seq 维（dim2）** 上拼：历史 K/V + 新 token 的 K/V。
```
past_k: [B, kv_heads, past_seq_len, head_dim]
   k:   [B, kv_heads,      S,       head_dim]
─────────────────────────── cat(dim=2)
   k:   [B, kv_heads, past_seq_len+S, head_dim]
```

| 阶段 | S | past_kv | cat 后 seq 长 |
|------|---|---------|--------------|
| prefill 首次 | 10 | None（跳过 cat） | 10 |
| 生成第 11 个 | 1 | seq=10 | 11 |
| 生成第 12 个 | 1 | seq=11 | 12 |

核心：decode 阶段每次只 forward 1 个新 token，历史 K/V 从 cache 取出拼上，不重算。

---

### Q14b. prefill vs decode：推理的两个阶段

LLM 推理生成分两段，KV Cache 正是连接它俩的桥梁：

```
prompt="今天天气怎么样" → 答"今天天气很好"
阶段1 Prefill：把整个 prompt 一次性并行喂进模型，算好全部 K/V 填进 cache
阶段2 Decode： 一个字一个字生成，每次只 forward 1 个新 token
```

**Prefill**：`model(input_ids, past_kv=None)` —— prompt 已知 → 整段并行算（像训练），把所有 token 的 K/V 填进 cache（"pre-**fill**" 名字由来），取最后一个 token 的 logits 预测第一个输出字。

**Decode**：`model(next_token, past_kv=cache)` —— 每步只喂 1 个新 token（S=1），历史 K/V 从 cache 取（Q14）。

| | Prefill | Decode |
|---|---|---|
| 处理 | 整个 prompt | 1 个新 token |
| S | prompt 长度（多） | 1 |
| 并行度 | 高（一次算完） | 低（逐个） |
| past_kv | None | 上一步的 cache |
| 瓶颈 | **compute-bound**（大批矩阵乘） | **memory-bound**（每吐1字都要从 HBM 搬全部权重+cache） |

> 为什么分两段：prompt 已知 → 能并行（prefill 快）；生成部分未知 → 必须逐个等（decode）。
> 体验对应：第一个字要等一下 = prefill 在处理 prompt（TTFT，prompt 越长越久）；之后字一个个蹦 = decode。
> 串联：decode 是 memory-bound（见 day01 Q11b 内存墙）；prefill 是 compute-bound。

---

### Q15. causal mask 的 `diagonal = total_S - S + 1` 为什么这么写？

`scores` 形状 `[B, num_heads, S, total_S]`，mask 是 `[S, total_S]`。
- `total_S = past_seq_len + S`（所有 key 总长），`S` = 这步新 query 数。
- 带 KV Cache 时 query 位置不从 0 起：**第 i 个 query 的真实位置 = past_seq_len + i**。

要遮住"key 位置 > query 位置"，即 `j > past_seq_len + i`，等价 `j - i >= (total_S - S) + 1`。
`torch.triu(M, diagonal=d)` 保留 `j - i >= d` 的位置填 `-inf` → 所以 `d = total_S - S + 1`。

例：past=2, S=3 → total_S=5, d=3：
```
        key→ 0    1    2    3    4
q(位置2)     ·    ·    ·   -inf -inf
q(位置3)     ·    ·    ·    ·   -inf
q(位置4)     ·    ·    ·    ·    ·
```
**纯 prefill**（past=0, S=total_S）→ d=1，退化成标准下三角 mask。
一个公式同时覆盖 prefill 和 decode 两种情况，不用写两套逻辑。

---

### Q16. `attn_weight = F.softmax(scores, dim=-1)` 在干什么？

把"原始相关度分数"变成"注意力权重"——一组 ≥0、加起来=1 的占比。

**softmax 在 torch 里就是三步**（不用记数学符号）：
```python
scores = torch.tensor([2.0, 1.0, 0.5])   # 某 query 对 3 个 key 的原始分数
exp = torch.exp(scores)          # ① 每个取指数 → [7.39, 2.72, 1.65]（拉开差距+变正数）
total = exp.sum()                # ② 全加起来当分母 → 11.76
attn_weight = exp / total        # ③ 各除以总和 → [0.63, 0.23, 0.14]，和=1
```
`F.softmax(scores, dim=-1)` 就是把这三步打包成一行。

**作用在哪一维**：`scores` 形状 `[B, num_heads, S, total_S]`，`dim=-1` = 最后一维 = `total_S`（所有 key）。
→ 即"每个 query 在所有 key 上分配注意力"，每行加起来=1。其他维不能做：B 跨样本、num_heads 各头独立、S 跨 query 都没意义。

**和 mask 的配合**（causal 闭环）：
```python
scores = torch.tensor([2.0, 1.0, 0.5, float("-inf")])  # 最后一个是被 mask 的未来 key
F.softmax(scores, dim=-1)   # → [0.63, 0.23, 0.14, 0.00]
```
因为 `torch.exp(float("-inf")) == 0` → 被 mask 的位置权重自动变 0 → query 看不到未来 token。

**这组权重接着用来**：`output = attn_weight @ v`，按占比加权平均所有 value 向量（60%的key0的v + 22%的key1的v + …）。所以 softmax 输出 = "这个 query 该从哪些 token、各取多少信息"的配方。

**⚠️ transformer 里有两个 softmax，别混**：

| | ① Attention softmax（本题） | ② 输出/采样 softmax |
|---|---|---|
| 在哪 | attention 层内部 | 模型最后、生成时（`torch.softmax(logits, dim=-1)`） |
| 作用维 | **key 维**（total_S） | **vocab 维**（词表） |
| 把什么→什么 | scores → **注意力权重**（query 对各 key 的关注度） | logits → **下一个 token 的概率分布** |
| 加起来=1 含义 | "注意力如何分配给各 token" | "下个词是各候选的概率" |
| mask | 有（causal，-inf） | 无（但有 temperature/top-p） |

→ eval 生成代码里 `torch.softmax(logits_last, dim=-1)` 是 **②**，后面 `multinomial` 按它采样下一个词；本题 `F.softmax(scores)` 是 **①**。面试主动点破这俩区别是加分点。

**scaled dot-product 的 `1/√d` 缩放**（score 计算那步，常被追问）：`scores = q @ k.T / sqrt(head_dim)`。d 维向量点积随 d 增大而变大；score 太大 → softmax 接近 one-hot → 梯度极小 → 训练停滞。除以 √d 让 score 方差稳定，与 head_dim 无关。**真正原因是防 softmax 饱和/梯度消失，不是泛泛"防数值太大"**。

---

### Q17. `generate_loss_mask`：为什么要 for 循环铺整段，能不能边找 end 边标？

`loss_mask[j]=1` = "第 j 个 token 要算 loss（要学着预测它）"。训练是**逐 token 预测**，所以 assistant 回答里**每个 token 都要标 1**，不是只标某一个。

原版（先纯找 end，再 for 一次性铺）：
```python
while end < len(input_ids):                    # 只负责找 eos 位置，不标
    if input_ids[end:end+len(eos)] == eos: break
    end += 1
for j in range(start, min(end + len(eos), self.max_length)):  # 再统一铺 start~eos
    loss_mask[j] = 1
```

**想合并成"边找边标"的写法**：
```python
while end < len(input_ids):
    if input_ids[end:end+len(eos)] == eos: break  # ← break 在标记前面！
    loss_mask[end] = 1
    end += 1
```
对**回答内容**等价，但有个 bug：**break 在 `loss_mask[end]=1` 之前 → eos 那个位置永远没被标**。

**关键：eos 必须算 loss。** 我们就是要模型学会输出 `<eos>` = 学会"**回答到这该停了**"。
- eos 标了 → 推理时模型会正常停下来。
- eos 没标 → 模型学不会何时停 → 推理时**停不下来、一直唠叨**（常见 bug）。

原版 `range(start, end + len(eos))` 里的 `end + len(eos)` 就是**故意把 eos 含进去**。
```
位置:   2    3    4    5    6
token: 推荐 机器 学习 实战 <eos>
原版:   1    1    1    1    1     ← eos 也标，学会停 ✅
合并:   1    1    1    1    0     ← eos 漏标，学不会停 ❌
```
另：原版 `min(..., self.max_length)` 还顺手做了截断保护（防越界）。

---

## 本日踩坑速查（手写 LoRA 时的 5 个 bug）

| 错 | 对 | 后果 |
|----|----|------|
| `class LoRa(nn.module)` | `nn.Module` | 大小写，基类错 |
| `model.name_modules()` | `named_modules()` | 拼写，AttributeError |
| `LoRa(...)` 没 `.to(device)` | `.to(device)` | GPU 时设备不一致 |
| `original_forward = model.forward` | `module.forward` | 存成整个模型的 forward → 无限递归 |
| `p.quires_grad = False` | `p.requires_grad` | **静默不报错**！凭空造了没用的新属性，底座没冻结，全模型在训练 |

> ⚠️ 最危险的是 `requires_grad` 拼错：Python 允许随便加新属性，拼错名字不报错，导致"以为冻结了其实没冻"。属性名一定要拼对。

---

## Part 4：Scaling Law（Chinchilla 最优配置）

### Q18. `tokens ≈ 20 × params` 这个 Chinchilla 规律到底怎么推算配置？

卡点不是"20 怎么乘"（那只是 `tokens = 20 * params`），而是 **20 这个数字从哪来、为什么是它**。

**知识锚点：你不是自由选参数和数据，你在花一笔固定的算力预算（compute, FLOPs）。**
真正稀缺的资源不是参数量也不是数据量，而是 GPU 算力。参数量 `N` 和数据量 `D` 都要花这笔算力，关系近似为：
```python
# 训练总算力 ≈ 6 × 参数量 × token 数
# 6：forward ≈ 2 FLOPs/param/token，backward ≈ 2 倍，合计 ≈ 6
compute = 6 * N * D   # 单位 FLOPs
```
预算 `compute` 定死后，`N` 和 `D` 只能此消彼长：模型做大 → 能喂的数据变少；想喂更多数据 → 模型得做小。

**20 是实验测出来的，不是公式推出来的。** Chinchilla（DeepMind, 2022）纯靠跑实验：
```python
for compute_budget in [小, 中, 大, ...]:        # 很多档算力
    for (N, D) in 所有满足 6*N*D == compute_budget 的组合:
        loss = train_and_eval(N, D)             # 真训出来测 loss
    记录这档预算下 loss 最低的 (N, D)
```
跑完几百个模型发现：**每档算力下 loss 最低的配置，D/N 都稳定在 ≈ 20**。就像 9.8 是测出来的重力加速度，不是推导。

**给定算力怎么推算配置**——联立两个约束解方程：
```python
C = 5.76e21          # 你的算力预算 (FLOPs)
# 约束1（预算）:    C = 6 * N * D
# 约束2（最优比例）: D = 20 * N
# 代入: C = 6*N*(20*N) = 120*N^2  →  N = sqrt(C/120)
N = (C / 120) ** 0.5
D = 20 * N
print(N, D)   # ≈ 6.9e9 参数, ≈ 1.4e11 tokens → 差不多 7B 配 140B
```
反向验证那张表（1B→20B、7B→140B、70B→1.4T）就是单纯把参数量乘 20：
```python
N = 7e9; D = 20 * N          # = 140B tokens ✓
C = 6 * N * D                # ≈ 5.9e21 FLOPs，训它需要的算力
```

**逻辑链**：有一笔算力 C → 联立 `C=6ND` 和 `D=20N` → 解出唯一最优 `(N,D)` → 体现为 `tokens ≈ 20 × params`。

【可深入】20 是"训练最省算力"的最优。但若之后要大量 **inference（部署）**，会故意把模型做小、数据喂超多（如 LLaMA 7B 喂了 1T+ tokens，远超 140B）——小模型推理便宜，值得训练时多花数据。这叫 **inference-aware scaling**，假设和 Chinchilla 不同。

---

## Part 4：推理阶段（Inference）完整流程

> 以 minimind 的 `eval_llm.py` 为地图。推理 vs 训练：推理没标准答案、不更新权重、只前向（`eval()`+`no_grad`），**自回归一个个吐 token**。核心两件事：① 高效前向（KV Cache、prefill/decode）② 怎么从输出分布挑下一个 token（采样）。

### Q19. 推理完整 pipeline（7 步）

```
① 加载模型+tokenizer → ② 拼 prompt → ③ tokenize → ④ generate(prefill+decode循环)
   → ⑤ 每步采样挑 token → ⑥ 停止判断 → ⑦ decode 回文字（流式）
```

| 步 | 代码 | 干什么 |
|---|---|---|
| ① 装模型 | `model.half().eval().to(device)` | `.half()`=FP16省显存提速；`.eval()`=关dropout；load 权重/LoRA |
| ② 拼 prompt | `tokenizer.apply_chat_template(conv, add_generation_prompt=True)` | 套对话模板（`<|im_start|>user...assistant`），让模型进入"该我回话"状态。pretrain 模型只加 bos |
| ③ tokenize | `tokenizer(inputs, return_tensors="pt")` | 文字→token id |
| ④ generate | `model.generate(input_ids, max_new_tokens=...)` | 内部=prefill+decode循环+KV Cache（=自己手写的那个 for 循环） |
| ⑤ 采样 | temperature/top_p/penalty | 从 logits 挑下一个 token（见 Q20） |
| ⑥ 停止 | `eos_token` 或 `max_new_tokens` | 模型说完 / 达上限兜底 |
| ⑦ 输出 | `TextStreamer` + `tokenizer.decode` | 边生成边打印（流式）+ id→文字 |

⚠️ 注意：tokenize 那两行（`tokenizer(prompt)...` + `generated_ids=prompt_ids[0].tolist()`）是**预处理**，不是 prefill。**prefill = 循环第一次 forward**（整段 prompt + `past_kv=None`）；之后 `prompt_ids` 换成单 token 转入 decode（同一循环共用代码）。

chat template 训练/推理用同一套 → 对得上才不答非所问（呼应 SFT/DPO 数据的 `apply_chat_template`）。

### Q20. 采样策略：怎么从 logits 挑下一个 token ⭐

每个 decode 步模型输出 logits（对词表每个 token 打分），"挑哪个"是控制生成质量/风格的核心旋钮：

```python
# a) greedy vs sampling
do_sample=False → argmax 取最高（确定但易死板重复）
do_sample=True  → 按概率随机采样（自然多样）

# b) temperature 温度：调随机程度
probs = softmax(logits / temperature)
# 小(0.2)→分布尖→保守确定(代码/事实)；大(1.2)→分布平→随机有创意(写作)。minimind 默认 0.85

# c) top-p / nucleus：砍长尾烂选项
top_p=0.95  # 只在"累计概率达95%的那撮高概率token"里采样，防蹦出离谱字
# 兄弟方案 top-k：只留概率最高的 k 个

# d) repetition penalty：防复读
# 对已生成过的 token 降低 logits，防"我我我"或整句循环
for tid in set(generated_ids):
    logits[tid] = logits[tid]/1.3 if logits[tid]>0 else logits[tid]*1.3
```

> 一句话：**temperature 调随机、top-p 砍长尾、repetition penalty 防复读**，三个一起控制"自然又不跑偏"。

### Q21. 推理阶段"该了解的环节"清单

| 环节 | 关键词 |
|---|---|
| 精度 | `.half()` FP16 省显存提速 |
| 模式 | `.eval()` + `no_grad` |
| prompt 构造 | chat template / system / 多轮 history |
| 高效前向 | prefill + decode + KV Cache（见 Q14/Q14b） |
| **采样策略** | temperature / top-p / top-k / repetition penalty / greedy（最该补，Q20） |
| 停止 | eos / max_new_tokens |
| 输出 | TextStreamer 流式 / decode |
| 多轮对话 | conversation 累积 + 截断 history |
| 性能指标 | tokens/s（decode速度）、TTFT（首字延迟，prefill 决定） |
| 进阶【未来】 | 量化 / batch 推理 / speculative decoding / vLLM |

---

## Part 5：logits 的统一理解 + loss/采样的 shape

### Q22. 算 loss 时 `x, y` 的 shape 是什么？

```python
x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
```

设 `logits = [B, T, V]`、`labels = [B, T]`：

| 变量 | 切片后 | view 后（送进 cross_entropy） |
|------|--------|------------------------------|
| `x`（预测） | `[B, T-1, V]` | `[B*(T-1), V]` |
| `y`（标签） | `[B, T-1]` | `[B*(T-1)]` |

- **切片是 autoregressive shift（错位一格）**：用位置 `t` 的 logits 预测位置 `t+1` 的 token，所以 `x` 砍掉最后一个位置（没有下一个 token 可预测），`y` 砍掉第一个 token（它没有前文）。长度都变 `T-1`。
- **view 是为了满足 `F.cross_entropy` 的接口**：input 要 `[N, C]`、target 要 `[N]`。把 `B` 和 `T-1` 两维压平 → `B*(T-1)` 个独立分类样本，每个是 `V` 维分类问题。
- `ignore_index=-100`：`y` 里值为 `-100` 的位置不算 loss —— 这就是 SFT loss mask 的实现（prompt 部分 label 设 -100，只对 answer 算 loss）。

---

### Q23. greedy decoding：`next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)`

```python
logits, _, past_key_values = model(input_ids, past_key_values=past_key_values)
next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
```

**logits 的 shape `[B, T, V]`，但因为有 KV Cache，`T` 两种情况不同：**
- 第一次（prefill）：`input_ids` 是整个 prompt → `T` = prompt token 数，如 `[1, 4, V]`。
- 之后（decode）：`input_ids = next_token` 只喂新的 1 个 token → `T=1`，logits 是 `[1, 1, V]`。历史在 `past_key_values` 里，不重算。

**逐步拆 shape：**
```python
logits                         # [B, T, V]
logits[:, -1, :]               # [B, V]   取最后一个位置（要预测序列后面接什么）
.argmax(dim=-1, keepdim=True)  # [B, 1]   在 V 维取分数最大 token 的 index
```
- 取 `-1`：自回归里位置 `t` 的 logits 预测 `t+1`，生成时只关心最后位置的输出。
- `argmax(dim=-1)`：在 vocab 维找最大分数 token 的 index —— 这就是 "greedy"，不采样。
- `keepdim=True`：保持 `[B, 1]`（不降成 `[B]`），好让下一步 `input_ids = next_token` 直接当 `[B, T=1]` 输入喂回模型。

---

### Q24. logits 在 pretrain / SFT / DPO 里到底代表什么？（统一理解）⭐

**结论：logits 永远是同一个东西 —— 模型最后一层对每个位置、对整个 vocab 的「未归一化分数(raw scores)」，shape 永远 `[B, T, V]`。三处含义完全不变，变的只是「拿这些分数去算什么 loss」。**

- logits 第 `i` 个值 = 「下一个 token 是词表第 `i` 个词」的未归一化分数。
- 过 `softmax` → 概率；过 `log_softmax` → log-probability(log-prob)。
- logits 本身不带任何"训练目标"信息，三阶段拿到的 logits 一模一样。

| 阶段 | 怎么用 logits | loss |
|------|--------------|------|
| **Pretrain** | 对所有位置算预测下一 token 的 cross-entropy | `CE(logits, labels)`，全部 token |
| **SFT** | 同 CE，但只对 answer 算（prompt 用 -100 mask） | `CE` + loss mask |
| **DPO** | 不直接 CE：先把 logits 转成整句 log-prob，再比较 chosen vs rejected | preference loss |

**关键洞察**：pretrain 和 SFT 是同一个 loss（cross-entropy），SFT 只多了 loss mask。DPO 才真正换了用法。

**DPO 里 logits → loss 的路径**：
```
1. logits ──log_softmax──> 每个位置每个token的 log-prob   [B, T, V]
2. 用真实 label gather ──> 每个位置「实际token」的 log-prob  [B, T]
3. 对 answer 求和 ──> 整句 log-prob（标量）  logp(answer)
```
一条数据算 4 个标量：`logp_policy(chosen/rejected)`、`logp_ref(chosen/rejected)`（ref 是冻结的 reference model），代入：
```
loss = -log σ( β · [ (logp_policy(chosen) - logp_ref(chosen))
                   - (logp_policy(rejected) - logp_ref(rejected)) ] )
```
直觉：让 policy 相比 reference 更抬高 chosen、压低 rejected 的概率。`β` 控制偏离 reference 的强度。

**连接**：cross-entropy 本质就是 `-log_prob(真实token)`。SFT 直接最小化 `-log_prob(answer)`（学会生成答案）；DPO 比较两个答案的 log_prob 差（学会偏好好答案）。两者都从 logits 取「真实 token 那一项的分数」，只是怎么用不同。

---

## Part 6：English Interview Version（英文面试讲解版）

> 把 Q22–Q24 的核心整理成英文口语表达，方便面试 / 读英文文档时复用。每段先给一句 "headline"，再展开。

### A. Loss computation & the autoregressive shift (Q22)

> "In a causal language model, the logits have shape `[batch, seq_len, vocab_size]`, and the labels are `[batch, seq_len]`. Before computing the loss we do an **autoregressive shift**: we take `logits[:, :-1, :]` and `labels[:, 1:]`, so the prediction at position `t` is matched against the actual token at position `t+1`. Both become length `T-1`.
>
> Then we flatten them to feed `cross_entropy`, which expects input `[N, C]` and target `[N]`. So `x` becomes `[B*(T-1), vocab_size]` and `y` becomes `[B*(T-1)]` — every position is treated as an independent classification over the vocabulary. The `ignore_index=-100` means any label set to -100 is skipped in the loss; that's exactly how the **SFT loss mask** works — we mask out the prompt tokens and only compute loss on the answer."

Key vocabulary: *autoregressive shift, cross-entropy, flatten, loss mask, ignore_index, classification over the vocabulary.*

### B. Greedy decoding step (Q23)

> "During generation the model returns logits of shape `[batch, seq_len, vocab_size]`. With a **KV cache**, the sequence length differs by phase: on the first forward pass (prefill) we feed the whole prompt, so `seq_len` equals the prompt length; after that (decode) we only feed the single newly generated token, so `seq_len` is 1 and the past context lives in `past_key_values`.
>
> To pick the next token we take `logits[:, -1, :]` — the **last position**, because in an autoregressive model the logits at the final position predict what comes next. That gives `[batch, vocab_size]`. We then `argmax` over the vocab dimension to get the index of the highest-scoring token — this is **greedy decoding**, no sampling. We keep `keepdim=True` so the shape stays `[batch, 1]`, which can be fed straight back into the model as the next input."

Key vocabulary: *KV cache, prefill vs decode, last position, argmax over the vocab dimension, greedy decoding, keepdim.*

### C. What logits mean across pretrain / SFT / DPO (Q24)

> "Logits are always the same thing — the model's final-layer **raw, unnormalized scores** over the entire vocabulary for each position, shape `[batch, seq_len, vocab_size]`. Their meaning never changes across training stages; what changes is **how we turn them into a loss**.
>
> In **pretraining**, we apply cross-entropy over all positions to predict the next token. In **SFT**, it's the same cross-entropy, but with a loss mask so only the answer tokens count. So pretrain and SFT share the same loss — SFT just adds masking.
>
> **DPO** is different. Instead of cross-entropy, we convert logits into a **sequence-level log-probability**: apply log-softmax, gather the log-prob of the actual token at each position, and sum over the answer. For each example we compute four scalars — the policy model's log-prob and the frozen reference model's log-prob, each for the chosen and the rejected response. The DPO loss is `-log σ(β · [(logp_policy_chosen - logp_ref_chosen) - (logp_policy_rejected - logp_ref_rejected)])`. Intuitively, it pushes the policy to raise the probability of the preferred answer and lower the probability of the rejected one, relative to the reference model.
>
> The unifying point: cross-entropy is just `-log_prob` of the true token. SFT minimizes that directly to learn to *generate* the answer; DPO compares the log-probs of two answers to learn a *preference*. Both read off the same logits — only the objective differs."

Key vocabulary: *raw/unnormalized scores, log-softmax, gather, sequence-level log-probability, policy vs reference model, preference loss, the objective differs.*

---

### D. Follow-up drill（面试官常接的追问）

> 套路:**先一句 headline 给结论,再补 1–2 句 why**,不要一上来堆细节。

**主题 A — Loss & autoregressive shift**

**Q1. "Why `-100` specifically? Could you use any other number?"**
> Headline: "-100 isn't magic — it's just PyTorch's default value for `ignore_index` in `cross_entropy`."
> 要点:无数学含义,纯 PyTorch 默认值。可传任何不与真实 token id 撞车的值(token id 非负 → -100 永远非法)。

**Q2. "After the shift, the sequences are length `T-1`. Doesn't that waste one token of data?"**
> Headline: "No — it's unavoidable, not wasteful. The last token has no 'next token' to predict, and the first token has no context to be predicted from."
> 要点:结构必然,非浪费。长序列里 `T-1 ≈ T`,可忽略。

**主题 B — Greedy decoding & KV Cache**

**Q3. "You said decode is memory-bound. Why memory-bound and not compute-bound?"**
> Headline: "Each decode step processes only one token, so there's almost no matrix-multiply work — but you still read all the model weights and the KV cache from HBM. The bottleneck is memory bandwidth, not FLOPs."
> 要点:decode 每步 1 token,算量极小,却要搬全部权重+KV cache。对比 prefill 整段并行 → compute-bound。(呼应 day01 内存墙)

**Q4. "What breaks if you forget `keepdim=True`?"**
> Headline: "The token tensor collapses from `[B, 1]` to `[B]`; fed back as the next input, the model expects a 2-D `[batch, seq_len]` tensor — shape error or wrong broadcasting."
> 要点:降维成 `[B]` → 喂回模型 shape 不匹配。`keepdim=True` 保住 `[B,1]`(T=1)。

**主题 C — logits 跨 pretrain/SFT/DPO**

**Q5. "In DPO, why do you need a frozen reference model at all? Why not just maximize the chosen log-prob?"**
> Headline: "The reference model is a regularizer — it keeps the policy from drifting too far and collapsing. Without it, the model could blow up the chosen probability while destroying its general language ability."
> 要点:ref = KL regularizer,防 over-optimize / reward hacking / mode collapse。`β` 控制允许偏离多少。这也是 DPO 用 log-prob **差值**而非绝对值的原因。

**Q6. "Cross-entropy and the DPO log-prob both come from logits. So is SFT just a special case of DPO?"**
> Headline: "Not quite — they optimize different things. SFT does *imitation* (maximize one answer's likelihood); DPO does *contrastive preference* (rank one answer above another). DPO needs paired chosen/rejected data; SFT only needs one target."
> 要点:目标不同——SFT 模仿单答案,DPO 对比排序成对数据。共同点只是"都从 logits 取真实 token 的 log-prob"。
