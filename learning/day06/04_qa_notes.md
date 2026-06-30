# Day 6 问答笔记：DPO 原理 + 手写实现

> 本篇是手写 `model/Mymodel_dpo.py` 时的问答整理，分三部分：
> ① DPO 是什么、和 SFT/PPO 的区别；② 三个核心函数（DPODataset / get_seq_logprob / dpo_loss）；③ 数字推演 + 踩坑。

---

## Part 0：LoRA 之后学什么 —— 进入对齐（alignment）阶段

主线：`pretrain → SFT → LoRA(高效微调) → [DPO → 蒸馏 → GRPO]`

推荐顺序（按依赖 + 难度爬坡）：
1. **DPO**：离线偏好，只换 loss，最接近 SFT（先学这个）
2. **Distillation**：离线，换成 KL loss + teacher 模型
3. **rollout_engine**：读懂"训练中如何实时生成采样"（GRPO/PPO 的基础设施）
4. **GRPO**：在线 RL，minimind 主推
5. **PPO**：更经典但更重（value model），理解 GRPO 后回看更轻松
6. **Agent**：最复杂，RL + 工具调用

> 关键分家：**离线偏好**（DPO/蒸馏，训练时不生成）vs **在线 RL**（PPO/GRPO/agent，依赖 rollout）。

---

## Part 1：DPO 是什么

**一句话**：用「好答案 vs 坏答案」的对比来教模型，不绕弯训练打分器（reward model）。

- **类比**：SFT 像"照菜谱抄"（给你标准答案模仿）；DPO 像"试吃对比"（告诉你 A 比 B 好，往 A 靠、离 B 远）。
- SFT 只告诉你"对的长什么样"，DPO 还告诉你"**错的长什么样**"——这正是对齐需要的信号。

### 和 SFT 的区别

| 维度 | SFT | DPO |
|------|-----|-----|
| 数据 | `(prompt, answer)` 一条 | `(prompt, chosen, rejected)` 一对好坏 |
| 学什么 | 模仿正确答案 | **拉开**好/坏的差距 |
| 模型数量 | 1 个 | **2 个**：policy + 冻结的 reference |
| 前置 | pretrain 后 | **必须先有 SFT 模型** |
| loss | cross-entropy | dpo_loss |

### 和 PPO/RLHF 的区别（面试考点）

```
传统 RLHF：preference → 训练 reward model → RL(PPO) rollout+打分+更新   （三步，重、不稳）
DPO：      preference →（数学推导）→ 直接一个 loss 更新 policy          （一步，轻、稳）
```
**DPO 为什么不需要 reward model 和 rollout？** 因为数学推导证明"先学 reward model 再做 RL"可以合并成一个 closed-form 的 loss，reward 被隐含在 policy 和 reference 的 log 概率比里（implicit reward）。

---

## Part 2：三个核心函数

### ① DPODataset —— 数据长什么样

每条样本 = 同一个 prompt 的 **chosen + rejected 两条对话**。对两条分别做 SFT 那三步（apply_chat_template → tokenizer 编码 → generate_loss_mask），返回 6 个张量：
```
x_chosen, y_chosen, mask_chosen        ┐ 好回答
x_rejected, y_rejected, mask_rejected  ┘ 坏回答
```
- `x = ids[:-1]`, `y = ids[1:]`：**next-token 错位**（teacher forcing），错位在 dataset 里就做好了。
- `loss_mask`：只在 assistant 回答处 = 1，prompt/padding = 0（只比较回答）。

> **错位直觉**：让"位置 i 的输入"对齐"位置 i 的标签 = 原序列第 i+1 个 token"。
> `x` 砍最后一个（末尾没有下一个可预测），`y` 砍第一个（开头不是任何位置的"下一个"）。

### ② get_seq_logprob —— 一条回答压成一个总分

```python
def get_seq_logprob(model, x, y, mask):
    logits, _, _ = model(x)                                   # [B, seq, vocab]
    log_probs = F.log_softmax(logits, dim=-1)                 # [B, seq, vocab]
    token_logp = torch.gather(log_probs, dim=2, index=y.unsqueeze(-1)).squeeze(-1)  # [B, seq]
    seq_logp = (token_logp * mask).sum(dim=-1)                # [B]
    return seq_logp
```

形状流：`x[B,seq] → logits[B,seq,vocab] → log_probs[B,seq,vocab] → token_logp[B,seq] → seq_logp[B]`

**logits / 概率 / log 概率 三者关系**：
```
logits ──softmax──▶ 概率(0~1,和=1) ──取log──▶ log概率(负数)
原始分数            归一化            越接近0越好、越负越差
```
- log 概率用加法代替乘法（整句概率 = 每 token 概率连乘 → 取 log 变 sum），且数值稳定、梯度好算。
- `log_softmax(dim=-1)`：在 **vocab 维**归一化（对每个位置的所有候选词）。

**gather 在干什么**（最易卡）：在每个 (样本,位置)，按 `y` 的值当列号，从那一排候选词里**挑出真实那个词的 log 概率**。
- `y.unsqueeze(-1)`：`[B,seq] → [B,seq,1]`，凑维度数（gather 要求 index 和源同维数）。
- `gather(dim=2)`：沿 vocab 维挑 → `[B,seq,1]`。
- `.squeeze(-1)`：去掉多余的 1 维 → `[B,seq]`。
- 直觉：模型对所有词都打了分，训练只关心"它给**正确答案**那个词打了多少分"。

### ③ dpo_loss —— 对比好坏

```python
def dpo_loss(ref_logp, policy_logp, beta=0.1):
    bs = policy_logp.shape[0] // 2
    chosen_policy, reject_policy = policy_logp[:bs], policy_logp[bs:]
    chosen_ref,    reject_ref    = ref_logp[:bs],    ref_logp[bs:]
    pi_logratios  = chosen_policy - reject_policy    # policy: 好−坏
    ref_logratios = chosen_ref    - reject_ref       # ref:   好−坏
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits).mean()
    return loss
```

**4 个值的网格**（chosen/rejected 是数据属性，policy/reference 是两个模型，互相正交）：
```
                好回答          坏回答
reference(冻结)  chosen_ref      reject_ref
policy(训练中)   chosen_policy   reject_policy
```
- `chosen_ref` vs `reject_ref`：**同一个冻结 ref 模型**分别看好/坏回答打的分。
- `ref_logratios`：ref 的"原始好坏偏好"，当**基准线**。
- 公式本质：
```
loss = -logsigmoid( β × [ (policy好−policy坏) − (ref好−ref坏) ] )
                          └─ policy 的偏好 ─┘   └─ 基准偏好 ─┘
```
**让 policy 的好坏偏好超过 ref 基准越多越好。** logits 越大 → loss 越小。

**reference 的作用**：当锚点。减去 ref 保证"在原有基础上改进偏好，而不是为拉大 margin 乱漂、把语言能力搞崩"。β 控制偏离幅度（0.1~0.5，太大学不动，太小 policy collapse 到 ref）。

---

## Part 3：数字推演（验证 dpo_loss）

```
policy_logp = [-5.0, -3.0, -6.5, -4.0]   # 上半好、下半坏
ref_logp    = [-5.2, -3.1, -5.0, -3.5]

pi_logratios  = [-5.0-(-6.5), -3.0-(-4.0)] = [1.5, 1.0]
ref_logratios = [-5.2-(-5.0), -3.1-(-3.5)] = [-0.2, 0.4]
logits        = [1.5-(-0.2), 1.0-0.4]      = [1.7, 0.6]   # 都>0 → policy 比 ref 更会区分
beta*logits   = [0.17, 0.06]
loss = mean(-logsigmoid([0.17,0.06])) ≈ 0.635
```
若某对学反了（pi_logratios 变负）→ logits 负 → loss 变大 → 梯度使劲纠正。

---

## Part 4：训练循环关键点 + 踩坑

```python
# ref 模型：深拷贝 + 冻结
ref_model = copy.deepcopy(model)
ref_model.eval()
ref_model.requires_grad_(False)   # 永久冻结（参数不更新）

for step, batch in enumerate(dpo_dataloader):   # ⚠️ 是 dataloader 不是 dataset
    x = torch.cat([batch["x_chosen"], batch["x_rejected"]]).to(device)   # dim=0 上下摞
    y = torch.cat([batch["y_chosen"], batch["y_rejected"]]).to(device)
    mask = torch.cat([batch["mask_chosen"], batch["mask_rejected"]]).to(device)  # ⚠️ 别漏 mask
    with torch.no_grad():                       # 临时不建图，省显存
        ref_logp = get_seq_logprob(ref_model, x, y, mask)
    policy_logp = get_seq_logprob(model, x, y, mask)
    loss = dpo_loss(ref_logp, policy_logp, beta=config.beta)
    optimizer.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
```

**两个冻结机制别混**：
| | 作用 | 针对 |
|---|---|---|
| `requires_grad_(False)` | 参数不更新（永久） | 模型参数，设一次 |
| `torch.no_grad()` | 这段前向不建图、不留激活（临时） | 一个代码块 |
ref 两个都用：声明永久冻结 + 每次 forward 省建图开销。policy **不能**加 no_grad（否则计算图断了，backward 传不回去）。

**torch.cat 要点**：`torch.cat([a, b], dim=0)` 第一参数是 list；dim=0 沿 batch 维上下摞（样本数翻倍，seq 不变）；除拼接维外其他维度必须一致（靠 padding="max_length" 保证等长）。

### 手写时踩的坑
| 错 | 对 | 后果 |
|----|----|------|
| `assitant` | `assistant` | bos_id 标记错，loss_mask 全错 |
| `padding="max_len"` | `padding="max_length"` | 参数名错 |
| `post_processing_chat(rejected)` | `(rejected_prompt)` | 传错对象 |
| `generate_loss_mask(chosen_input_ids)`（给 rejected 用） | 传 `rejected_input_ids` | 好坏 mask 串了 |
| `def train_dpo(ref_logp, ..., beta=config.beta)` | `def train_dpo():` | config 未定义，NameError（默认参数定义时求值） |
| `enumerate(dpo_dataset)` | `enumerate(dpo_dataloader)` | 拿到单条样本没 batch 维 |
| 漏定义 `mask` | cat mask_chosen/mask_rejected | NameError |

---

## 设计差异（对照参考 trainer/train_dpo.py）

| | 我的版本 | 参考 train_dpo.py |
|---|---|---|
| `*mask` + `.sum()` | 在 get_seq_logprob 里 | 在 dpo_loss 里 |
| dpo_loss 收到的 | 已是 `[B]` 总分 | 还是 `[B,seq]` 每 token |

两种都对。我的写法职责更分离：get_seq_logprob 负责"回答→分"，dpo_loss 只负责"对比"。

---

## Part 5：待修复 bug 清单（2026-06-26 review，明天自己改）

文件 `model/Mymodel_dpo.py`，结构和逻辑大方向已正确，剩 4 个 bug + 1 处清理：

| # | 位置 | 问题 | 严重度 |
|---|------|------|--------|
| 1 | L26 | `assitant` → `assistant` | 🔴 静默不学习 |
| 2 | L80,82 | 标记循环崩溃 + 注释没解开 | 🔴 崩溃 |
| 3 | L52 | rejected mask 传成 chosen | 🔴 数据错 |
| 4 | L168 | `config.beta` 不存在 | 🔴 崩溃 |
| 5 | L153-158 | 死代码（x_chosen 等没用上） | 🟡 清理 |

**Bug 1（最隐蔽）** — bos_id 拼写：
```python
self.bos_id=tokenizer(f"{tokenizer.bos_token}assistant\n", ...)   # assitant → assistant
```
拼错 → bos_id 和真实 input_ids 对不上 → generate_loss_mask 匹配不到 → loss_mask 全 0 → 模型学不到东西（无报错，loss 恒定）。

**Bug 2** — generate_loss_mask 标记循环（L80 解开注释 / L82 改写）：
```python
# 把 [start, end+eos] 标 1（min 两参数取小，防越界）
for j in range(start, min(end + len(self.eos_id), len(input_ids))):
    loss_mask[j] = 1
```
原来的 `min(end+self.max_len)` 只传一个 int → TypeError 崩溃。

**Bug 3** — rejected mask 传错输入（L52）：
```python
rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)   # 不是 chosen_input_ids
```

**Bug 4** — config.beta 不存在（Config 里没这个字段）。二选一：
- 简单：`dpo_loss(ref_logp, policy_logp, beta=0.1)`
- 规范：Config.__init__ 里加 `self.beta = 0.1`，再用 `config.beta`（推荐）

**清理 5** — 删掉 L153-158 没用上的 6 行，直接：
```python
x    = torch.cat([batch["x_chosen"],    batch["x_rejected"]]).to(config.device)
y    = torch.cat([batch["y_chosen"],    batch["y_rejected"]]).to(config.device)
mask = torch.cat([batch["mask_chosen"], batch["mask_rejected"]]).to(config.device)
```

> 改完 4 个 🔴 就能跑 `python model/Mymodel_dpo.py`。

---

## Part 4：后训练全景 + 蒸馏 + GRPO + CISPO

### Q. 后训练几种方法怎么定位？

| 方法 | 一句话 | 要采样 | 要 reward |
|---|---|---|---|
| SFT | 模仿固定答案 | 否 | 否 |
| DPO | 从"好/坏一对"学偏好 | 否 | 否（偏好对代替） |
| GRPO/CISPO | 模型自己生成→打分→强化好的 | **是（在线）** | **是** |
| 蒸馏 | 小模型模仿大模型输出分布 | 否 | 否（压缩，非对齐） |

关键：蒸馏=**压缩模型**；GRPO=**RL 对齐**，两类完全不同的东西。

---

### Q. 知识蒸馏（Knowledge Distillation）

**目标**：把大而强的 Teacher 的能力塞进小而便宜的 Student。

**核心：学软标签，不只学硬标签**
```
硬标签：正确答案是"猫"，其它全错
软标签：Teacher 给整个词表的分布 —— 猫0.7 狗0.2 老虎0.08 桌子0.001
```
软标签里"错误选项之间的相对关系"= **dark knowledge**，信息量远大于硬标签。

**loss（train_distillation.py:63）= 两部分加权**
```python
# ① 硬标签：和真实答案比（普通 SFT 交叉熵）
ce_loss = F.cross_entropy(student_logits, true_labels)
# ② 软标签：和 Teacher 分布比（KL 散度）
teacher_probs     = F.softmax(teacher_logits / T, dim=-1)
student_log_probs = F.log_softmax(student_logits / T, dim=-1)
kl_loss = F.kl_div(student_log_probs, teacher_probs) * (T ** 2)
# ③ 合并
loss = alpha * ce_loss + (1 - alpha) * kl_loss   # alpha=1纯SFT, 0纯蒸馏, 0.5各半
```
- **KL 散度**：衡量两个分布差多远，目标让 Student 贴近 Teacher。
- **temperature T>1**：把分布拉平，让小概率 token 的信号显现（dark knowledge 更明显）。minimind 默认 1.5。
- `*(T**2)`：补偿——除以 T 让梯度变小，乘回 T² 保持量级。
- **最大不同**：同时加载两个模型，Teacher 全程 `eval()`+`requires_grad_(False)` 只 forward。

---

### Q. GRPO（Group Relative Policy Optimization）

**目标**：RL 后训练——模型自己生成回答→reward 打分→强化高分、抑制低分。在线（边训边生成自己的数据）。

**流程（train_grpo.py）：**
1. **一题多答**：每个 prompt 采样 G 个回答（num_generations）→ "Group" 由来。
2. **打分**（:37）：格式分（`<think>`、长度）+ reward model 质量分 + 重复惩罚。
3. **advantage（精髓, :121）**：
```python
grouped = rewards.view(-1, num_generations)
mean_r = grouped.mean(dim=1); std_r = grouped.std(dim=1)
advantage = (rewards - mean_r) / (std_r + 1e-4)   # 比组内平均好多少
```
   - "Relative" 由来：不看绝对分，看**在同题这组里比平均好/差**。>0 鼓励，<0 抑制。
   - **省 critic**：PPO 要额外训 value/critic 模型估基准分；GRPO 直接用**组内平均当基准**，省掉整个 critic（比 PPO 简洁的关键）。
4. **policy loss（:134）**：
```python
ratio = torch.exp(per_token_logps - old_per_token_logps)
clipped_ratio = torch.clamp(ratio, 1-eps, 1+eps)
per_token_loss = -min(ratio*advantage, clipped_ratio*advantage) - beta*per_token_kl
```
   - clip：限制单步改动防训崩；KL 惩罚：拉住别离 ref 太远。

---

### Q. CISPO（GRPO 变体，改裁剪方式）

**解决的问题**：标准 GRPO/PPO 的 clip 有副作用——token 的 ratio 越界时 `min`+clip 让它**梯度归零**（这步不学了）。但越界 token 往往是最关键的"分叉点"token → 丢掉了最重要的信号。

**CISPO 怎么改（train_grpo.py:135）**：
```python
clamped_ratio = torch.clamp(ratio, max=epsilon_high).detach()  # 只截上限当权重 + detach
per_token_loss = -(clamped_ratio * advantage * per_token_logps - beta*per_token_kl)
```
- 不用 `min` 裁 token，只把 ratio 截顶当**权重**（detach 只当系数、不传梯度）。
- 梯度经 `per_token_logps` 照常流回**每个 token** → 没有 token 被裁成 0。

| | 裁剪对象 | 越界 token |
|---|---|---|
| GRPO/PPO | 裁 loss（min+clip） | 梯度归零，丢掉 |
| CISPO | 裁 ratio（当权重 detach） | 仍更新，保留 |

---

### 三者一句话

```
蒸馏 ：压缩。Teacher软标签 → Student用KL贴近。不采样、不要reward。
GRPO ：RL对齐。一题多答→打分→组内相对advantage→强化好的。要采样要reward、省critic。
CISPO：GRPO变体。改裁剪，不丢越界token梯度，保关键信号。
```

### Q1. 蒸馏/冻结模型时 `eval()` vs `requires_grad_(False)` vs `torch.no_grad()` 有什么区别？（Layer 6）

冻结一个 teacher（蒸馏）或 ref_model（DPO）时常一起出现这三者，但它们是**三件独立的事**，容易混。

| 写法 | 管什么 | 范围/性质 |
|------|--------|----------|
| `model.eval()` | forward 的**行为**（关 dropout / 用 BN 的 running stats）| 模式开关 |
| `model.requires_grad_(False)` | **参数级**冻结，不算/不存梯度 | 永久，in-place 批量 |
| `with torch.no_grad():` | 这段代码**完全不建计算图** | 代码块级，临时 |

**① `eval()`** —— 只影响"训练/推理行为不同"的层。在 MyModel 里就是 **Dropout**：
- train() 模式：随机丢弃神经元 → forward 带随机性
- eval() 模式：dropout 关闭 → 输出**确定、可复现**
- teacher 必须 eval()，否则同一输入每次分布都不同，student 学的目标会飘。

**② `requires_grad_(False)`** —— 把模型**所有参数**的 requires_grad 一次设成 False（结尾 `_` = in-place 批量，和 LoRA 里 `mark_only_lora_trainable` 同一个概念）。效果：参数不再算/存梯度 → ① 不被训练更新 ② 省大量显存。

**③ 三者正交，缺一不可的误区**：
- 以为"eval() 了就不算梯度" —— ❌ 错！eval() 只关 dropout，梯度照样算。
- 只 requires_grad_(False) 不 eval() —— dropout 还开着，分布带噪声。
- 所以 teacher 要**两个都设**：稳定(eval) + 冻结(requires_grad)。

**④ 和 no_grad 的配合**：模型层面 `eval()` + `requires_grad_(False)` 表明"这是冻结的 teacher"；forward 时再套 `with torch.no_grad():` 让那次前向彻底不建图、最省显存。

```python
teacher.eval()
teacher.requires_grad_(False)
with torch.no_grad():
    teacher_logits = teacher(input_ids).logits   # 稳定 + 零梯度开销
```

**一句话**：eval 管 forward 行为（关 dropout 求稳定），requires_grad_ 管参数冻结（不训练省显存），no_grad 管某段代码不建图；三者正交，teacher/ref_model 通常三个一起用。

### Q2. 知识蒸馏里 temperature（除以 T）为什么能软化分布？为什么还要乘 T²？（Layer 6）

## 原理：softmax 看的是「差距」，除以 T 缩小差距

softmax = `exp(x) / exp(x).sum()`，关键在 `exp()` 会**指数放大** logits 间的差距 → 差距大就尖、差距小就平。

除以 T(>1) 把所有 logits 的差距**等比例缩小** → 喂给 exp 的差距变小 → 放大效应变弱 → 分布变平（软化）。

```python
T=1:  softmax([2, 1, 0])    = [0.67, 0.24, 0.09]   # 尖
T=2:  softmax([1, 0.5, 0])  = [0.51, 0.31, 0.19]   # 平，次优词冒头
T=4:  softmax([.5,.25,0])   = [0.40, 0.31, 0.29]   # 更平
```
第3个词 0.09→0.19→0.29，T 越大越有"存在感" = 把次优词的 dark knowledge 捞出来。

**两个极端**：T→0 退化成 one-hot（最尖）；T→∞ 变均匀分布（最平）。T 就是"尖↔平"旋钮。
**名字来源**：物理 Boltzmann 分布——高温=分散、低温=集中。

## 为什么乘 T²

除以 T 会让梯度也等比缩小（大约 1/T² 量级）。乘回 `T**2` 把梯度量级补回来，使蒸馏 loss 的梯度和硬标签 CE 的梯度**量级可比**，方便 `alpha*CE + (1-alpha)*KL` 加权时两边平衡、调 T 时学习动态稳定。

## 💼 面试版回答（口头精简，30 秒能说完）

> "蒸馏里 temperature 用来**软化 teacher 的输出分布**。softmax 靠 exp 指数放大 logits 差距，分布往往很尖、把次优 token 的信息淹没了；**把 logits 除以 T 等比缩小差距，分布变平**，那些'次优但相关'的 token 概率显现出来——这就是蒸馏要传的 dark knowledge，比 one-hot 硬标签信息量大。
> 至于**乘 T²**：因为除以 T 会让 KL 的梯度按 1/T² 缩小，乘回 T² 让它的梯度量级和硬标签 CE 保持可比，这样两个 loss 加权才平衡、调 T 时训练才稳定。
> 极端情况：T→1 是原始分布，T→0 退化成 one-hot，T→∞ 趋于均匀。"

**追问可能**：T 一般取多少？→ 1~4 常用（本项目 1.5）。T 太大分布太平、信息糊；太小接近硬标签、失去软化意义。

### Q3. 蒸馏 KL loss 里，为什么 student 用 log_softmax 防数值问题，teacher 用普通 softmax 却不用担心？（Layer 6）

## 数值问题是什么
`log(softmax(x))` 不稳：某 logit 很负时 softmax 可能**下溢成 0.0**，`log(0.0)=-inf` → loss/梯度炸。
`log_softmax` 用 **log-sum-exp 技巧**直接算 log，不经过会下溢的中间小数，所以稳。理论上 teacher 也有这风险，但下面两点让它免疫。

## 根因：公式里两个 log 的「乘数」不对称
`F.kl_div` 逐元素算 `target*(log target - input) = p*log(p) - p*log(q)`：

| 项 | 谁的 log | 乘数 | 安全？ |
|----|---------|------|-------|
| `p*log(p)` | teacher | **p 自己** | ✅ p→0 时 p·log(p)→0，自动归零 |
| `-p*log(q)` | student | **p（teacher）** | ⚠️ q→0 时不会自洽，还带梯度 |

- teacher 的 log **乘自己** → 小概率把自身 log 压回 0（PyTorch 也按 0·log0=0 处理），普通 softmax 够用。
- student 的 log **乘的是 p 不是 q** → q 下溢时 `-p·log(q)` 变巨大 + 坏梯度 → 必须 log_softmax。

## 第二个原因：teacher 没梯度
teacher 在 no_grad + detach 里，`p*log(p)` 是**常量**，不参与 backward；数值稳定最要命的是带梯度的项（student 那项，梯度≈ q−p）。死的标准答案不怕噪声，活的、要更新的才怕。

## 术语
numerical stability / underflow / log-sum-exp trick / KL divergence / log-probs / detach

## 💼 面试版回答
> "`F.kl_div` 第一个参数要 log 概率、第二个要普通概率，所以 student 取 log、teacher 不取。至于数值稳定：KL 逐元素是 `p·log p − p·log q`。teacher 的 log 那项是 `p·log p`，**乘的是它自己**，p→0 时整项趋于 0，自动安全，而且 teacher 是 detach 的常量没有梯度；student 的 log 那项是 `−p·log q`，**乘的是 teacher 的 p 不是 q**，一旦 q 下溢就会变成巨大值还带坏梯度。所以只有 student 必须用 `log_softmax`（log-sum-exp 直接算 log，避开下溢）。"
> **追问**：log_softmax 为什么比 log(softmax) 稳？→ 它不显式算出会下溢的中间概率，直接用 logsumexp 得到 log 概率。
