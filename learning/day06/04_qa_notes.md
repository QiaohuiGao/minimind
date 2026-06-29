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
