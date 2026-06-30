# 项目 & Post 灵感库

> 用途：收集那些"**值得写成 blog post / 值得自己做成小项目**"的问题和思路。
> 来源：学习过程中遇到的、有洞察、有坑、有"原来如此"时刻的点。
> 和 `04_qa_notes.md` 的区别：那是"学懂某个知识点"，这里是"**能对外输出 / 能立项**的选题"。

---

## 怎么用这个文件

每条想法记四样东西：
- **Hook**：一句话钩子（post 的标题/卖点）
- **核心**：要讲清楚的关键洞察或要做的事
- **难点/前置**：动手前要先有什么
- **状态**：💡灵感 / 🔨进行中 / ✅已产出（附链接）

---

## 💡 Idea 1：把 Qwen / Llama 蒸馏进一个小模型 —— 真正的坑不是算力

**类型**：post + project
**Hook**：「你以为蒸馏大模型缺的是显卡？其实卡在 vocab 对不齐」

**核心洞察**（来自 day05~ 的讨论）：
- 蒸馏要的是 teacher 的**概率分布（logits / soft label）**，不是文本结果。
- 开源模型（Qwen/Llama）直接 `teacher(ids).logits` 就能拿到完整分布，不需要 API。
- **真正的拦路虎**：logit 蒸馏要逐位置、在整个 vocab 维度比 KL → 要求 teacher 和 student **同 tokenizer / 同 vocab_size**。Qwen 词表 ~15万、minimind 6400，维度对不上、序列也对不齐。
- 三条出路：
  - **A. 黑盒蒸馏（最实用）**：用 Qwen 生成回答 → 当 SFT 数据训小模型。绕开 vocab，但丢了 soft 分布（本质是"用大模型造数据做 SFT"，Alpaca 同款）。
  - **B. 换 tokenizer**：student 直接用 Qwen 的 vocab 重训，之后能做真·logit 蒸馏。工程量大。
  - **C. 跨 tokenizer 蒸馏**：ULD / MinED 等研究级方法对齐不同词表分布。【待深入】

**Post 角度**：用一张"维度对不上"的图，讲清"白盒 vs 黑盒蒸馏"的本质区别，破除"蒸馏=有显卡就行"的误解。
**Project 角度**：实测方案 A —— 拿 Qwen 造一份小数据，SFT 自己的 mymodel，对比蒸馏前后效果。
**难点/前置**：理解 logits/softmax/KL（✅已会）；跑通一次 SFT（🔄进行中）。
**状态**：💡灵感

---

## 💡 Idea 2：比 minimind 更小的 mymodel —— 核心是「让代码先跑起来」

**类型**：project（可拆成 post 系列）
**Hook**：「从零手写一个能训练的最小 LLM，目标不是性能，是『跑通』」

**核心思路**：
- 做一个**比 minimind 还小**的可训练 LLM，强调"**端到端能跑**"而非效果。
- 参数极小（tiny vocab / 小 hidden_size / 少层），让它在 **CPU / MPS 上几分钟就能跑一轮**，降低试错成本。
- 全流程**自己手写**：pretrain → SFT → LoRA → DPO（→ 蒸馏），每个阶段最小可运行版本。
- 这其实就是你**正在做的事**（`model/Mymodel*.py` 系列）—— 可以直接把它产品化/文章化。

**Post 角度（系列）**：每个训练阶段一篇"手写 + 踩坑实录"：
1. 手写一个最小 Transformer（RoPE/GQA/RMSNorm/SwiGLU）让它能 forward
2. Pretrain：next-token prediction + loss mask 的真相
3. SFT：只对 assistant 回答算 loss（附 loss_mask 可视化）
4. LoRA：monkey-patch forward + 闭包坑 + B 零初始化
5. DPO：preference data + ref_model + dpo_loss
6. （选）蒸馏：soft label vs hard label

**为什么有价值**：市面教程要么太理论、要么直接 import 现成库。"**全手写 + 最小可跑 + 踩坑实录**"这个定位稀缺，初学者最需要。
**难点/前置**：你已经走到 LoRA/DPO，素材现成（qa_notes + 代码 + bug 清单都在）。
**状态**：🔨进行中（代码在 `model/Mymodel*.py`，笔记在 `learning/`）

---

## 🔨 Idea 3：边学 LLM，边给自己造一个「学习引导系统」

**类型**：post + project（已有 v1，可继续做）
**Hook**：「学 LLM 的副产品，是一个能记住我学到哪、不重复教我的学习 OS」

**核心思路**：
- 把"边做边学"的流程**系统化**：学习者状态（会什么/不会什么/学到哪）存成单一数据源，工具能查现状、给问题归类、推荐下一步、自动去重笔记。
- 已落地 v1（见 `learning_system/`，纯 stdlib、规则驱动）：
  - `learner_profile.json` 单一数据源（解决"跨会话失忆"）
  - 6 模块：Profiler / Classifier(Layer0-7) / Planner / Knowledge Tracker / Artifact Generator / Curriculum Planner
  - Q&A 自动去重（difflib）+ 总索引
- 设计文档：`system_design_memo.md`（从画像、分类法到 Agent 架构）。

**Post 角度**：
- 元叙事："The best way to learn X is to build a tool that teaches X" —— 学 LLM 的过程本身产出一个学习系统。
- 技术点：为什么先做**规则驱动**（可解释、零依赖、马上能跑），再谈升级成**模型驱动**（用 LLM 判断 Level、生成解释）—— 一个"先让它跑起来"的工程哲学，和 Idea 2 同源。
- 反差点：很多人想"用 AI 学习"，方向是找现成工具；这个是**自己造**，而且造的过程就是最好的学习。

**Project 角度（v2 路线）**：
- 把关键词分类器换成**模型驱动**（甚至用自己训的 mymodel 来分类/答题）—— 让 Idea 2 的模型反哺这个系统，形成闭环。
- 加 `error_log` / `reading_list` / quiz 自测（衡量学习效果）。

**为什么有价值**：把"学习方法论 + 可运行代码 + LLM 应用"三件事缝在一起，故事完整、可复现、有 meta 趣味。
**难点/前置**：v1 已完成；v2 需要 mymodel 能跑（依赖 Idea 2）。
**状态**：🔨进行中（v1 已产出：`learning_system/` + `system_design_memo.md`）

---

## 模板（复制下面这块加新想法）

```
## 💡 Idea N：<标题>
**类型**：post / project / both
**Hook**：「一句话钩子」
**核心**：要讲清的洞察 / 要做的事
**难点/前置**：动手前要有什么
**状态**：💡灵感
```
