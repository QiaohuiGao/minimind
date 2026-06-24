# Day 7 Prereading: PPO 原理 + 总复盘、交付、工程化思维

## 今日目标

今天两件事：补完 PPO（GRPO 的前身，理解它能让整个 RL 链路串起来）；然后做总复盘，把一周训练流程整理成可复现报告。

---

## Part 1：PPO（Proximal Policy Optimization）

### 必须掌握的基础概念

- Policy（π）：这里就是 LLM，输入 prompt，输出 token 分布。
- Value function / Critic：预测当前状态的未来累计 reward，是一个独立的网络。
- Advantage（A）：实际 reward 减去 critic 的预期 reward，衡量这次回答"比预期好多少"。
- Clip ratio：`clip(π_new/π_old, 1-ε, 1+ε)`，限制每步 policy 更新幅度，防止训练崩溃。
- Rollout：用当前 policy 生成回答，收集 (prompt, response, reward, old_logp) 数据。
- KL penalty：防止 policy 偏离 reference model 太远（reward hacking 的防护）。

**PPO policy loss：**

```
ratio = π_new(a|s) / π_old(a|s)
L = min(ratio × A,  clip(ratio, 1-ε, 1+ε) × A)
```

直觉：advantage > 0（好回答）就鼓励提高概率，但提高幅度不超过 clip 范围。

### PPO vs GRPO：核心差异

| | PPO | GRPO |
|--|-----|------|
| Critic | 需要单独训练 value network | 不需要，用组内相对 reward 代替 |
| Advantage | reward - value（TD） | (reward - group_mean) / group_std |
| 复杂度 | 高（两个网络） | 低（一个网络） |
| 稳定性 | 较好（有 value 估计） | 较弱（方差更大） |

GRPO 的思路：与其训练一个 value network，不如对同一个 prompt 多采样几个回答，用回答之间的相对好坏作为 advantage——简单且够用。

### 面试考点

- clip ratio 解决了什么问题？没有它会发生什么？
- critic 网络的作用是什么？为什么 GRPO 能去掉它？
- on-policy 和 off-policy 有什么区别？PPO 是哪种？
- PPO 的 rollout 阶段和 update 阶段分别做什么？

### 项目中要看的文件

- `trainer/train_ppo.py`
- `trainer/rollout_engine.py`

---

## Part 2：总复盘与工程化思维

### 必须掌握的基础概念

- Reproducibility：别人能按你的记录复现结果。
- Ablation：只改变一个变量观察影响。
- Baseline：每次改动都要和基线比较。
- Checkpoint hygiene：权重命名、保存、恢复、版本管理清晰。
- Evaluation discipline：固定测试集、记录失败、不挑样例。

## 面试考点

- 从零训练一个小 LLM 的完整流程是什么？
- Pretrain、SFT、DPO/GRPO、Distillation 的顺序和作用是什么？
- 训练中你最关注哪些指标？
- 如果效果不好，你如何定位是数据、模型、优化还是评测问题？
- 如何把研究代码变成可靠训练 pipeline？

## 工业要求

- 命令可复制。
- 数据路径明确。
- 权重命名明确。
- 每个实验有目的、有结果、有结论。
- 失败样例和成功样例一样重要。
- 不把一次随机生成当作模型能力结论。

## 今日交付

1. `training_log.md`
2. `eval_report.md`
3. `code_understanding.md`
4. `final_exam_answers.md`

建议结构：

```text
环境
数据
模型配置
训练命令
loss 记录
推理评测
失败分析
后训练理解
下一步计划
```

## 今日注意事项

- 不要隐藏没跑通的部分。
- 不要只写结论，要写证据。
- 不要凭感觉说模型变好，要给对比样例。
