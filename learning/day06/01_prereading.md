# Day 6 Prereading: DPO + 蒸馏与 GRPO/CISPO 后训练

## 今日目标

今天覆盖三个后训练方向：DPO（alignment 入门首选）、蒸馏（知识迁移）、GRPO（RL 对齐）。推荐按顺序先理解 DPO 再看 GRPO，因为 DPO 是最简单的 alignment 算法，理解它能让 GRPO 的设计动机更清晰。

---

## Part 1：DPO（Direct Preference Optimization）

### 必须掌握的基础概念

- Preference data：每条样本是 (prompt, chosen_response, rejected_response) 三元组。
- Reference model（π_ref）：冻结的 SFT 模型，防止 policy 偏离太远。
- DPO loss：不需要 reward model，直接从 preference 对中推导 policy 更新方向。
- β 参数：控制 policy 偏离 reference 的幅度，越大越保守。
- Implicit reward：DPO 本质上仍是 RL，只是把 reward 和 policy 优化合并成一个闭合形式解。

**DPO loss 公式：**

```
L = -log σ( β × (log π(chosen|x)/π_ref(chosen|x) - log π(rejected|x)/π_ref(rejected|x)) )
```

直觉：chosen 的 log prob 相对 ref 提高，rejected 的相对降低，两者差距越大 loss 越低。

### 面试考点

- DPO 为什么不需要 reward model 和 rollout？
- DPO 和 PPO 本质上的区别是什么？
- reference model 在 DPO 里扮演什么角色？
- β 太小或太大分别有什么问题？
- DPO 的数据要求和 SFT 有什么不同？

### 工业要求

- DPO 数据质量比数量重要：chosen 和 rejected 的差距要明显。
- 训练前必须跑 SFT 得到 base model，DPO 在 SFT 基础上做。
- 监控指标：chosen reward margin（chosen - rejected log ratio 差）。
- β 通常取 0.1~0.5，太大不收敛，太小 policy 会 collapse 到 reference。

### 项目中要看的文件

- `trainer/train_dpo.py`
- `dataset/lm_dataset.py`（DPO 数据集部分）

### 今日推荐命令

```bash
python trainer/train_dpo.py \
  --hidden_size 768 \
  --from_weight full_sft \
  --epochs 1 \
  --learning_rate 1e-5
```

---

## Part 2：蒸馏与 GRPO/CISPO

### 必须掌握的基础概念

### 蒸馏

- Black-box distillation：学习 teacher 生成的硬标签答案。
- White-box distillation：学习 teacher logits/probability distribution。
- Temperature：软化概率分布。
- KL divergence：衡量 teacher/student 分布差异。

### GRPO/CISPO

- On-policy rollout：用当前模型实时采样。
- Reward model：给回答打分。
- Reference model：限制 policy 偏离原模型。
- Advantage：回答比同组其他回答好多少。
- KL penalty：防止 reward hacking 和分布漂移。
- CISPO：把 clipped ratio 作为权重，保留 logprob 梯度路径。

## 面试考点

- DPO、PPO、GRPO 的主要区别是什么？
- GRPO 为什么可以不训练 critic？
- 为什么每个 prompt 要采样多个 responses？
- KL penalty 防什么？
- 蒸馏中的 temperature 为什么要乘 `T^2`？
- Reward hacking 是什么？

## 工业要求

- RL 后训练必须监控 reward、KL、response length、advantage std。
- 只看 reward 上升很危险，可能是 reward hacking。
- 小模型 RL 容易奖励稀疏，要选择能力边界内任务。
- 蒸馏要确认 teacher/student vocab 对齐或正确裁剪。
- 后训练前必须保留 SFT baseline，便于回退和对比。

## 项目中要看的文件

- `trainer/train_grpo.py`
- `trainer/train_agent.py`
- `trainer/rollout_engine.py`
- `trainer/train_distillation.py`
- `dataset/lm_dataset.py::RLAIFDataset`
- `trainer/trainer_utils.py::LMForRewardModel`

## 今日建议

如果 reward model 已准备好：

```bash
cd trainer
python train_grpo.py --debug_mode --debug_interval 5
```

如果没有 reward model：

- 先只读代码。
- 或选择 `train_distillation.py` 做理解。
