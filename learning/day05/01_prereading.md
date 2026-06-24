# Day 5 Prereading: LoRA 微调 + 评测与推理参数

## 今日目标

今天两件事：理解 LoRA 作为 full finetune 的轻量替代；建立评测习惯，能判断模型哪里变好了，哪里只是看起来变好了。

---

## Part 1：LoRA 微调

### 必须掌握的基础概念

- LoRA（Low-Rank Adaptation）：只更新少量新增参数，冻结原始权重。
- Low-rank decomposition：把权重更新分解为两个小矩阵乘积 BA（B: d×r，A: r×k，r 远小于 d/k）。
- rank r：控制 LoRA 容量，越大越接近 full finetune，典型值 4/8/16/32。
- alpha scaling：实际更新 = (α/r) × BA，控制 LoRA 更新幅度。
- Target modules：通常对 q_proj、v_proj 加 LoRA；可选加 k_proj、o_proj。
- Adapter merging：推理前把 BA 合并进原始权重 W' = W + (α/r)BA，不增加推理开销。

### 面试考点

- LoRA 为什么有效？为什么 full finetune 的权重变化是低秩的？
- rank 太小或太大分别有什么问题？
- LoRA 合并进原始权重后，推理速度变了吗？
- LoRA 和 full SFT 分别适合什么场景？
- QLoRA 在 LoRA 基础上加了什么？

### 工业要求

- LoRA 适合：显存受限、多任务切换（每个任务保留独立 adapter）、快速迭代。
- Full SFT 适合：数据充足、追求最佳性能、长期单一任务。
- 保存时只保存 LoRA weights（小很多），不保存完整模型。
- 合并前必须确认 alpha/rank 配置和训练时一致。

### 项目中要看的文件

- `model/model_lora.py`
- `trainer/train_lora.py`

### 今日推荐命令

```bash
python trainer/train_lora.py \
  --hidden_size 768 \
  --from_weight full_sft \
  --epochs 1 \
  --learning_rate 1e-4
```

---

## Part 2：评测与推理参数

### 必须掌握的基础概念

- Offline eval：固定测试集，离线比较。
- Human eval：人工主观评分。
- Regression eval：新权重不能明显退化旧能力。
- Sampling parameters：temperature、top_p、top_k、repetition_penalty。
- Hallucination：模型生成看似合理但错误的信息。
- Exposure bias：训练时看标准前缀，推理时看自己生成的前缀。

## 面试考点

- 为什么 loss 不能完全代表生成质量？
- temperature 和 top_p 分别控制什么？
- 为什么小模型容易重复和幻觉？
- 如何设计一个低成本的 LLM eval？
- 如何判断 SFT 是否过拟合？

## 工业要求

- 每次训练后必须跑固定评测集。
- 评测 prompt 要覆盖：身份、事实、代码、总结、推理、工具调用。
- 生成评测要固定随机种子或多次采样。
- 不要只展示最好样例，要记录失败样例。
- 小模型不要用大模型标准苛责，但要诚实记录边界。

## 项目中要看的文件

- `eval_llm.py`
- `scripts/eval_toolcall.py`
- `scripts/serve_openai_api.py`
- `scripts/web_demo.py`

## 今日固定评测集

```text
1. 你是谁？
2. 解释什么是机器学习。
3. 为什么天空是蓝色的？
4. 请用 Python 写一个斐波那契函数。
5. 比较猫和狗作为宠物的优缺点。
6. 推荐一些中国美食。
7. 把下面这段话总结成三点：...
8. 如果明天下雨，我应该如何出门？
```

## 今日注意事项

- 同一个 prompt 至少用 2 组采样参数测试。
- 把失败样例原样保存。
- 评估时区分：语言流畅、指令跟随、事实正确、格式正确。
