# Day 2 问答笔记：SFT Dataset 详解

---

## Q1. Pretrain 和 SFT 的数据格式有什么区别？

**Pretrain：**
```json
{"text": "秋日早晨，清风拂面……"}
```
一段连续原始文本，模型学习预测每一个下一个 token。

**SFT：**
```json
{"conversations": [
  {"role": "user",      "content": "你好"},
  {"role": "assistant", "content": "你好啊"}
]}
```
多轮对话，user/assistant 交替，一条样本可以包含多轮。

---

## Q2. SFT 为什么只对 assistant 部分计算 loss？

因为训练目标是让模型学会"怎么回答"，而不是"怎么提问"。

```
token:  <bos> user \n 你好 <eos> \n <bos> assistant \n 你好啊 <eos>
label:  -100  -100 -100 -100 -100 -100  -100    -100  -100  你好啊  <eos>
                    ↑ user 全部 -100                   ↑ 只有这里参与 loss
```

如果 user prompt 也算 loss，模型会学"如何提问"而不是"如何回答"，训练目标错误。

---

## Q3. `generate_labels` 怎么知道哪里是 assistant 的回答？

用 `<bos>assistant\n` 这个 token 序列定位，而不是单独找 `assistant` 这个词。

原因：对话内容里可能出现"assistant"这个词（比如"你作为一个 assistant…"），单独匹配会误判。`<bos>` 是特殊 token，不会出现在普通文字里，所以 `<bos>assistant\n` 只会出现在 assistant 回答的开头，定位准确。

```python
self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
```

扫描逻辑：
```
全部初始化为 -100
→ 找到 <bos>assistant\n → 打开 label（从这里开始 = 真实 token id）
→ 找到 <eos>\n         → 关闭 label
→ 继续扫描下一轮
```

---

## Q4. `pre_processing_chat` 的核心作用是什么？

20% 概率在 conversations 前面随机插一条 system prompt。

**为什么随机，不是每条都加？**
- 80% 没有 system prompt → 模型学会没有指令时正常回答
- 20% 有 system prompt  → 模型学会按指令调整行为
- 两种情况都见过 → 有没有 system 都能正常工作

**为什么选 10 种不同的 system prompt？**

固定用同一句，模型会把那句话和"正确行为"强绑定。随机选 10 种，模型学到的是"system prompt 是在描述角色"这个概念本身。

---

## Q5. tool call 数据为什么要"直接透传"不做任何处理？

tool call 数据的 system 消息里带有 `tools` 字段（工具定义），结构特殊：

```json
{"role": "system", "content": "你是助手", "tools": "[{\"name\": \"get_weather\"}]"}
```

如果再插一条随机 system prompt，就变成两条 system 消息，chat template 处理不了，工具定义也可能被覆盖。所以 tool call 数据完整保留，不做任何修改。

---

## Q6. `role: system + tools` 和 `role: tool` 有什么区别？

| | `role: system` + `tools` 字段 | `role: tool` |
|--|--|--|
| 时机 | 对话最开始 | assistant 调用工具之后 |
| 作用 | 声明"有哪些工具可以用" | 返回工具的执行结果 |
| 内容 | 工具定义（名字、参数、描述） | 工具实际运行的返回值 |

完整 tool call 对话流程：
```
system  → "你可以用 get_weather 工具"    ← 工具手册（定义）
user    → "今天天气怎样？"
assistant → 调用 get_weather(city="北京") ← 模型决定用工具
tool    → "晴，25°C"                     ← 工具执行结果
assistant → "今天北京天气晴朗，25°C。"    ← 模型拿到结果后回答
```

---

## Q7. `Features` 声明是干什么的？

告诉 `load_dataset` 每个字段的结构和类型，防止自动推断出错。

SFT 数据有些样本有 `reasoning_content`，有些没有。不声明的话，`load_dataset` 看第一条没有这个字段，后面遇到有的就报错。

声明之后，缺失的字段自动补 `None`，每条样本都保证有完整的 5 个 key。

`Value('string')` 是 HuggingFace `datasets` 库里的类型声明类，类似 SQL 建表时声明列类型。

---

## Q8. `post_processing_chat` 为什么要清除空 `<think>` 标签？

reasoning 数据里，没有思考内容的 assistant 回答经过 `apply_chat_template` 渲染后会带上空标签：

```
<think>

</think>

实际回答内容
```

80% 概率移除，避免模型学到"每次回答前输出一个空思考"这种无意义行为。保留 20% 让模型知道可以不思考直接回答。

---

## Q9. SFT 和 Pretrain 的核心差异总结

| | Pretrain | SFT |
|--|--|--|
| 数据格式 | `{"text": "..."}` | `{"conversations": [...]}` |
| Loss 算在哪 | 每个非 pad token | 只算 assistant 回答部分 |
| label 构造 | `labels = input_ids.clone()` | `generate_labels()` 扫描定位 |
| 目的 | 学语言规律 | 学怎么回答问题 |

训练循环本身几乎一样，唯一本质差异是 **label 的哪些位置参与 loss**。
