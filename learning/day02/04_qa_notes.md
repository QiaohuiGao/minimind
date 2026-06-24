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

---

## Q10. 手动读文件 vs `load_dataset` 有什么区别？

**手动读文件（全量加载）：**
```python
with open(data_path, 'r') as f:
    for line in f:
        self.dataset.append(json.loads(line))
```
把所有数据一次性读进内存，存成 Python list。简单直接，适合小数据集（< 1GB）。

**`load_dataset`（按需读取）：**
```python
self.samples = load_dataset('json', data_files=data_path, split='train')
```
HuggingFace `datasets` 库，底层用 Apache Arrow 格式，数据存在磁盘，训练时按需读取，不全部加载进内存。

| | 手动读文件 | `load_dataset` |
|--|-----------|---------------|
| 内存占用 | 全量加载 | 按需读取 |
| 适合数据量 | 小（< 1GB） | 大（几十 GB） |
| 代码复杂度 | 简单 | 需要安装 `datasets` |

**【待深入】** Apache Arrow 格式的具体机制、`datasets` 的 streaming 模式、多进程加载原理。

---

## Q11. LoRA 和 QLoRA 有什么区别？

**LoRA：** 解决"训练哪些参数"的问题。原始权重 W 保持不变，旁边插两个小矩阵 A（d×r）和 B（r×d），r << d（如 r=8）。forward 时：`output = W·x + B·A·x`，只有 A 和 B 参与训练，参数量约为原来的 1%。

**QLoRA：** 在 LoRA 基础上，再解决"原始权重占多少显存"的问题。把原始权重 W 从 float16（16位）量化成 int4（4位），显存压缩到 1/4，LoRA 的 A/B 矩阵仍用 bf16 训练。

```
LoRA:   W[fp16]  +  A/B[bf16 可训练]   → 原始权重占满显存
QLoRA:  W[int4]  +  A/B[bf16 可训练]   → 原始权重压缩到 1/4
```

以 7B 模型为例：

| 方式 | 显存需求 |
|------|---------|
| Full SFT (bf16) | ~28 GB |
| LoRA (bf16) | ~14 GB |
| QLoRA (int4) | ~6 GB |

**代价：** int4 量化会损失一点精度，效果略差于 LoRA，但差距很小。训练时 int4 权重会临时 dequantize 回 bf16 做 forward，梯度只流向 A/B 矩阵。

**【待深入】** 量化（Quantization）的具体原理、int4 精度损失的量化分析、`bitsandbytes` 库的使用。
