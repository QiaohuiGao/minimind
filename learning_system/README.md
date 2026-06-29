# learning_system —— LLM 学习引导系统

把 [`learning/system_design_memo.md`](../learning/system_design_memo.md) 里设计的「学习引导 Agent」落地成可运行的程序。
纯 Python 标准库，无第三方依赖。

## 它解决什么

- **跨会话记忆**：学习者会什么、学到哪，存在 `learning/learner_profile.json`（单一数据源）。
- **不重复教学**：教过的概念记进 `known_concepts`，下次直接用。
- **笔记不重复累积**：加问答时自动去重（difflib 相似度），并能一键生成总索引。

## 6 个模块（对应设计文档 6.1）

| 模块 | 文件 | 作用 |
|------|------|------|
| Learner Profiler + Knowledge Tracker | `profile.py` | 读写学习者状态、判断概念是否已掌握 |
| Question Classifier | `classifier.py` | 把问题归类到 Layer 0–7（关键词规则）|
| Response Planner + Curriculum Planner | `planner.py` | 推荐回答策略 + 课程下一步 |
| Artifact Generator | `artifacts.py` | 问答笔记追加（去重）+ 总索引 |

## 用法（在仓库根目录运行）

```bash
python3 -m learning_system status                 # 看学习者现状
python3 -m learning_system classify "RoPE 是什么"  # 问题归类 + 回答策略建议
python3 -m learning_system next                    # 课程下一步推荐
python3 -m learning_system learned "DPO"           # 标记学会某概念（写回 profile）
python3 -m learning_system add-qa --day 5 --q "问题" --a "答案" --layer 7   # 加问答（自动去重）
python3 -m learning_system index                   # 重建 learning/qa_index.md
```

也可作为库调用：

```python
from learning_system import LearnerProfile, classifier, planner
p = LearnerProfile()
print(p.knows("LoRA"))                 # True
print(classifier.classify("KV cache 怎么省显存"))
print(planner.next_topics(p.curriculum["completed_stages"]))
```

## 现状与未来

当前是**规则驱动**（关键词分类 / difflib 去重 / 依赖图推进），轻量、可解释、随处能跑。
未来可把规则替换成**模型驱动**（用 LLM 甚至 minimind 自己判断 Level、生成解释），见设计文档第六章。
