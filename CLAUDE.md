# CLAUDE.md — minimind 项目指令

## 回答语言风格（重要）

回答这个用户时遵循以下双语风格：

- **一般解释用中文**：正文、说明、推理、举例都用中文，让用户读得顺。
- **专业术语保留英文**：technical terms（如 learning rate、batch size、gradient、overfitting、attention、checkpoint、epoch 等）直接用英文，不要硬翻成中文（必要时可在英文后用括号附一句中文解释）。
- **可能不好读/不熟悉的词用英文**：命令、参数名、库名、API、文件名、代码标识符等一律保留英文原文。
- 目的：让用户既能用母语顺畅理解概念，又能记住和识别英文专业词汇（方便看英文文档、论文、报错）。

## 项目背景

- 用户在通过 minimind 项目**边做边学**大语言模型（从预训练到 SFT/RL）。
- 用户是 Python / 大模型初学者，环境用 `~/Project/minimind/.venv`（venv 虚拟环境，非 conda）。
- 每次会话把有价值的问答整理进 `learning/dayNN/04_qa_notes.md`（详见学习笔记习惯）。
