"""
模块 ③+⑥：Response Planner + Curriculum Planner

- Response Planner: 根据 Layer + 学习者 Level/风格，推荐「怎么答」（类比/torch代码/公式/实验 + 深度）
  对应 memo 第三章「回答策略矩阵」。
- Curriculum Planner: 根据课程依赖图 + 已完成阶段，推荐「学什么」
  对应 memo 第九章「课程路线图」。
"""

# ---------------- Response Planner ----------------

# 每层默认的回答策略组件（memo 3.1）。注意：本学习者偏好 torch 代码胜过数学符号。
STRATEGY_BY_LAYER = {
    0: ["step-by-step 命令", "少讲原理"],
    1: ["类比", "torch 代码示例"],
    2: ["类比", "torch 代码 (替代数学公式)", "对照参考实现"],
    3: ["类比", "torch 代码 (替代数学公式)", "实验验证"],
    4: ["类比", "数字估算", "实验验证"],
    5: ["类比", "torch 代码", "实验验证"],
    6: ["类比", "torch 代码 (替代数学公式)", "对比表"],
    7: ["类比", "架构图", "对照参考实现"],
}


def plan_response(layer_id: int, level: str = "L3"):
    """给出针对某层问题的回答策略建议。"""
    components = STRATEGY_BY_LAYER.get(layer_id, ["类比", "代码"])
    # 按 Level 调整深度（memo 3.2）
    if level.startswith("L0") or level.startswith("L1"):
        depth = "先给 80% 简化版，确认理解再补细节；多用类比，少上公式"
    elif level.startswith("L4") or level.startswith("L5"):
        depth = "可直接深入底层/权衡取舍，省略基础铺垫"
    else:  # L2-L3
        depth = "类比锚定 → torch 代码展开 → 连接已学概念；按需补深"
    return {"components": components, "depth": depth}


# ---------------- Curriculum Planner ----------------

# 课程依赖图（memo 9.2）：(阶段, [前置依赖])
ROADMAP = [
    ("tokenizer", []),
    ("pretrain", []),
    ("full_sft", ["pretrain"]),
    ("lora", ["full_sft"]),
    ("distillation", ["pretrain"]),
    ("dpo", ["full_sft"]),
    ("grpo", ["dpo"]),
    ("ppo", ["dpo"]),
    ("rollout_engine", ["ppo", "grpo"]),
    ("train_agent", ["ppo", "grpo"]),
]


def next_topics(completed_stages):
    """返回所有「前置全部完成、但自己还没完成」的阶段 = 当前可以学的。"""
    done = set(completed_stages)
    ready = []
    for stage, deps in ROADMAP:
        if stage in done:
            continue
        if all(d in done for d in deps):
            ready.append(stage)
    return ready
