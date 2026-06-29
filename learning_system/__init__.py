"""
LLM 学习引导系统 (learning_system)

把 system_design_memo.md 里设计的「学习引导 Agent」落地成可运行的程序。
6 个模块（对应 memo 6.1）：
    ① Learner Profiler  + ④ Knowledge Tracker  → profile.LearnerProfile
    ② Question Classifier                        → classifier.classify
    ③ Response Planner  + ⑥ Curriculum Planner   → planner
    ⑤ Artifact Generator (含 Q&A 去重/索引, 方向D) → artifacts

单一数据源：learning/learner_profile.json
"""
from .profile import LearnerProfile
from . import classifier, planner, artifacts

__all__ = ["LearnerProfile", "classifier", "planner", "artifacts"]
