"""统一路径配置：所有模块从这里取路径，避免各写各的。"""
from pathlib import Path

# 本文件在 repo/learning_system/config.py → 上一级就是仓库根目录
REPO_ROOT = Path(__file__).resolve().parent.parent
LEARNING_DIR = REPO_ROOT / "learning"
PROFILE_PATH = LEARNING_DIR / "learner_profile.json"
QA_INDEX_PATH = LEARNING_DIR / "qa_index.md"
