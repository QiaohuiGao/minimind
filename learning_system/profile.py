"""
模块 ①+④：Learner Profiler + Knowledge Tracker

learner_profile.json 是整个系统的「单一数据源」(single source of truth)：
记录学习者会什么 (known_concepts)、不会什么 (blind_spots)、学到哪 (curriculum_state)。
所有模块都读它/写它；想看现状直接打开那个 json 即可。
"""
from __future__ import annotations
import json
from datetime import date

from .config import PROFILE_PATH


class LearnerProfile:
    def __init__(self, path=PROFILE_PATH):
        self.path = path
        with open(path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    # ---------------- 读 ----------------
    @property
    def _lp(self):
        return self.data["learner_profile"]

    @property
    def level(self):
        return self._lp["current_level"]

    @property
    def known(self):
        return self._lp["known_concepts"]

    @property
    def blind_spots(self):
        return self._lp["blind_spots"]

    @property
    def curriculum(self):
        return self.data["curriculum_state"]

    def knows(self, concept: str) -> bool:
        """模糊判断：只要已知概念里有谁包含这个词（不分大小写）就算会。"""
        c = concept.strip().lower()
        return any(c in k.lower() for k in self.known)

    # ---------------- 写 ----------------
    def add_concept(self, concept: str, save=True):
        """标记学会了一个概念；同时从盲点里移除（如果在的话）。"""
        if not self.knows(concept):
            self.known.append(concept)
        self._remove_blindspot(concept)
        if save:
            self.save()

    def add_blindspot(self, item: str, save=True):
        if item not in self.blind_spots:
            self.blind_spots.append(item)
        if save:
            self.save()

    def _remove_blindspot(self, concept: str):
        c = concept.strip().lower()
        self._lp["blind_spots"] = [b for b in self.blind_spots if c not in b.lower()]

    def set_level(self, level: str, save=True):
        self._lp["current_level"] = level
        if save:
            self.save()

    def set_stage(self, stage: str, save=True):
        self.curriculum["current_stage"] = stage
        if save:
            self.save()

    def complete_stage(self, stage: str, save=True):
        """把一个课程阶段标记为已完成（用于 Curriculum Planner 算下一步）。"""
        done = self.curriculum.setdefault("completed_stages", [])
        if stage not in done:
            done.append(stage)
        if save:
            self.save()

    def set_next(self, topic: str, save=True):
        self.curriculum["next_recommended"] = topic
        if save:
            self.save()

    def save(self):
        self.data["last_updated"] = date.today().isoformat()
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
            f.write("\n")

    # ---------------- 展示 ----------------
    def summary(self) -> str:
        lp, cs = self._lp, self.curriculum
        lines = [
            f"👤 学习者: {lp.get('name', '?')}  |  Level: {lp['current_level']}",
            f"🎯 当前阶段: {cs.get('current_stage', '?')}",
            f"➡️  下一步推荐: {cs.get('next_recommended', '?')}",
            "",
            f"✅ 已掌握概念 ({len(self.known)}):",
            "   " + "、".join(self.known),
            "",
            f"🕳  盲点 ({len(self.blind_spots)}):",
            "   " + "、".join(self.blind_spots),
            "",
            "🔄 进行中:",
        ]
        for t in cs.get("in_progress", []):
            lines.append(f"   - {t}")
        return "\n".join(lines)
