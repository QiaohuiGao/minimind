"""
模块 ⑤：Artifact Generator + Q&A 笔记自动化（方向 D）

解决 memo 第 7.3 提的痛点：「笔记如何避免重复累积 / 如何建索引」。
- add_qa_note: 往 learning/dayNN/04_qa_notes.md 追加一条 Q&A，追加前【去重】
- build_index: 扫描所有 dayNN/04_qa_notes.md 的 Q 标题，生成总索引 learning/qa_index.md
"""
import re
import difflib

from .config import LEARNING_DIR, QA_INDEX_PATH, INTERVIEW_PATH

INTERVIEW_MARKER = "💼"  # 笔记里标了这个的小节 = 面试版回答

# 匹配 "## Q1. xxx" 或 "### Q3. xxx" 这类问答标题
QA_HEADER_RE = re.compile(r"^#{2,4}\s*(Q\d+[.、．]?\s*.+?)\s*$", re.M)


def _day_note_path(day):
    """day 可以是数字 5 或字符串 'day05'。"""
    name = day if str(day).startswith("day") else f"day{int(day):02d}"
    return LEARNING_DIR / name / "04_qa_notes.md"


def existing_questions(day):
    """读出某天 qa_notes 里已有的所有问题标题文本。"""
    path = _day_note_path(day)
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    return [m.strip() for m in QA_HEADER_RE.findall(text)]


def _normalize(q):
    # 去掉 Qn 编号和标点，只比内容，避免编号不同被当成不重复
    q = re.sub(r"^Q\d+[.、．]?\s*", "", q)
    return re.sub(r"[\s，。？?！!、,.]+", "", q).lower()


def is_duplicate(new_q, existing, threshold=0.82):
    """和已有问题比相似度（difflib），超过阈值算重复。返回最像的那条或 None。"""
    n = _normalize(new_q)
    best, best_ratio = None, 0.0
    for e in existing:
        r = difflib.SequenceMatcher(None, n, _normalize(e)).ratio()
        if r > best_ratio:
            best, best_ratio = e, r
    return best if best_ratio >= threshold else None


def add_qa_note(day, question, answer, layer=None, force=False):
    """
    追加一条 Q&A 到当天笔记。返回 (status, info)。
    status: 'added' | 'duplicate' | 'created'
    去重：若已有高度相似的问题，默认跳过（force=True 可强制追加）。
    """
    path = _day_note_path(day)
    existing = existing_questions(day)

    if not force:
        dup = is_duplicate(question, existing)
        if dup:
            return ("duplicate", dup)

    # 自动编号：当天已有 N 条 → 新的是 QN+1
    next_n = len(existing) + 1
    layer_tag = f"（Layer {layer}）" if layer is not None else ""
    block = f"\n### Q{next_n}. {question}{layer_tag}\n\n{answer}\n"

    created = not path.exists()
    if created:
        path.parent.mkdir(parents=True, exist_ok=True)
        name = path.parent.name
        path.write_text(f"# {name} 问答笔记\n", encoding="utf-8")

    with open(path, "a", encoding="utf-8") as f:
        f.write(block)
    return ("created" if created else "added", f"Q{next_n}")


def build_index():
    """扫描所有 learning/day*/04_qa_notes.md，生成总索引 qa_index.md。"""
    days = sorted(LEARNING_DIR.glob("day*/04_qa_notes.md"))
    lines = ["# Q&A 总索引", "", "> 由 `learning_system` 自动生成，汇总各天问答笔记的问题标题。", ""]
    total = 0
    for note in days:
        day = note.parent.name
        qs = existing_questions(day)
        if not qs:
            continue
        rel = note.relative_to(LEARNING_DIR)
        lines.append(f"## {day} ({len(qs)} 条)")
        for q in qs:
            lines.append(f"- [{q}]({rel})")
            total += 1
        lines.append("")
    lines.insert(3, f"**共 {total} 条问答，覆盖 {len([d for d in days])} 天。**\n")
    QA_INDEX_PATH.write_text("\n".join(lines), encoding="utf-8")
    return total


def _split_qa_blocks(text):
    """把一篇笔记按 Q 标题切成 [(标题, 正文), ...]。"""
    matches = list(QA_HEADER_RE.finditer(text))
    blocks = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        blocks.append((m.group(1).strip(), text[start:end]))
    return blocks


def _extract_interview_section(body):
    """从一个 Q 正文里抽出 💼 面试版那一小节的内容（不含小节标题本身）。"""
    captured, capturing = [], False
    for line in body.splitlines():
        is_heading = line.lstrip().startswith("#")
        if is_heading and INTERVIEW_MARKER in line:
            capturing = True            # 命中面试版小节标题，开始抓（跳过标题行本身）
            continue
        if capturing:
            if is_heading and line.lstrip().startswith("## "):
                break                   # 遇到下一个同级小节，停
            captured.append(line)
    return "\n".join(captured).strip() or None


def build_interview_prep():
    """扫描所有 dayNN 笔记，抽出标了 💼 的面试问答，汇总成 interview_prep.md。"""
    days = sorted(LEARNING_DIR.glob("day*/04_qa_notes.md"))
    out = ["# 面试速记", "", "> 由 `learning_system interview` 自动抽取各天笔记里标了 💼 面试版 的问答。", ""]
    count = 0
    for note in days:
        blocks = _split_qa_blocks(note.read_text(encoding="utf-8"))
        items = [(t, _extract_interview_section(b)) for t, b in blocks if INTERVIEW_MARKER in b]
        items = [(t, s) for t, s in items if s]
        if not items:
            continue
        out.append(f"## {note.parent.name}\n")
        for title, section in items:
            out.append(f"### {title}\n")
            out.append(section + "\n")
            count += 1
    out.insert(3, f"**共 {count} 条面试问答。**\n")
    INTERVIEW_PATH.write_text("\n".join(out), encoding="utf-8")
    return count
