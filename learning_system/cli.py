"""
命令行入口：把 6 个模块串成可用工具。

用法（在仓库根目录）：
    python -m learning_system status                  # 看学习者现状
    python -m learning_system classify "RoPE 是什么"   # 给问题归类 + 回答策略建议
    python -m learning_system next                     # 课程下一步推荐
    python -m learning_system learned "DPO"            # 标记学会了某概念
    python -m learning_system add-qa --day 5 --q "..." --a "..."   # 加一条问答（自动去重）
    python -m learning_system index                    # 重建 Q&A 总索引
"""
import argparse

from .profile import LearnerProfile
from . import classifier, planner, artifacts


def cmd_status(args):
    print(LearnerProfile().summary())


def cmd_classify(args):
    question = args.question
    prof = LearnerProfile()
    res = classifier.classify(question)
    if not res:
        print("❓ 没命中任何 Layer 关键词，建议人工判断。")
        return
    print(f"📂 归类: Layer {res['layer']} — {res['name']}")
    print(f"   命中关键词: {', '.join(res['matched'])}")
    plan = planner.plan_response(res["layer"], prof.level)
    print(f"🧭 回答策略: {' → '.join(plan['components'])}")
    print(f"   深度建议 (按 {prof.level}): {plan['depth']}")
    # 顺带提示这个概念学习者会不会
    if prof.knows(question):
        print("   💡 学习者似乎已接触过相关概念，可直接用、少铺垫。")


def cmd_next(args):
    prof = LearnerProfile()
    done = prof.curriculum.get("completed_stages", [])
    ready = planner.next_topics(done)
    print(f"✅ 已完成阶段: {', '.join(done) or '（无）'}")
    print(f"➡️  现在可以学: {', '.join(ready) or '（已学完全部）'}")
    print(f"📌 profile 里的推荐: {prof.curriculum.get('next_recommended', '?')}")


def cmd_learned(args):
    prof = LearnerProfile()
    prof.add_concept(args.concept)
    print(f"✅ 已记录学会: {args.concept}")
    print(f"   当前掌握 {len(prof.known)} 个概念。")


def cmd_add_qa(args):
    status, info = artifacts.add_qa_note(args.day, args.q, args.a, layer=args.layer)
    if status == "duplicate":
        print(f"⚠️  疑似重复，已跳过。最相似的已有问题：\n   {info}")
        print("   如确认要加，重跑并带 --force（CLI 暂未开放，可在代码里设 force=True）。")
    else:
        print(f"✅ 已写入 day{int(args.day):02d}/04_qa_notes.md 作为 {info}（{status}）")


def cmd_index(args):
    total = artifacts.build_index()
    print(f"✅ 已重建 Q&A 总索引：learning/qa_index.md（共 {total} 条）")


def cmd_interview(args):
    total = artifacts.build_interview_prep()
    print(f"✅ 已生成面试速记：learning/interview_prep.md（共 {total} 条面试问答）")


def main(argv=None):
    p = argparse.ArgumentParser(prog="learning_system", description="LLM 学习引导系统")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="看学习者现状").set_defaults(func=cmd_status)

    sp = sub.add_parser("classify", help="问题归类 + 回答策略")
    sp.add_argument("question")
    sp.set_defaults(func=cmd_classify)

    sub.add_parser("next", help="课程下一步").set_defaults(func=cmd_next)

    sp = sub.add_parser("learned", help="标记学会某概念")
    sp.add_argument("concept")
    sp.set_defaults(func=cmd_learned)

    sp = sub.add_parser("add-qa", help="追加一条问答（自动去重）")
    sp.add_argument("--day", required=True)
    sp.add_argument("--q", required=True, help="问题")
    sp.add_argument("--a", required=True, help="答案")
    sp.add_argument("--layer", type=int, default=None)
    sp.set_defaults(func=cmd_add_qa)

    sub.add_parser("index", help="重建 Q&A 总索引").set_defaults(func=cmd_index)

    sub.add_parser("interview", help="抽取 💼 面试版生成面试速记").set_defaults(func=cmd_interview)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
