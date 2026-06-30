"""
基于 Mymodel 的简化版知识蒸馏（Knowledge Distillation）

核心思想：让小模型 Student 模仿大模型 Teacher 的"输出分布"（软标签），
         而不只是模仿数据里的"正确答案"（硬标签）。

和 SFT 的差别只有两处：
  ① 多加载一个 Teacher 模型（冻结，只 forward 出 logits）
  ② loss = alpha * CE(硬标签) + (1-alpha) * KL(软标签)

数据格式与 SFT 完全相同，所以直接复用 Mymodelsftdataset。
"""
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import wandb

from Mymodel import Config, MyModel
from Mymodel_sft import Mymodelsftdataset


# ─────────────────────────────────────────────────────────────────────
# 核心：蒸馏 loss = KL 散度（让 Student 分布贴近 Teacher 分布）
# ─────────────────────────────────────────────────────────────────────
def distillation_loss(student_logits, teacher_logits, temperature=1.5):
    """
    student_logits / teacher_logits: [N, vocab]  N=参与蒸馏的 token 数（只取 assistant 部分）
    返回：KL 散度 loss（标量）

    为什么 Teacher 用 softmax、Student 用 log_softmax：
      F.kl_div 的接口约定 —— 第一个参数传 log 概率，第二个传普通概率。
    为什么 Teacher 要 no_grad + detach：
      Teacher 冻结，梯度不能流向它，否则浪费显存/报错。
    temperature > 1：把分布"拉平"，让小概率 token 的判断细节（dark knowledge）显现。
    """
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()
        #teacher 是固定标准答案。.detach() 确保 teacher_probs 是常量目标,backward 时梯度只进 student、不回 teacher。

    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    # KL(teacher || student)：衡量两个分布差多远，目标让 Student 贴近 Teacher
    kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")

    # T**2 补偿：除以 T 让梯度等比例缩小，乘回 T² 保持梯度量级不变
    return kl * (temperature ** 2)


def train_distillation():
    # 蒸馏和 SFT 一样用较小 lr，避免破坏已学到的能力
    config = Config(learning_rate=1e-5)

    wandb.init(
        project="mymodel-distill",
        name=f"distill-lr{config.learning_rate}-epochs{config.epochs}",
        config=vars(config),
    )

    tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(__file__))

    # ① 数据：和 SFT 完全相同，直接复用
    data_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "sft_t2t_mini.jsonl")
    dataset = Mymodelsftdataset(data_path, tokenizer, config.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=8)

    # ② 两个模型（蒸馏唯一的结构特殊点）
    # ----- Student：要训练的模型，从已有权重初始化（通常是 SFT 权重）-----
    student = MyModel(config).to(config.device)
    student_init = os.path.join(os.path.dirname(__file__), "my_model_sft.pt")
    student.load_state_dict(torch.load(student_init, map_location=config.device))

    # ----- Teacher：更强的模型，冻结，只 forward -----
    # 简化版：用相同结构、但加载一个训得更好的权重当 Teacher。
    # （进阶可换成更大模型：teacher = MyModel(Config(hidden_size=更大, num_layers=更多))）
    teacher = MyModel(config).to(config.device)
    teacher_weight = os.path.join(os.path.dirname(__file__), "my_model_teacher.pt")
    teacher.load_state_dict(torch.load(teacher_weight, map_location=config.device))
    teacher.eval()                    # 【模式开关】关 dropout，让 teacher 每次输出确定、可复现
    # 【参数级 · 永久】把 teacher 所有参数的 requires_grad 一次设为 False（结尾 _ = in-place 批量）
    #   · 作用对象 = 参数(weights)：标记它们「不需要梯度」，长期有效
    #   · 效果 = teacher 永远不会被训练更新 + 不为参数分配梯度/optimizer 状态（省显存）
    #   ↔ 注意和下面 ④ 处的 torch.no_grad() 区别：那个管的是「运算」，不是「参数」
    teacher.requires_grad_(False)

    optimizer = torch.optim.AdamW(student.parameters(), lr=config.learning_rate)

    alpha = 0.5        # loss = alpha*CE + (1-alpha)*KL（0.5 = 各一半）
    temperature = 1.5  # 蒸馏温度（1.0~2.0 常用）

    student.train()
    for epoch in range(config.epochs):
        total_loss = 0
        for step, (input_ids, labels) in enumerate(dataloader):
            input_ids = input_ids.to(config.device)
            labels = labels.to(config.device)

            # ③ Student 前向（有梯度）。MyModel 传 labels 会顺便算好 CE loss
            student_logits, ce_loss, _ = student(input_ids, labels)

            # ④ Teacher 前向（只要 logits）
            # 【代码块级 · 临时】torch.no_grad()：这个 with 块里的运算【不建计算图、不存中间激活值】
            #   · 作用对象 = 这段 forward 的「运算过程」，只在块内有效，出了块自动恢复
            #   · 为什么：teacher 永远不 backward，没必要为它建图 → 这次前向几乎不占额外显存
            #
            # 🔑 和上面 requires_grad_(False) 的区别（这就是你想搞清的点）：
            #     requires_grad_(False) → 管「参数」要不要梯度，永久冻结 teacher 的 weights
            #     torch.no_grad()       → 管「这段运算」建不建图，临时省这次 forward 的显存
            #   两者【正交、互不替代】：一起用，teacher 才彻底惰性（参数不更新 + forward 不建图）
            #   附带好处：no_grad 里产出的 teacher_logits 自动 detached → 当 KL 的「固定目标」，梯度只进 student
            with torch.no_grad():
                teacher_logits, _, _ = teacher(input_ids)

            # ⑤ next-token 对齐 + 只在 assistant 部分（labels != -100）算 KL
            #    MyModel 内部 CE 用的是 logits[:, :-1] 对 labels[:, 1:]，这里保持一致
            s_logits = student_logits[:, :-1, :]          # [B, S-1, vocab]
            t_logits = teacher_logits[:, :-1, :]
            shift_labels = labels[:, 1:]                  # [B, S-1]
            mask = shift_labels != -100                   # 只保留 assistant token

            kl_loss = distillation_loss(
                s_logits[mask],      # [N, vocab]  布尔索引后自动展平成参与蒸馏的 token
                t_logits[mask],
                temperature=temperature,
            )

            # ⑥ 合并：硬标签 + 软标签
            loss = alpha * ce_loss + (1 - alpha) * kl_loss

            # ⑦ 标准五步
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            if step % 100 == 0:
                avg = total_loss / (step + 1)
                print(f"Epoch {epoch+1}/{config.epochs} | Step {step}/{len(dataloader)} "
                      f"| loss {loss.item():.4f} | ce {ce_loss.item():.4f} | kl {kl_loss.item():.4f} | avg {avg:.4f}")
                wandb.log({"loss": loss.item(), "ce_loss": ce_loss.item(),
                           "kl_loss": kl_loss.item(), "avg_loss": avg}, step=step)

        print(f"Epoch {epoch+1} 完成，平均 loss {total_loss/len(dataloader):.4f}")

    # ⑧ 只保存 Student（Teacher 是借来打分的，不存）
    save_path = os.path.join(os.path.dirname(__file__), "my_model_distill.pt")
    torch.save(student.state_dict(), save_path)
    print(f"Student 权重已保存至 {save_path}")
    wandb.finish()


if __name__ == "__main__":
    train_distillation()
