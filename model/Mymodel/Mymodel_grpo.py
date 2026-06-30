"""
最小可跑的 GRPO（Group Relative Policy Optimization），基于 MyModel。

GRPO 一句话：让模型自己生成一组(G个)回答 → 打分 → 用「组内平均分」当 baseline
            算 advantage → 强化高于平均的回答、抑制低于平均的（+ KL penalty 防跑偏）。
核心创新：用「组内相对」省掉了 PPO 的 critic 模型。

整体流程（train_grpo 里逐步对应 ①~⑥）：
  ① 准备 policy(要训) + reference(冻结，算 KL 用) 两个模型
  ② rollout：对每个 prompt 采样 G 个回答
  ③ reward：给每个回答打分（这里用规则占位，真实场景换 reward model）
  ④ advantage：组内归一化 (r - mean) / std   ← GRPO 的灵魂
  ⑤ loss：clip 后的 policy gradient + beta * KL(policy‖ref)
  ⑥ 标准五步更新（只更新 policy）

为「最小可跑」做的简化：
  - reward 用规则（token 多样性，反重复），不训 reward model
  - 不用 KV cache（generate 每步重跑全序列，代码更直观）
  - 不用 wandb，直接 print
  - prompt 硬编码几条
你可以在这些点上「逐步深化」。
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import copy
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from model.Mymodel.Mymodel import Config, MyModel


# ─────────────────────────────────────────────────────────────────────
# ③ reward：最小占位（规则奖励）。真实 GRPO 这里换成 reward model 或可验证奖励（如答案对错）
# ─────────────────────────────────────────────────────────────────────
def reward_fn(resp_token_ids):
    """玩具奖励：鼓励「不重复」—— 回答里不同 token 的占比越高，分越高（0~1）。
    语言无关、可验证，GRPO 真的能把它优化上去，适合验证流程跑通。"""
    if len(resp_token_ids) == 0:
        return 0.0
    return len(set(resp_token_ids)) / len(resp_token_ids)


# ─────────────────────────────────────────────────────────────────────
# ② rollout：采样回答
# ─────────────────────────────────────────────────────────────────────
def generate(model, prompt_ids, max_new_tokens, eos_id, temperature=1.0):
    """从 prompt 出发，按概率采样（multinomial）生成一条回答。返回完整序列的 token id 列表。"""
    ids = prompt_ids.clone()  # [1, P]
    for _ in range(max_new_tokens):  # 逐 token 自回归生成，最多 max_new_tokens 个
        with torch.no_grad():        # 生成阶段不需要梯度
            logits, _, _ = model(ids)
        next_logits = logits[:, -1, :] / temperature  # 只取最后位置的 logits；除以 T 调节随机度(T小=保守, T大=发散)
        probs = F.softmax(next_logits, dim=-1)         # logits → 概率分布
        # multinomial = 按概率"加权抽签"(不是均匀随机)：概率高的常被选，最差的词概率≈它本身,极低。
        # 这里故意不用 argmax：GRPO 要对同一 prompt 生成 G 个【不同】答案才能组内比较，argmax 每次都一样就没法学。
        # 注：本实现从全词表采样(只用 temperature)，没加 top-k/top-p；要砍长尾烂词可再加那两个过滤。
        next_token = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_token], dim=1)  # 把新 token 接到序列末尾，喂给下一轮
        if next_token.item() == eos_id:  # 生成到结束符就停
            break
    return ids[0].tolist()


def sample_group(model, prompt_ids, G, max_new_tokens, eos_id, pad_id, temperature=1.0):
    """对同一个 prompt 采样 G 个回答，padding 成同长，并给出 response_mask。
    返回 batch_ids [G, Lmax]、response_mask [G, Lmax]（1=生成的回答 token，0=prompt/pad）。"""
    P = prompt_ids.shape[1]
    seqs = [generate(model, prompt_ids, max_new_tokens, eos_id, temperature) for _ in range(G)]
    Lmax = max(len(s) for s in seqs)
    batch, mask = [], []
    for s in seqs:
        pad = Lmax - len(s)
        batch.append(s + [pad_id] * pad)
        m = [0] * Lmax
        for i in range(P, len(s)):  # 只有「回答」部分算 loss，prompt 和 pad 都置 0
            m[i] = 1
        mask.append(m)
    return torch.tensor(batch), torch.tensor(mask, dtype=torch.float)


# ─────────────────────────────────────────────────────────────────────
# 算「回答 token」的 log-prob（next-token 对齐）
# ─────────────────────────────────────────────────────────────────────
def response_logprobs(model, input_ids):
    """返回每个位置「预测下一个 token」的 log 概率，shape [B, L-1]。"""
    logits, _, _ = model(input_ids)                       # [B, L, V]
    logp = F.log_softmax(logits[:, :-1, :], dim=-1)       # 用前 L-1 个预测后 L-1 个
    targets = input_ids[:, 1:].unsqueeze(-1)              # [B, L-1, 1]
    token_logp = logp.gather(-1, targets).squeeze(-1)     # 取出真实 token 的 log 概率 [B, L-1]
    return token_logp


# ─────────────────────────────────────────────────────────────────────
# ⑤ GRPO loss
# ─────────────────────────────────────────────────────────────────────
def grpo_loss(logp_new, logp_old, logp_ref, advantages, mask, clip_eps=0.2, beta=0.04):
    """
    logp_new/old/ref: [B, L-1]  当前/采样时/参考模型 的 token log 概率
    advantages: [B]   每条回答一个（组内归一化后的优势）
    mask: [B, L-1]    1=回答 token

    ratio = exp(new - old)：当前策略相对采样时策略的概率比（PPO/GRPO 同款）
    clip：限制 ratio 不要偏离 1 太多，防止一步更新太猛
    KL penalty(k3 估计，恒 ≥0)：拉住 policy 别离 reference 太远
    """
    adv = advantages.unsqueeze(1)                         # [B,1] 广播到每个 token
    ratio = torch.exp(logp_new - logp_old)               # [B, L-1]
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    policy_term = torch.min(surr1, surr2)                 # clip 后的代理目标（要最大化）

    diff = logp_ref - logp_new
    kl = torch.exp(diff) - diff - 1                       # k3 KL 估计，无偏且 ≥0

    per_token = -(policy_term - beta * kl)                # 最大化目标 → 取负当 loss
    return (per_token * mask).sum() / mask.sum().clamp(min=1)  # 只在回答 token 上平均


def _pick_checkpoint():
    here = os.path.dirname(__file__)
    for name in ["my_model_sft.pt", "my_model.pt"]:   # GRPO 通常从 SFT 权重起；没有就用 pretrain
        p = os.path.join(here, name)
        if os.path.exists(p):
            return p
    return None


def train_grpo(group_size=4, max_new_tokens=24, steps=20, temperature=1.0, lr=1e-6):
    config = Config(learning_rate=lr)
    tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(__file__))
    eos_id, pad_id = tokenizer.eos_token_id, tokenizer.pad_token_id

    # ① policy（要训练）+ reference（冻结，算 KL 用）
    policy = MyModel(config).to(config.device)
    ckpt = _pick_checkpoint()
    if ckpt:
        policy.load_state_dict(torch.load(ckpt, map_location=config.device))
        print(f"已加载底座权重: {os.path.basename(ckpt)}")
    else:
        print("⚠️ 没找到 my_model_sft.pt / my_model.pt，用随机权重跑（仅验证流程）")

    # reference = policy 的冻结副本（回忆蒸馏 teacher：eval 关 dropout + 冻结参数）
    ref = copy.deepcopy(policy)
    ref.eval()
    ref.requires_grad_(False)

    # policy 用 eval() 关 dropout（让 logp_old 和 logp_new 一致；eval 不影响梯度）
    policy.eval()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=config.learning_rate)

    prompts = [
        "你好，请介绍一下你自己。",
        "用一句话解释什么是机器学习。",
        "讲个简短的笑话。",
    ]

    for step in range(steps):
        prompt = prompts[step % len(prompts)]
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(config.device)
        P = prompt_ids.shape[1]

        # ② rollout：采样一组回答
        ids, mask = sample_group(policy, prompt_ids, group_size, max_new_tokens, eos_id, pad_id, temperature)
        ids, mask = ids.to(config.device), mask.to(config.device)

        # ③ reward：给每条回答打分（取回答 token 算规则奖励）
        rewards = []
        for g in range(group_size):
            resp = [int(t) for t, m in zip(ids[g].tolist(), mask[g].tolist()) if m == 1]
            rewards.append(reward_fn(resp))
        rewards = torch.tensor(rewards, device=config.device)  # [G]

        # ④ advantage：组内归一化（GRPO 灵魂，省掉 critic）
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)  # [G]

        # 采样时策略 / 参考策略 的 log-prob（都不带梯度）
        with torch.no_grad():
            logp_old = response_logprobs(policy, ids)
            logp_ref = response_logprobs(ref, ids)
        mask_shift = mask[:, 1:]  # 和 [B, L-1] 的 logp 对齐

        # ⑤ 当前策略 log-prob（带梯度）+ 算 loss
        logp_new = response_logprobs(policy, ids)
        loss = grpo_loss(logp_new, logp_old, logp_ref, advantages, mask_shift)

        # ⑥ 标准五步（只更新 policy）
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        print(f"step {step:3d} | loss {loss.item():+.4f} | "
              f"reward mean {rewards.mean().item():.3f} max {rewards.max().item():.3f}")

    save_path = os.path.join(os.path.dirname(__file__), "my_model_grpo.pt")
    torch.save(policy.state_dict(), save_path)
    print(f"GRPO 权重已保存至 {save_path}")


if __name__ == "__main__":
    train_grpo()
