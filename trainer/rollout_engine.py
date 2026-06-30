"""
# 如果使用sglang加速，需通过以下命令首先启动（transformers格式）模型：
python -m sglang.launch_server --model-path ./minimind-3 --attention-backend triton --host 0.0.0.0 --port 8998

================================================================================
rollout_engine.py —— RL（GRPO/PPO）的「经验收集车间」
================================================================================
知识锚点：rollout = RL 里让模型「实际下场表演」。RL 训练分两阶段交替：
  ① rollout（本文件）：给一堆 prompt，让模型真的生成答案 → 拿到「它说了什么 + 每个词的 logprob」
  ② update（GRPO 训练器）：给答案打分(reward)，好的拉高、差的压低 → 更新模型
     ↑ 然后回到 ①，用更新后的模型再生成 ……
生成（推理）很慢，是 RL 的主要瓶颈，所以单独成模块，支持切换不同推理后端。

本文件结构（从上到下）：
  ① compute_per_token_logps()   工具函数：算「每个生成 token 的 logprob」
  ② RolloutResult (@dataclass)  数据容器：rollout 的产出打包
  ③ RolloutEngine (ABC)         抽象基类：定义引擎必须有的两个方法
  ④ TorchRolloutEngine          实现A：原生 PyTorch，简单/正确/慢（本地学习、debug）
  ⑤ SGLangRolloutEngine         实现B：SGLang HTTP 服务，复杂/快（正经训练、大模型）
  ⑥ create_rollout_engine()     工厂函数：按字符串造对应引擎
================================================================================
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import requests
import torch
import torch.distributed as dist
from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional, Tuple
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer


# ===== ① 工具函数：计算每个生成 token 的 logprob =====
# 作用：给定 model + 一条完整序列，返回「最后 n_keep 个生成 token，模型各给了多少 log 概率」。
# 为什么 GRPO 需要它：更新时要比较「新模型 vs 生成时旧模型」对同一个词的概率比值
#   (importance sampling ratio)，所以生成完得回头把每个 token 的 logprob 算出来存好。
def compute_per_token_logps(model, input_ids: Tensor, n_keep: int, attention_mask: Optional[Tensor] = None) -> Tensor:
    if n_keep <= 0:  # 没有生成段（纯 prompt）→ 返回空，避免后续越界
        return input_ids.new_empty((input_ids.size(0), 0), dtype=torch.float32)
    # DDP 包装过的模型要先 .module 取出真身，才能直接 forward
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    input_ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids
    # forward 取 logits；[:, :-1, :] 砍最后一个 → next-token 错位对齐：位置 t 的 logits 预测 t+1 的词
    logits = unwrapped(input_ids, attention_mask=attention_mask, logits_to_keep=n_keep + 1).logits[:, :-1, :]
    per_token_logps = []
    # 逐条样本：把 logits 过 log_softmax 变 log 概率，再 gather 出「实际生成的那个词」对应的值
    for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):  # input_ids[:, -n_keep:] = 只看生成段
        ids_row = ids_row.detach().clone() if ids_row.is_inference() else ids_row
        per_token_logps.append(
            # log_softmax → 每个词的 log 概率; gather(dim=1, 真实token索引) → 只取真实生成那个词的 log 概率
            torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1)
        )
    return torch.stack(per_token_logps)  # [B, n_keep] 每个生成 token 一个 logprob


# ===== ② 数据容器：rollout 的产出打包成一个结构体 =====
# rollout 一次产出的所有东西，交给 GRPO 训练器去打分、算 advantage、更新。
@dataclass
class RolloutResult:
    output_ids: Tensor       # 完整序列 [prompt + completion]，形状 [B*num_gen, P+R]
    completion_ids: Tensor   # 只有「生成的那段」，形状 [B*num_gen, R]
    per_token_logps: Tensor  # 每个生成 token 的 logprob（来自①），形状 [B*num_gen, R]
    completions: List[str]   # decode 回的人类可读文本 → 拿去算 reward 用
    prompt_lens: Tensor      # 每条 prompt 多长 → 用来切分 prompt / completion
    completion_mask: Tensor  # 哪些是真 token、哪些是 padding → 算 loss 时排除 pad

# ===== ③ 抽象基类：定义「任何 rollout 引擎都必须实现这两个方法」=====
# 抽象类 = 定标准。下面 Torch / SGLang 两个具体引擎都实现这俩接口，
# GRPO 训练器只认这个接口 → 可无缝替换后端（慢但简单 ↔ 快但复杂）。
class RolloutEngine(ABC):
    tokenizer = None
    @abstractmethod
    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int, temperature: float = 0.8) -> RolloutResult:
        pass  # 生成答案 → 返回 RolloutResult
    @abstractmethod
    def update_policy(self, model: torch.nn.Module):
        pass  # 把最新模型权重同步给推理引擎（on-policy 硬要求，见下方实现）


# ===== ④ 实现A：PyTorch 原生推理引擎（简单/正确/慢，适合本地学习、debug）=====
class TorchRolloutEngine(RolloutEngine):
    def __init__(self, policy_model: torch.nn.Module, tokenizer, device: str = "cuda", autocast_ctx=None):
        self.policy_model = policy_model       # 策略模型（在训的那个），生成就直接用它
        self.tokenizer = tokenizer
        self.device = device
        self.autocast_ctx = autocast_ctx       # 混合精度上下文（可选）

    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int, temperature: float = 0.8) -> RolloutResult:
        model = self.policy_model.module if isinstance(self.policy_model, DistributedDataParallel) else self.policy_model
        ctx = self.autocast_ctx if self.autocast_ctx else nullcontext()
        with torch.no_grad(), ctx:  # 生成阶段不需要梯度（梯度在 update 阶段才算）
            output_ids = model.generate(
                # ★ GRPO 的「组」：每条 prompt 复制 num_generations 份 → 对同一问题生成多个不同答案
                #   (do_sample=True + temperature 保证答案不同)。组内互相比较是 GRPO 区别于 DPO 的核心。
                input_ids=prompt_ids.repeat_interleave(num_generations, dim=0),
                attention_mask=attention_mask.repeat_interleave(num_generations, dim=0),
                max_new_tokens=max_new_tokens,
                do_sample=True,            # 采样（非 greedy）→ 才能生成多样的答案
                temperature=temperature,
                num_return_sequences=1,    # 每个输入返回1条（多样性已靠上面复制+采样实现）
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            ).clone()  # [B*num_gen, P+R]  P=prompt长, R=生成长
            prompt_len = prompt_ids.size(1)
            completion_ids = output_ids[:, prompt_len:]  # [B*num_gen, R] 切出「只有生成的那段」
            full_mask = (output_ids != self.tokenizer.pad_token_id).long()  # 非 pad 为1
            # 回头算每个生成 token 的 logprob（GRPO 更新要用，见①）
            per_token_logps = compute_per_token_logps(self.policy_model, output_ids, completion_ids.size(1), attention_mask=full_mask)
        completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)  # 转文本→算reward
        return RolloutResult(output_ids, completion_ids, per_token_logps, completions,
                             prompt_ids.new_full((output_ids.size(0),), prompt_len),                  # prompt_lens：本引擎左对齐，长度都一样
                             attention_mask.new_ones(output_ids.size(0), completion_ids.size(1)))     # completion_mask：本引擎里生成段全有效

    def update_policy(self, model: torch.nn.Module):
        # 模型就在本地内存里，换个引用即可（下一轮 rollout 自动用更新后的权重）
        self.policy_model = model


# ===== ⑤ 实现B：SGLang HTTP API 推理引擎（复杂/快，适合正经训练、大模型）=====
# 推理跑在另一个进程/服务器（SGLang），本类通过 HTTP 把 prompt 发过去、拿回结果。
# 代价：generate 走网络、权重要存盘再热加载（见 update_policy），比 Torch 版复杂得多。
class SGLangRolloutEngine(RolloutEngine):
    def __init__(self, base_url: str, model_path: str, shared_ckpt_path: str = "./sglang_ckpt", timeout: int = 120):
        self.base_url = base_url.rstrip('/')        # SGLang 服务地址
        self.shared_ckpt_path = shared_ckpt_path    # 训练端存权重、服务端读权重的共享目录
        self.timeout = timeout
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.http = requests

    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int, temperature: float = 0.8) -> RolloutResult:
        # 去除左侧 padding tokens，只保留有效 token（HTTP 传变长 list，不需要 padding）
        input_ids_list = []
        for ids, mask in zip(prompt_ids, attention_mask):
            valid_ids = ids[mask.bool()].tolist()  # mask 为1的才是真 token
            input_ids_list.append(valid_ids)
        # ★ 同 Torch 版的「组」：每条 prompt 复制 num_generations 份
        all_input_ids = [ids for ids in input_ids_list for _ in range(num_generations)]

        # 组装 HTTP 请求体：输入 + 采样参数 + 要求返回 logprob
        payload = {
            "input_ids": all_input_ids,
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "stop_token_ids": [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id else [],
            },
            "return_logprob": True,  # 让服务端直接返回每个 token 的 logprob（省得本地再算）
        }

        # 发请求给 SGLang 服务，拿生成结果
        resp = self.http.post(f"{self.base_url}/generate", json=payload, timeout=self.timeout)
        resp.raise_for_status()

        results = resp.json()
        if not isinstance(results, list):
            results = [results]  # 单条时包成 list，统一处理

        all_output_ids, all_completion_ids, all_logprobs = [], [], []
        completions = []

        # 逐条解析服务端返回：取 生成token / logprob / 文本
        for i, result in enumerate(results):
            meta = result.get("meta_info", {})
            completion_ids = meta.get("output_ids", result.get("output_ids", []))  # 生成的 token ids
            raw_logprobs = meta.get("output_token_logprobs", [])

            # SGLang 的 logprob 可能是 [logprob, token_id, ...] 的元组，也可能是裸数字 → 统一抽出 logprob
            logprobs = []
            for item in raw_logprobs:
                if isinstance(item, (list, tuple)) and len(item) >= 1:
                    logprobs.append(item[0])
                elif isinstance(item, (int, float)):
                    logprobs.append(item)

            # 对齐保护：logprob 数量和生成 token 数量对不齐时，补齐/截断
            if len(logprobs) < len(completion_ids):
                logprobs = [0.0] * (len(completion_ids) - len(logprobs)) + logprobs
            elif len(logprobs) > len(completion_ids):
                logprobs = logprobs[-len(completion_ids):] if completion_ids else []
            prompt = all_input_ids[i]
            full_output = prompt + completion_ids  # 拼回完整序列 [prompt + completion]
            all_output_ids.append(full_output)
            all_completion_ids.append(completion_ids)
            all_logprobs.append(logprobs)
            completions.append(self.tokenizer.decode(completion_ids, skip_special_tokens=True))

        # 变长的 list 要 padding 成同长度的 tensor，才能打包进 RolloutResult
        device = prompt_ids.device
        max_comp_len = max(1, max(len(ids) for ids in all_completion_ids))    # 最长生成段
        max_out_len = max(len(ids) for ids in all_input_ids) + max_comp_len   # 最长完整序列

        def pad_to_tensor(seqs, max_len, pad_val=0):  # 右侧补 pad_val 到 max_len，再转 tensor
            return torch.tensor([s + [pad_val] * (max_len - len(s)) for s in seqs], device=device)

        pad_id = self.tokenizer.pad_token_id
        # 打包成统一的 RolloutResult（和 Torch 版返回同样的结构，对训练器透明）
        return RolloutResult(
            output_ids=pad_to_tensor(all_output_ids, max_out_len, pad_val=pad_id),
            completion_ids=pad_to_tensor(all_completion_ids, max_comp_len, pad_val=pad_id),
            per_token_logps=pad_to_tensor(all_logprobs, max_comp_len, pad_val=0.0),
            completions=completions,
            prompt_lens=torch.tensor([len(ids) for ids in all_input_ids], device=device),  # 每条 prompt 真实长度（变长）
            completion_mask=torch.tensor([[1] * len(ids) + [0] * (max_comp_len - len(ids)) for ids in all_completion_ids], device=device),  # 真 token 1 / pad 0
        )

    # update_policy：on-policy 的硬要求——每训一步，推理服务端必须换成最新权重，
    # 否则就是「用旧模型的经验训新模型」。SGLang 在另一个进程，所以要：存盘 → HTTP 通知热加载。
    def update_policy(self, model: torch.nn.Module):
        ok = True
        if not dist.is_initialized() or dist.get_rank() == 0:  # 多卡时只让 rank0 存权重（避免重复/冲突）
            try:
                unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
                unwrapped = getattr(unwrapped, '_orig_mod', unwrapped)  # 解开 torch.compile 包装
                abs_path = os.path.abspath(self.shared_ckpt_path)
                state_dict = {k: v.detach().half().cpu() for k, v in unwrapped.state_dict().items()}  # 转 fp16+cpu 存盘，省空间
                unwrapped.save_pretrained(abs_path, state_dict=state_dict, safe_serialization=False)
                self.tokenizer.save_pretrained(abs_path)
                # 通知 SGLang 服务从磁盘热加载新权重
                resp = self.http.post(f"{self.base_url}/update_weights_from_disk", json={"model_path": abs_path}, timeout=self.timeout)
                if resp.status_code != 200: print(f"[SGLANG WARNING] update_weights 失败: {resp.status_code}, {resp.text}")
                ok = resp.status_code == 200
            except Exception as e:
                print(f"[SGLANG WARNING] update_weights 异常: {e}"); ok = False
        if dist.is_initialized():  # 多卡：rank0 的成败广播给所有进程，再一起对齐（barrier）
            ok_t = torch.tensor(int(ok), device=next(model.parameters()).device)
            dist.broadcast(ok_t, src=0); dist.barrier(); ok = bool(ok_t.item())
        if not ok: raise RuntimeError("SGLang update_policy failed")
        return ok

    def flush_cache(self) -> bool:  # 清空 SGLang 的 KV cache（换权重后旧缓存失效）
        resp = self.http.post(f"{self.base_url}/flush_cache", timeout=30)
        return resp.status_code == 200

    def health(self) -> bool:  # 探测 SGLang 服务是否在线
        try:
            resp = self.http.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except:
            return False


# ===== ⑥ 工厂函数：按 engine_type 字符串造对应引擎 =====
# 训练脚本只需传 "torch" 或 "sglang"，不用关心两个类的构造细节。
def create_rollout_engine(
    engine_type: str = "torch",
    policy_model: torch.nn.Module = None,
    tokenizer = None,
    device: str = "cuda",
    autocast_ctx = None,
    sglang_base_url: str = None,
    sglang_model_path: str = None,
    sglang_shared_path: str = None,
) -> RolloutEngine:
    if engine_type == "torch":
        return TorchRolloutEngine(policy_model, tokenizer, device, autocast_ctx)
    elif engine_type == "sglang":
        return SGLangRolloutEngine(sglang_base_url, sglang_model_path, sglang_shared_path)
    else:
        raise ValueError(f"不支持的引擎类型: {engine_type}")
