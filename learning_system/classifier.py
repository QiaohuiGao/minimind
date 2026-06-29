"""
模块 ②：Question Classifier —— 把问题归类到 Layer 0-7（关键词规则，不依赖 LLM）

对应 system_design_memo.md 第二章「七层分类法」。
原理很朴素：每层挂一组关键词，命中最多的那层即归类结果。
"""

# layer_id: (中文名, [关键词...])
LAYERS = {
    0: ("Environment 环境/工具", ["venv", "conda", "pip", "路径", "cuda", "驱动", "权限",
                                  "import", "sys.path", "目录", "安装", "__file__", "os.path"]),
    1: ("Data 数据", ["dataset", "tokenizer", "loss mask", "label", "jsonl", "数据", "分词",
                     "padding", "truncation", "conversations", "数据集", "样本"]),
    2: ("Architecture 架构", ["attention", "rope", "ffn", "norm", "embedding", "gqa", "swiglu",
                             "rmsnorm", "residual", "head", "transformer", "架构", "层"]),
    3: ("Training 训练", ["loss", "optimizer", "scheduler", "learning rate", "梯度", "gradient",
                         "backward", "warmup", "adamw", "overfit", "batch", "训练", "收敛"]),
    4: ("Hardware 硬件", ["gpu", "hbm", "flash attention", "nvidia-smi", "显存", "memory",
                         "显卡", "device", "fp16", "bf16", "硬件"]),
    5: ("Inference 推理", ["kv cache", "generation", "生成", "量化", "quantization", "temperature",
                          "sampling", "greedy", "推理", "decode", "vlm", "多模态"]),
    6: ("Alignment 对齐", ["sft", "dpo", "ppo", "grpo", "reward", "偏好", "preference", "rlhf",
                          "对齐", "蒸馏", "distillation", "微调"]),
    7: ("Systems 系统", ["分布式", "distributed", "tp", "dp", "pp", "zero", "lora", "并行",
                        "deepspeed", "parallel"]),
}


def classify(question: str):
    """返回归类结果 dict；没命中任何关键词返回 None。"""
    q = question.lower()
    scores = {}
    for lid, (name, kws) in LAYERS.items():
        hits = [k for k in kws if k.lower() in q]
        if hits:
            scores[lid] = hits
    if not scores:
        return None
    best = max(scores, key=lambda l: len(scores[l]))
    return {
        "layer": best,
        "name": LAYERS[best][0],
        "matched": scores[best],
        "all_hits": {LAYERS[l][0]: kw for l, kw in scores.items()},
    }
