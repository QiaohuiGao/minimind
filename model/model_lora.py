"""
LoRA (Low-Rank Adaptation) 低秩适配器

核心思想：不修改原模型的权重 W，而是在旁边加一条"旁路" ΔW = B × A
    原始: output = W @ x              (W是冻结的预训练权重)
    LoRA: output = W @ x + B @ A @ x  (只训练 A 和 B，参数极少)

为什么有效：
    W 是 [768, 768] = 589,824 个参数
    A 是 [768, 16] + B 是 [16, 768] = 24,576 个参数 (仅 4.2%)
    低秩假设: 微调只需要改变权重矩阵的一个低秩子空间，不需要改全部参数

整体流程：
    1. 加载预训练好的模型（Full SFT 之后的权重）
    2. apply_lora(): 给每个方阵 Linear 层挂上 LoRA 旁路
    3. 冻结原始权重，只训练 LoRA 参数（A 和 B 矩阵）
    4. save_lora(): 只保存 LoRA 权重（很小，几MB）
    5. 部署时: merge_lora() 把 LoRA 权重合并回原模型 → W' = W + B×A
"""
import torch
from torch import optim, nn


# ════════════════════════════════════════════════════════════════════════════════
#                           LoRA 模块定义
# ════════════════════════════════════════════════════════════════════════════════
# 一条低秩旁路: x → A (降维) → B (升维) → ΔW·x
#
# 参数量对比 (以 q_proj 768→768 为例):
#   原始 W:  768 × 768 = 589,824
#   LoRA:    768 × 16 + 16 × 768 = 24,576  (仅 4.2%)
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank  # rank = 秩，控制旁路的"宽度"，越大表达力越强但参数越多
        self.A = nn.Linear(in_features, rank, bias=False)    # A: 降维矩阵 [768 → 16]
        self.B = nn.Linear(rank, out_features, bias=False)   # B: 升维矩阵 [16 → 768]
        self.A.weight.data.normal_(mean=0.0, std=0.02)  # A 高斯随机初始化
        self.B.weight.data.zero_()                       # B 全零初始化 → 训练开始时 ΔW = B×A = 0，不影响原模型

    def forward(self, x):
        # x → A(降维) → B(升维) = ΔW·x
        return self.B(self.A(x))


# ════════════════════════════════════════════════════════════════════════════════
#                         挂载 LoRA 到模型
# ════════════════════════════════════════════════════════════════════════════════
# 找到所有方阵 Linear 层 (in == out，即 q_proj/o_proj 等 768→768 的层)
# 给它们加上 LoRA 旁路，修改 forward: output = 原始(x) + LoRA(x)
def apply_lora(model, rank=16):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.in_features == module.out_features:
            # 只给方阵层加 LoRA (q_proj, o_proj 等)
            # k_proj/v_proj 是 768→384 不是方阵，不加
            lora = LoRA(module.in_features, module.out_features, rank=rank).to(model.device)
            setattr(module, "lora", lora)  # 把 LoRA 挂到模块上
            original_forward = module.forward

            # 修改 forward: 原始输出 + LoRA旁路输出
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)  # W·x + B·A·x

            module.forward = forward_with_lora


# ════════════════════════════════════════════════════════════════════════════════
#                       LoRA 权重的 加载/保存/合并
# ════════════════════════════════════════════════════════════════════════════════
def load_lora(model, path):
    """加载 LoRA 权重（只有 A 和 B 矩阵，很小）"""
    state_dict = torch.load(path, map_location=model.device)
    # 去掉 DDP 的 'module.' 前缀
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            module.lora.load_state_dict(lora_state)


def save_lora(model, path):
    """只保存 LoRA 权重（不保存原模型权重，所以文件很小）"""
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            clean_name = name[7:] if name.startswith("module.") else name
            lora_state = {f'{clean_name}.lora.{k}': v.cpu().half() for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)


def merge_lora(model, lora_path, save_path):
    """合并 LoRA 到原模型: W' = W + B×A，合并后不再需要 LoRA 模块"""
    load_lora(model, lora_path)
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {k: v.cpu().half() for k, v in raw_model.state_dict().items() if '.lora.' not in k}
    for name, module in raw_model.named_modules():
        if isinstance(module, nn.Linear) and '.lora.' not in name:
            state_dict[f'{name}.weight'] = module.weight.data.clone().cpu().half()
            if hasattr(module, 'lora'):
                # 关键一步: W' = W + B × A (矩阵乘法合并)
                state_dict[f'{name}.weight'] += (module.lora.B.weight.data @ module.lora.A.weight.data).cpu().half()
    torch.save(state_dict, save_path)
