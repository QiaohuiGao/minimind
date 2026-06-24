import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import SFTDataset
from model.model_lora import save_lora, apply_lora
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
#  LoRA Fine-tuning 整体架构协作关系
#
#  训练脚本 (train_lora.py)
#    ├── SFTDataset               数据层：与 Full SFT 完全相同（对话格式 + loss mask）
#    ├── DataLoader               批处理
#    ├── MiniMindForCausalLM      模型层（结构不变，但被 apply_lora 修改了 forward）
#    │     └── apply_lora()       向每个 Attention 的 q_proj/v_proj 注入低秩矩阵：
#    │           原始 W [d×d]  →  W（frozen）+ B[d×r] × A[r×d]（可训练）
#    │           forward: output = W·x + B·A·x，r << d（如 r=8）
#    ├── lora_params only         optimizer 只接收名字含 'lora' 的参数（~1%）
#    │     其余参数 requires_grad=False，完全冻结
#    ├── GradScaler               混合精度
#    └── save_lora()              只保存 LoRA 矩阵（几 MB），不保存完整模型
#
#  数据流（一个 step）：
#    input_ids [B, S] ──→ model.forward（含 LoRA）──→ logits [B, S, 6400]
#    → CE loss（只算 assistant 部分）→ backward
#    → 梯度只流向 A/B 矩阵（原始 W 梯度为 None）→ optimizer.step
#
#  [与 Full SFT 的核心区别]：
#    起点权重：full_sft checkpoint（站在更高起点）
#    训练参数：只有 LoRA A/B 矩阵（Full SFT 训练全部参数）
#    保存内容：只保存 LoRA 权重（Full SFT 保存完整模型）
#    显存需求：更低（原始权重不需要存梯度和 optimizer state）
#    适用场景：领域微调（医疗/法律/金融），多个 LoRA 按需切换
#
#  [与 QLoRA 的区别]：
#    LoRA：原始权重保持 fp16/bf16（显存较大）
#    QLoRA：原始权重量化为 int4（显存压缩到 1/4，需要 bitsandbytes 库）
# ═══════════════════════════════════════════════════════════════════════════


def train_epoch(epoch, loader, iters, lora_params, start_step=0, wandb=None):
    # [与 pretrain/full_sft 相同] 训练循环骨架完全一致
    start_time = time.time()
    last_step = start_step
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        last_step = step
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            # [与 full_sft 不同] clip 的是 lora_params，不是 model.parameters()
            # 因为只有 LoRA 参数有梯度，clip 全参数没意义
            torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            lora_save_path = f'{args.save_dir}/{args.lora_name}_{lm_config.hidden_size}{moe_suffix}.pth'
            # [与 pretrain/full_sft 不同] 只保存 LoRA 参数（几 MB），不保存完整模型（几 GB）
            # 推理时：加载原始模型 + 叠加 LoRA 权重 = 完整能力
            save_lora(model, lora_save_path)
            lm_checkpoint(lm_config, weight=args.lora_name, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()

        del input_ids, labels, res, loss

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind LoRA Fine-tuning")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument("--lora_name", type=str, default="lora_medical", help="LoRA权重名称(如lora_identity/lora_medical等)")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=10, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/lora_medical.jsonl", help="LoRA训练数据路径")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练，默认full_sft")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-LoRA", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.lora_name, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-LoRA-{args.lora_name}-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、应用LoRA、冻结非LoRA参数 ==========
    # [与 pretrain 不同] pretrain 是随机初始化；LoRA 从已有权重（默认 full_sft）开始
    # [与 full_sft 不同] full_sft 从 pretrain 权重开始；LoRA 从 full_sft 权重开始（站在巨人肩上）
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)

    # [LoRA 核心操作 1] 向模型的 Attention q_proj/v_proj 注入低秩适配矩阵
    # 原始 W 保持不变，新增两个小矩阵 A(d×r) 和 B(r×d)，r << d（如 r=8, d=512）
    # forward 时：output = W·x + B·A·x，只有 A 和 B 参与训练
    apply_lora(model)

    # 统计参数，直观看到 LoRA 有多"轻量"
    total_params = sum(p.numel() for p in model.parameters())
    lora_params_count = sum(p.numel() for name, p in model.named_parameters() if 'lora' in name)
    Logger(f"LLM 总参数量: {total_params / 1e6:.3f} M")
    Logger(f"LoRA 参数量: {lora_params_count / 1e6:.3f} M")
    Logger(f"LoRA 参数占比: {lora_params_count / total_params * 100:.2f}%")

    # [LoRA 核心操作 2] 冻结原始模型的全部参数，只让 LoRA 的 A/B 矩阵可训练
    # [与 pretrain/full_sft 不同] 两者是 model.parameters() 全部可训练
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True   # LoRA 矩阵：可训练
            lora_params.append(param)
        else:
            param.requires_grad = False  # 原始权重：冻结，不更新

    # ========== 6. 定义数据和优化器 ==========
    # [与 full_sft 相同] 数据格式一样用 SFTDataset（对话格式 + loss mask）
    # [与 full_sft 不同] 数据量更小，通常是特定领域数据（如医疗、法律）
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    # [与 pretrain/full_sft 不同] optimizer 只传 lora_params，不是 model.parameters()
    # 这样 optimizer 完全不知道冻结参数的存在，节省内存和计算
    optimizer = optim.AdamW(lora_params, lr=args.learning_rate)
    
    # ========== 7. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 8. 编译和分布式包装 ==========
    # [与 full_sft 不同] LoRA 用 monkey-patch 替换 forward，与 torch.compile 不兼容，强制关闭
    if args.use_compile == 1:
        args.use_compile = 0
        Logger('[LoRA] monkey-patch forward 与 torch.compile 不兼容，use_compile 已自动关闭')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 9. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, lora_params, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), lora_params, 0, wandb)
    
    # ========== 10. 清理分布进程 ==========
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()