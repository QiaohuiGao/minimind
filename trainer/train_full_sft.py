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
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
#  Full SFT 整体架构协作关系
#
#  训练脚本 (train_full_sft.py)
#    ├── SFTDataset               数据层：读对话 jsonl → apply_chat_template
#    │     ├── input_ids          完整对话的 token 序列
#    │     └── labels             只有 assistant 回答部分是真实 token id，
#    │                            user/system 部分全部置为 -100（不参与 loss）
#    ├── DataLoader               批处理：切 batch，多进程预加载
#    ├── MiniMindForCausalLM      模型层（同 pretrain，结构不变）
#    ├── AdamW optimizer          优化器：更新全部模型参数（同 pretrain）
#    ├── GradScaler               混合精度
#    └── lm_checkpoint            存档
#
#  数据流（一个 step）：
#    jsonl {"conversations":[...]} → apply_chat_template → tokenize
#    → input_ids [B, S]  ──→ model.forward ──→ logits [B, S, 6400]
#    → labels    [B, S]       loss = CE(logits[:,:-1,:], labels[:,1:], ignore_index=-100)
#                                           ↑ -100 的位置自动跳过，只算 assistant 部分
#    → loss.backward → clip_grad_norm → optimizer.step
#
#  [与 Pretrain 的核心区别]：
#    起点权重：pretrain checkpoint（不是随机初始化）
#    labels：只有 assistant token 有值，其余 -100（loss mask）
#    learning rate：1e-5，比 pretrain 小 50 倍（微调不能步子太大）
#
#  [与 LoRA 的核心区别]：
#    Full SFT 更新全部参数（显存占用大，效果上限高）
#    LoRA 只更新 ~1% 的参数（显存占用小，适合资源受限场景）
# ═══════════════════════════════════════════════════════════════════════════


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    last_step = start_step
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        last_step = step

        # cosine schedule：动态调整 lr，每一步都更新
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ---- forward ----
        with autocast_ctx:  # 混合精度：forward 用 bfloat16，节省显存+加速
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss  # aux_loss 是 MoE 的负载均衡 loss；非 MoE 时为 0
            loss = loss / args.accumulation_steps  # 梯度累积：把 loss 均摊，等价于更大的 batch

        # ---- backward ----
        scaler.scale(loss).backward()  # scaler 防止 float16 梯度下溢（bfloat16 时是 no-op）

        # 每累积 N 步才真正更新一次参数
        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)                                     # 还原被 scale 放大的梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 梯度裁剪
            scaler.step(optimizer)   # 更新参数
            scaler.update()          # 调整 scaler 的放大系数
            optimizer.zero_grad(set_to_none=True)  # 清空梯度（set_to_none 比置 0 更省显存）

        # ---- 日志 ----
        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps  # 还原累积前的真实 loss
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss  # 去掉 MoE 辅助 loss 的纯语言模型 loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        # ---- checkpoint 保存 ----
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            # DDP 包装后真正的模型在 .module 里；torch.compile 后真正的模型在 ._orig_mod 里
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            # 存为 fp16（half）节省磁盘空间，推理时再转回 bfloat16
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            # 同时保存完整训练状态（含 optimizer/scaler/epoch/step），用于断点续训
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer,
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scaler=scaler)
            model.train()
            del state_dict

        del input_ids, labels, res, loss

    # epoch 结束时若还有未 flush 的累积梯度，补一次更新
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Full SFT")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='full_sft', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--head_dim', default=None, type=int, help="每个注意力头的维度，None表示用hidden_size//num_heads的默认值")
    parser.add_argument('--max_seq_len', default=768, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_t2t_mini.jsonl", help="训练数据路径")
    parser.add_argument('--from_weight', default='pretrain', type=str, help="基于哪个权重训练，为none则不基于任何权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    head_dim_kwargs = {"head_dim": args.head_dim} if args.head_dim is not None else {}
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe), **head_dim_kwargs)
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # CPU 不支持 autocast，用 nullcontext 占位（什么都不做）
    # GPU 用 autocast：forward 自动用低精度，backward 自动恢复，不用手动转
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 4. 配 wandb（实际用的是 swanlab，API 兼容 wandb）==========
    wandb = None
    if args.use_wandb and is_main_process():
        # import swanlab as wandb
        import wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None # 续训时取回上次的实验id
        resume = 'must' if wandb_id else None                     # 有id就接续上次曲线，否则新建
        wandb_run_name = f"MiniMind-Full-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. 定义模型、数据、优化器 ==========
    # init_model：内部做了 MiniMindForCausalLM(config) + load_state_dict(pretrain 权重)
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    
    # SFTDataset：读对话格式 JSONL，只有 assistant 部分的 token 参与 loss（其余位置 label=-100）
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    
    # 多卡时用 DistributedSampler 保证每张卡拿到不同的数据切片
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    
    # GradScaler 配合 float16 防梯度下溢；bfloat16 不需要（enabled=False 时是 no-op）
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========== 6. 从 checkpoint 恢复训练状态（断点续训）==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        # 恢复模型权重、optimizer 的 momentum/variance、scaler 放大系数、训练进度
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)  # torch.compile：把模型编译成优化的 kernel，通常提速 10-30%
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])  # 多卡 DDP 包装

    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)  # 多卡：每个 epoch 重新 shuffle
        
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        # SkipBatchSampler：断点续训时跳过已训练的 step，不从头重来
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        # pin_memory=True：数据锁页内存，CPU→GPU 传输更快
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)

    # ========== 9. 清理分布式进程 ==========
    if dist.is_initialized():
        dist.barrier()          # 等所有卡都跑完再继续
        dist.destroy_process_group()