"""
═══════════════════════════════════════════════════════════════════════════════
 MiniMind 预训练脚本 (Pretrain)
═══════════════════════════════════════════════════════════════════════════════
 目标：从零训练一个"文本续写"的 base 模型（学语言规律，还不会对话）。
 核心任务：next-token prediction（给定前文，预测下一个 token）。

 整体数据流：
   jsonl({"text":...}) → PretrainDataset 编码 → DataLoader 拼batch
   → model.forward 算 loss → backward 反向传播 → optimizer 更新权重 → 存档

 运行示例：
   python train_pretrain.py --data_path ../dataset/pretrain_t2t_mini.jsonl
═══════════════════════════════════════════════════════════════════════════════
"""
import os
import sys

# __package__ + sys.path：让本脚本在 trainer/ 目录下运行时，也能 import 到上级目录的
# model/、dataset/ 等包（把项目根目录加入模块搜索路径）。
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL 冲突的 workaround（issue #771），必须在 torch 前导入
import argparse                                          # 解析命令行参数（--batch_size 等）
import time                                              # 计时，用于估算剩余训练时间
import warnings
import torch
import torch.distributed as dist                         # 分布式训练（多卡）
from contextlib import nullcontext                        # 空上下文管理器（CPU 时占位用）
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel     # 多卡并行包装器(DDP)
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig           # 模型配置类
from dataset.lm_dataset import PretrainDataset            # 预训练数据集类(本地 dataset/ 文件夹)
# 一批训练辅助工具：学习率调度/日志/主进程判断/检查点/分布式初始化/随机种子/建模型/跳batch
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')                         # 屏蔽各种警告，让日志干净


# ═══════════════════════════════════════════════════════════════════════════
#  训练一个 epoch（把数据集完整过一遍）
#  参数：epoch=当前轮次, loader=数据加载器, iters=本轮总步数,
#       start_step=起始步(续训用), wandb=实验记录器(可为None)
# ═══════════════════════════════════════════════════════════════════════════
def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()                              # 记录开始时间，用于算 ETA
    last_step = start_step
    # 遍历每个 batch。enumerate(..., start=...) 让 step 从 start_step+1 开始计数
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)             # 把数据搬到 GPU。input_ids 形状[B, L]
        labels = labels.to(args.device)                   # 标签同样搬 GPU。pad 位置为 -100(不算loss)
        last_step = step

        # —— 学习率调度：按当前训练进度动态计算 lr（warmup + cosine 衰减）——
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:        # 把新 lr 写进优化器
            param_group['lr'] = lr

        # —— 前向传播（在混合精度上下文里，省显存、提速）——
        with autocast_ctx:
            res = model(input_ids, labels=labels)         # 模型计算，返回含 loss/logits 的输出对象
            loss = res.loss + res.aux_loss                # 主损失 + MoE均衡损失(没开MoE时aux=0)
            # 梯度累积：把 loss 除以累积步数，使多步累积的梯度等价于一个大 batch
            loss = loss / args.accumulation_steps

        # —— 反向传播：算梯度。scaler 用于 float16 混合精度下的梯度缩放，防止下溢 ——
        scaler.scale(loss).backward()

        # —— 每累积满 accumulation_steps 步，才真正更新一次权重 ——
        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)                                      # 先把梯度还原回真实尺度
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 梯度裁剪，防梯度爆炸
            scaler.step(optimizer)                                          # 更新权重
            scaler.update()                                                 # 更新缩放因子
            optimizer.zero_grad(set_to_none=True)                           # 清空梯度，准备下一轮累积

        # —— 每隔 log_interval 步打印一次日志（也在最后一步打印）——
        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps            # 还原成未除以累积步的真实loss
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss           # 纯预测损失 = 总loss - aux
            current_lr = optimizer.param_groups[-1]['lr']
            # 估算本轮剩余时间(分钟) = 平均每步耗时 × 剩余步数 / 60
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            # 若开了 wandb/swanlab，把指标上传到网页仪表盘画曲线
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        # —— 每隔 save_interval 步保存一次模型（只在主进程存，避免多卡重复写）——
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()                                                    # 切到推理模式（关 dropout 等）
            moe_suffix = '_moe' if lm_config.use_moe else ''                # MoE 模型文件名加后缀区分
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            # 取出"裸模型"：剥掉 DDP 包装(.module)和 torch.compile 包装(._orig_mod)
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()                             # 取出所有权重
            # 存权重：转半精度(half)省空间、搬回CPU再存
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            # 另存一份完整检查点(含优化器/scaler/epoch/step)，用于断点续训
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()                                                   # 切回训练模式
            del state_dict

        del input_ids, labels, res, loss                                    # 手动释放，缓解显存压力

        # —— 实验用：跑到指定步数自动停（--max_steps>0 时生效），免得手动 Ctrl+C ——
        if args.max_steps and step >= args.max_steps:
            Logger(f'达到 max_steps={args.max_steps}，自动停止本次训练')
            break

    # —— 收尾：若最后剩了不足一个累积周期的梯度，补一次更新，别浪费 ——
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


# ═══════════════════════════════════════════════════════════════════════════
#  主程序入口（直接运行本文件时才执行）
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # —— 定义所有命令行参数，每个都有默认值，可在启动时用 --xxx 覆盖 ——
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数(调试时设0)")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument("--max_steps", type=int, default=0, help="跑到第几步自动停(0=不限制,跑满epoch)；做对照实验时设如1000") #实验所用
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    # ----- 架构超参(做对照实验用；都会传进 MiniMindConfig 并记入 wandb) -----
    parser.add_argument('--num_attention_heads', default=8, type=int, help="Q注意力头数(必须能被 num_key_value_heads 整除；不整除hidden_size时需同时显式传偶数head_dim)")
    parser.add_argument('--num_key_value_heads', default=4, type=int, help="KV头数(GQA分组，必须整除num_attention_heads；=Q头数即标准MHA)")
    parser.add_argument('--head_dim', default=0, type=int, help="每个注意力头维度(0=auto=hidden_size//num_attention_heads)；显式传必须为偶数(RoPE要求)，可与hidden_size解耦")
    parser.add_argument('--intermediate_size', default=0, type=int, help="Dense FFN中间维度(0=auto=ceil(hidden_size*pi/64)*64)")
    parser.add_argument('--dropout', default=0.0, type=float, help="Dropout丢弃率(预训练大数据通常0；>0用于正则)")
    parser.add_argument('--rope_theta', default=1e6, type=float, help="RoPE基础频率base(越大支持有效上下文越长)")
    parser.add_argument('--max_position_embeddings', default=3276, type=int, help="最大位置数(必须>=max_seq_len，以及推理时prefill+max_new_tokens)")
    # ----- MoE专用超参(仅 use_moe=1 时生效) -----
    parser.add_argument('--num_experts', default=4, type=int, help="MoE专家总数(仅use_moe=1)")
    parser.add_argument('--num_experts_per_tok', default=1, type=int, help="每token激活专家数top-k(须 1<=k<=num_experts；仅use_moe=1)")
    parser.add_argument('--moe_intermediate_size', default=0, type=int, help="每个MoE专家FFN中间维度(0=auto=intermediate_size；仅use_moe=1)")
    parser.add_argument('--router_aux_loss_coef', default=5e-4, type=float, help="MoE负载均衡aux_loss权重(仅use_moe=1)")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t_mini.jsonl", help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb/swanlab可视化")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    parser.add_argument("--run_name", type=str, default="", help="wandb运行名(自己命名实验，最清晰；留空则自动从改动的超参生成)")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()                            # 解析，结果存进 args

    # ========== 1. 初始化分布式环境和随机种子 ==========
    local_rank = init_distributed_mode()                  # 多卡时初始化进程组，单卡返回默认值
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"  # 多卡时每个进程绑定自己的GPU
    # 设随机种子保证可复现；多卡时每张卡用不同种子(+rank)，避免数据增强等完全雷同
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查检查点 ==========
    os.makedirs(args.save_dir, exist_ok=True)             # 创建权重保存目录(已存在则忽略)
    # 用命令行参数构造模型配置对象。0=auto 的派生项(head_dim/intermediate_size/moe_intermediate_size)仅在>0时覆盖，否则交给config按公式算
    config_kwargs = dict(
        hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe),
        num_attention_heads=args.num_attention_heads, num_key_value_heads=args.num_key_value_heads,
        dropout=args.dropout, rope_theta=args.rope_theta, max_position_embeddings=args.max_position_embeddings,
        num_experts=args.num_experts, num_experts_per_tok=args.num_experts_per_tok, router_aux_loss_coef=args.router_aux_loss_coef,
    )
    if args.head_dim > 0: config_kwargs["head_dim"] = args.head_dim                          # 必须为偶数(RoPE要求)
    if args.intermediate_size > 0: config_kwargs["intermediate_size"] = args.intermediate_size
    if args.moe_intermediate_size > 0: config_kwargs["moe_intermediate_size"] = args.moe_intermediate_size
    # 安全校验：尽早报清晰错误，避免模型深处的 RuntimeError(约束经对抗式验证)
    assert args.num_attention_heads % args.num_key_value_heads == 0, \
        f"num_attention_heads({args.num_attention_heads}) 必须能被 num_key_value_heads({args.num_key_value_heads}) 整除 (GQA约束)"
    _eff_head_dim = args.head_dim if args.head_dim > 0 else (args.hidden_size // args.num_attention_heads)
    assert _eff_head_dim % 2 == 0, \
        f"head_dim({_eff_head_dim}) 必须为偶数 (RoPE约束)；若 hidden_size//num_attention_heads 为奇数请显式传偶数 --head_dim"
    assert args.max_position_embeddings >= args.max_seq_len, \
        f"max_position_embeddings({args.max_position_embeddings}) 必须 >= max_seq_len({args.max_seq_len})"
    if bool(args.use_moe):
        assert 1 <= args.num_experts_per_tok <= args.num_experts, \
            f"需满足 1 <= num_experts_per_tok({args.num_experts_per_tok}) <= num_experts({args.num_experts})"
    lm_config = MiniMindConfig(**config_kwargs)
    # 若开启续训，加载上次的检查点数据；否则为 None
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None

    # ========== 3. 设置混合精度(AMP) ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # CPU 用空上下文(不做混合精度)；GPU 用 autocast 自动混合精度(部分算子用低精度，省显存提速)
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 4. 配置 wandb/swanlab 实验追踪(可视化训练曲线) ==========
    wandb = None                                          # 默认不开启
    if args.use_wandb and is_main_process():              # 加了 --use_wandb 且是主进程才开
        import wandb                                      # 使用 wandb.ai（已 wandb login 登录）
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None  # 续训时取回上次的实验id
        resume = 'must' if wandb_id else None             # 有id就接续上次曲线，否则新建
        # 运行名优先用 --run_name(你自己命名最清晰)；留空才自动从"与默认值不同的超参"生成
        if args.run_name:
            wandb_run_name = args.run_name
        else:
            _SKIP_KEYS = {"save_dir", "save_weight", "device", "dtype", "num_workers", "data_path",
                          "from_weight", "from_resume", "use_wandb", "wandb_project", "run_name", "use_compile",
                          "log_interval", "save_interval", "max_steps", "epochs"}
            _ABBR = {"hidden_size": "h", "num_hidden_layers": "L", "batch_size": "bs", "learning_rate": "lr",
                     "accumulation_steps": "acc", "grad_clip": "gc", "max_seq_len": "seq",
                     "use_moe": "moe", "num_attention_heads": "nah", "num_key_value_heads": "nkv", "head_dim": "hd",
                     "intermediate_size": "inter", "dropout": "do", "rope_theta": "rope", "max_position_embeddings": "maxpos",
                     "num_experts": "ne", "num_experts_per_tok": "nek", "moe_intermediate_size": "moeinter", "router_aux_loss_coef": "aux"}
            _defaults = {a.dest: a.default for a in parser._actions if a.dest != "help"}
            _parts = []
            for k, v in vars(args).items():
                if k in _SKIP_KEYS or k not in _defaults or v == _defaults[k]:
                    continue                              # 跳过非超参字段 / 未改动的参数
                val = f"{v:g}" if isinstance(v, float) else (int(v) if isinstance(v, bool) else v)
                _parts.append(f"{_ABBR.get(k, k)}{val}")
            wandb_run_name = "-".join(_parts) if _parts else "baseline"
        # config=vars(args)：把全部超参记进 wandb，可用对比表/平行坐标图分析"超参→loss"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume, config=vars(args))  # 启动记录

    # ========== 5. 创建模型、数据集、优化器 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)  # 建模型(可基于已有权重)+分词器
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)  # 加载预训练数据集
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None     # 多卡时给每张卡分不同数据
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))                # float16梯度缩放器(bf16不需要)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)                   # AdamW优化器

    # ========== 6. 从检查点恢复训练状态(续训) ==========
    start_epoch, start_step = 0, 0
    if ckp_data:                                          # 若有检查点，恢复模型/优化器/scaler/进度
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========== 7. (可选)编译加速 + 多卡并行包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)                      # torch.compile：首次编译慢，之后每步更快
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])  # DDP包装，多卡同步梯度

    # ========== 8. 开始训练(按 epoch 循环) ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)  # 多卡时每轮重设种子，保证各卡数据划分一致且每轮打乱
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()  # 单卡时手动打乱样本顺序
        # 续训时跳过已训练过的 step(只在恢复的那一轮跳)
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        # SkipBatchSampler：能跳过前 skip 个 step 的批采样器
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        # DataLoader：多进程加载数据并拼成 batch。pin_memory 加速 CPU→GPU 拷贝
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)  # 续训：总步数要加回跳过的
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)                  # 正常从头训这一轮

    # ========== 9. 清理分布式进程组 ==========
    if dist.is_initialized():
        dist.barrier()                                    # 等所有进程都到这(同步)
        dist.destroy_process_group()                      # 销毁进程组，干净退出
