"""
LoRA 练习框架 —— 在 MyModel 的 pretrain / full-SFT 权重之上做 LoRA 微调

═══════════════════════════════════════════════════════════════════════
LoRA 核心思想：冻结原权重 W，旁边挂一条「低秩旁路」 ΔW = B @ A，只训练 A、B
    output = W·x  +  (B·A)·x
            └─冻结─┘   └──只训练这两个小矩阵──┘

为什么省参数（以 q_proj: 256→256 为例，rank=8）：
    原始 W: 256×256        = 65536 个参数（冻结，不训练）
    LoRA  : 256×8 + 8×256  = 4096  个参数（仅约 6%，只训这些）

练习方法：
    把下面带  # TODO  的地方自己填完。每个 TODO 都给了提示（公式/坑）。
    卡住了 → 对照同目录 model_lora.py（那是完整参考实现）。
═══════════════════════════════════════════════════════════════════════
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from torch import nn
import wandb
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# 注意类名是 MyModel（不是 Mymodel）；用 model.xxx 前缀 + 上面的 sys.path 保证哪个目录都能跑
from model.Mymodel import MyModel, Config
from model.Mymodel_sft import Mymodelsftdataset

class LoRa(nn.module):
    def __init__(self,in_feature:int, out_feature:int,rank:int):
        super().__init__()
        self.A=nn.Linear(in_feature,rank,bias=False)
        self.B=nn.Linear(rank,out_feature,bias=False)
        self.A.weight.data.normal_(mean=0,std=0.02)
        self.B.weight.data.zero_()
    
    def forward(self,x):
        return self.B(self.A(x))

def apply_lora(model, rank:int):
    device=next(model.parameters()).device
    for name, module in model.name_modules():
        if isinstance(module,nn.Linear)and module.in_features==module.out_features:
            lora=LoRa(module.in_features,module.out_features,rank)
            setattr(module, "lora", lora) 
            original_forward=model.forward
            def new_forward(x,_orig=original_forward,_lora=lora):
                return _orig(x)+_lora(x)
            module.forward=new_forward

def mark_only_lora_trainable(model):
    lora_params=[]
    for name, p in model.named_parameters():
        if "lora" in name:
            p.requires_grad=True
            lora_params.append(p)
        else:
            p.quires_grad=False
    return lora_params

def save_lora(model,path):
    state_dict={}
    for name, module in model.named_modules():
        if hasattr(module, "lora"):
            for k, v in module.lora.state_dict().items():
                state_dict[f"{name}.lora.{k}"]=v
    torch.save(state_dict,path)          
    
def run_train_lora():
    config=Config()
    wandb.init(
        project="mymodel-lora",
        name=f"lora-lr{config.learning_rate}-epochs{config.epochs}",
        config=vars(config)
    )
    tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(__file__))
    
    # 数据：复用 SFT 的 dataset（LoRA 也是在做指令微调）
    sft_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "sft_t2t_mini.jsonl")
    dataset = Mymodelsftdataset(sft_path, tokenizer, config.max_seq_len)
    
    
    val_size = int(len(dataset) * 0.05)
    train_size = len(dataset) - val_size
    train_data, eval_data = torch.utils.data.random_split(dataset, [train_size, val_size])
    dataloader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(eval_data, batch_size=config.batch_size, shuffle=False, num_workers=8)

    # ① 建模型 + 加载「已经训练好」的权重作为底座
    #    LoRA 一般挂在 full-SFT 之后的权重上；没有的话用 pretrain 的 my_model.pt 也行
    model = MyModel(config).to(config.device)
    base_ckpt = os.path.join(os.path.dirname(__file__), "my_model_sft.pt")  # 或 "my_model.pt"
    model.load_state_dict(torch.load(base_ckpt, map_location=config.device))

    # ② 挂 LoRA + 冻结底座，只留 LoRA 可训练
    apply_lora(model, rank=8)
    lora_params = mark_only_lora_trainable(model)

    # ③ optimizer 只优化 LoRA 参数（这是 LoRA 省显存/省算力的关键）
    optimizer = torch.optim.AdamW(lora_params, lr=config.learning_rate)
    
    # ④ 训练循环 —— 和 train_mymodel_sft 几乎一模一样
    model.train()
    for epoch in range(config.epochs):
        total_loss = 0  # 累积整个 epoch 的 loss，放 step 循环外
        for step, (input_ids, labels) in enumerate(dataloader):
            input_ids = input_ids.to(config.device)
            labels = labels.to(config.device)

            # forward：底座被冻结，梯度只会流到 LoRA 的 A、B 上
            _, loss, _ = model(input_ids, labels)

            optimizer.zero_grad()
            loss.backward()
            # 只对 LoRA 参数做梯度裁剪（底座本来就不更新，传 lora_params 即可）
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            optimizer.step()

            total_loss += loss.item()

            if step % 100 == 0:
                avg_loss = total_loss / (step + 1)
                print(f"Epoch {epoch+1}/{config.epochs} | Step {step}/{len(dataloader)} | Loss: {loss.item():.4f} | Avg: {avg_loss:.4f}")
                wandb.log({"train_loss": loss.item(), "avg_loss": avg_loss}, step=step)

                if step % 500 == 0:
                    model.eval()
                    val_total = 0
                    with torch.no_grad():
                        for eval_ids, eval_labels in val_dataloader:
                            eval_ids = eval_ids.to(config.device)
                            eval_labels = eval_labels.to(config.device)
                            _, eval_loss, _ = model(eval_ids, eval_labels)
                            val_total += eval_loss.item()
                    val_avg = val_total / len(val_dataloader)
                    model.train()  # 评估完切回训练模式
                    print(f">> val loss: {val_avg:.4f}")
                    wandb.log({"val_loss": val_avg}, step=step)

        print(f"Epoch {epoch+1} 完成，平均 Loss: {total_loss / len(dataloader):.4f}")

    # ⑤ 只保存 LoRA 权重（不存底座，文件很小）
    save_lora(model, os.path.join(os.path.dirname(__file__), "my_model_lora.pt"))
    wandb.finish()
    
if __name__ == "__main__":
    run_train_lora()
