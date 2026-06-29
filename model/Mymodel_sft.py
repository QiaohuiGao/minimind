import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import torch
import wandb
import random
from transformers import AutoTokenizer
from torch.utils.data import DataLoader,Dataset
from datasets import Features, Value, load_dataset
# SFT dataset 用本文件下面的 Mymodelsftdataset；只从 Mymodel 导入模型和配置
from model.Mymodel import Config, MyModel

def pre_processing_chat(conversation, add_system_ratio=0.2):
    # if conversation.get("role")=="system" or isinstance(conversation["tools"],str):
    #     return conversation
    if any(conv.get('tools') for conv in conversation): 
        return conversation
    
    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    
    if conversation[0].get("role")!="system":
        if random.random()<add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversation#直接加最简洁
    return conversation    

def post_processing_chat(prompt,empty_think_ratio=0.2):
    # apply_chat_template 渲染出来的空思考是 "<think>\n\n</think>\n\n"（带换行），
    # 不是 "<think></think>"；要匹配真实渲染结果才删得掉
    if '<think>\n\n</think>\n\n' in prompt and random.random() > empty_think_ratio:
        prompt = prompt.replace("<think>\n\n</think>\n\n", "")
    return prompt

    
class Mymodelsftdataset(Dataset):
    def __init__(self, sftdata_path,tokenizer, max_len):
        self.max_len=max_len
        self.tokenizer=tokenizer
        #Features = 告诉库原始数据格式是什么（防止推断出错）defines the internal structure of a dataset. It is used to specify the underlying serialization format.
        feature=Features({'conversations':[{"role":Value('string'),'content': Value('string'), 'reasoning_content': Value('string'), 'tools': Value('string'), 'tool_calls': Value('string')}]})
        #conversation:
            #role:
            #content:
            #tools
            #reasoning_content
            #tool_calls
        # data_files 是复数，不是 data_file
        self.datasets=load_dataset("json", data_files=sftdata_path, split="train", features=feature)
        #load_dataset(path, data_files=None, split=None, features=None, ...)
        # 命名要和 generate_labels 里用的 self.bos_ids / self.eos_ids 一致
        self.bos_ids=self.tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_ids=self.tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
    
    def __len__(self):
        return len(self.datasets)
    
    def create_chat_prompt(self,conversations):
        # [{"role":"user","content":"你好"}, {"role":"assistant","content":"你好啊"}]
        #             ↓ create_chat_prompt
        # "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n你好啊<|im_end|>\n"
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            # 从 system 消息里提取 tools 字段（JSON 字符串 → Python list）
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            # tool_calls 同样从 JSON 字符串转成 Python list
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])
            messages.append(message)
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, tools=tools
        )
        
    def generate_labels(self, input_ids):
        # 全部初始化为 -100（不参与 loss）
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            # 找到 <bos>assistant\n 的起始位置
            if input_ids[i: i + len(self.bos_ids)] == self.bos_ids:
                start = i + len(self.bos_ids)
                end = start
                # 找对应的 <eos>\n 结束位置
                while end < len(input_ids):
                    if input_ids[end: end + len(self.eos_ids)] == self.eos_ids:
                        break
                    end += 1
                # 把 assistant 回答部分（含 eos）的 label 设为真实 token id
                for j in range(start, min(end + len(self.eos_ids), self.max_len)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_ids) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        sample=self.datasets[index]
        # 字段名是 conversations（复数）
        conversations=pre_processing_chat(sample["conversations"])
        # 漏了这一步：先把 conversations 渲染成字符串，才能 post_processing / 编码
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt)#delete empty thinking process
        # self.max_len 不是 self.max_length
        input_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids[:self.max_len]

        pad_input_ids=self.max_len-len(input_ids)
        
        input_ids=input_ids + pad_input_ids * [self.tokenizer.pad_token_id]
        
        labels=self.generate_labels(input_ids)
        
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)              
                            
def train_mymodel_sft():
    # SFT 用更小的 learning rate，避免破坏 pretrain 学到的语言能力
    config = Config(learning_rate=1e-5)

    wandb.init(
        project="mymodel-sft",
        name=f"sft-lr{config.learning_rate}-epochs{config.epochs}",
        config=vars(config)
    )

    # tokenizer 在 model/ 目录下（tokenizer.json + tokenizer_config.json）
    tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(__file__))

    # Dataset 是纯 CPU 数据对象，不能 .to(device)；模型才需要 .to(device)
    # 用本文件里的 Mymodelsftdataset；用绝对路径，避免依赖运行时的 cwd
    sft_data_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "sft_t2t_mini.jsonl")
    dataset = Mymodelsftdataset(sft_data_path, tokenizer, config.max_seq_len)

    val_size = int(len(dataset) * 0.05)
    train_size = len(dataset) - val_size
    train_data, eval_data = torch.utils.data.random_split(dataset, [train_size, val_size])

    dataloader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(eval_data, batch_size=config.batch_size, shuffle=False, num_workers=8)

    model = MyModel(config).to(config.device)

    # SFT 从 pretrain checkpoint 开始，不是随机初始化（关键步骤）
    pretrain_path = os.path.join(os.path.dirname(__file__), "my_model.pt")
    model.load_state_dict(torch.load(pretrain_path, map_location=config.device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)  # optim 不是 optimal

    model.train()
    for epoch in range(config.epochs):
        total_loss = 0  # 在 step 循环外初始化，才能累积整个 epoch 的 loss
        for step, (input_ids, labels) in enumerate(dataloader):
            input_ids = input_ids.to(config.device)
            labels = labels.to(config.device)

            # SFT 训练不用 KV cache（完整序列一次过），不传 past_key_values
            _, loss, _ = model(input_ids, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # nn.utils 不是 utils；带下划线的 in-place 版本
            optimizer.step()  # step 是方法调用，必须加 ()

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
                            val_total += eval_loss.item()  # .item() 把 tensor 转成 float，否则 tensor 累加会保留计算图

                    val_avg = val_total / len(val_dataloader)
                    model.train()
                    print(f">> val loss: {val_avg:.4f}")
                    wandb.log({"val_loss": val_avg}, step=step)

        print(f"Epoch {epoch+1} 完成，平均 Loss: {total_loss / len(dataloader):.4f}")

    # state_dict() 是参数字典（可序列化）；parameters() 是迭代器，不能直接 save
    save_path = os.path.join(os.path.dirname(__file__), "my_model_sft.pt")
    torch.save(model.state_dict(), save_path)
    print(f"SFT 权重已保存至 {save_path}")

    wandb.finish()  # 训练全部结束后调用，不能放在 epoch 循环里

if __name__ == "__main__":
    train_mymodel_sft()

# 1) LoRA 模块：一条低秩旁路  x --A(降维)--> rank --B(升维)--> out
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank=8):
        super().__init__()
        self.rank = rank
        #1: 定义 A、B 两个 nn.Linear（都 bias=False）
        self.A = nn.Linear(in_features, rank, ...)    # 降维：in -> rank
        self.B = nn.Linear(rank, out_features, ...)   # 升维：rank -> out
    
        # 2: 初始化（关键！）
        # A 用高斯随机：self.A.weight.data.normal_(mean=0.0, std=0.02)
        # B 用全零    ：self.B.weight.data.zero_()
        # 为什么 B 要全零？ → 训练刚开始时 ΔW = B@A = 0，输出 == 原模型，
        #                      保证不会一上来就破坏 pretrain/SFT 学到的能力
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        self.B.weight.data.zero_()

    def forward(self, x):
        # 先降维再升维，得到 ΔW·x
        return self.B(self.A(x))


# 2) 给模型挂 LoRA：找到所有「方阵」Linear，加旁路并改写它的 forward
def apply_lora(model, rank=8):
    device = next(model.parameters()).device
    for name, module in model.named_modules():#随便取模型的一个参数，看它在哪个设备
        if isinstance(module,nn.Linear) and module.in_feature==module.out_features:
            lora = LoRA(module.in_features, module.out_features, rank).to(device)
            setattr(module, "lora", lora)   # 之后 module.lora 就能拿到
            original_forward = module.forward
            def new_forward(x, _orig=original_forward, _lora=lora):
                return _orig(x) + _lora(x)
            module.forward = new_forward

# 3) 冻结原模型，只训练 LoRA 的 A、B
def mark_only_lora_trainable(model):
    """把非 LoRA 参数 requires_grad=False；返回要交给 optimizer 的 LoRA 参数列表"""
    lora_params = []
    # 遍历 model.named_parameters()
    for name, p in model.named_parameters():
        
        if 'lora' in name:   # 名字里带 lora 的才训练
            p.requires_grad = True
            lora_params.append(p)
        else:
            p.requires_grad = False   # 其余全部冻结
    return lora_params


# 4) 只存 / 只读 LoRA 权重（文件很小，几 MB）
def save_lora(model, path):
    state_dict = {}
    # 遍历 named_modules，只收集有 .lora 的层
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            # key 建议存成  f'{name}.lora.{k}'  方便 load 时对应回去
            for k, v in module.lora.state_dict().items():
                state_dict[f'{name}.lora.{k}'] = v
    torch.save(state_dict, path)


def load_lora(model, path):
    state_dict = torch.load(path, map_location=next(model.parameters()).device)
    # 遍历 named_modules，给每个有 .lora 的层装回它自己的 A、B
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            # 从 state_dict 里筛出属于本层的 key，并去掉前缀变成 "A.weight"/"B.weight"
            sub = {k.replace(f'{name}.lora.', ''): v
                    for k, v in state_dict.items() if f'{name}.lora.' in k}
            module.lora.load_state_dict(sub)

# 5) 训练主流程（orchestration 已搭好；训练循环留给你填）
def run_train_lora():
    # LoRA 可训参数很少，通常用比 full-SFT 稍大的 learning rate
    config = Config(learning_rate=1e-4)

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

    # ④ 训练循环 —— 和 train_mymodel_sft 几乎一模一样，自己写一遍加深印象
    model.train()
    for epoch in range(config.epochs):
        total_loss = 0
        for step, (input_ids, labels) in enumerate(dataloader):
            input_ids = input_ids.to(config.device)
            labels = labels.to(config.device)
            # TODO（对照 train_mymodel_sft）:
            #   _, loss, _ = model(input_ids, labels)
            #   optimizer.zero_grad(); loss.backward()
            #   torch.nn.utils.clip_grad_norm_(lora_params, 1.0)   # 注意：clip 的是 lora_params
            #   optimizer.step()
            #   total_loss += loss.item()
            #   定期 print + wandb.log + 跑 val
            pass
        print(f"Epoch {epoch+1} 完成，平均 Loss: {total_loss / len(dataloader):.4f}")

    # ⑤ 只保存 LoRA 权重（不存底座，文件很小）
    save_lora(model, os.path.join(os.path.dirname(__file__), "my_model_lora.pt"))
    wandb.finish()