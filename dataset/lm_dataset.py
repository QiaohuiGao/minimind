from torch.utils.data import Dataset
import torch
import json
import os
import random
from datasets import load_dataset, Features, Sequence, Value
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def pre_processing_chat(conversations, add_system_ratio=0.2):
    # tool call 数据包含 tools 字段，格式特殊，不做任何修改直接透传
    if any(conv.get('tools') for conv in conversations): return conversations

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
    # 20% 概率随机在 conversations 最前面插一条 system prompt
    # 让模型学会接受 system 指令，增加数据多样性；已有 system 时跳过
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations

def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    # apply_chat_template 渲染 reasoning 数据时，没有思考内容的 assistant 回答会带上空 <think> 标签
    # 80% 概率移除，避免模型学到"空思考"这种无意义行为；保留 20% 让模型知道可以不思考
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        # 编码文本，预留 2 个位置给 bos/eos，超长截断
        tokens = self.tokenizer(str(sample['text']), add_special_tokens=False, max_length=self.max_length - 2, truncation=True).input_ids
        # 首尾加 bos/eos token
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        # 不足 max_length 的在末尾补 pad，DataLoader 要求同 batch 内长度一致
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        # pretrain：label 和 input 完全一样，每个位置都预测下一个 token
        labels = input_ids.clone()
        # pad 位置设为 -100，cross_entropy 自动跳过，不计入 loss
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels


class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 显式声明 JSONL 字段结构，防止 huggingface datasets 自动推断类型出错
        # conversations 是列表，每条消息有 role/content/reasoning_content/tools/tool_calls
        features = Features({'conversations': [{'role': Value('string'), 'content': Value('string'), 'reasoning_content': Value('string'), 'tools': Value('string'), 'tool_calls': Value('string')}]})
        self.samples = load_dataset('json', data_files=jsonl_path, split='train', features=features)

        # 提前把 "<bos>assistant\n" 编码成 token id 列表
        # 作用：在 generate_labels 里用它定位每一轮 assistant 回答的起始位置
        # add_special_tokens=False：只对这段文字编码，不再额外插入 bos/eos
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids

        # 提前把 "<eos>\n" 编码成 token id 列表
        # 作用：在 generate_labels 里定位 assistant 回答的结束位置
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        # 把 conversations 列表渲染成模型输入的格式化字符串
        messages = []
        tools = None
        for message in conversations:
            # huggingface dataset 返回的是只读对象，转成普通 dict 才能修改字段
            message = dict(message)

            # role=system + tools 字段：工具定义，在对话最开始声明"有哪些工具可以用"
            # 例：{"role":"system","content":"你是助手","tools":"[{\"name\":\"get_weather\"}]"}
            # apply_chat_template 需要单独接收 tools 参数，所以这里从 system 消息里提取出来
            # tools 在 JSONL 里存成 JSON 字符串，需要先 json.loads() 解析成 dict/list
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]

            # role=tool：工具执行完之后返回的结果，出现在 assistant 调用工具之后
            # 例：{"role":"tool","content":"晴，25°C"}
            # 注意区分：role=system+tools 是"工具手册"（定义），role=tool 是"工具反馈"（结果）
            # tool_calls 同样存成 JSON 字符串，解析回来让 apply_chat_template 正确渲染
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])

            messages.append(message)

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,              # 只生成文本字符串，不做编码；__getitem__ 里统一编码
            add_generation_prompt=False, # 训练时不加末尾的 "<bos>assistant\n" 提示词
                                         # 推理时才需要，告诉模型"现在轮到你回答"
            tools=tools                  # 有 tool 定义时渲染进模板，没有则为 None
        )
        # 输出示例：
        # <bos>user\n你好<eos>\n<bos>assistant\n你好啊<eos>\n<bos>user\n再见<eos>\n...

    def generate_labels(self, input_ids):
        # SFT 核心：只对 assistant 回答部分计算 loss，user/system 全部忽略
        labels = [-100] * len(input_ids)  # 第一步：全部设为 -100（全部忽略）

        i = 0
        while i < len(input_ids):
            # 检查从当前位置开始是否匹配 "<bos>assistant\n" 的 token 序列
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)  # assistant 实际回答内容的第一个 token
                end = start
                # 从 start 往后扫，找到 "<eos>\n" 的位置，确定这轮回答在哪里结束
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # 把 start 到 end+eos 这一段的 label 设回真实 token id
                # 包含：reasoning <think>...</think> + 实际回答 + <eos>
                # min(..., self.max_length) 防止越界
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                # 跳到这轮 eos 之后，继续找下一轮 assistant 回答
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1  # 当前位置不是 assistant 开头，往后移一位继续扫描
        return labels
        # 结果示意（简化）：
        # token:  <bos> user \n 你好 <eos> \n <bos> asst \n 你好啊 <eos>
        # label:  -100  -100 -100 -100 -100 -100 -100 -100 -100  你好啊 <eos>

    def __getitem__(self, index):
        sample = self.samples[index]

        # 20% 概率随机插入 system prompt，增加数据多样性
        conversations = pre_processing_chat(sample['conversations'])

        # 渲染成格式化字符串：<bos>user\n你好<eos>\n<bos>assistant\n你好啊<eos>\n<bos>user\n再见<eos>\n...
        prompt = self.create_chat_prompt(conversations)

        # 80% 概率清除空的 <think>\n\n</think>，避免模型学到空思考
        prompt = post_processing_chat(prompt)

        # 编码成 token id，超过 max_length 直接截断（长对话后半段丢弃）
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]

        # 不足 max_length 的在末尾补 pad token，DataLoader 要求同 batch 内长度一致
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        # 生成 label mask：只有 assistant 回答位置是真实 token id，其余是 -100
        labels = self.generate_labels(input_ids)

        # # === 调试打印（取消注释可逐 token 查看 label 是否正确）===
        # print(f"\n--- Sample {index} ---")
        # for i, (x, y) in enumerate(zip(input_ids[:-1], labels[1:])):
        #     print(f"{i:3d}: X={self.tokenizer.decode([x])!r:16s} ---> Y={self.tokenizer.decode([input_ids[i+1]])!r:16s} label={y}")
        # # ================

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        self.samples = load_dataset('json', data_files=file_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        chosen = sample['chosen']  # 是一个 list，里面包含若干 {role, content}
        rejected = sample['rejected']  # 同上
        chosen_prompt = self.tokenizer.apply_chat_template(chosen, tokenize=False, add_generation_prompt=False)
        chosen_prompt = post_processing_chat(chosen_prompt)

        rejected_prompt = self.tokenizer.apply_chat_template(rejected, tokenize=False, add_generation_prompt=False)
        rejected_prompt = post_processing_chat(rejected_prompt)
        chosen_encoding = self.tokenizer(chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length')
        rejected_encoding = self.tokenizer(rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length')

        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)
        # input_ids:  <bos>user 你好<eos> <bos>assistant 你好呀<eos>  <pad> <pad>
        # loss_mask:    0    0   0   0      0      0       1 1 1 1      0    0
        #                                                 └─assistant 回答─┘
        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)
        
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    def generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


class RLAIFDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024, thinking_ratio=0.5):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.thinking_ratio = thinking_ratio  # 按概率开启 thinking
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        conversations = pre_processing_chat(conversations)
        use_thinking = random.random() < self.thinking_ratio
        return self.tokenizer.apply_chat_template(
            conversations[:-1],
            tokenize=False,
            open_thinking=use_thinking,
            add_generation_prompt=True
        )
    def __getitem__(self, index):
        sample = self.samples[index]
        prompt = self.create_chat_prompt(sample['conversations'])

        return {
            'prompt': prompt,
            'answer': ""
        }

class AgentRLDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.samples)

    def parse_conversations(self, conversations):
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            messages.append(message)
        return messages[:-1], tools

    def __getitem__(self, index):
        sample = self.samples[index]
        messages, tools = self.parse_conversations(sample['conversations'])
        return {'messages': messages, 'tools': tools, 'gt': sample['gt']}


if __name__ == "__main__":
    pass