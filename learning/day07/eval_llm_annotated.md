# eval_llm.py 详细标注

---

## init_model()：加载模型和 tokenizer

```python
def init_model(args):
    # 加载 tokenizer（分词器），从 args.load_from 指定的路径读取
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)

    if 'model' in args.load_from:
        # load_from 默认是 'model'，说明用原生 torch 权重加载
        # 先按超参数建好空模型结构
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,           # 隐藏层维度
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling
        ))

        moe_suffix = '_moe' if args.use_moe else ''   # MoE 模型文件名加后缀

        # 拼出权重文件路径，比如：./out/pretrain_768.pth
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'

        # 把文件里的 weights 装进刚建好的空模型
        # strict=True：文件里的 key 和模型结构必须完全匹配，否则报错
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)

        if args.lora_weight != 'None':
            # 如果指定了 LoRA 权重，在 base model 上叠加 LoRA 层
            apply_lora(model)
            load_lora(model, f'./{args.save_dir}/{args.lora_weight}_{args.hidden_size}.pth')
    else:
        # load_from 是一个具体路径，说明用 transformers 格式加载（如 HuggingFace 模型）
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)

    get_model_params(model, model.config)   # 打印参数量

    # .half()：weights 转成 FP16，省显存
    # .eval()：关掉 dropout，推理模式
    # .to(device)：移到 GPU
    return model.half().eval().to(args.device), tokenizer
```

---

## main()：命令行参数 + 对话主循环

```python
def main():
    # argparse：让你在命令行传参，不用改代码
    # 比如：python eval_llm.py --weight pretrain --hidden_size 256
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_from', default='model')         # 加载方式
    parser.add_argument('--save_dir', default='out')            # 权重目录
    parser.add_argument('--weight', default='full_sft')         # 权重名前缀
    parser.add_argument('--hidden_size', default=768, type=int) # 隐藏层维度（要和训练时一致）
    parser.add_argument('--num_hidden_layers', default=8)
    parser.add_argument('--use_moe', default=0)
    parser.add_argument('--max_new_tokens', default=8192)       # 最多生成多少 token
    parser.add_argument('--temperature', default=0.85)          # 越大越随机，越小越确定
    parser.add_argument('--top_p', default=0.95)                # nucleus sampling 阈值
    parser.add_argument('--historys', default=0)                # 带几轮历史对话
    parser.add_argument('--show_speed', default=1)              # 是否打印 tokens/s
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    # 自动测试用的预设 prompt 列表
    prompts = ['你有什么特长？', '为什么天空是蓝色的', ...]

    conversation = []   # 存历史对话，用于多轮
    model, tokenizer = init_model(args)

    # 让用户选：自动跑预设 prompt，还是手动输入
    input_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))

    # TextStreamer：让 token 边生成边打印，不用等全部生成完
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    for prompt in prompt_iter:
        # 保留最近 N 轮历史（多轮对话用）
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})

        if 'pretrain' in args.weight:
            # pretrain 模型没有对话格式，直接用 bos_token + 文字续写
            inputs = tokenizer.bos_token + prompt
        else:
            # SFT/RL 模型用 chat_template 包装成对话格式
            # 比如：<|user|>你好<|assistant|>
            inputs = tokenizer.apply_chat_template(conversation, ...)

        inputs = tokenizer(inputs, return_tensors="pt").to(args.device)

        # model.generate()：HuggingFace 封装好的生成函数
        # do_sample=True：启用随机采样（temperature / top_p 才会生效）
        # streamer：边生成边打印
        generated_ids = model.generate(
            inputs=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            streamer=streamer,
            top_p=args.top_p,
            temperature=args.temperature,
        )

        # 解码：只取新生成的部分（去掉原始 prompt 的 token）
        response = tokenizer.decode(
            generated_ids[0][len(inputs["input_ids"][0]):],
            skip_special_tokens=True
        )

        conversation.append({"role": "assistant", "content": response})

        # 计算生成速度：新生成的 token 数 / 用时
        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        print(f'[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s')
```

---

## 和你的 Mymodel.py 的对比

| | minimind eval_llm.py | 你的 Mymodel.py |
|--|--|--|
| 参数传入 | argparse 命令行 | Config 类写死 |
| 模型加载 | `load_state_dict` + 路径拼接 | 同样方式 |
| 生成方式 | `model.generate()`（HF封装） | 手写 for 循环 |
| streaming | TextStreamer 边生成边打印 | 无，生成完再打印 |
| 多轮对话 | conversation 列表 + chat_template | 无 |
| pretrain vs SFT | 自动判断用哪种 prompt 格式 | 只有 pretrain 格式 |
