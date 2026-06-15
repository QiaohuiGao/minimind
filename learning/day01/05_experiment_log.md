# Day 1 实验记录：超参数对训练的影响（控制变量法）

> **目标**：通过"每次只改一个变量"的对照实验，亲手感受各超参数对 loss 和训练的影响。
> **方法**：控制变量法——固定其他所有参数，只变一个，对比 loss 曲线。

---

## 第 0 步：准备小数据子集（让每个实验只跑几分钟）

跑满整份数据太慢，先切一个小子集用于快速实验：

```bash
cd ~/Project/minimind
head -n 50000 dataset/pretrain_t2t_mini.jsonl > dataset/pretrain_tiny.jsonl
```
> 5 万条样本，batch=128 时约 390 步/epoch，单次实验约 3~5 分钟。

之后所有实验都用 `--data_path ../dataset/pretrain_tiny.jsonl`。

**基准配置（不变的部分）**：`epochs=1, batch_size=128, hidden_size=768, num_hidden_layers=8`

---

## 实验指令清单

> 跑前 `cd ~/Project/minimind/trainer` 并确认 `(.venv)`。加 `--use_wandb` 可在网页看曲线；不加则看终端 loss。

### 组 A：学习率 learning_rate（最重要，先做这组）
```bash
python train_pretrain.py --data_path ../dataset/pretrain_tiny.jsonl --epochs 1 --learning_rate 1e-5   # A1 极小
python train_pretrain.py --data_path ../dataset/pretrain_tiny.jsonl --epochs 1 --learning_rate 5e-4   # A2 默认
python train_pretrain.py --data_path ../dataset/pretrain_tiny.jsonl --epochs 1 --learning_rate 5e-3   # A3 偏大
python train_pretrain.py --data_path ../dataset/pretrain_tiny.jsonl --epochs 1 --learning_rate 5e-2   # A4 极大(可能发散)
```

### 组 B：批大小 batch_size（固定 lr=5e-4）
```bash
python train_pretrain.py --data_path ../dataset/pretrain_tiny.jsonl --epochs 1 --batch_size 16    # B1
python train_pretrain.py --data_path ../dataset/pretrain_tiny.jsonl --epochs 1 --batch_size 128   # B2
python train_pretrain.py --data_path ../dataset/pretrain_tiny.jsonl --epochs 1 --batch_size 512   # B3
```

### 组 C：模型大小（固定 lr=5e-4, batch=128）
```bash
python train_pretrain.py --data_path ../dataset/pretrain_tiny.jsonl --epochs 1 --hidden_size 256 --num_hidden_layers 4   # C1 小模型
python train_pretrain.py --data_path ../dataset/pretrain_tiny.jsonl --epochs 1 --hidden_size 768 --num_hidden_layers 8   # C2 默认
python train_pretrain.py --data_path ../dataset/pretrain_tiny.jsonl --epochs 1 --hidden_size 1024 --num_hidden_layers 12 # C3 大模型
```

> 💡 看每个实验启动时打印的 `Model Params: xxM` 记下参数量；训练中用另一个终端 `nvidia-smi` 看显存。

---

## 📝 观察表（你来填）

> 填法：**实验前**先填"我的预测"那列（猜一猜）；**实验后**填实际观察，最后写一句"结论"。

| 编号 | 改变的变量 | 我的预测(实验前猜) | 起始loss(step100) | 中段loss(约step300) | 末尾loss | 现象(平稳/震荡/发散/NaN) | 参数量&显存 | 结论(我的理解) |
|------|-----------|------------------|-----------------|-------------------|---------|----------------------|-----------|--------------|
| A1 | lr=1e-5 |  |  |  |  |  | — |  |
| A2 | lr=5e-4 |  |  |  |  |  | — |  |
| A3 | lr=5e-3 |  |  |  |  |  | — |  |
| A4 | lr=5e-2 |  |  |  |  |  | — |  |
| B1 | batch=16 |  |  |  |  |  |  |  |
| B2 | batch=128 |  |  |  |  |  |  |  |
| B3 | batch=512 |  |  |  |  |  |  |  |
| C1-h256 | hidden 256 |  |  |  |  |  | 8.33M / __G |  |
| C(基准) | hidden 768 |  |  |  |  |  | __M / __G | =默认 |
| C2-h512 | hidden 512 |  |  |  |  |  | 30.0M / __G |  |
| C3-h1024 | hidden 1024 |  |  |  |  |  | 112M / __G |  |
| D1-nah4 | heads 4 (kv4) |  |  |  |  |  | __M / __G |  |
| D(基准) | heads 8 (kv4) |  |  |  |  |  | __M / __G | =默认 |
| D2-nah16 | heads 16 (kv4) |  |  |  |  |  | __M / __G |  |
| D3-nah32 | heads 32 (kv4) |  |  |  |  |  | __M / __G |  |
| E1-MQA | kv 1 (Q8) |  |  |  |  |  | __M / __G |  |
| E2-nkv2 | kv 2 (Q8) |  |  |  |  |  | __M / __G |  |
| E(基准) | kv 4 (Q8) |  |  |  |  |  | __M / __G | =默认 GQA |
| E3-MHA | kv 8 (Q8) |  |  |  |  |  | __M / __G |  |
| F1-hd48 | head_dim 48 |  |  |  |  |  | __M / __G |  |
| F2-hd64 | head_dim 64 |  |  |  |  |  | __M / __G |  |
| F(基准) | head_dim 96 |  |  |  |  |  | __M / __G | =默认 |
| F3-hd128 | head_dim 128 |  |  |  |  |  | __M / __G |  |
| F4-hd256 | head_dim 256 |  |  |  |  |  | __M / __G |  |
| G1-ffn1024 | inter 1024 |  |  |  |  |  | __M / __G |  |
| G2-ffn1536 | inter 1536 |  |  |  |  |  | __M / __G |  |
| G(基准) | inter 2432 |  |  |  |  |  | __M / __G | =默认 |
| G3-ffn3072 | inter 3072 |  |  |  |  |  | __M / __G |  |
| G4-ffn4096 | inter 4096 |  |  |  |  |  | __M / __G |  |
| G5-ffn6144 | inter 6144 |  |  |  |  |  | __M / __G |  |
| H(基准) | dropout 0.0 |  |  |  |  |  | — | =默认 |
| H1-drop0.05 | dropout 0.05 |  |  |  |  |  | — |  |
| H2-drop0.1 | dropout 0.1 |  |  |  |  |  | — |  |
| H3-drop0.2 | dropout 0.2 |  |  |  |  |  | — |  |
| H4-drop0.3 | dropout 0.3 |  |  |  |  |  | — |  |
| I1-theta1e4 | rope_theta 1e4 |  |  |  |  |  | — |  |
| I2-theta1e5 | rope_theta 1e5 |  |  |  |  |  | — |  |
| I(基准) | rope_theta 1e6 |  |  |  |  |  | — | =默认 |
| I3-theta1e7 | rope_theta 1e7 |  |  |  |  |  | — |  |
| J(基准) | Dense (no MoE) |  |  |  |  |  | __M / __G | =默认 |
| J1-e4k1 | MoE 4专家 top1 |  |  |  |  |  | __M / __G |  |
| J2-e8k1 | MoE 8专家 top1 |  |  |  |  |  | __M / __G |  |
| J3-e8k2 | MoE 8专家 top2 |  |  |  |  |  | __M / __G |  |
| J4-e16k2 | MoE 16专家 top2 |  |  |  |  |  | __M / __G |  |

---

## 🤔 实验后回答这些问题（加深理解）

**组 A（学习率）：**
1. 哪个 lr 的 loss 降得最快又稳？
2. lr 太小（A1）会怎样？为什么？
3. lr 太大（A4）会怎样？loss 是震荡、变成 NaN、还是上升？
4. 你能总结出"学习率过大 / 过小 / 合适"分别的表现吗？

**组 B（批大小）：**
1. batch 越大，loss 曲线更平滑还是更抖？为什么？（提示：梯度噪声）
2. batch 改变时，单步速度、显存怎么变？
3. 回忆之前学的：batch 变大时，学习率是否也该跟着变？试试 batch=512 配 lr=1e-3 看看？

**组 C（模型大小）：**
1. 模型越大，参数量、显存、每步速度怎么变？
2. 在相同步数下，大模型的 loss 一定更低吗？（注意：大模型可能需要更多数据/步数才训得动）
3. 小模型(C1)的 loss 会不会很快就降不动了（容量不够）？

**综合：**
- 如果只给你 5 分钟训练时间，你会怎么配这些参数？
- 如果追求最终效果最好（时间不限），又会怎么配？

---

## 实验小贴士

- **不用每个都跑满**：loss 曲线的"形状"在前几百步就能看出差别，看够了可 `Ctrl+C` 停。
- **发散的样子**：loss 不降反升、或变成 `nan`/`inf`，就是 lr 太大了。
- **wandb 对比**：加 `--use_wandb`，多条曲线会自动叠加（运行名含 epochs/batch/lr，改这三个能区分；改 hidden_size 时运行名不区分，靠本表记录）。
- 做完记得把有价值的结论回填到 `04_qa_notes.md`。
