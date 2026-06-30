# Day 5 英文面试表达：LLM 核心概念怎么讲

> **Part A**：attention 一条链（embedding→RoPE→KV cache→mask→softmax）。
> **Part B**：attention 之外（normalization、FFN/SwiGLU、weight decay、LoRA、DPO、inference、scaling）。
> 每步结构：**一句话点核心动词 → 展开 → 加分项 → 术语**。
> 面试技巧：先抛核心动词，随口报 shape，别背公式（讲 intuition + 类比）。

---

## 0. 开场总览：整个 modern LLM 架构怎么讲

> 被问 "describe a transformer / LLM architecture" 时用。主干：input→embedding→blocks→输出。
> 先用**精简版**铺框架；面试官想听更多就接**完整版**。
> ⚠️ 注意：这是**整个模型**的总览（从 token id 到 vocab）；下面 Part A 是**单个 attention 层内部**的细节，别混。

### 0a. 精简版（~60 秒，默认用这个）

> "Most modern LLMs share a common, transformer-based architecture. The input is a sequence of **token IDs**. An **embedding layer** first maps each token ID into a high-dimensional vector, so each token's meaning lives in a continuous space.
>
> The vectors then pass through a stack of **identical blocks**, which we call layers. Each block has two sublayers with **pre-norm** and **residual connections**: an **attention sublayer**, where tokens **exchange information across positions**, and a **feed-forward sublayer**, which processes **each token independently** and adds the non-linearity and most of the model's capacity.
>
> After the last block, a final norm and an **output projection** map the hidden vector back to **vocabulary size**, and **softmax** turns it into a probability distribution over the next token."

### 0b. 完整版（~69 秒，想多讲一点 / 被追问展开时）

> "Most modern LLMs share a common, transformer-based architecture. The input is a sequence of **token IDs**. First, an **embedding layer** maps each token ID into a high-dimensional vector, so the meaning of each token lives in a continuous space.
>
> Then the data goes through a stack of **identical blocks** — we call them layers. Each block has two sublayers: an **attention sublayer** and a **feed-forward sublayer**. We use **pre-norm**, so each sublayer starts with normalization, usually **RMSNorm**. Inside attention, we project the input into **query, key, and value**, apply **rotary position embedding to the query and key**, and let each token attend to the others — this is how tokens **exchange information across positions**.
>
> The **feed-forward network** then does the opposite: it processes **each token independently**. It projects the vector up to a wider dimension, applies a non-linear activation, and projects it back down — in modern models this is a **gated SwiGLU**. This is where most of the model's parameters and non-linearity live; without it, stacking attention would just collapse into one linear map. Both sublayers are wrapped in **residual connections**, which keep gradients flowing in deep networks.
>
> After the final block, we apply one more norm, then an **output projection** that maps the hidden vector back to **vocabulary size**, and finally **softmax** turns those scores into a probability distribution over the next token."

> ⚠️ 易错点：映射回 vocab size 的是 **output projection（线性层 / lm_head）**，softmax 只负责把 logits 归一化成概率——别说成 "softmax 映射到词表"。

### 0.1 架构总览深挖（讲完 0a/0b 后逐术语被追问）

**Q: Why pre-norm instead of post-norm?**
> "Pre-norm puts normalization **inside** each sublayer, before attention or FFN, so the residual path stays clean — the input flows straight through with a constant gradient. That makes deep models much more stable to train; post-norm tends to need careful warmup and can have vanishing gradients when you stack many layers."

**Q: What do query, key, and value stand for?**
> "They're three linear projections of the same input. The **query** is what the current token is looking for, the **key** is what each token offers as an index, and the **value** is the actual content. We match a query against all keys to get attention scores, then use those scores to take a weighted sum of the values."

**Q: What is rotary position embedding for?**
> "Attention itself is order-agnostic — it doesn't know token positions. RoPE injects position by **rotating** the query and key vectors by an angle proportional to their position. Because it's a rotation, the dot product between a query and a key ends up depending only on their **relative** distance, which also helps the model generalize to longer sequences."

**Q: How does the feed-forward network work, and why gated SwiGLU?**
> "The FFN processes each token on its own: it projects the vector **up** to a wider dimension, applies a non-linearity, and projects it back **down**. Modern models use **SwiGLU** — a gated version, `down(silu(gate(x)) * up(x))`. The gate is a learned, input-dependent filter, and SiLU is smoother than ReLU; empirically it gives lower loss at the same parameter count."

**Q: What is a residual connection and why do we need it?**
> "A residual connection adds the sublayer's input back to its output — `x + f(x)`. It gives gradients a direct highway back through the network, so they don't vanish in deep stacks, and it lets each layer learn just a small **correction** rather than rebuild the whole representation."

**Q: What is the output projection and why do we need it?**
> "The model works in hidden dimension, but we need a score for every token in the vocabulary. The output projection is a linear layer mapping the final hidden vector to a vector of size **vocabulary**, giving one logit per possible next token. Softmax then turns those logits into probabilities. Many models **tie** this matrix with the embedding to save parameters."

术语补充：pre-norm vs post-norm, query/key/value, relative position, gated activation, gradient highway, weight tying.

---

## 1. Embedding

> "The embedding layer is essentially a **lookup table** of shape `[vocab_size, hidden_size]`. Each token ID indexes into one row to get its dense vector. So the first argument is the **number of rows — the vocabulary size**, and the second is the **output dimension**."

术语：lookup table, token ID, indexes into, dense vector, vocabulary size.

---

## 2. RoPE (Rotary Position Embedding)

> "RoPE encodes position by **rotating** the query and key vectors. We split each vector into pairs and rotate each pair by an angle that depends on **the token's position and the pair's frequency**. Lower dimensions rotate fast (high frequency), higher dimensions rotate slowly — similar to the second hand versus the hour hand on a clock. Since the rotation only depends on position, not on the token content, we **precompute** these angles once and reuse them."

加分项：
> "A nice property is that the dot product after rotation naturally encodes **relative** position, which helps length generalization."

术语：rotate, frequency, precompute, relative position.

### 2.1 被追问细节：frequency / angle / 到底预计算了啥

**核心类比 — position = time：**
> "The core trick is that we treat the token's **position** as if it were **time**. In physics the angle you've rotated equals angular velocity times time; here the rotation angle equals the **frequency** times the **position**."

**frequency 从哪来：**
> "Each pair of dimensions gets its own **angular frequency**. We design them to decay geometrically — the first pair rotates at frequency one, and each later pair rotates exponentially slower, down to one over ten-thousand. Fast-rotating dimensions tell **nearby** tokens apart; slow-rotating dimensions encode **long-range** distance — like the second hand versus the hour hand on a clock."

**angle 怎么算：**
> "For every position and every pair, the angle is just **frequency times position**. We compute this with an **outer product**, giving a table of angles for all positions at once."

**预计算的是哪个：**
> "We convert each angle into a **unit complex number** — cosine plus i sine — and cache that. The key insight: these angles depend **only on position and dimension, never on the token content**, so the table is identical for any input — compute once, reuse forever."

**怎么用上（追问 how it's applied）：**
> "At runtime we view each query/key vector as complex numbers and **multiply** by these precomputed factors. Multiplying by a unit complex number is exactly a rotation, and the dot product then depends only on the **relative** distance."

**高频追问 — 为什么比 absolute/learned 好：**
> "Two reasons. First, it's **relative** — the attention score naturally depends on the distance between tokens, not their absolute positions. Second, it **extrapolates** better to lengths longer than training, because it's a fixed function, not a learned lookup table."

术语：angular frequency, decay geometrically, outer product, unit complex number, extrapolate.

---

## 3. KV Cache

> "During autoregressive decoding, we generate one token at a time. Without caching, we'd recompute the keys and values for the entire prefix every step, which is wasteful. The **KV cache** stores the past keys and values — each of shape `[batch, num_kv_heads, past_seq_len, head_dim]` — and we just **concatenate** the new token's K/V along the sequence dimension. So each decode step only does a forward pass for **one** new token."

加分项 (GQA)：
> "Note the K/V have `num_kv_heads`, not `num_heads`, because of **grouped-query attention** — fewer K/V heads are shared across query heads to save memory."

术语：autoregressive, decoding, prefix, concatenate along the sequence dimension, prefill vs decode.

---

## 4. Causal Mask

> "Because it's an **autoregressive** language model, each position may only look at itself and **earlier** tokens, never future ones. During training we process the whole sequence in **parallel**, so we apply a **causal mask**: before softmax, we set the scores at all future positions to **minus infinity**. After softmax they become zero weight, so a token can't 'cheat' by attending to the words it's supposed to predict. At inference this is automatic — with a KV cache we only have past tokens anyway."

被追问"怎么实现 / 那个 diagonal offset"：
> "It's just an **upper-triangular** mask — block everything to the right of the diagonal. With a KV cache the query's real position is offset by the cached length, so the boundary shifts by `past_seq_len`; that's where the `total_S - S + 1` diagonal offset comes from. Conceptually it's still 'no looking right of the diagonal.'"

术语：autoregressive, causal mask, attend to future tokens, parallel training, upper-triangular.

> 讲法要点：**先 why（autoregressive，不能偷看未来=训练作弊），再 how（future→-inf→softmax 归零）**；`total_S - S + 1` 降级成追问，别让实现细节淹没主答案。

---

## 5. Scaled Dot-Product Attention（Softmax + Weighted Sum）

> "For each query, we compute a **dot product with every key** to get a similarity score, and **scale it by one over the square root of the head dimension**. That scaling matters — without it, large head dimensions make the dot products huge, pushing softmax into a region where gradients vanish. Then **softmax** normalizes each query's scores, over the **key dimension**, into weights that are non-negative and sum to one — masked positions are minus infinity, so they get zero weight. Finally, the output is a **weighted sum of the value vectors**: each query reads most from the tokens it attends to most."

被追问"为什么除以 √d"：
> "The dot product of two d-dimensional vectors grows with d. If the scores get too large, softmax becomes nearly one-hot and its gradient gets tiny, so training stalls. Dividing by √d keeps the score variance stable regardless of head dimension."

术语：scaled dot-product, similarity score, square root of head dimension, softmax saturation, weighted sum over values.

> 讲法要点：补上 **score 计算 + `1/√d` 缩放**（高频考点）；缩放的真正原因是**防 softmax 饱和→梯度消失**，不是泛泛"防数值太大"。

---

## 万能收尾句（展示理解深度）

> "So the whole attention operation is really: *for each query, decide how much to read from every other token, then read it.*"

---

## 面试小技巧

1. **每段先抛核心动词**：embedding=lookup, RoPE=rotate, KV cache=concatenate & reuse, mask=block future, softmax=normalize。
2. **形状随口报**：`[B, num_heads, S, total_S]` 这种 shape 显得真写过代码。
3. **被追问推导**：用 "equivalent to ... for integers" 这种干脆说法。
4. **别背公式**：讲 intuition + shape + 一个类比（钟表、查表）。

---

# Part B：Attention 之外的概念（面试口语版）

> 结构同上：**一句话核心 → 展开 → 加分项 → 术语**。涵盖 normalization、FFN/SwiGLU、weight decay、LoRA、DPO、inference、scaling 等。

## 6. Normalization (RMSNorm vs LayerNorm)

> "RMSNorm is a simplified LayerNorm. LayerNorm subtracts the **mean**, divides by the standard deviation, then applies a scale **and a bias**. RMSNorm drops the mean-centering and the bias entirely — it just divides by the **root-mean-square** and applies a scale.
>
> There are two benefits. **First, it's computationally cheaper**: fewer operations — no mean, no subtraction, no bias — so fewer FLOPs and fewer memory reads and writes. **Second, the quality is basically the same**, because the useful part of normalization is the **rescaling**, not the centering.
>
> And that speedup matters more than it looks: normalization is a tiny fraction of total FLOPs, but because it's **memory-bound**, it takes a **disproportionate share of wall-clock time** — so making it simpler gives a real win."

被追问"具体省了什么计算"：
> "LayerNorm needs **two reductions** over the vector — one for the mean, one for the variance. RMSNorm needs only **one** — the mean of squares. It also skips the per-element mean subtraction and the bias add. Fewer passes over the data and fewer ops."

术语：root-mean-square, mean-centering, bias term, fewer reductions, memory-bound, disproportionate runtime.

> 讲法要点：**两条独立原因**——① 计算更便宜（fewer ops/reductions，第一性优势）② 质量不掉（rescaling 才是关键，centering 可省）。memory-bound 是"解释为什么这点小省也值得"的**放大器**，不是"快"的唯一来源——别混。

## 7. Memory bandwidth & the "memory wall"

> "What limits data movement between HBM and SRAM is **memory bandwidth**, roughly **bus width times clock frequency**. Both are capped by physics — pin count, power, signal integrity. Compute grows much faster, so the gap keeps widening — the **memory wall**."

加分项：
> "Whether an op is memory-bound depends on its **arithmetic intensity** — FLOPs per byte. Low-intensity ops like normalization and elementwise add are memory-bound, which is why FlashAttention and kernel fusion focus on **moving less data**, not computing less."

术语：memory bandwidth, bus width, memory wall, arithmetic intensity, kernel fusion.

## 8. Why FFN, and why SwiGLU

> "A transformer block has two roles: **attention mixes information across tokens**, and the **feed-forward network processes each token independently**. The FFN holds most of the non-linearity and most of the parameters — without it, stacking attention would collapse into a single linear map."

SwiGLU：
> "Modern LLMs replace the ReLU FFN with **SwiGLU**, a gated variant: `down(silu(gate(x)) * up(x))`. The gate is a learned, input-dependent filter, and SiLU is smoother than ReLU. Empirically lower loss at the same parameter count, so PaLM, LLaMA, Qwen adopted it."

术语：mix across tokens, per-token processing, non-linearity, gated activation, SwiGLU.

## 9. Architecture ratios (consensus hyperparameters)

> "A few ratios are remarkably stable. The **FFN ratio** `d_ff/d_model` is about **4** for ReLU FFNs, or **8/3** for SwiGLU to keep parameter count equal. **Head dimension** is almost always **64 or 128**, so you scale by adding heads, not widening them. And `num_heads × head_dim` usually equals `d_model`."

加分项：
> "These aren't sharp optima — there's a wide flat **basin**, say ratio 1 to 10 for the FFN, where loss barely changes. So pick the consensus value and spend your tuning budget on learning rate and data."

术语：FFN ratio, head dimension, flat basin, consensus value.

## 10. Weight decay (the LLM-specific twist)

> "Classically weight decay is L2 regularization against overfitting. But in **LLM pretraining** it's **not about overfitting** — pretraining is essentially single-epoch, so there's little to overfit. Validation loss just tracks training loss regardless of the decay strength."

真实作用：
> "Its real role is that it **interacts with the learning-rate schedule** — combined with cosine decay it lowers the final training loss. So in LLMs it's an optimization knob, not a regularizer."

术语：L2 regularization, single-epoch, interacts with the LR schedule, optimization knob.

## 11. LoRA

> "LoRA freezes the pretrained weights and learns a **low-rank update** `ΔW = B·A` alongside each linear layer, with a small rank like 8. You only train and store those tiny matrices, so the optimizer state shrinks dramatically — that's the memory win."

加分项（B=0 init）：
> "B is initialized to **zero** so `ΔW` starts at zero and the model initially behaves exactly like the pretrained one — a harmless start that won't destroy what pretraining learned."

术语：low-rank update, rank, freeze the base, optimizer state, zero-init.

## 12. DPO (preference alignment)

> "DPO aligns a model using pairs of a **chosen** and a **rejected** response to the same prompt. It trains the policy so that, **relative to a frozen reference model**, it raises the likelihood of the chosen response and lowers the rejected one. The reference acts as an anchor so the model doesn't drift from its SFT start."

加分项（vs RLHF）：
> "Unlike PPO-based RLHF, DPO needs **no separate reward model and no sampling loop** — it's a simple classification-style loss directly on preference pairs, so it's more stable and cheaper."

术语：chosen/rejected pair, policy vs reference model, anchor, preference alignment.

### 12.2 How the DPO loss is actually computed（计算链：四个 log-prob → 一个 loss）

> 被追问 "walk me through how the DPO loss is computed" 时用。核心：**四个 sequence-level log-prob → 两个 implicit reward → 一个 logistic loss。**

> "DPO works on **preference pairs** — for the same prompt, a **chosen** and a **rejected** response. The loss is built from **four sequence-level log-probabilities**.
>
> First, for each response I compute its **log-probability under the policy model**: take `log_softmax` over the logits, **gather** the log-prob of each actual answer token, and **sum** over the answer tokens. I do the same under the **frozen reference model**. That gives four numbers — policy and reference log-probs, each for chosen and rejected.
>
> Then for each response I form an **implicit reward** — the policy log-prob **minus** the reference log-prob — which measures how far the policy has shifted from the reference. The loss is `-log σ(β · (chosen_reward − rejected_reward))`, where **β** controls how strongly we trust the preference signal.
>
> Intuitively, minimizing it **raises the chosen response's probability and lowers the rejected one — always relative to the reference**, so the policy can't drift too far from its SFT start."

计算链（脑里的骨架）：
```python
# 4 个 sequence-level log-prob：logp = log_softmax(logits).gather(真实token).sum(回答部分)
logp_policy_chosen, logp_policy_rejected      # policy（在训）
logp_ref_chosen,    logp_ref_rejected         # reference（冻结, no_grad）
# 2 个 implicit reward（相对 reference）
chosen_reward   = logp_policy_chosen   - logp_ref_chosen
rejected_reward = logp_policy_rejected - logp_ref_rejected
# loss
loss = -log(sigmoid(β * (chosen_reward - rejected_reward))).mean()
```

加分追问：
> **为什么用差值不用绝对值**："The reward is *relative to the reference* — `logp_policy − logp_ref`. The difference makes the reference an anchor / implicit KL penalty, so the policy improves preference without collapsing its general ability."
>
> **σ 和 log 哪来的**："It's the **Bradley-Terry** preference model — the probability that chosen beats rejected is `σ(reward difference)`, and we minimize its negative log-likelihood. So it's really just a **binary classification** loss on which response is preferred."

术语：sequence-level log-probability, log_softmax + gather + sum, implicit reward = logp_policy − logp_ref, β / KL strength, Bradley-Terry, binary classification loss.

## 13. Inference: prefill vs decode

> "Generation has two phases. **Prefill** processes the whole prompt in one parallel forward pass and fills the KV cache — it's **compute-bound**. **Decode** then generates one token at a time reusing the cache, so each step is cheap in compute but **memory-bound**, reloading weights and cache from HBM for a single token."

加分项：
> "That's why the first token has higher latency — **time-to-first-token** is dominated by prefill — while later tokens stream out quickly."

术语：prefill, decode, compute-bound, memory-bound, time-to-first-token.

## 14. Sampling strategies

> "At each decode step the model outputs logits over the vocabulary, and sampling decides the next token. **Temperature** scales the logits before softmax to control randomness; **top-p** (nucleus) restricts sampling to the smallest set covering 95% of probability, cutting the long tail; and **repetition penalty** lowers logits of already-generated tokens to avoid loops."

术语：logits, temperature, top-p / nucleus sampling, repetition penalty, greedy vs sampling.

## 15. Scaling laws (Chinchilla)

> "Chinchilla found that for compute-optimal training you want roughly **20 tokens per parameter**. Given a compute budget, you solve two constraints — compute ≈ 6·N·D and D ≈ 20·N — to get the optimal model and dataset size."

加分项：
> "In practice people often train smaller models on far more than 20× data, because it makes **inference** cheaper — that's inference-aware scaling, a different objective from pure compute-optimality."

术语：compute-optimal, tokens per parameter, compute budget, inference-aware scaling.

## 16. Cross-entropy / how the loss is computed

> "Cross-entropy measures **how surprised the model is by the correct token**. At each position the model outputs **logits** over the vocabulary; we apply **softmax** to turn them into a probability distribution, then the loss is the **negative log of the probability it assigned to the true token**. If it's confident and right, the loss is near zero; if it puts low probability on the truth, the loss blows up. We average this over all positions in the batch.
>
> For a language model there's one key detail: it's **next-token prediction**, so we **shift by one** — the logits at position *i* are scored against the **true token at position i+1**. In code that's `logits[:, :-1]` against `labels[:, 1:]`. We also pass an **ignore_index**, usually -100, so that padding and the prompt tokens don't contribute to the loss — only the positions we actually want the model to learn."

加分项（被追问 loss 和 gradient 的关系时）：
> "A neat property is that the gradient of cross-entropy through softmax is just **predicted probability minus the one-hot of the true label** — so for the correct token the gradient is `p − 1`, pushing its logit up, and for every wrong token it's `p`, pushing them down. The bigger the mistake, the bigger the push. That clean form is one reason softmax + cross-entropy is the default."

万能收尾：
> "So the loss is just *how wrong*, and its gradient is literally *predicted minus true* — that's what backprop pushes down."

术语：logits, softmax, negative log-likelihood, next-token prediction, shift by one, ignore_index, gradient = probs − one-hot.

> 讲法要点：**先 intuition（how surprised），别从公式起手**；主动报 **shift-by-one + ignore_index**（区分"看过博客"和"真跑过训练"）；梯度那句 `probs − one-hot` 降级成追问加分项，别一开始就讲（信息过载）。

---

# Part C：模拟追问 Q&A（层层深挖，练"被压力测试"）

> 用法：自己先读 Q，蒙住 A 试着答，再对照。面试官最爱顺着一个点一直挖到底。

## 13.1 Inference 深挖

**Q: Walk me through what happens when a user sends a prompt.**
> A: "First we build the prompt with a chat template and tokenize it. Then **prefill**: one parallel forward pass over the whole prompt that fills the KV cache and produces logits for the first new token. Then a **decode loop**: each step feeds the single previous token, reuses the cache, samples the next token, until we hit an end-of-sequence token or a max length."

**Q: Why is prefill compute-bound but decode memory-bound?**
> A: "Prefill processes many tokens at once, so it's a big matrix multiply — high arithmetic intensity, compute-bound. Decode processes **one** token per step, but still has to load all the weights and the whole KV cache from HBM — lots of bytes moved for very little compute, so it's memory-bound."

**Q: So how would you speed up decode?**
> A: "Since it's memory-bound, reduce data movement: quantize the weights and the KV cache, use **batching** to amortize weight loads across requests, or use **speculative decoding** to generate several tokens per verification step. Frameworks like vLLM also use paged KV cache to pack memory efficiently."

**Q: What's the difference between temperature and top-p?**
> A: "Temperature **reshapes** the whole distribution — lower is sharper and more deterministic. Top-p **truncates** it — it keeps only the smallest set of tokens covering p of the probability mass, then renormalizes. They're complementary: temperature controls how peaked, top-p controls how much tail you allow."

**Q: Greedy decoding is deterministic and picks the most likely token — why not always use it?**
> A: "Greedy is locally optimal but globally repetitive and bland, and it can get stuck in loops. Sampling adds diversity, which matters for open-ended generation. For factual or code tasks you do lean toward low temperature or greedy."

## 12.1 DPO 深挖

**Q: Why do you need a frozen reference model in DPO?**
> A: "It's the anchor. The loss rewards the policy for preferring chosen over rejected **relative to the reference**, not in absolute terms. Without it, the model could blow up the probability of the chosen response and drift far from its SFT behavior, degrading general quality. The reference keeps it close — it's like an implicit KL penalty."

**Q: How is DPO different from RLHF with PPO?**
> A: "RLHF trains a separate reward model, then uses PPO to optimize the policy against it with an online sampling loop — complex and unstable. DPO skips both: it derives a loss that turns preference pairs directly into a classification-style objective on the policy and reference log-probabilities. No reward model, no sampling loop."

**Q: What data does DPO need, and where does it come from?**
> A: "Pairs of chosen and rejected responses to the **same** prompt. They come from human preference annotations, or from ranking multiple model samples — for example with an LLM-as-a-judge — and labeling the better one chosen, the worse one rejected."

**Q: What's the beta hyperparameter?**
> A: "Beta controls how strongly the loss trusts the preference data versus staying near the reference. Larger beta pushes harder on the preference signal; smaller beta keeps the policy closer to the reference. It plays the role of the KL strength."

**Q: Could DPO make the model worse?**
> A: "Yes — if the preference data is noisy or biased, or beta is too large, the model can overfit the preferences and lose fluency or general capability. That's why you keep the reference anchor and often monitor with held-out evals."

术语补充：speculative decoding, paged KV cache, implicit KL penalty, LLM-as-a-judge, beta / KL strength.
