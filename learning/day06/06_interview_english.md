# Day 6 英文面试表达：Attention 内部实现细节

> 接 day05 的 06_interview_english.md。这篇聚焦**手撕 attention 时容易被追问的实现细节**：softmax 在哪一维、dropout 到底干嘛、多头怎么合并。
> 结构同前：**一句话核心 → 展开 → 加分项 → 术语**。

---

## 1. Why is softmax applied over the key dimension, not the query dimension?

> "The attention scores have shape batch, heads, **query length**, **key length**. Both of the last two are sequence dimensions, but they play different roles: the **row** is the query — the token that's asking — and the **column** is the key — every token being looked at. We apply softmax over the **key dimension**, so that **each query independently** turns its scores into a distribution over all the keys it can see, summing to one. That answers 'for this token, how should I split my attention across the others.'"

被追问"为什么不沿 query 维":
> "Normalizing over the query dimension would make different queries compete for the same key, which is meaningless — what one token attends to is independent of what another token attends to. Each query owns its own attention budget."

术语：query length vs key length, normalize over the key dimension, attention distribution, independent per query.

---

## 2. What is attention dropout — does it drop the small / unimportant weights?

> "No — that's a common misconception. Dropout doesn't look at magnitude. During **training**, it **randomly** zeroes out a fraction of the attention weights — say ten percent — regardless of whether they're large or small, and scales the rest up by one over one-minus-p to keep the expectation. Its purpose is **regularization**: it stops the model from over-relying on a few fixed attention connections. At **inference** time, with `model.eval()`, dropout is **turned off** entirely — nothing is dropped."

被追问"那小权重怎么处理":
> "Small weights aren't explicitly removed — they just naturally contribute little after softmax because their value is near zero. That's a property of softmax, not of dropout."

术语：randomly zeroes out, regularization, scale by 1/(1-p), train vs eval, not magnitude-based.

---

## 3. After multiplying attention weights by V, why the transpose and reshape? / How are heads combined?

> "Two steps. First, **attention-weights times V** gives, for each query and each head, a weighted average of the value vectors — shape batch, heads, query length, head-dim. Then I **merge the heads** back into a single per-token vector: I transpose to bring the sequence dimension in front of the heads, then reshape to concatenate head-dim across all heads, giving batch, query length, hidden-size."

被追问"为什么必须先 transpose 再 reshape":
> "Reshape just merges adjacent dimensions in memory order. I want to concatenate all the **heads of the same token**, so I first transpose to put the token dimension before the heads — that makes one token's heads contiguous — and only then reshape. After that, the output projection maps the concatenated vector back to the model dimension."

术语：weighted average of values, merge heads, transpose then reshape, contiguous in memory, output projection.

---

## 4. What does the output projection (o_proj) do — isn't it just reshaping?

> "No — reshape only **concatenates** the heads side by side; at that point the heads haven't interacted, since each one was computed independently. The output projection is a **learned linear layer** that **mixes information across heads** into a unified representation, and projects it back to the model dimension so it can re-enter the residual stream. Even when the input and output dimensions are equal, it's not the identity — it's a learned mixing matrix."

术语：concatenate heads, learned mixing, residual stream, not identity.

---

## 5. What is a residual connection and why is it essential?

> "A residual connection is `x = x + f(x)` — the sublayer's input is added straight back to its output, so the sublayer only learns a **small correction** instead of rebuilding the whole representation. We need it for two reasons. **First, gradient flow**: the `+x` path has a derivative of one, so it acts as a highway that carries gradients back without vanishing — that's what lets us stack dozens of layers. **Second, easier optimization**: each layer starts close to the identity, so adding layers never makes things worse, and the model refines gradually."

被追问"和 pre-norm 怎么配合":
> "With pre-norm, normalization is applied only to the **copy** fed into the sublayer; the residual adds back the **un-normalized** input, so the residual path stays a clean, unnormalized highway all the way through the network."

术语：identity mapping, gradient highway, vanishing gradient, small correction, clean residual path.

---

## 一句话串联（手撕 attention 收尾）

> "So the output side is: weight the values, concatenate the heads back into one vector per token, mix them with the output projection, and add the result back through a residual connection."
