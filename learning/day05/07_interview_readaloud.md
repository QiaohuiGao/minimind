# Interview Read-Aloud Script (pure English)

> Read these out loud 2–3 times. No Chinese — practice fluency. Source: 06_interview_english.md.

---

## Overview

In a transformer attention layer, the data flows through a few key stages: token embedding, applying rotary position encoding to Q and K, optionally appending cached K/V, computing scaled dot-product scores, applying a causal mask, normalizing with softmax, and finally a weighted sum over V.

## 1. Embedding

The embedding layer is essentially a lookup table of shape vocab-size by hidden-size. Each token ID indexes into one row to get its dense vector. So the first argument is the number of rows — the vocabulary size — and the second is the output dimension.

## 2. RoPE

RoPE encodes position by rotating the query and key vectors. We split each vector into pairs and rotate each pair by an angle that depends on the token's position and the pair's frequency. Lower dimensions rotate fast, higher dimensions rotate slowly — like the second hand versus the hour hand on a clock. Since the rotation only depends on position, not on the token content, we precompute these angles once and reuse them. A nice property is that the dot product after rotation naturally encodes relative position, which helps length generalization.

## 3. KV Cache

During autoregressive decoding, we generate one token at a time. Without caching, we'd recompute the keys and values for the entire prefix every step, which is wasteful. The KV cache stores the past keys and values, and we concatenate the new token's K/V along the sequence dimension. So each decode step only does a forward pass for one new token. Note the K/V use num-kv-heads, not num-heads, because of grouped-query attention — fewer K/V heads are shared across query heads to save memory.

## 4. Causal Mask

The causal mask prevents each query from attending to future tokens. We add minus infinity to the score wherever the key position is later than the query's position. With a KV cache, the query's real position is offset by the cached length, so the masking boundary shifts accordingly. The boundary is always a forty-five-degree line, so a single diagonal offset captures it for both prefill and decode.

## 5. Softmax and Weighted Sum

Softmax turns the raw scores into attention weights that are non-negative and sum to one. We apply it over the last dimension, the key dimension, so each query distributes its attention across all keys. The masked positions become minus infinity, and since e-to-the-minus-infinity is zero, they get zero weight. Finally we take a weighted sum over the value vectors to produce the output.

So the whole attention operation is really: for each query, decide how much to read from every other token, then read it.

## 6. RMSNorm vs LayerNorm

RMSNorm is a simplified LayerNorm. LayerNorm computes both the mean and variance and re-centers the data; RMSNorm skips the mean entirely and only divides by the root-mean-square. The insight is that the benefit comes mostly from the rescaling, not the centering, so dropping the mean and the bias term makes it faster with almost no quality loss. Normalization layers are memory-bound, not compute-bound — the bottleneck is moving the activation tensor between HBM and on-chip SRAM, not the arithmetic.

## 7. Memory Bandwidth and the Memory Wall

What limits data movement between HBM and SRAM is memory bandwidth, roughly bus width times clock frequency. Both are capped by physics — pin count, power, and signal integrity. Compute grows much faster, so the gap keeps widening — that's the memory wall. Whether an op is memory-bound depends on its arithmetic intensity, that is, FLOPs per byte. Low-intensity ops like normalization are memory-bound, which is why FlashAttention and kernel fusion focus on moving less data, not computing less.

## 8. Why FFN, and why SwiGLU

A transformer block has two roles: attention mixes information across tokens, and the feed-forward network processes each token independently. The FFN holds most of the non-linearity and most of the parameters — without it, stacking attention would collapse into a single linear map. Modern LLMs replace the ReLU FFN with SwiGLU, a gated variant. The gate is a learned, input-dependent filter, and SiLU is smoother than ReLU. Empirically it gives lower loss at the same parameter count, so PaLM, LLaMA, and Qwen all adopted it.

## 9. Architecture Ratios

A few ratios are remarkably stable. The FFN ratio, d-ff over d-model, is about four for ReLU FFNs, or about eight-thirds for SwiGLU to keep the parameter count equal. The head dimension is almost always sixty-four or one-twenty-eight, so you scale by adding heads, not widening them. These aren't sharp optima — there's a wide flat basin where loss barely changes. So pick the consensus value and spend your tuning budget on learning rate and data.

## 10. Weight Decay

Classically, weight decay is L2 regularization against overfitting. But in LLM pretraining it's not about overfitting — pretraining is essentially single-epoch, so there's little to overfit. Its real role is that it interacts with the learning-rate schedule — combined with cosine decay, it lowers the final training loss. So in LLMs it's better thought of as an optimization knob than a regularizer.

## 11. LoRA

LoRA freezes the pretrained weights and learns a low-rank update, delta-W equals B times A, alongside each linear layer, with a small rank like eight. You only train and store those tiny matrices, so the optimizer state shrinks dramatically — that's the memory win. B is initialized to zero so that delta-W starts at zero and the model initially behaves exactly like the pretrained one — a harmless start that won't destroy what pretraining learned.

## 12. DPO

DPO aligns a model using pairs of a chosen and a rejected response to the same prompt. It trains the policy so that, relative to a frozen reference model, it raises the likelihood of the chosen response and lowers the rejected one. The reference acts as an anchor so the model doesn't drift from its SFT starting point. Unlike PPO-based RLHF, DPO needs no separate reward model and no sampling loop — it's a simple classification-style loss directly on preference pairs, so it's more stable and cheaper.

## 13. Prefill vs Decode

Generation has two phases. Prefill processes the whole prompt in one parallel forward pass and fills the KV cache — it's compute-bound. Decode then generates one token at a time, reusing the cache, so each step is cheap in compute but memory-bound, because we reload the weights and cache from HBM for a single token. That's why the first token has higher latency — time-to-first-token is dominated by prefill — while later tokens stream out quickly.

## 14. Sampling

At each decode step the model outputs logits over the vocabulary, and the sampling strategy decides the next token. Temperature scales the logits before softmax to control randomness; top-p, or nucleus sampling, restricts sampling to the smallest set of tokens covering ninety-five percent of probability, cutting the long tail; and repetition penalty lowers the logits of already-generated tokens to avoid loops.

## 15. Scaling Laws

Chinchilla found that for compute-optimal training, you want roughly twenty tokens per parameter. Given a compute budget, you solve two constraints — compute is about six times N times D, and D is about twenty times N — to get the optimal model size and dataset size. In practice, people often train smaller models on far more data than that, because it makes inference cheaper — that's inference-aware scaling.
