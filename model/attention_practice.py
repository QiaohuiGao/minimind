import transformers as tf
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import time
import matplotlib.pyplot as plt
import seaborn
from torch.autograd import Variable
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
 


# # ==================== Annotated Transformer (学习用) ====================

# class Encoder(nn.Module):
#     def __init__(self, layer, N):
#         super(Encoder, self).__init__()
#         self.layers = clones(layer, N)
#         self.norm = LayerNorm(layer.size)

#     def forward(self, x, mask):
#         for layer in self.layers:
#             x = layer(x, mask)
#         return self.norm(x)


# class SublayerConnection(nn.Module):
#     def __init__(self, size, dropout):
#         super(SublayerConnection, self).__init__()
#         self.norm = LayerNorm(size)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x, sublayer):
#         return x + self.dropout(sublayer(self.norm(x)))


# class EncoderLayer(nn.Module):
#     def __init__(self, size, self_attn, feed_forward, dropout):
#         super(EncoderLayer, self).__init__()
#         self.self_attn = self_attn
#         self.feed_forward = feed_forward
#         self.sublayer = clones(SublayerConnection(size, dropout), 2)
#         self.size = size

#     def forward(self, x, mask):
#         x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
#         return self.sublayer[1](x, self.feed_forward)


# class Decoder(nn.Module):
#     def __init__(self, layer, N):
#         super(Decoder, self).__init__()
#         self.layers = clones(layer, N)
#         self.norm = LayerNorm(layer.size)

#     def forward(self, x, memory, src_mask, tgt_mask):
#         for layer in self.layers:
#             x = layer(x, memory, src_mask, tgt_mask)
#         return self.norm(x)


# class DecoderLayer(nn.Module):
#     def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
#         super(DecoderLayer, self).__init__()
#         self.size = size
#         self.self_attn = self_attn
#         self.src_attn = src_attn
#         self.feed_forward = feed_forward
#         self.sublayer = clones(SublayerConnection(size, dropout), 3)

#     def forward(self, x, memory, src_mask, tgt_mask):
#         m = memory
#         x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
#         x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
#         return self.sublayer[2](x, self.feed_forward)


# def subsequent_mask(size):
#     attn_shape = (1, size, size)
#     subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
#     return torch.from_numpy(subsequent_mask) == 0


# def attention(query, key, value, mask=None, dropout=None):
#     d_k = query.size(-1)
#     scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
#     if mask is not None:
#         scores = scores.masked_fill(mask == 0, -1e9)
#     p_attn = F.softmax(scores, dim=-1)
#     if dropout is not None:
#         p_attn = dropout(p_attn)
#     return torch.matmul(p_attn, value), p_attn


# class MultiHeadsAttention(nn.Module):
#     def __init__(self, heads, d_model, dropout=0.1):
#         super(MultiHeadsAttention, self).__init__()
#         assert d_model % heads == 0
#         self.d_k = d_model // heads
#         self.heads = heads
#         self.linears = clones(nn.Linear(d_model, d_model), 4)
#         self.attn = None
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, query, key, value, mask=None):
#         if mask is not None:
#             mask = mask.unsqueeze(1)
#         nbatches = query.size(0)
#         query, key, value = [l(x).view(nbatches, -1, self.heads, self.d_k).transpose(1, 2)
#                              for l, x in zip(self.linears, (query, key, value))]
#         x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
#         x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.heads * self.d_k)
#         return self.linears[-1](x)


# class PostionwiseFeedForward(nn.Module):
#     def __init__(self, d_model, d_ff, dropout=0.1):
#         super(PostionwiseFeedForward, self).__init__()
#         self.w_1 = nn.Linear(d_model, d_ff)
#         self.w_2 = nn.Linear(d_ff, d_model)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         return self.w_2(self.dropout(F.relu(self.w_1(x))))


# class Embeddings(nn.Module):
#     def __init__(self, d_model, vocab):
#         super(Embeddings, self).__init__()
#         self.lut = nn.Embedding(vocab, d_model)
#         self.d_model = d_model

#     def forward(self, x):
#         return self.lut(x) * math.sqrt(self.d_model)


# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(dropout)
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) *
#                              -(math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)
#         self.register_buffer("pe", pe)

#     def forward(self, x):
#         x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
#         return self.dropout(x)


# class EncoderDecoder(nn.Module):
#     def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
#         super(EncoderDecoder, self).__init__()
#         self.encoder = encoder
#         self.decoder = decoder
#         self.src_embed = src_embed
#         self.tgt_embed = tgt_embed
#         self.generator = generator

#     def forward(self, src, tgt, src_mask, tgt_mask):
#         return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

#     def encode(self, src, src_mask):
#         return self.encoder(self.src_embed(src), src_mask)

#     def decode(self, memory, src_mask, tgt, tgt_mask):
#         return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


# class Generator(nn.Module):
#     def __init__(self, d_model, vocab):
#         super(Generator, self).__init__()
#         self.proj = nn.Linear(d_model, vocab)

#     def forward(self, x):
#         return F.log_softmax(self.proj(x), dim=-1)


# def clones(module, N):
#     return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# class LayerNorm(nn.Module):
#     def __init__(self, features, eps=1e-6):
#         super(LayerNorm, self).__init__()
#         self.a_2 = nn.Parameter(torch.ones(features))
#         self.b_2 = nn.Parameter(torch.zeros(features))
#         self.eps = eps

#     def forward(self, x):
#         mean = x.mean(-1, keepdim=True)
#         std = x.std(-1, keepdim=True)
#         return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
#     c = copy.deepcopy
#     attn = MultiHeadsAttention(h, d_model)
#     ff = PostionwiseFeedForward(d_model, d_ff, dropout)
#     position = PositionalEncoding(d_model, dropout)
#     model = EncoderDecoder(
#         Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
#         Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
#         nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
#         nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
#         Generator(d_model, tgt_vocab))
#     for p in model.parameters():
#         if p.dim() > 1:
#             nn.init.xavier_uniform_(p)
#     return model


# class Batch:
#     def __init__(self, src, trg=None, pad=0):
#         self.src = src
#         self.src_mask = (src != pad).unsqueeze(-2)
#         if trg is not None:
#             self.trg = trg[:, :-1]
#             self.trg_y = trg[:, 1:]
#             self.trg_mask = self.make_std_mask(self.trg, pad)
#             self.ntokens = (self.trg_y != pad).data.sum()

#     @staticmethod
#     def make_std_mask(tgt, pad):
#         tgt_mask = (tgt != pad).unsqueeze(-2)
#         tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
#         return tgt_mask


# def run_epoch(data_iter, model, loss_compute):
#     start = time.time()
#     total_tokens = 0
#     total_loss = 0
#     tokens = 0
#     for i, batch in enumerate(data_iter):
#         out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
#         loss = loss_compute(out, batch.trg_y, batch.ntokens)
#         total_loss += loss
#         total_tokens += batch.ntokens
#         tokens += batch.ntokens
#         if i % 50 == 1:
#             elapsed = time.time() - start
#             print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
#                   (i, loss / batch.ntokens, tokens / elapsed))
#             start = time.time()
#             tokens = 0
#     return total_loss / total_tokens


# global max_src_in_batch, max_tgt_in_batch


# def batch_size_fn(new, count, sofar):
#     global max_src_in_batch, max_tgt_in_batch
#     if count == 1:
#         max_src_in_batch = 0
#         max_tgt_in_batch = 0
#     max_src_in_batch = max(max_src_in_batch, len(new.src))
#     max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
#     src_elements = count * max_src_in_batch
#     tgt_elements = count * max_tgt_in_batch
#     return max(src_elements, tgt_elements)


# class NoamOpt:
#     def __init__(self, model_size, factor, warmup, optimizer):
#         self.optimizer = optimizer
#         self._step = 0
#         self.warmup = warmup
#         self.factor = factor
#         self.model_size = model_size
#         self._rate = 0

#     def step(self):
#         self._step += 1
#         rate = self.rate()
#         for p in self.optimizer.param_groups:
#             p["lr"] = rate
#         self._rate = rate
#         self.optimizer.step()

#     def rate(self, step=None):
#         if step is None:
#             step = self._step
#         return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


# def get_std_opt(model):
#     return NoamOpt(model.src_embed[0].d_model, 2, 4000,
#                    torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


# class LabelSmoothing(nn.Module):
#     def __init__(self, size, padding_idx, smoothing=0.0):
#         super(LabelSmoothing, self).__init__()
#         self.criterion = nn.KLDivLoss(reduction='sum')
#         self.padding_idx = padding_idx
#         self.confidence = 1.0 - smoothing
#         self.smoothing = smoothing
#         self.size = size
#         self.true_dist = None

#     def forward(self, x, target):
#         assert x.size(1) == self.size
#         true_dist = x.data.clone()
#         true_dist.fill_(self.smoothing / (self.size - 2))
#         true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
#         true_dist[:, self.padding_idx] = 0
#         mask = torch.nonzero(target.data == self.padding_idx)
#         if mask.dim() > 0:
#             true_dist.index_fill_(0, mask.squeeze(), 0.0)
#         self.true_dist = true_dist
#         return self.criterion(x, Variable(true_dist, requires_grad=False))


# ==================== MiniMind 风格的现代 Attention ====================

class Config(nn.Module):
    def __init__(self, hidden_size=128, hidden_layer=2, use_moe=False, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.hidden_layer = hidden_layer
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0)

        self.vocab_size = kwargs.get("vocab_size", 640)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)

        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attn_heads = kwargs.get("num_attn_heads", 4)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 2)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attn_heads)

        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)

        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)

        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)

        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 1024,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None

        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)


def procompute_freqs_cis(dim: int, end: int = int(2 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    attn_factor = 1.0

    if rope_scaling is not None:
        orig_max = rope_scaling.get("original_max_position_embeddings", 2048)
        factor = rope_scaling.get("factor", 16)
        beta_fast = rope_scaling.get("beta_fast", 32.0)
        beta_slow = rope_scaling.get("beta_slow", 1.0)
        attn_factor = rope_scaling.get("attention_factor", 1.0)
        if end / orig_max > 1.0:
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low = max(math.floor(inv_dim(beta_fast)), 0)
            high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), dim=-1)
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (x[:, :, :, None, :]
            .expand(batch_size, slen, num_key_value_heads, n_rep, head_dim)
            .reshape(batch_size, slen, num_key_value_heads * n_rep, head_dim))


# class Attention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.num_key_value_heads = config.num_attn_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attn_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.is_causal = True

        self.q_proj = nn.Linear(config.hidden_size, config.num_attn_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attn_heads * self.head_dim, config.hidden_size, bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape

        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        xq, xk = self.q_norm(xq), self.k_norm(xk)

        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq = xq.transpose(1, 2)
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)

        use_flash = (self.flash and (seq_len > 1)
                     and (not self.is_causal or past_key_value is None)
                     and (attention_mask is None or torch.all(attention_mask == 1)))

        if use_flash:
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.is_causal)
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)

            if self.is_causal:
                causal_mask = torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
                scores[:, :, :, -seq_len:] += causal_mask

            if attention_mask is not None:
                scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9

            attn_weights = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = self.attn_dropout(attn_weights) @ xv

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv
#input: normalized x? 
class Attention(nn.Module):
    def __init__(self, config:Config):
        super().__init__()
        self.num_key_value_heads = config.num_attn_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attn_heads      
        self.n_local_kv_heads = self.num_key_value_heads     
        self.n_rep = self.n_local_heads // self.n_local_kv_heads 
        self.head_dim = config.head_dim                      
        self.is_causal = True          #推理必须是true
        
        # Q/K/V/O 线性投影 (无偏置，现代LLM标配)
        # proj = projection (线性投影/矩阵乘法)
        self.q_proj = nn.Linear(config.hidden_size, config.num_attn_heads * self.head_dim, bias=False)    # q_proj = query_projection: 768 → 8*96=768
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)      # k_proj = key_projection: 768 → 4*96=384
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)      # v_proj = value_projection: 768 → 4*96=384
        self.o_proj = nn.Linear(config.num_attn_heads * self.head_dim, config.hidden_size, bias=False)    # o_proj = output_projection: 768 → 768
        
         # QK归一化 (Qwen3的设计，防止注意力分数随训练爆炸)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        
        # 检测是否支持Flash Attention (PyTorch 2.0+的SDPA)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn
                              
    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _=x.shape # hidden_size (已知，丢弃)
        print(f"\n{'='*60}")
        print(f"[Attention Input] x.shape = {x.shape}")
        print(f"  bsz={bsz}, seq_len={seq_len}, hidden_size={x.shape[-1]}")
        print(f"  x[:1,:2,:8] = {x[:1,:3,:8]}")
        
        # Step 1: 线性投影得到Q, K, V
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        print(f"\n[Step 1: Linear Projection]")
        print(f"  xq.shape = {xq.shape}  (hidden_size → num_heads * head_dim)")
        print(f"  xk.shape = {xk.shape}  (hidden_size → num_kv_heads * head_dim)")
        print(f"  xv.shape = {xv.shape}  (hidden_size → num_kv_heads * head_dim)")
        
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)      
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)   
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)  
        print(f"\n[Step 1b: Reshape to multi-head]")
        print(f"  xq.shape = {xq.shape}  [B, S, n_heads={self.n_local_heads}, head_dim={self.head_dim}]")
        print(f"  xk.shape = {xk.shape}  [B, S, n_kv_heads={self.n_local_kv_heads}, head_dim={self.head_dim}]")
        print(f"  xv.shape = {xv.shape}  [B, S, n_kv_heads={self.n_local_kv_heads}, head_dim={self.head_dim}]")
        
        # Step 2: QK归一化 (防止注意力分数随维度增大而爆炸)
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        print(f"\n[Step 2: QK RMSNorm]")
        print(f"  xq normalized: mean={xq.mean().item():.4f}, std={xq.std().item():.4f}")
        print(f"  xk normalized: mean={xk.mean().item():.4f}, std={xk.std().item():.4f}")

        # Step 3: 应用RoPE旋转位置编码
        cos, sin = position_embeddings
        print(f"\n[Step 3: RoPE]")
        print(f"  cos.shape = {cos.shape}, sin.shape = {sin.shape}")
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        print(f"  After RoPE: xq.shape = {xq.shape}, xk.shape = {xk.shape}")
        
        # Step 4: KV Cache (推理加速，拼接历史K/V)
        if past_key_value is not None:
            print(f"\n[Step 4: KV Cache]")
            print(f"  Before cat: xk.shape = {xk.shape}")
            print(f"  past_key.shape = {past_key_value[0].shape}")
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
            print(f"  After cat:  xk.shape = {xk.shape} (历史 + 当前)")
        else:
            print(f"\n[Step 4: KV Cache] past_key_value is None, 不拼接")
        past_kv = (xk, xv) if use_cache else None

        # Step 5: GQA repeat (如果 n_rep > 1，复制KV头以匹配Q头数)
        print(f"\n[Step 5: Transpose for attention]")
        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))
        print(f"  xq.shape = {xq.shape}  [B, n_heads, S, head_dim]")
        print(f"  xk.shape = {xk.shape}  [B, n_kv_heads, S, head_dim]")
        
        # Step 6: 注意力计算 (两条路径)
        use_flash = self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1))
        print(f"\n[Step 6: Attention Computation]")
        print(f"  Flash Attention? {use_flash}")
        
        if use_flash:
            print(f"  → 使用 Flash Attention (SDPA)")
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)
        else:
            print(f"  → 使用手动注意力计算")
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            print(f"  scores.shape = {scores.shape}  [B, n_heads, S_q, S_kv]")
            print(f"  scores range: [{scores.min().item():.4f}, {scores.max().item():.4f}]")
            
            if self.is_causal: 
                causal_mask = torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
                scores[:, :, :, -seq_len:] += causal_mask
                print(f"  Applied causal mask (上三角=-inf)")
                print(f"  scores after mask range: [{scores.min().item():.4f}, {scores.max().item():.4f}]")
                
            if attention_mask is not None: 
                scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
                print(f"  Applied padding mask")
            
            attn_weights = F.softmax(scores.float(), dim=-1).type_as(xq)
            print(f"  attn_weights.shape = {attn_weights.shape}")
            print(f"  attn_weights[0,0,0,:8] = {attn_weights[0,0,0,:min(8,attn_weights.shape[-1])]}  (第一个head, 第一个query token对前8个key的注意力)")
            output = self.attn_dropout(attn_weights) @ xv
        
        print(f"  output.shape = {output.shape}  [B, n_heads, S, head_dim]")
        
        # Step 7: 合并多头 + 输出投影
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        print(f"\n[Step 7: Merge heads + Output projection]")
        print(f"  After merge: output.shape = {output.shape}  [B, S, hidden_size]")
        output = self.resid_dropout(self.o_proj(output))
        print(f"  After o_proj: output.shape = {output.shape}")
        print(f"  output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        print(f"{'='*60}\n")
        return output, past_kv

class FeedFoward(nn.Module):
    def __init__(self, config: Config, intermediate_size: int = None):
        #x ─→ gate_proj ─→ SiLU ─┐
        #                        ├─→ 逐元素相乘 ─→ down_proj ─→ 输出
        # x ─→ up_proj ──────────┘
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)  # gate_projection: 768 → 2432 (门控投影)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)  # down_projection: 2432 → 768 (降维投影)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)    # up_projection: 768 → 2432 (升维投影)
        self.act_fn = ACT2FN[config.hidden_act]  # act_fn = activation_function, SiLU = x * sigmoid(x)
        
    def forward(self,x):
        return self.down_proj(self.act_fn(self.gate_proj(x))*self.up_proj(x))

class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)  # Router: 768 → 4
        self.experts = nn.ModuleList([FeedForward(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)])  # 4个独立的FFN
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)  # x_flat = x_flattened, [B*S, 768] 把batch和seq展平为一维token序列

        # Step 1: Router计算每个token对每个Expert的分数
        scores = F.softmax(self.gate(x_flat), dim=-1)  # [B*S, 4] 概率分布

        # Step 2: 选出每个token的top-k Expert (k=1)
        # topk_weight = top_k_routing_weight (路由权重), topk_idx = top_k_expert_index (选中的Expert编号)
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)  # [B*S, 1]
        # 归一化权重 (top-1时其实就是1.0，top-k>1时确保权重和为1)
        if self.config.norm_topk_prob: topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)

        # Step 3: 分发token到各Expert并加权聚合
        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)  # 哪些token被分配到Expert i
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()     # 被选中的token索引
                weight = topk_weight[mask].view(-1, 1)               # 对应的路由权重
                # Expert处理 → 乘权重 → 累加到输出
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                # 训练时: 即使Expert未被选中，也让它参与计算图(0*params)
                # 否则DDP会报错"某些参数没有梯度"
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())

        # Step 4: 负载均衡辅助loss (鼓励token均匀分布到各Expert)
        # 如果没有这个loss，Router倾向于把所有token都发给同一个Expert("赢者通吃")
        if self.training and self.config.router_aux_loss_coef > 0:
            # load: 每个Expert实际接收的token比例 [num_experts]
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            # aux_loss = (实际负载分布 · 路由概率分布) * num_experts * coef
            # 当所有Expert负载均匀时，此loss最小
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()

        return y.view(batch_size, seq_len, hidden_dim)

class Block(nn.Module):
    def __init__(self, layers, config:Config):
        super().__init__()
        self.self_attn=Attention(config)
        self.input_layernorm=RMSNorm(Config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm=RMSNorm(config.hidden_size,eps=config.rms_norm_eps)
        self.mlp=FeedFoward(config)if not config.use_moe else MOEFeedForward(config)
        
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
 
class MyModel(PreTrainedModel, GenerationMixin):
    config_class=PretrainedConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}  # 声明权重绑定关系
        
    def __init__(self,config:Config=None):
        self.config=config or MiniMindConfig()
        super().__init__()
        self.mode=MyModel(config) # Transformer主干
        # lm_head = language_model_head (语言模型输出头): 768 → 6400
        self.lm_head=nn.Linear(self.config.hidden_size,self.config.vocab_size,bias=False)
        # 权重绑定: lm_head和embedding共享同一份权重矩阵，减少参数量
        if self.config.tie_word_embeddings: 
            self.model.embed_tokens.weight = self.lm_head.weight
        self.post_init()  # HuggingFace的初始化后处理
    
    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        # Step 1: 通过Transformer主干得到hidden_states
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)

        # Step 2: lm_head投影到词表空间
        # logits_to_keep: 优化——只计算最后N个位置的logits(节省显存)
        # 推理时只需最后1个位置，RL训练时只需completion部分
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])  # [B, S, 6400]

        # Step 3: 计算loss (仅训练时有labels)
        loss = None
        if labels is not None:
            # 自回归: 用位置t的logits预测位置t+1的token
            # logits[:-1] 预测 labels[1:]
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)  # -100位置不算loss

        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
        
        
    # 自回归生成 (推理)
    # 自定义generate，不用HuggingFace的默认实现
    # 支持: KV Cache / temperature / top_k / top_p / repetition_penalty / streaming
    #
    # 每步循环:
    #   1. 只输入新token(利用KV Cache)
    #   2. 取最后位置logits
    #   3. temperature缩放 → repetition penalty → top_k截断 → top_p截断
    #   4. 采样得到next_token
    #   5. 拼接到input_ids，重复直到EOS或达到max长度
    @torch.inference_mode
    def generate(self, inputs = None, attention_mask=None, ax_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, 
                 eos_token_id=2, streamer=None, use_cache=True, 
                 num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
       