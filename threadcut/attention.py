"""GQA-aware attention for Turing GPUs (T4, GTX 16xx).

transformers' SDPA path asks PyTorch for `enable_gqa=True` whenever there is no mask, which is
every decode step. On sm75 neither flash nor mem-efficient kernels support that, so PyTorch falls
back to the math kernel: it expands the KV cache 4x and upcasts it to fp32 on every token. Cost
grows with context, and decode fell from ~13 to ~3.6 tok/s at 10k tokens on a T4.

Decode (one query token): split-KV Triton kernel (flash_decode.py) over the cache as it is, or,
without Triton, SDPA with each group's query heads folded into a short query sequence.
Prefill: expand KV once and use SDPA's mem-efficient kernel with an explicit mask.
"""
import torch
import torch.nn.functional as F
from transformers import AttentionInterface
from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

try:
    from .flash_decode import decode_attention
except ImportError:  # no Triton: fall back to SDPA with folded query heads
    decode_attention = None

NAME = "threadcut"


def repeat_kv(x, n):
    b, h, s, d = x.shape
    return x if n == 1 else x[:, :, None].expand(b, h, n, s, d).reshape(b, h * n, s, d)


def attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    b, hq, q, d = query.shape
    hkv = key.shape[1]
    g = hq // hkv
    scaling = scaling if scaling is not None else d ** -0.5
    if q == 1 and attention_mask is None:
        # The g query heads sharing a KV head become a length-g query sequence with no mask, which
        # SDPA's mem-efficient kernel handles directly against the unexpanded cache.
        if decode_attention is not None and query.is_cuda and b == 1:
            out = decode_attention(query.reshape(hkv, g, d), key[0], value[0], scaling).reshape(1, hq, 1, d)
        else:
            out = F.scaled_dot_product_attention(query.reshape(b, hkv, g, d), key, value,
                                                 scale=scaling).reshape(b, hq, 1, d)
    else:
        is_causal = attention_mask is None and q > 1 and q == key.shape[2]
        out = F.scaled_dot_product_attention(query, repeat_kv(key, g), repeat_kv(value, g),
                                             attn_mask=attention_mask, scale=scaling, is_causal=is_causal)
    return out.transpose(1, 2).contiguous(), None


AttentionInterface.register(NAME, attention)
AttentionMaskInterface.register(NAME, sdpa_mask)
