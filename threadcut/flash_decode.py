"""Split-KV decode attention (flash-decoding) in Triton.

One decode step has a single query token, so per KV head there are only g = heads/kv_heads query
rows. Library kernels parallelize over (kv_head, query block), which is 8 thread blocks for Qwen3,
and each walks the whole cache alone; on a T4 that leaves most SMs idle and cost grows with context.
Here each program takes one KV head and one slice of the cache, keeps an online softmax, and writes
a partial output plus its log-sum-exp; the slices are merged exactly afterwards.
"""
import os

import torch
import triton
import triton.language as tl

if os.name == "nt" and "CC" not in os.environ:  # triton-windows ships TinyCC next to itself
    _tcc = os.path.join(os.path.dirname(triton.__file__), "runtime", "tcc", "tcc.exe")
    if os.path.exists(_tcc):
        os.environ["CC"] = _tcc


@triton.jit
def _partial(Q, K, V, O, L, n, scale,
             sq_h, sq_g, sk_h, sk_n, sv_h, sv_n, so_s, so_h, so_g, sl_s, sl_h,
             G: tl.constexpr, GP: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr, SPLIT: tl.constexpr):
    h = tl.program_id(0)
    s = tl.program_id(1)
    offs_g = tl.arange(0, GP)
    offs_d = tl.arange(0, D)
    q = tl.load(Q + h * sq_h + offs_g[:, None] * sq_g + offs_d[None, :], mask=offs_g[:, None] < G,
                other=0.0).to(tl.float32)
    m = tl.full([GP], float("-inf"), tl.float32)
    l = tl.zeros([GP], tl.float32)
    acc = tl.zeros([GP, D], tl.float32)
    start = s * SPLIT
    for off in range(start, start + SPLIT, BLOCK):
        offs_n = off + tl.arange(0, BLOCK)
        valid = offs_n < n
        k = tl.load(K + h * sk_h + offs_n[:, None] * sk_n + offs_d[None, :], mask=valid[:, None], other=0.0)
        v = tl.load(V + h * sv_h + offs_n[:, None] * sv_n + offs_d[None, :], mask=valid[:, None], other=0.0)
        # g is 2-4 and the step is memory-bound, so plain fp32 FMAs instead of tensor cores (tl.dot
        # in fp16 does not lower on sm75 in Triton 3.3).
        qk = tl.sum(q[:, None, :] * k.to(tl.float32)[None, :, :], axis=2) * scale
        qk = tl.where(valid[None, :], qk, float("-inf"))
        m_new = tl.maximum(m, tl.max(qk, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp(qk - m_safe[:, None])
        alpha = tl.exp(m - m_safe)
        l = l * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.sum(p[:, :, None] * v.to(tl.float32)[None, :, :], axis=1)
        m = m_new
    out = acc / tl.where(l == 0, 1.0, l)[:, None]
    lse = tl.where(l == 0, float("-inf"), m + tl.log(l))
    tl.store(O + s * so_s + h * so_h + offs_g[:, None] * so_g + offs_d[None, :], out, mask=offs_g[:, None] < G)
    tl.store(L + s * sl_s + h * sl_h + offs_g, lse, mask=offs_g < G)


def decode_attention(q, k, v, scale, split=256, block=16):
    """q: [hkv, g, d]; k, v: [hkv, n, d] (last dim contiguous). Returns [hkv, g, d]."""
    hkv, g, d = q.shape
    n = k.shape[1]
    splits = triton.cdiv(n, split)
    o = torch.empty(splits, hkv, g, d, device=q.device, dtype=torch.float32)
    lse = torch.empty(splits, hkv, g, device=q.device, dtype=torch.float32)
    _partial[(hkv, splits)](q, k, v, o, lse, n, scale,
                            q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1),
                            o.stride(0), o.stride(1), o.stride(2), lse.stride(0), lse.stride(1),
                            G=g, GP=max(2, triton.next_power_of_2(g)), D=d, BLOCK=block, SPLIT=split)
    w = torch.softmax(lse, dim=0)[..., None]
    return (o * w).sum(0).to(q.dtype)
