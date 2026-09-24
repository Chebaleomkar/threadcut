"""KV cache surgery: drop the pruned span B and move C left without re-encoding it.

Keys are cached after RoPE, and RoPE rotations compose: rotating a key stored at position p by
-d radians-per-frequency gives exactly the key at position p - d. Values carry no position, so
they move as they are. What does not change is C's content, which was computed while B was still
visible; that residue is the "subconscious" part.
"""
import torch

from .match import Split


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def shift_keys(keys, delta, inv_freq):
    """Move post-RoPE keys by `delta` positions (negative = left). Math in fp32."""
    angle = delta * inv_freq.float().to(keys.device)
    emb = torch.cat((angle, angle))
    k = keys.float()
    return (k * emb.cos() + rotate_half(k) * emb.sin()).to(keys.dtype)


def splice(cache, s: Split, inv_freq):
    """Rewrite every layer of a DynamicCache from A·B·C·(tail) to A·C."""
    device = cache.layers[0].keys.device
    keep = torch.cat((torch.arange(0, s.a), torch.arange(s.c0, s.c0 + s.c))).to(device)
    for layer in cache.layers:
        k = layer.keys.index_select(2, keep)
        v = layer.values.index_select(2, keep)
        if s.c and s.gap:
            k[:, :, s.a:] = shift_keys(k[:, :, s.a:], -s.gap, inv_freq)
        layer.keys, layer.values = k, v
