"""KV cache surgery: drop the pruned span B and move C left without re-encoding it.

Keys are cached after RoPE, and RoPE rotations compose: rotating a key stored at position p by
-d radians-per-frequency gives exactly the key at position p - d. Values carry no position, so
they move as they are. What does not change is C's content, which was computed while B was still
visible; that residue is the "subconscious" part.
"""
import torch
from transformers import DynamicCache
from transformers.cache_utils import DynamicLayer

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
    if all(isinstance(layer, GrowingLayer) for layer in cache.layers):
        for layer in cache.layers:
            layer.splice(s, inv_freq)
        return
    device = cache.layers[0].keys.device
    keep = torch.cat((torch.arange(0, s.a), torch.arange(s.c0, s.c0 + s.c))).to(device)
    for layer in cache.layers:
        k = layer.keys.index_select(2, keep)
        v = layer.values.index_select(2, keep)
        if s.c and s.gap:
            k[:, :, s.a:] = shift_keys(k[:, :, s.a:], -s.gap, inv_freq)
        layer.keys, layer.values = k, v


class GrowingLayer(DynamicLayer):
    """DynamicLayer backed by a preallocated buffer, so appending a token writes in place instead of
    re-copying the whole cache with torch.cat (~1.5 GB per token for Qwen3-4B at 10k context).
    Grows by 1.25x (not 2x) so a 32k context still fits next to the weights on a 16 GB T4."""

    def lazy_initialization(self, key_states, value_states):
        super().lazy_initialization(key_states, value_states)
        self.n, self.kbuf, self.vbuf = 0, None, None

    def _reserve(self, need, like):
        cap = 0 if self.kbuf is None else self.kbuf.shape[-2]
        if need <= cap:
            return
        new = max(need, int(cap * 1.25), cap + 2048)
        kb = like.new_empty((*like.shape[:-2], new, like.shape[-1]))
        vb = torch.empty_like(kb)
        if self.n:
            kb[..., :self.n, :] = self.kbuf[..., :self.n, :]
            vb[..., :self.n, :] = self.vbuf[..., :self.n, :]
        self.kbuf, self.vbuf = kb, vb

    def _view(self):
        self.keys, self.values = self.kbuf[..., :self.n, :], self.vbuf[..., :self.n, :]

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        add = key_states.shape[-2]
        self._reserve(self.n + add, key_states)
        self.kbuf[..., self.n:self.n + add, :] = key_states
        self.vbuf[..., self.n:self.n + add, :] = value_states
        self.n += add
        self._view()
        return self.keys, self.values

    def crop(self, max_length):
        self.n = min(self.n, max_length)
        self._view()

    def splice(self, s: Split, inv_freq):
        """In-place A·B·C -> A·C: only C is touched, so a prune costs O(|C|), not O(cache)."""
        if s.c:
            k = self.kbuf[..., s.c0:s.c0 + s.c, :]
            k = shift_keys(k, -s.gap, inv_freq) if s.gap else k.clone()
            v = self.vbuf[..., s.c0:s.c0 + s.c, :].clone()
            self.kbuf[..., s.a:s.a + s.c, :] = k
            self.vbuf[..., s.a:s.a + s.c, :] = v
        self.n = s.a + s.c
        self._view()


def new_cache(config):
    cache = DynamicCache(config=config)
    cache.layers = [GrowingLayer() for _ in cache.layers]
    return cache
