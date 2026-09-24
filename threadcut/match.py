"""Split a new request against the cached token sequence (the Subconscious Cache rule).

cached = A · B · C · (stale tail)      new = A · C · D

A is the shared prefix, B is the span the agent pruned, C is reused from the cache and shifted
left by len(B), and D is the only part the model has to run. With suffix matching off this
degrades to a plain prefix cache: reuse A, recompute everything after it.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Split:
    a: int   # len(A)
    c0: int  # C starts at cached[c0]; B = cached[a:c0]
    c: int   # len(C); D = new[a + c:]

    @property
    def gap(self):
        return self.c0 - self.a


def lcp(x, y):
    n = min(len(x), len(y))
    if n == 0:
        return 0
    diff = np.flatnonzero(np.asarray(x[:n]) != np.asarray(y[:n]))
    return int(diff[0]) if len(diff) else n


def split(cached, new, min_suffix=32, suffix=True):
    # Always leave at least one token of D so the model produces logits for the next token.
    a = min(lcp(cached, new), len(new) - 1)
    rest = new[a:]
    if not suffix or len(rest) <= min_suffix or len(cached) - (a + 1) < min_suffix:
        return Split(a, a, 0)

    cached_arr = np.asarray(cached)
    windows = np.lib.stride_tricks.sliding_window_view(cached_arr[a + 1:], min_suffix)
    starts = np.flatnonzero((windows == np.asarray(rest[:min_suffix])).all(axis=1)) + a + 1
    best = Split(a, a, 0)
    for j in starts:
        c = min(lcp(cached[j:], rest), len(rest) - 1)
        if c > best.c:
            best = Split(a, int(j), c)
    return best
