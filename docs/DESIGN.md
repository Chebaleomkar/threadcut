# threadcut design notes

A mini version of Subconscious's OrangeLine runtime: an inference engine that prunes a coding
agent's working memory and reuses the KV cache on both sides of the cut (Subconscious Cache).

## The cache rule

The engine keeps one cached token sequence. A new request is split against it:

```
cached = A · B · C · (stale tail)        new = A · C · D
```

- `A`: longest common prefix, reused as is.
- `B`: the span the agent pruned. Its KV is dropped and its GPU memory freed.
- `C`: the longest run of the new request that follows `A` and also appears in the cache after `A`
  (at least `min_suffix` tokens). Its KV is kept and moved left by `len(B)`.
- `D`: everything after `C`. This is the only part the model runs.

With suffix matching off, the same engine is a plain prefix cache: reuse `A`, recompute `C · D`.

## Moving C without re-encoding it

Qwen3 caches keys after RoPE. RoPE rotates each key by an angle proportional to its position, and
rotations compose, so rotating a cached key by `-len(B)` positions gives exactly the key at its new
position. That is one elementwise multiply per key, done in fp32. Values have no position and are
copied. Test T2 checks that this matches a fresh computation to fp16 precision.

What the shift cannot change is the content of `C`: its keys and values were computed while `B` was
still visible. That leftover information is the "subconscious" part. `experiments/drift.py`
measures it.

## Keeping the history byte-identical

An agent sends its whole history back every step, and it re-serializes the model's replies (for
example, tool-call JSON with different spacing). If the tokens differ, nothing matches. So the
engine remembers the exact token ids it generated for each reply, keyed by the reply's content and
parsed tool calls, and splices those ids back in when the reply comes back as history.

## Pruning policy (Auto Compaction, rule-based)

This follows the TIM paper's subtask stack with threshold `k ∈ {0, 1, 2}`:

- A subtask is an assistant tool call plus its tool results.
- A subtask is complete once another assistant message follows it.
- All but the last `k` completed subtasks lose their tool-result messages. The call stays as the
  record of what was done.
- The engine deletes whole messages and never inserts stubs. Chat-template special tokens bound
  every message, so a deletion removes exactly one contiguous token span, and each step adds at most
  one new span. Every step therefore lands on the prefix cache or the suffix cache.

## Model choice

The engine uses Qwen3-4B-Instruct-2507 in fp16 on a T4. It is pure attention with plain RoPE, so
the KV shift is exact. TIM-9B is a hybrid model with recurrent layers, whose state cannot be cut
at a token boundary. The engine asserts plain RoPE and full attention on load.

## Experiments

1. Correctness (`tests/`): T1 shows rotation composition, T2 splice exactness, T3 equality with HF
   `generate` when nothing is pruned.
2. Agent benchmark (`kaggle/run/run.py`): Pi CLI on 6 multi-bug Python tasks in three modes:
   no pruning, pruning + prefix cache, and pruning + suffix reuse.
3. Drift (`experiments/drift.py`): on each pruned step, KL to the unpruned model for a fresh
   recompute versus the spliced cache.
