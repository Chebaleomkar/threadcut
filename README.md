# threadcut

**A mini Subconscious Cache on a free Kaggle T4.** I built a small inference engine that prunes a
coding agent's working memory mid-run and reuses the KV cache on both sides of the cut, so the
suffix after a pruned span is shifted into place instead of re-encoded. Then I ran the Pi coding
agent on it and measured what that saves, what it breaks, and what the reused suffix "remembers"
about the tokens that were cut.

This is my reproduction, at toy scale, of the idea behind Subconscious's OrangeLine runtime
(Subconscious Cache plus Auto Compaction) and the TIM paper
([Luo et al., 2025](https://arxiv.org/abs/2507.16784)). It is not their code; everything here is
written from their public descriptions.

<!-- RESULTS -->

## How it works

```
Pi coding agent ──OpenAI chat API──► threadcut server ──► engine (PyTorch, Qwen3-4B fp16, one T4)
                                      │ 1. subtask pruning (paper's stack, k = 1)
                                      │ 2. render history; splice in the exact ids of past replies
                                      │ 3. split against the cache:  cached A·B·C   new A·C·D
                                      ▼
                                    drop B, rotate C's keys left by |B|, prefill only D
```

- **Cache rule** (`threadcut/match.py`): `A` is the shared prefix, `B` the pruned span, `C` the
  longest run after `A` that is still in the cache, `D` the only tokens the model runs.
- **KV surgery** (`threadcut/kv.py`): Qwen3 caches keys after RoPE, and rotations compose, so
  moving `C` left by `|B|` positions is one elementwise rotation per key. No re-encoding. The
  values are copied as they are. `C` keeps states that were computed while `B` was visible.
- **Byte-identical history** (`threadcut/chat.py`): agents re-serialize tool calls, and different
  JSON spacing means different tokens, which means no cache hit. The engine remembers the exact
  token ids of every reply and splices them back in.
- **Pruning policy**: the TIM paper's subtask stack. A tool call plus its result is a subtask; all
  but the last `k` completed subtasks lose their tool-result message. The engine deletes whole
  messages only, so each step removes one contiguous span, and every step lands on either the
  prefix cache or the suffix cache.
- **Turing-friendly decode** (`threadcut/flash_decode.py`, `threadcut/attention.py`): see
  "Engineering notes" below.

Design notes: [docs/DESIGN.md](docs/DESIGN.md).

## Correctness

`pytest tests` (13 tests, CPU and GPU):

- **T1**: rotating a post-RoPE key by `-d` equals RoPE at `p - d`.
- **T2**: splice exactness. When `C` never saw `B`, dropping `B` and shifting `C` gives the same
  next-token logits as computing `A·C` from scratch.
- **T3**: with no pruning, the engine (custom attention, Triton decode, in-place cache) produces
  the same greedy tokens as stock transformers `generate` with SDPA.
- An agent loop through the chat layer lands a suffix hit on every step after a prune.

## Reproduce

```bash
pip install torch transformers==5.12.1 fastapi uvicorn pytest
python -m pytest tests                       # uses Qwen3-0.6B locally if models/Qwen3-0.6B exists
python tasks/check_solvable.py               # every benchmark task fails as shipped, passes when fixed
python -m threadcut.server --model Qwen/Qwen3-4B-Instruct-2507 --k 1 --trace runs/trace.jsonl
PI_CODING_AGENT_DIR=pi-config pi --model threadcut/qwen3-4b -p "fix the failing tests"
```

The full benchmark runs on Kaggle's free 2xT4 from the terminal: `python kaggle/push.py run`
ships the source as a private dataset and starts `kaggle/run/run.py`. It runs the tests, then
6 tasks x 3 modes on both GPUs, then the drift replay.
