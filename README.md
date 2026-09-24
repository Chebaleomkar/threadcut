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

## Results

### 1. Pruning on a normal prefix cache costs more than not pruning. Suffix reuse makes it free.

![cost](docs/cost.png)

I recorded 18 real Pi agent conversations with Qwen3-4B on a Kaggle T4 (6 bug-fixing tasks x 3
modes, 664 agent steps). Then I replayed each conversation under three cache policies, using the
engine's exact rules. Replaying each run under its own policy reproduces the measured traces for
all 18 runs, token for token.

| Policy | Tokens in prompts | Tokens computed (prefill) | vs. no pruning | Mean peak context |
|---|---|---|---|---|
| No pruning + prefix cache | 11,774,948 | 132,459 | 1.00x | 11,021 |
| Pruning (k=1) + prefix cache | 5,447,634 | 361,331 | **2.73x** | 5,745 |
| Pruning (k=1) + suffix reuse | 5,447,634 | 132,459 | **1.00x** | 5,745 |

- Pruning halves the context the model attends to and the KV memory it holds. On a standard
  prefix cache, though, every prune invalidates everything after the cut, so the GPU re-prefills
  2.73x more tokens than if the agent had never pruned at all. This is the problem Subconscious
  describes, and here it is measured.
- With suffix reuse, the engine computes exactly the tokens that are new at each step, the same
  132,459 as an append-only agent. The prune is free in prefill, and the context stays half the
  size.
- In a single live run, suffix reuse served 98.7% of a 189-step agent's 2.2M prompt tokens from
  cache (27,893 computed).

### 2. Engineering notes: making decode not collapse on a T4

The first pilot decoded at 13.6 tok/s at 1.5k context and 3.6 tok/s at 10k. Profiling turned up
two causes, and both grow with context length:

1. **Attention fell back to PyTorch's math kernel.** transformers asks for `enable_gqa=True` on
   every decode step. On Turing GPUs (sm75), neither flash nor mem-efficient SDPA supports that,
   so each token expanded the KV cache 4x and upcast it to fp32. Folding GQA into the query length
   did not fix it, because the mem-efficient kernel then launched only 8 thread blocks. I wrote a
   **split-KV flash-decoding kernel in Triton** (`threadcut/flash_decode.py`): one program per
   (KV head, cache slice), with an online softmax and an exact log-sum-exp merge. It matches SDPA
   to 3e-5 and is 3-6x faster at long context (0.7 ms vs 4.0 ms at 8k on a GTX 1650).
2. **`DynamicCache` copies the whole cache on every token** (`torch.cat`), about 1.5 GB per token
   for Qwen3-4B at 10k. I replaced it with a **growable preallocated buffer** (1.25x growth, so a
   32k context still fits beside the weights on 16 GB). The splice also became in place, so a
   prune costs O(|C|) instead of O(whole cache).

Decode on Qwen3-0.6B / GTX 1650 went from 4.1 to 12.8 tok/s at 8k context and from 1.3 to
7.4 tok/s at 12k. With no pruning, the engine still produces the same greedy tokens as stock
transformers `generate`.

### 3. When harness-level agent runs fail

- **Malformed tool calls ended 14 of 18 runs in the first benchmark.** When Qwen3-4B edits code
  containing quotes, it writes invalid JSON in about 2% of its replies (14 of 668 tool-call blocks).
  A standard Hermes-style parser (which is also what vLLM does) then returns the reply as plain
  text. The agent reads that as "done" and stops mid-fix. The engine now forwards the call with
  its raw arguments, so the agent harness reports the error and the model can retry. All 14 real
  cases are now forwarded.
- **Loops under aggressive pruning.** With k=1, an agent that has fixed 4 of 5 bugs can lose the
  details of its earlier edits. It then re-runs the same test command with the same reasoning until
  it times out (the last 8 steps of the pilot were identical). A stub summary of the pruned output
  would help the model but breaks the A·C·D shape (new tokens in the middle). That is a real design
  tension for runtime pruning.

<!-- V2 -->

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
