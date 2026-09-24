"""What does the reused suffix remember? Replays recorded agent steps and compares three contexts.

For every step where pruning removed a span and the engine reused the suffix after it, we score the
reply the agent actually produced (teacher forcing, first --n tokens) under:
  full     the unpruned history                 (the model with no pruning at all)
  fresh    the pruned history, recomputed       (a standard prefix cache after a prune)
  surgery  the pruned history, spliced cache    (suffix reuse: C keeps states computed with B visible)
and report KL(full || fresh), KL(full || surgery) and top-1 agreement with `full`.
If surgery is closer to full than fresh is, information from the pruned span survives in the
reused suffix states: the "subconscious" part of Subconscious Cache.

Usage: python -m experiments.drift --model Qwen/Qwen3-4B-Instruct-2507 --k 1 --out drift.jsonl dumps/*.jsonl
"""
import argparse
import json

import torch
import torch.nn.functional as F
from transformers import DynamicCache

from threadcut.chat import Renderer, parse_reply, prune
from threadcut.engine import Engine


@torch.no_grad()
def reply_logits(engine, cache, first_logits, teacher):
    """Logits predicting teacher[0..n-1], given logits for teacher[0] and a cache ending before it."""
    rest = engine.model(input_ids=torch.tensor([teacher[:-1]], device=engine.device),
                        past_key_values=cache, use_cache=True).logits[0] if len(teacher) > 1 else None
    out = first_logits.float()
    return torch.cat((out, rest.float())) if rest is not None else out


@torch.no_grad()
def fresh_logits(engine, ids, teacher):
    cache = DynamicCache(config=engine.model.config)
    first = engine._forward(ids, cache)
    return reply_logits(engine, cache, first, teacher)


def compare(ref, other):
    lp, lq = F.log_softmax(ref, -1), F.log_softmax(other, -1)
    kl = (lp.exp() * (lp - lq)).sum(-1)
    return {"kl_mean": kl.mean().item(), "kl_max": kl.max().item(),
            "top1_agree": (ref.argmax(-1) == other.argmax(-1)).float().mean().item()}


@torch.no_grad()
def replay(engine, steps, k, n, max_rows=None, max_prompt=None):
    """max_rows samples pruned steps evenly across the run; max_prompt skips steps whose unpruned
    context is too long to recompute twice next to the live cache (T4 memory)."""
    renderer = Renderer(engine.tok)
    engine.reset()
    rows = []
    pruned_steps = [i for i, st in enumerate(steps) if prune(st["messages"], k)[1]]
    if max_rows and len(pruned_steps) > max_rows:
        pruned_steps = [pruned_steps[round(j * (len(pruned_steps) - 1) / (max_rows - 1))] for j in range(max_rows)]
    measure = set(pruned_steps)
    for i, st in enumerate(steps):
        messages, tools, gen = st["messages"], st["tools"], st["gen_ids"]
        kept, n_pruned = prune(messages, k)
        ids = renderer.render(kept, tools)
        first, s = engine.load(ids)
        teacher = gen[:n]
        full_ids = renderer.render(messages, tools) if i in measure else None
        if s["reused_suffix"] and teacher and full_ids and (max_prompt is None or len(full_ids) <= max_prompt):
            surgery = reply_logits(engine, engine.cache, first, teacher)
            fed = len(teacher) - 1
            fresh = fresh_logits(engine, ids, teacher)
            torch.cuda.empty_cache()
            full = fresh_logits(engine, full_ids, teacher)
            torch.cuda.empty_cache()
            rows.append({"step": st["step"], "prompt_tokens": s["prompt_tokens"],
                         "pruned_tokens": s["dropped_from_cache"], "reused_suffix": s["reused_suffix"],
                         "scored_tokens": len(teacher),
                         "fresh_vs_full": compare(full, fresh), "surgery_vs_full": compare(full, surgery),
                         "surgery_vs_fresh": compare(fresh, surgery)})
        else:
            fed = 0
        if gen[fed:]:
            engine.model(input_ids=torch.tensor([gen[fed:]], device=engine.device),
                         past_key_values=engine.cache, use_cache=True, logits_to_keep=1)
        engine.ids = ids + gen
        content, calls, _ = parse_reply(engine.tok.decode(gen))
        renderer.remember(content, calls, gen)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--k", type=int, required=True, help="the k the dumps were recorded with")
    p.add_argument("--n", type=int, default=64, help="reply tokens to score per step")
    p.add_argument("--min-suffix", type=int, default=32)
    p.add_argument("--max-rows", type=int, default=25, help="pruned steps measured per run")
    p.add_argument("--max-prompt", type=int, default=10000, help="skip steps with a longer unpruned context")
    p.add_argument("--out", required=True)
    p.add_argument("dumps", nargs="+")
    a = p.parse_args()
    engine = Engine(a.model, suffix=True, min_suffix=a.min_suffix)
    with open(a.out, "w", encoding="utf-8") as f:
        for path in a.dumps:
            steps = [json.loads(line) for line in open(path, encoding="utf-8")]
            for row in replay(engine, steps, a.k, a.n, a.max_rows, a.max_prompt):
                f.write(json.dumps({"run": path, **row}) + "\n")
                print(path, row["step"], "fresh KL", round(row["fresh_vs_full"]["kl_mean"], 4),
                      "surgery KL", round(row["surgery_vs_full"]["kl_mean"], 4), flush=True)


if __name__ == "__main__":
    main()
