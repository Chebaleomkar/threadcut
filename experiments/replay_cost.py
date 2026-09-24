"""Trajectory-matched cost: replay the same recorded conversations under three cache policies.

The agent benchmark cannot compare cost across modes directly, because each mode's agent takes a
different path of a different length. Here every recorded conversation is replayed step by step
under each policy, with exactly the engine's rules (same pruning, rendering, and A·B·C·D split),
so the only thing that changes is the policy. No GPU: this is token accounting with the
tokenizer. Prefill seconds are estimated from the T4 throughput measured in the traces.

Usage: python -m experiments.replay_cost --tokenizer models/Qwen3-4B-tok results/ docs/
"""
import argparse
import json
import pathlib

import numpy as np
from transformers import AutoTokenizer

from threadcut.chat import Renderer, parse_reply, prune
from threadcut.match import split

POLICIES = {"No pruning + prefix cache": (None, False),
            "Pruning (k=1) + prefix cache": (1, False),
            "Pruning (k=1) + suffix reuse": (1, True)}


def replay(tok, steps, k, suffix, min_suffix=32):
    renderer, cached = Renderer(tok), []
    prompt = computed = peak = 0
    for st in steps:
        kept, _ = prune(st["messages"], k)
        ids = renderer.render(kept, st["tools"])
        s = split(cached, ids, min_suffix, suffix) if cached else None
        reused = (s.a + s.c) if s else 0
        prompt, computed, peak = prompt + len(ids), computed + len(ids) - reused, max(peak, len(ids))
        cached = ids + st["gen_ids"]
        content, calls, _ = parse_reply(tok.decode(st["gen_ids"]))
        renderer.remember(content, calls, st["gen_ids"])
    return {"steps": len(steps), "prompt_tokens": prompt, "computed_tokens": computed, "peak_context": peak}


def prefill_rate(traces):
    """Least-squares fit ttft = overhead + computed / rate over all measured T4 steps."""
    x = np.array([t["computed_tokens"] for t in traces], float)
    y = np.array([t["ttft_s"] for t in traces], float)
    slope, overhead = np.polyfit(x, y, 1)
    return 1 / slope, overhead


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", required=True)
    p.add_argument("results")
    p.add_argument("docs")
    a = p.parse_args()
    res, docs = pathlib.Path(a.results), pathlib.Path(a.docs)
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    traces = [json.loads(l) for f in sorted((res / "traces").glob("*.jsonl")) for l in open(f, encoding="utf-8")]
    rate, overhead = prefill_rate(traces)
    rows = []
    for f in sorted((res / "dumps").glob("*.jsonl")):
        steps = [json.loads(l) for l in open(f, encoding="utf-8")]
        for name, (k, suffix) in POLICIES.items():
            rows.append({"run": f.stem, "policy": name, **replay(tok, steps, k, suffix)})
            print(rows[-1], flush=True)
    (docs / "replay_cost.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    base = {r["run"]: r for r in rows if r["policy"] == next(iter(POLICIES))}
    lines = [f"Same {len(base)} recorded conversations ({sum(b['steps'] for b in base.values()):,} agent steps), "
             f"replayed under each policy. Prefill time estimated at the measured T4 rate of "
             f"{rate:,.0f} tokens/s (+{overhead * 1000:.0f} ms per step).", "",
             "| Policy | Tokens in prompts | Tokens computed (prefill) | vs. no pruning | Est. prefill time | Mean peak context |",
             "|---|---|---|---|---|---|"]
    total_base = sum(b["computed_tokens"] for b in base.values())
    for name in POLICIES:
        rs = [r for r in rows if r["policy"] == name]
        comp = sum(r["computed_tokens"] for r in rs)
        secs = sum(r["computed_tokens"] / rate + r["steps"] * overhead for r in rs)
        lines.append(f"| {name} | {sum(r['prompt_tokens'] for r in rs):,} | {comp:,} | {comp / total_base:.2f}x | "
                     f"{secs:,.0f} s | {np.mean([r['peak_context'] for r in rs]):,.0f} |")
    (docs / "replay_cost.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
