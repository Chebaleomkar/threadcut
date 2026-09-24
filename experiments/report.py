"""Turn a Kaggle results folder into tables (markdown) and charts (PNG).

Usage: python -m experiments.report results/ docs/
"""
import json
import math
import pathlib
import random
import statistics as st
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODES = {"full": ("No pruning", "#2a78d6"), "prune_prefix": ("Pruning + prefix cache", "#eb6834"),
         "prune_suffix": ("Pruning + suffix reuse", "#1baf7a")}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def paired_stats(fresh, surg, iters=10000, seed=0):
    """Mean of (fresh - surgery) KL with a 95% bootstrap CI, and an exact two-sided sign test."""
    d = [f - s for f, s in zip(fresh, surg)]
    rng = random.Random(seed)
    means = sorted(st.mean(rng.choices(d, k=len(d))) for _ in range(iters))
    wins, n = sum(x > 0 for x in d), sum(x != 0 for x in d)
    tail = sum(math.comb(n, i) for i in range(min(wins, n - wins) + 1)) / 2 ** n
    return st.mean(d), means[int(0.025 * iters)], means[int(0.975 * iters)], wins, n, min(1.0, 2 * tail)


def trailing_loop(trace):
    """Length of the run of identical steps (same reply length and tool calls) the trace ends with."""
    sig = [(t["completion_tokens"], tuple(t["tool_calls"])) for t in trace]
    n = 1 if sig else 0
    while n < len(sig) and sig[-n - 1] == sig[-1]:
        n += 1
    return n


def load(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")] if path.exists() else []


def style(ax, title, ylabel):
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=12)
    ax.set_ylabel(ylabel, color=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def bar(out, modes, values, title, ylabel, fmt):
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    labels = [MODES[m][0] for m in modes]
    bars = ax.bar(labels, values, color=[MODES[m][1] for m in modes], width=0.5)
    for b, v in zip(bars, values):
        ax.annotate(fmt(v), (b.get_x() + b.get_width() / 2, b.get_height()), ha="center", va="bottom",
                    xytext=(0, 4), textcoords="offset points", color=INK, fontsize=10)
    style(ax, title, ylabel)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main():
    res, docs = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    docs.mkdir(parents=True, exist_ok=True)
    runs = load(res / "results.jsonl")
    modes = [m for m in MODES if any(r["mode"] == m for r in runs)]
    lines = ["| Mode | Tasks passed | Agent steps | Tokens in prompts | Tokens computed (prefill) | Reused from cache | Prefill time (s) | Decode time (s) | Wall time (s) | Peak context |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    agg = {}
    for m in modes:
        rs = [r for r in runs if r["mode"] == m]
        a = {k: sum(r[k] for r in rs) for k in ("steps", "prompt_tokens", "computed_tokens", "reused_tokens",
                                                "ttft_s", "decode_s", "wall_s")}
        a["passed"], a["n"] = sum(r["passed"] for r in rs), len(rs)
        a["peak"] = max(r["peak_prompt_tokens"] for r in rs)
        agg[m] = a
        lines.append(f"| {MODES[m][0]} | {a['passed']}/{a['n']} | {a['steps']} | {a['prompt_tokens']:,} | "
                     f"{a['computed_tokens']:,} | {a['reused_tokens'] / max(a['prompt_tokens'], 1):.0%} | "
                     f"{a['ttft_s']:.0f} | {a['decode_s']:.0f} | {a['wall_s']:.0f} | {a['peak']:,} |")
    per_task = ["| Task | " + " | ".join(MODES[m][0] for m in modes) + " |", "|---|" + "---|" * len(modes)]
    for t in sorted({r["task"] for r in runs}):
        cells = []
        for m in modes:
            r = next((r for r in runs if r["task"] == t and r["mode"] == m), None)
            if r is None:
                cells.append("n/a")
                continue
            loop = trailing_loop(load(res / "traces" / f"{m}__{t}.jsonl"))
            cells.append(f"{'pass' if r['passed'] else 'fail'}{' (timeout)' if r['pi_status'] == 'timeout' else ''}, "
                         f"{r['steps']} steps, {r['computed_tokens']:,} computed"
                         + (f", looped last {loop}" if loop >= 3 else ""))
        per_task.append(f"| {t} | " + " | ".join(cells) + " |")

    if modes:
        bar(docs / "computed_tokens.png", modes, [agg[m]["computed_tokens"] for m in modes],
            "Prefill tokens the GPU actually computed (all tasks)", "tokens", lambda v: f"{v:,.0f}")
        bar(docs / "prefill_time.png", modes, [agg[m]["ttft_s"] for m in modes],
            "Total prefill time across all agent steps", "seconds", lambda v: f"{v:,.0f} s")

    drift = load(res / "drift.jsonl")
    # A looping agent repeats the same reply; keep one row per (run, reply length, pruned span) so
    # repeated steps do not inflate the count.
    seen, unique = set(), []
    for d in drift:
        key = (d["run"], d["scored_tokens"], d["pruned_tokens"], round(d["fresh_vs_full"]["kl_mean"], 3))
        if key not in seen:
            seen.add(key)
            unique.append(d)
    n_dupes, drift = len(drift) - len(unique), unique
    dlines = []
    if drift:
        fresh = [d["fresh_vs_full"]["kl_mean"] for d in drift]
        surg = [d["surgery_vs_full"]["kl_mean"] for d in drift]
        closer = sum(s < f for s, f in zip(surg, fresh))
        t1f = st.mean(d["fresh_vs_full"]["top1_agree"] for d in drift)
        t1s = st.mean(d["surgery_vs_full"]["top1_agree"] for d in drift)
        dlines = ["| Context after a prune | Mean KL to unpruned model | Median KL | Top-1 agreement with unpruned |",
                  "|---|---|---|---|",
                  f"| Recomputed from scratch (prefix cache) | {st.mean(fresh):.4f} | {st.median(fresh):.4f} | {t1f:.1%} |",
                  f"| Spliced cache (suffix reuse) | {st.mean(surg):.4f} | {st.median(surg):.4f} | {t1s:.1%} |",
                  "", f"Suffix reuse was closer to the unpruned model on {closer}/{len(drift)} pruned steps "
                      f"({sum(d['scored_tokens'] for d in drift):,} reply tokens scored, {len({d['run'] for d in drift})} runs, "
                      f"{n_dupes} repeated loop steps removed)."]
        mean_d, lo, hi, wins, n, p = paired_stats(fresh, surg)
        dlines += ["", f"Paired difference KL(fresh) - KL(surgery): mean {mean_d:+.4f}, 95% bootstrap CI "
                       f"[{lo:+.4f}, {hi:+.4f}]; sign test {wins}/{n}, p = {p:.3g}."]
        fig, ax = plt.subplots(figsize=(5.5, 5), dpi=150)
        hi = max(fresh + surg) * 1.05
        ax.plot([0, hi], [0, hi], color=MUTED, linewidth=1, linestyle="--")
        ax.scatter(fresh, surg, s=36, color=MODES["prune_suffix"][1], edgecolor="white", linewidth=1.5, zorder=3)
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        ax.set_xlabel("KL(unpruned || recomputed after prune)", color=MUTED)
        style(ax, "Each dot is one pruned agent step.\nBelow the line: the reused suffix stayed closer\n"
                  "to the unpruned model than a fresh recompute", "KL(unpruned || spliced cache)")
        fig.tight_layout()
        fig.savefig(docs / "drift.png")
        plt.close(fig)

    md = ["## Agent benchmark (totals)", "", *lines, "", "## Per task", "", *per_task]
    if dlines:
        md += ["", "## What the reused suffix remembers", "", *dlines]
    (docs / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
