"""Check that the cost replay reproduces every measured run exactly, token for token.

Each run is replayed with the configuration it was recorded under (kaggle/run/run.py MODES) and
its total computed tokens and step count are compared with the engine's own trace.

Usage: python -m experiments.validate_replay --tokenizer Qwen/Qwen3-4B-Instruct-2507 data/v1 data/v2
"""
import argparse
import gzip
import json
import pathlib

from transformers import AutoTokenizer

from experiments.replay_cost import replay

RECORDED = {"full": (None, True), "prune_prefix": (1, False), "prune_suffix": (1, True)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", required=True)
    p.add_argument("results", nargs="+")
    a = p.parse_args()
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    ok = n = 0
    for res in map(pathlib.Path, a.results):
        for f in sorted((res / "dumps").glob("*.jsonl*")):
            run = f.name.split(".")[0]
            k, suffix = RECORDED[run.split("__")[0]]
            with (gzip.open(f, "rt", encoding="utf-8") if f.suffix == ".gz" else open(f, encoding="utf-8")) as fh:
                steps = [json.loads(l) for l in fh]
            trace = [json.loads(l) for l in open(res / "traces" / f"{run}.jsonl", encoding="utf-8")]
            r = replay(tok, steps, k, suffix)
            same = r["computed_tokens"] == sum(t["computed_tokens"] for t in trace) and r["steps"] == len(trace)
            ok, n = ok + same, n + 1
            if not same:
                print("MISMATCH", res.name, run, r["computed_tokens"], sum(t["computed_tokens"] for t in trace))
    print(f"{ok} of {n} runs reproduce their measured traces exactly")


if __name__ == "__main__":
    main()
