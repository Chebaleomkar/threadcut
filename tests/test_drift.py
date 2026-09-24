import json
import os

import pytest
import torch
from transformers import AutoTokenizer

from experiments.drift import replay
from threadcut.chat import Renderer, normalize, parse_reply, prune
from threadcut.engine import Engine

SMALL = os.environ.get("THREADCUT_TEST_MODEL", "models/Qwen3-0.6B")
TEMPLATE = "models/Qwen3-4B-tok"


def test_replay_measures_suffix_steps():
    if not (os.path.isdir(SMALL) and os.path.isdir(TEMPLATE) and torch.cuda.is_available()):
        pytest.skip("real model or CUDA not available")
    tok = AutoTokenizer.from_pretrained(TEMPLATE)
    eng = Engine(SMALL, device="cuda", min_suffix=16, tokenizer=tok)
    r = Renderer(tok)
    msgs = [{"role": "system", "content": "You are a coding agent."}, {"role": "user", "content": "Inspect the files."}]
    call = lambda i: [{"id": "c", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": f"cat f{i}.py"})}}]
    steps = []
    for i in range(4):
        kept, _ = prune(normalize(msgs), 1)
        out, _ = eng.generate(r.render(kept), max_new_tokens=24)
        content, _ = parse_reply(tok.decode(out))
        r.remember(content, call(i), out)
        steps.append({"step": i + 1, "messages": normalize(msgs), "tools": None, "gen_ids": out})
        msgs += [{"role": "assistant", "content": content, "tool_calls": call(i)},
                 {"role": "tool", "tool_call_id": "c", "content": f"# f{i}.py\n" + "y = 2\n" * 30}]
    rows = replay(eng, steps, k=1, n=8)
    assert rows, "expected at least one suffix-reuse step"
    for row in rows:
        assert row["reused_suffix"] > 0
        assert 0 <= row["surgery_vs_full"]["top1_agree"] <= 1
