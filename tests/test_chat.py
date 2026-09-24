import json
import os

import pytest
import torch
from transformers import AutoTokenizer

from threadcut.chat import Renderer, normalize, parse_reply, prune

SMALL = os.environ.get("THREADCUT_TEST_MODEL", "models/Qwen3-0.6B")
TEMPLATE = "models/Qwen3-4B-tok"
TOOLS = [{"type": "function", "function": {"name": "bash", "description": "run a shell command",
          "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}}]


def call(name, **args):
    return {"id": "c", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def convo(n_subtasks):
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    for i in range(n_subtasks):
        msgs += [{"role": "assistant", "content": "", "tool_calls": [call("bash", command=f"cmd{i}")]},
                 {"role": "tool", "tool_call_id": "c", "content": f"output {i}"}]
    return msgs


def test_prune_keeps_last_k_completed_and_the_live_one():
    msgs = convo(4)  # subtasks 0..2 are completed; 3 is live (its result was just returned)
    kept, n = prune(msgs, 1)
    assert n == 2
    outputs = [m["content"] for m in kept if m["role"] == "tool"]
    assert outputs == ["output 2", "output 3"]
    assert sum(m["role"] == "assistant" for m in kept) == 4  # calls stay as the record
    assert prune(msgs, None) == (msgs, 0)
    assert prune(msgs, 0)[1] == 3


def test_parse_reply():
    content, calls = parse_reply('Let me look.\n<tool_call>\n{"name": "bash", "arguments": {"command": "ls"}}\n</tool_call>')
    assert content == "Let me look." and calls[0]["function"]["name"] == "bash"
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "ls"}


@pytest.fixture(scope="module")
def tok():
    if not os.path.isdir(TEMPLATE):
        pytest.skip("tokenizer not available")
    return AutoTokenizer.from_pretrained(TEMPLATE)


def test_render_splices_remembered_reply_ids(tok):
    r = Renderer(tok)
    gen = tok.encode('<tool_call>\n{"name":"bash","arguments":{"command":"ls"}}\n</tool_call>', add_special_tokens=False)
    msgs = normalize([{"role": "user", "content": "hi"}])
    prompt = r.render(msgs, TOOLS)
    content, calls = parse_reply(tok.decode(gen))
    r.remember(content, calls, gen)
    # The agent sends the call back re-serialized differently; the rendered history must still
    # begin with exactly prompt + generated ids.
    echoed = {"role": "assistant", "content": None, "tool_calls": [call("bash", command="ls")]}
    nxt = r.render(normalize([{"role": "user", "content": "hi"}, echoed,
                              {"role": "tool", "tool_call_id": "c", "content": "a.py"}]), TOOLS)
    assert nxt[:len(prompt) + len(gen)] == prompt + gen


def test_agent_loop_hits_suffix_cache_after_prune(tok):
    if not (os.path.isdir(SMALL) and torch.cuda.is_available()):
        pytest.skip("real model or CUDA not available")
    from threadcut.engine import Engine
    eng = Engine(SMALL, device="cuda", min_suffix=16, tokenizer=tok)
    r = Renderer(tok)
    msgs = [{"role": "system", "content": "You are a coding agent. Always use the bash tool."},
            {"role": "user", "content": "List files, then print each file, one tool call per step."}]
    stats = []
    for i in range(5):
        kept, n_pruned = prune(normalize(msgs), 1)
        out, s = eng.generate(r.render(kept, TOOLS), max_new_tokens=48)
        s["pruned"] = n_pruned
        stats.append(s)
        content, calls = parse_reply(tok.decode(out))
        # Force a tool-call shaped history even if the small model rambles, so pruning has work to do.
        calls = calls or [call("bash", command=f"cat file{i}.py")]
        r.remember(content, calls, out)
        msgs.append({"role": "assistant", "content": content, "tool_calls": calls})
        msgs.append({"role": "tool", "tool_call_id": "c", "content": f"# file{i}.py\n" + "x = 1\n" * 40})
    after_prune = [s for s in stats if s["pruned"] and s["dropped_from_cache"]]
    assert after_prune, stats
    for s in after_prune:
        assert s["reused_suffix"] > 0, s
        assert s["computed_tokens"] < s["prompt_tokens"] - s["reused_prefix"], s
