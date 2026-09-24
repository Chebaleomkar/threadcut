"""OpenAI chat messages <-> token ids, tool-call parsing, and the subtask pruning policy.

Two details keep the cache hitting across agent steps:
- The engine remembers the exact token ids it generated for each reply. When the agent sends that
  reply back as history (re-serialized, possibly with different JSON spacing), we splice in the
  original ids instead of re-rendering, so the history is byte-identical to what is cached.
- Pruning deletes whole messages only. <|im_start|>/<|im_end|> are hard token boundaries, so a
  deletion removes one contiguous token span and never changes tokenization on either side.
"""
import json
import re
import uuid

TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
SLOT = re.compile("⁣(\\d+)⁣")  # invisible-separator placeholder, never produced by agents


def text_of(content):
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


def reply_key(content, tool_calls):
    calls = []
    for tc in tool_calls or []:
        fn = tc.get("function", tc)
        args = fn.get("arguments")
        try:
            args = json.loads(args) if isinstance(args, str) else args
        except json.JSONDecodeError:
            pass
        calls.append((fn.get("name"), json.dumps(args, sort_keys=True)))
    return (text_of(content).strip(), tuple(calls))


def _parse_call(body):
    """(name, arguments-json-string, well_formed) for one <tool_call> body, or None if hopeless.

    Small models often write invalid JSON when a tool argument contains code (unescaped quotes).
    Dropping such a call makes the reply look like a final answer and silently ends the agent's
    run, so a call whose name is recoverable is forwarded with its raw argument text; the agent
    harness then reports the bad arguments back to the model, which can retry.
    """
    try:
        obj = json.loads(body, strict=False)
        return obj["name"], json.dumps(obj.get("arguments", {})), True
    except (json.JSONDecodeError, KeyError, TypeError):
        name = re.search(r'"name"\s*:\s*"([^"]+)"', body)
        if not name:
            return None
        args = re.search(r'"arguments"\s*:\s*(.*)\}\s*$', body, re.S)
        return name.group(1), args.group(1).strip() if args else "{}", False


def parse_reply(text):
    """Split generated text into (content, OpenAI tool_calls, number of malformed calls)."""
    calls, malformed = [], 0
    for m in TOOL_CALL.finditer(text):
        parsed = _parse_call(m.group(1))
        if parsed is None:
            return text.strip(), [], 1
        name, args, ok = parsed
        malformed += not ok
        calls.append({"id": f"call_{uuid.uuid4().hex[:12]}", "type": "function",
                      "function": {"name": name, "arguments": args}})
    return TOOL_CALL.sub("", text).strip(), calls, malformed


def normalize(messages):
    out = []
    for m in messages:
        role = "system" if m["role"] == "developer" else m["role"]
        msg = {"role": role, "content": text_of(m.get("content"))}
        if m.get("tool_calls"):
            msg["tool_calls"] = m["tool_calls"]
        out.append(msg)
    return out


def prune(messages, k):
    """Subtask pruning with a stack of size k (the TIM paper's threshold, k in {0, 1, 2}).

    A subtask is an assistant tool call plus its tool results. Once the agent has reacted to a
    subtask (another assistant message follows it) it is complete. All but the last k completed
    subtasks lose their tool-result messages; the call itself stays as the record of what was done.
    Returns (kept messages, number of subtasks pruned).
    """
    if k is None:
        return messages, 0
    groups, current = [], None
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            current = [] if m.get("tool_calls") else None
            if current is not None:
                groups.append(current)
        elif m["role"] == "tool" and current is not None:
            current.append(i)
    groups = [g for g in groups if g]
    completed = [g for g in groups if any(m["role"] == "assistant" for m in messages[g[-1] + 1:])]
    doomed = completed[:max(len(completed) - k, 0)]
    drop = {i for g in doomed for i in g}
    return [m for i, m in enumerate(messages) if i not in drop], len(doomed)


class Renderer:
    def __init__(self, tokenizer, max_remembered=512):
        self.tok = tokenizer
        self.raw = {}  # reply_key -> generated token ids
        self.max_remembered = max_remembered

    def remember(self, content, tool_calls, ids):
        self.raw[reply_key(content, tool_calls)] = list(ids)
        while len(self.raw) > self.max_remembered:
            self.raw.pop(next(iter(self.raw)))

    def render(self, messages, tools=None):
        msgs, slots = [], []
        for m in messages:
            ids = self.raw.get(reply_key(m["content"], m.get("tool_calls"))) if m["role"] == "assistant" else None
            if ids is None:
                msgs.append(m)
            else:
                msgs.append({"role": "assistant", "content": f"⁣{len(slots)}⁣"})
                slots.append(ids)
        text = self.tok.apply_chat_template(msgs, tools=tools or None, add_generation_prompt=True, tokenize=False)
        out = []
        for i, part in enumerate(SLOT.split(text)):
            out += slots[int(part)] if i % 2 else self.tok.encode(part, add_special_tokens=False)
        return out
