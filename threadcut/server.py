"""OpenAI-compatible /v1/chat/completions over the threadcut engine.

Run: python -m threadcut.server --model Qwen/Qwen3-4B-Instruct-2507 --k 1 --trace runs/x.jsonl
Every request appends one JSON line to the trace with what was sent, pruned, reused and computed.
"""
import argparse
import json
import threading
import time
import uuid

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .chat import Renderer, normalize, parse_reply, prune
from .engine import Engine


def build_app(engine, k, trace_path, model_name, temperature=0.0, max_tokens=4096):
    app = FastAPI()
    renderer = Renderer(engine.tok)
    lock = threading.Lock()
    step = {"n": 0}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "threadcut"}]}

    def run(body):
        messages = normalize(body["messages"])
        tools = body.get("tools")
        kept, n_pruned = prune(messages, k)
        with lock:
            full_tokens = len(renderer.render(messages, tools)) if n_pruned else None
            ids = renderer.render(kept, tools)
            out, stats = engine.generate(ids, max_new_tokens=min(body.get("max_tokens") or max_tokens, max_tokens),
                                         temperature=temperature)
            text = engine.tok.decode(out)
            content, tool_calls = parse_reply(text)
            renderer.remember(content, tool_calls, out)
            step["n"] += 1
            rec = {"step": step["n"], "t": time.time(), "k": k, "suffix": engine.suffix,
                   "n_messages": len(messages), "subtasks_pruned": n_pruned,
                   "full_prompt_tokens": full_tokens or stats["prompt_tokens"], **stats,
                   "tool_calls": [tc["function"]["name"] for tc in tool_calls],
                   "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None}
            if trace_path:
                with open(trace_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
        finish = "tool_calls" if tool_calls else ("length" if stats["finish"] == "length" else "stop")
        usage = {"prompt_tokens": stats["prompt_tokens"], "completion_tokens": stats["completion_tokens"],
                 "total_tokens": stats["prompt_tokens"] + stats["completion_tokens"],
                 "prompt_tokens_details": {"cached_tokens": stats["reused_tokens"]}, "threadcut": rec}
        return content, tool_calls, finish, usage

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        try:
            content, tool_calls, finish, usage = await run_in_thread(run, body)
        except ValueError as e:
            return JSONResponse({"error": {"message": str(e), "type": "invalid_request_error",
                                           "code": "context_length_exceeded"}}, status_code=400)
        cid, created = f"chatcmpl-{uuid.uuid4().hex[:16]}", int(time.time())
        if not body.get("stream"):
            msg = {"role": "assistant", "content": content or None}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            return {"id": cid, "object": "chat.completion", "created": created, "model": model_name,
                    "choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": usage}

        def chunk(delta, finish_reason=None, **extra):
            return "data: " + json.dumps({"id": cid, "object": "chat.completion.chunk", "created": created,
                                          "model": model_name, "choices": [{"index": 0, "delta": delta,
                                          "finish_reason": finish_reason}], **extra}) + "\n\n"

        def events():
            yield chunk({"role": "assistant", "content": ""})
            if content:
                yield chunk({"content": content})
            for i, tc in enumerate(tool_calls):
                yield chunk({"tool_calls": [{"index": i, **tc}]})
            yield chunk({}, finish)
            if (body.get("stream_options") or {}).get("include_usage"):
                yield "data: " + json.dumps({"id": cid, "object": "chat.completion.chunk", "created": created,
                                             "model": model_name, "choices": [], "usage": usage}) + "\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


async def run_in_thread(fn, *args):
    import asyncio
    return await asyncio.get_running_loop().run_in_executor(None, fn, *args)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--name", default="qwen3-4b")
    p.add_argument("--chat-template-from", help="take the chat template from another tokenizer")
    p.add_argument("--k", type=int, default=None, help="subtask stack size; omit for no pruning")
    p.add_argument("--no-suffix", action="store_true", help="prefix cache only")
    p.add_argument("--min-suffix", type=int, default=32)
    p.add_argument("--max-ctx", type=int, default=32768)
    p.add_argument("--device", default="cuda")
    p.add_argument("--trace")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    engine = Engine(a.model, device=a.device, suffix=not a.no_suffix, min_suffix=a.min_suffix, max_ctx=a.max_ctx)
    if a.chat_template_from:
        from transformers import AutoTokenizer
        engine.tok.chat_template = AutoTokenizer.from_pretrained(a.chat_template_from).chat_template
    uvicorn.run(build_app(engine, a.k, a.trace, a.name), host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
