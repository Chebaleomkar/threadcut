"""A single-sequence inference engine with prefix + suffix KV reuse.

One cached sequence (one agent at a time). Each request is split against it (match.split), the
cache is spliced to A·C (kv.splice), and only D is prefilled before decoding.
"""
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from .kv import splice
from .match import split


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class Engine:
    def __init__(self, model_path, device="cuda", dtype=torch.float16, suffix=True, min_suffix=32,
                 max_ctx=32768, prefill_chunk=2048, model=None, tokenizer=None):
        self.device = torch.device(device)
        self.tok = tokenizer or AutoTokenizer.from_pretrained(model_path)
        self.model = model or AutoModelForCausalLM.from_pretrained(
            model_path, dtype=dtype, attn_implementation="sdpa").to(self.device)
        self.model.eval()
        cfg = self.model.config
        rope = getattr(cfg, "rope_parameters", None) or {}
        assert rope.get("rope_type", "default") == "default", f"KV shifting needs plain RoPE, got {rope}"
        assert all(t == "full_attention" for t in (getattr(cfg, "layer_types", None) or ["full_attention"]))
        self.inv_freq = self.model.model.rotary_emb.inv_freq
        self.suffix, self.min_suffix = suffix, min_suffix
        self.max_ctx, self.prefill_chunk = max_ctx, prefill_chunk
        self.stop_ids = {self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|im_end|>")}
        self.reset()

    def reset(self):
        self.ids, self.cache = [], None

    @torch.no_grad()
    def _forward(self, ids, cache):
        """Run `ids` on top of `cache` in chunks; return logits for the last position."""
        logits = None
        for i in range(0, len(ids), self.prefill_chunk):
            x = torch.tensor([ids[i:i + self.prefill_chunk]], device=self.device)
            logits = self.model(input_ids=x, past_key_values=cache, use_cache=True, logits_to_keep=1).logits[:, -1]
        return logits

    def load(self, new_ids):
        """Bring the cache to `new_ids` with as little compute as possible. Returns (last logits, stats)."""
        s = split(self.ids, new_ids, self.min_suffix, self.suffix) if self.cache is not None else None
        if s is None or (s.a == 0 and s.c == 0):
            self.cache, reused = DynamicCache(config=self.model.config), 0
        else:
            splice(self.cache, s, self.inv_freq)
            reused = s.a + s.c
        logits = self._forward(new_ids[reused:], self.cache)
        stats = {
            "prompt_tokens": len(new_ids),
            "reused_prefix": s.a if s else 0,
            "reused_suffix": s.c if s else 0,
            "reused_tokens": reused,
            "dropped_from_cache": len(self.ids) - reused,
            "computed_tokens": len(new_ids) - reused,
        }
        self.ids = list(new_ids)
        return logits, stats

    def _pick(self, logits, temperature):
        if temperature <= 0:
            return int(logits.argmax(-1))
        probs = torch.softmax(logits.float() / temperature, -1)
        return int(torch.multinomial(probs, 1))

    @torch.no_grad()
    def generate(self, new_ids, max_new_tokens=1024, temperature=0.0):
        if len(new_ids) + 1 > self.max_ctx:
            raise ValueError(f"prompt of {len(new_ids)} tokens exceeds max_ctx {self.max_ctx}")
        max_new_tokens = min(max_new_tokens, self.max_ctx - len(new_ids))
        sync(self.device)
        t0 = time.perf_counter()
        logits, stats = self.load(new_ids)
        sync(self.device)
        t1 = time.perf_counter()
        out, finish = [], "length"
        for _ in range(max_new_tokens):
            nxt = self._pick(logits, temperature)
            if nxt in self.stop_ids:
                finish = "stop"
                break
            out.append(nxt)
            x = torch.tensor([[nxt]], device=self.device)
            logits = self.model(input_ids=x, past_key_values=self.cache, use_cache=True).logits[:, -1]
        sync(self.device)
        t2 = time.perf_counter()
        self.ids += out
        kv_tokens = self.cache.get_seq_length()
        layer = self.cache.layers[0]
        kv_bytes = 2 * kv_tokens * layer.keys[0, :, 0].numel() * layer.keys.element_size() * len(self.cache.layers)
        stats.update(completion_tokens=len(out), finish=finish, ttft_s=t1 - t0, decode_s=t2 - t1,
                     kv_tokens=kv_tokens, kv_mb=kv_bytes / 2**20)
        return out, stats
