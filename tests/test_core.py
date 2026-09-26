import os

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, Qwen3Config

from threadcut.engine import Engine
from threadcut.kv import GrowingLayer, shift_keys, rotate_half, splice
from threadcut.match import Split, split

SMALL = os.environ.get("THREADCUT_TEST_MODEL", "models/Qwen3-0.6B")
# The tiny CPU model only needs a Qwen tokenizer; use the local copy if present, else the Hub.
TOKENIZER = SMALL if os.path.isdir(SMALL) else "Qwen/Qwen3-0.6B"


def tiny_engine():
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096)
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32, attn_implementation="sdpa")
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    return Engine(None, device="cpu", model=model, tokenizer=tok, min_suffix=4, prefill_chunk=7), 1e-4


def real_engine():
    if not (os.path.isdir(SMALL) and torch.cuda.is_available()):
        pytest.skip("real model or CUDA not available")
    return Engine(SMALL, device="cuda", dtype=torch.float16, min_suffix=4), 3e-2


@pytest.fixture(params=["tiny", "real"], scope="module")
def eng(request):
    return tiny_engine() if request.param == "tiny" else real_engine()


def rand_ids(n, seed, vocab=500):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(10, vocab, (n,), generator=g).tolist()


def kv_alone(model, ids, start):
    """KV for `ids` placed at positions start.., computed with no other context visible."""
    cache = DynamicCache(config=model.config)
    x = torch.tensor([ids], device=model.device)
    pos = torch.arange(start, start + len(ids), device=model.device)[None]
    model(input_ids=x, position_ids=pos, past_key_values=cache, use_cache=True)
    return cache


def concat(*caches):
    out = caches[0]
    for c in caches[1:]:
        for lo, li in zip(out.layers, c.layers):
            lo.keys = torch.cat((lo.keys, li.keys), 2)
            lo.values = torch.cat((lo.values, li.values), 2)
    return out


def test_split_rule():
    A, B, C, D = [1, 2, 3], [7, 7], [4, 5, 6, 8, 9], [10, 11]
    assert split(A + B + C, A + C + D, min_suffix=3) == Split(3, 5, 5)
    assert split(A + B + C, A + C + D, min_suffix=3, suffix=False) == Split(3, 3, 0)
    assert split(A + C, A + C + D, min_suffix=3) == Split(8, 8, 0)          # plain append
    assert split(A + C, A + C, min_suffix=3).a == len(A + C) - 1            # identical: still compute 1


def test_T1_shift_is_rotation_composition(eng):
    engine, atol = eng
    rot = engine.model.model.rotary_emb
    k = torch.randn(1, 2, 5, rot.inv_freq.numel() * 2)
    def rope_at(x, p):
        cos, sin = rot(x, torch.full((1, x.shape[2]), p, dtype=torch.long))
        return x * cos[:, None].float() + rotate_half(x) * sin[:, None].float()
    assert torch.allclose(shift_keys(rope_at(k, 900), -350, rot.inv_freq), rope_at(k, 550), atol=1e-4)


def test_T2_splice_matches_fresh_compute(eng):
    engine, atol = eng
    m = engine.model
    A, B, C, D = rand_ids(9, 1), rand_ids(13, 2), rand_ids(11, 3), rand_ids(5, 4)
    with torch.no_grad():
        # Reference: A, then C placed right after A without seeing A, then D on top.
        ref = concat(kv_alone(m, A, 0), kv_alone(m, C, len(A)))
        ref_logits = m(input_ids=torch.tensor([D], device=m.device), past_key_values=ref).logits[0, -1]
        # Test: cache holds A · B · C (C placed after B, not seeing it); the engine drops B and shifts C.
        engine.cache = concat(kv_alone(m, A, 0), kv_alone(m, B, len(A)), kv_alone(m, C, len(A) + len(B)))
        engine.ids = A + B + C
        logits, stats = engine.load(A + C + D)
    assert (stats["reused_prefix"], stats["reused_suffix"], stats["computed_tokens"]) == (9, 11, 5)
    assert torch.allclose(logits[0].float(), ref_logits.float(), atol=atol, rtol=0)


def hf_generate(m, ids, n, stop):
    """Reference decode with stock transformers SDPA attention (independent of threadcut.attention)."""
    impl = m.config._attn_implementation
    m.set_attn_implementation("sdpa")
    try:
        return m.generate(torch.tensor([ids], device=m.device), max_new_tokens=n, do_sample=False,
                          eos_token_id=list(stop))[0, len(ids):].tolist()
    finally:
        m.set_attn_implementation(impl)


def test_growing_layer_splice_matches_copy_splice():
    torch.manual_seed(0)
    k, v = torch.randn(1, 2, 40, 16), torch.randn(1, 2, 40, 16)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, 16, 2).float() / 16))
    g = GrowingLayer()
    g.update(k, v)
    plain = DynamicCache(config=Qwen3Config(num_hidden_layers=1))
    plain.layers[0].update(k, v)
    for s in (Split(10, 17, 20), Split(12, 12, 0)):
        grown = GrowingLayer()
        grown.update(k, v)
        grown.splice(s, inv_freq)
        ref = DynamicCache(config=Qwen3Config(num_hidden_layers=1))
        ref.layers[0].update(k, v)
        splice(ref, s, inv_freq)
        assert torch.allclose(grown.keys, ref.layers[0].keys) and torch.allclose(grown.values, ref.layers[0].values)
        grown.update(k[:, :, :3], v[:, :, :3])  # appending after a splice keeps working
        assert grown.get_seq_length() == s.a + s.c + 3


def test_T3_no_pruning_matches_hf_generate(eng):
    engine, _ = eng
    engine.reset()
    m = engine.model
    p1 = rand_ids(20, 5)
    out1, s1 = engine.generate(p1, max_new_tokens=12)
    ref1 = hf_generate(m, p1, 12, engine.stop_ids)
    assert out1 == [t for t in ref1 if t not in engine.stop_ids][:len(out1)]
    p2 = p1 + out1 + rand_ids(6, 6)
    out2, s2 = engine.generate(p2, max_new_tokens=12)
    assert s2["reused_tokens"] == len(p1) + len(out1) and s2["computed_tokens"] == 6
    ref2 = hf_generate(m, p2, 12, engine.stop_ids)
    assert out2 == [t for t in ref2 if t not in engine.stop_ids][:len(out2)]
