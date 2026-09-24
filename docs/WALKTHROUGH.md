# threadcut code walkthrough

This follows **one request** from the Pi agent through the engine, file by file, in the order the
code actually runs. Every code block is copied from the repo with its line numbers.

The whole engine is 640 lines in `threadcut/`:

| File | Lines | Job |
|---|---|---|
| `server.py` | 140 | HTTP front door: receives the agent's chat request, returns the reply |
| `chat.py` | 130 | Messages to tokens, pruning policy, parsing tool calls |
| `match.py` | 48 | Compares the new request with the cache: finds A, B, C, D |
| `engine.py` | 104 | Runs the model: reuse cache, compute only D, generate the reply |
| `kv.py` | 98 | KV cache surgery: drop B, shift C; a faster cache buffer |
| `attention.py` | 51 | Attention function that stays fast on T4 GPUs |
| `flash_decode.py` | 69 | Triton GPU kernel for decode attention |

---

## 0. Two ideas you need first

### What the KV cache is

A transformer reads tokens one after another. For every token, in every layer, it computes a
**key** (K) and a **value** (V) vector. Later tokens "look back" at earlier tokens through these
K and V vectors. Computing them is the expensive part of reading a prompt (the *prefill*).

The **KV cache** keeps those K and V vectors in GPU memory, so the next request does not have to
recompute them. Qwen3-4B has 36 layers, and for each token it stores 8 key heads and 8 value heads
of 128 numbers per layer: about 147 KB per token. A 10,000-token conversation is about 1.5 GB of cache.

### The A · B · C · D picture

Every step, the agent sends the whole conversation again. We compare it with what is cached:

```
cached (last step):   [ A: system + task + early work ][ B: old test log ][ C: later work + last reply ]
new    (this step):   [ A: system + task + early work ]                   [ C: later work + last reply ][ D: new tool result ]
                                                        ^ pruning deleted B
```

- **A**: the shared start. Reused as is.
- **B**: what pruning removed. Thrown away.
- **C**: what comes after the cut and is still the same. A normal engine recomputes it. **We reuse it.**
- **D**: the genuinely new part. The only thing the model actually runs.

**Worked example** (real numbers from a pilot step): prompt 2,159 tokens.
A = 1,532 tokens reused, B = 12 tokens dropped, C = 461 tokens reused, D = 166 tokens computed.
A normal prefix cache would have computed C + D = 627 tokens. We computed 166.

---

## 1. `server.py`: the front door

Pi talks to us like it would talk to OpenAI: it POSTs to `/v1/chat/completions` with the full
message list. This is the heart of the handler (`run`, lines 44-74 of the file):

```python
def run(body):
    messages = normalize(body["messages"])            # 1
    tools = body.get("tools")                         # 2
    with lock:                                        # 3
        k, renderer = state["k"], state["renderer"]
        kept, n_pruned = prune(messages, k)           # 4
        full_tokens = len(renderer.render(messages, tools)) if n_pruned else None   # 5
        ids = renderer.render(kept, tools)            # 6
        out, stats = engine.generate(ids, ...)        # 7
        text = engine.tok.decode(out)                 # 8
        content, tool_calls, malformed = parse_reply(text)   # 9
        renderer.remember(content, tool_calls, out)   # 10
        ... write one line to the trace file ...      # 11
```

1. **`normalize`** cleans up the OpenAI message format (e.g. content given as a list of parts
   becomes one string; the `developer` role becomes `system`).
2. **`tools`** is the list of tools Pi offers the model (read, edit, bash, ...). They go into
   the prompt so the model knows how to call them.
3. **`lock`**: the engine has one cache for one conversation, so requests run one at a time.
4. **`prune`** applies the pruning rule and returns the shorter message list (section 2.2).
5. Only for the logs: how long the prompt *would* have been without pruning.
6. **`render`** turns messages into token ids (section 2.3).
7. **`engine.generate`** does all the real work: cache matching, surgery, compute D, decode the
   reply (section 4).
8-9. Turn the generated tokens back into text, and split out any tool calls
   (`<tool_call>{...}</tool_call>`) into the OpenAI format Pi expects.
10. **`remember`** stores the exact token ids of this reply (section 2.3 explains why this matters).
11. Every step appends a line with the stats (prompt, reused, computed, times). That file is what
    the charts and the viewer are built from.

`/admin/config` (lines 27-38) lets the Kaggle conductor switch modes between runs (pruning on or
off, suffix reuse on or off) and clears the cache, without reloading the 8 GB model.

---

## 2. `chat.py`: messages, pruning, and tool calls

### 2.1 Normalizing (lines 70-78)

```python
def normalize(messages):
    out = []
    for m in messages:
        role = "system" if m["role"] == "developer" else m["role"]
        msg = {"role": role, "content": text_of(m.get("content"))}
        if m.get("tool_calls"):
            msg["tool_calls"] = m["tool_calls"]
        out.append(msg)
    return out
```

Every message becomes `{role, content-as-string, tool_calls?}`. Nothing clever; it just makes the
rest of the code simpler.

### 2.2 The pruning rule (lines 81-103)

This is the TIM paper's "subtask stack" rule. A *subtask* is one tool call plus its result, e.g.
"read ledger/account.py" followed by the file's contents.

```python
def prune(messages, k):
    if k is None:                          # 89: no pruning mode
        return messages, 0
    groups, current = [], None
    for i, m in enumerate(messages):       # 92: walk the conversation
        if m["role"] == "assistant":
            current = [] if m.get("tool_calls") else None     # 94: a tool call starts a new subtask
            if current is not None:
                groups.append(current)
        elif m["role"] == "tool" and current is not None:
            current.append(i)              # 98: remember the index of each tool RESULT
    groups = [g for g in groups if g]
    completed = [g for g in groups if any(m["role"] == "assistant" for m in messages[g[-1] + 1:])]  # 100
    doomed = completed[:max(len(completed) - k, 0)]          # 101
    drop = {i for g in doomed for i in g}                    # 102
    return [m for i, m in enumerate(messages) if i not in drop], len(doomed)
```

- Lines 92-98 group the messages: each assistant message that calls a tool opens a group, and the
  tool-result messages after it are collected into that group (only their indexes).
- Line 100: a subtask is **completed** once the model has already replied after seeing its
  result. The newest result (which the model has not reacted to yet) is never pruned.
- Line 101: keep the newest `k` completed subtasks and mark all older ones for pruning. With
  `k=1`, only the most recent completed tool output survives.
- Line 102-103: delete the tool-**result** messages of the doomed subtasks. The assistant's tool
  *call* stays, so the model still sees "I ran pytest" even though the output is gone.

Why delete whole messages? In Qwen's chat format every message is wrapped in special tokens
(`<|im_start|>` ... `<|im_end|>`). Deleting a whole message removes exactly one contiguous run of
tokens and never changes how the text on either side is tokenized. That is what makes the
cached C still match (section 3).

### 2.3 Rendering, and why we remember reply ids (lines 106-130)

```python
class Renderer:
    def __init__(self, tokenizer, max_remembered=512):
        self.tok = tokenizer
        self.raw = {}  # reply_key -> generated token ids

    def remember(self, content, tool_calls, ids):
        self.raw[reply_key(content, tool_calls)] = list(ids)
        ...

    def render(self, messages, tools=None):
        msgs, slots = [], []
        for m in messages:
            ids = self.raw.get(reply_key(m["content"], m.get("tool_calls"))) if m["role"] == "assistant" else None
            if ids is None:
                msgs.append(m)
            else:
                msgs.append({"role": "assistant", "content": f"⁣{len(slots)}⁣"})   # a placeholder
                slots.append(ids)
        text = self.tok.apply_chat_template(msgs, tools=tools or None, add_generation_prompt=True, tokenize=False)
        out = []
        for i, part in enumerate(SLOT.split(text)):
            out += slots[int(part)] if i % 2 else self.tok.encode(part, add_special_tokens=False)
        return out
```

**The problem it solves.** The model generated, say,
`<tool_call>{"name": "read", "arguments": {"path": "a.py"}}</tool_call>`. Pi parses that, and
next step sends it back as structured JSON. When we turn it back into text, the spacing or key
order can differ slightly, so the **tokens differ**, and the cache stops matching at that point.

**The fix.** After each reply, `remember` stores the exact generated token ids, keyed by the
reply's text and tool calls (`reply_key` normalizes the JSON so formatting differences do not
matter). When rendering, any assistant message we recognize is swapped for an invisible
placeholder (`⁣` is an invisible separator character). The chat template renders the rest as
text, then we cut the text at the placeholders, tokenize the pieces, and paste the original ids
back in. Result: the history is byte-for-byte what the cache holds.

### 2.4 Parsing tool calls (lines 37-67)

```python
def _parse_call(body):
    try:
        obj = json.loads(body, strict=False)                     # normal case
        return obj["name"], json.dumps(obj.get("arguments", {})), True
    except (json.JSONDecodeError, KeyError, TypeError):
        name = re.search(r'"name"\s*:\s*"([^"]+)"', body)       # broken JSON: salvage the tool name
        if not name:
            return None
        args = re.search(r'"arguments"\s*:\s*(.*)\}\s*$', body, re.S)
        return name.group(1), args.group(1).strip() if args else "{}", False
```

The model sometimes writes **invalid JSON** when a tool argument contains code with quotes. The
first benchmark taught us that if we drop such a call, the reply looks like a final answer and Pi
ends the whole run. So we salvage the tool name, forward the raw arguments, and let Pi report
"bad arguments" back to the model, which can then retry. `parse_reply` runs this for every
`<tool_call>` block and counts how many were malformed.

---

## 3. `match.py`: finding A, B, C and D

This is the core idea, in 48 lines.

```python
@dataclass(frozen=True)
class Split:
    a: int   # len(A)
    c0: int  # C starts at cached[c0]; B = cached[a:c0]
    c: int   # len(C); D = new[a + c:]

    @property
    def gap(self):
        return self.c0 - self.a        # len(B)
```

A `Split` is just three numbers that describe the picture from section 0. With the worked
example: `a = 1532`, `c0 = 1544` (so B is 12 tokens), `c = 461`.

```python
def lcp(x, y):                                          # "longest common prefix"
    n = min(len(x), len(y))
    if n == 0:
        return 0
    diff = np.flatnonzero(np.asarray(x[:n]) != np.asarray(y[:n]))   # positions where they differ
    return int(diff[0]) if len(diff) else n             # first difference = length of the match
```

`lcp` compares two token lists position by position (NumPy does it in one vectorized step) and
returns how many tokens match from the start.

```python
def split(cached, new, min_suffix=32, suffix=True):
    a = min(lcp(cached, new), len(new) - 1)                                 # 35
    rest = new[a:]                                                          # 36
    if not suffix or len(rest) <= min_suffix or len(cached) - (a + 1) < min_suffix:
        return Split(a, a, 0)                                               # 37-38
    cached_arr = np.asarray(cached)
    windows = np.lib.stride_tricks.sliding_window_view(cached_arr[a + 1:], min_suffix)   # 41
    starts = np.flatnonzero((windows == np.asarray(rest[:min_suffix])).all(axis=1)) + a + 1   # 42
    best = Split(a, a, 0)
    for j in starts:                                                        # 44
        c = min(lcp(cached[j:], rest), len(rest) - 1)                       # 45
        if c > best.c:
            best = Split(a, int(j), c)
    return best
```

- **Line 35:** A is the common prefix. The `len(new) - 1` cap makes sure at least one new token
  is always computed, because we need the model's output for the last position to start
  generating.
- **Line 36:** `rest` is everything in the new request after A. It should begin with C.
- **Lines 37-38:** if suffix reuse is off (the "normal prefix cache" mode), or there is too little
  left to bother, return "A only". Everything after A gets recomputed.
- **Line 41:** C must appear somewhere in the cache after A. We take the first 32 tokens of `rest`
  as a fingerprint and look for it: `sliding_window_view` makes a view of every 32-token window
  of the cache (no copying).
- **Line 42:** compare every window with the fingerprint at once; `starts` are the cache
  positions where the fingerprint matches. Requiring 32 matching tokens avoids false matches on
  common short sequences.
- **Lines 44-47:** for each candidate start `j`, measure how far the match extends (`lcp` again)
  and keep the longest. That longest run is C, and `j` is where it sits in the cache (`c0`).

Everything before `c0` and after A is B (dropped). Everything in `new` after A + C is D.

---

## 4. `engine.py`: running the model

### 4.1 Setup (lines 21-37)

```python
self.model = model or AutoModelForCausalLM.from_pretrained(
    model_path, dtype=dtype, attn_implementation=attention.NAME).to(self.device)     # 26-27
...
assert rope.get("rope_type", "default") == "default", ...                            # 31
assert all(t == "full_attention" for t in (...layer_types...))                       # 32
self.inv_freq = self.model.model.rotary_emb.inv_freq                                 # 33
self.stop_ids = {self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|im_end|>")}  # 36
```

- **Lines 26-27:** load Qwen3-4B in fp16 with *our* attention function (section 6) instead of the
  default one.
- **Lines 31-32:** safety checks. The KV shift in section 5 is only exact for plain RoPE and full
  attention. A model with scaled RoPE, sliding windows or recurrent layers (like TIM-9B) would be
  refused instead of silently giving wrong answers.
- **Line 33:** `inv_freq` is the list of RoPE rotation speeds, one per pair of dimensions. The
  surgery needs it.
- **Line 36:** the tokens that mean "the model is done with its reply".

### 4.2 `load`: bring the cache up to date (lines 51-69)

```python
def load(self, new_ids):
    s = split(self.ids, new_ids, self.min_suffix, self.suffix) if self.cache is not None else None   # 53
    if s is None or (s.a == 0 and s.c == 0):
        self.cache, reused = new_cache(self.model.config), 0         # 55: nothing reusable, start fresh
    else:
        splice(self.cache, s, self.inv_freq)                         # 57: A·B·C -> A·C
        reused = s.a + s.c
    logits = self._forward(new_ids[reused:], self.cache)            # 59: compute ONLY D
    stats = {...}
    self.ids = list(new_ids)                                         # 68: the cache now holds new_ids
    return logits, stats
```

- `self.ids` is the list of tokens currently in the cache. `split` compares it with the new
  request (line 53).
- Line 57: `splice` edits the cache in GPU memory, removing B and moving C left (section 5).
- Line 59: this is the payoff. `new_ids[reused:]` is exactly D. The model runs on D only, on top
  of the cache that already holds A and C.
- The stats record how much was reused and computed; these become the numbers in the results.

### 4.3 `_forward`: prefill in chunks (lines 43-49)

```python
for i in range(0, len(ids), self.prefill_chunk):
    x = torch.tensor([ids[i:i + self.prefill_chunk]], device=self.device)
    logits = self.model(input_ids=x, past_key_values=cache, use_cache=True, logits_to_keep=1).logits[:, -1]
```

D can be thousands of tokens. Feeding 2,048 at a time bounds the peak GPU memory. Each call adds
those tokens' K and V to the cache (`use_cache=True`). `logits_to_keep=1` asks for the
prediction at the last position only, which avoids building a huge `[tokens x 151k vocabulary]`
array we would not use.

### 4.4 `generate`: the decode loop (lines 78-104)

```python
logits, stats = self.load(new_ids)                  # 84: prefill (only D)
...
for _ in range(max_new_tokens):
    nxt = self._pick(logits, temperature)           # 89: greedy = highest-probability token
    if nxt in self.stop_ids:
        finish = "stop"
        break
    out.append(nxt)
    x = torch.tensor([[nxt]], device=self.device)
    logits = self.model(input_ids=x, past_key_values=self.cache, use_cache=True).logits[:, -1]   # 95
...
self.ids += out                                     # 98: the reply is now in the cache too
```

This is standard autoregressive decoding, written by hand instead of calling `generate()` so we
can time the two phases separately: **TTFT** (time to first token, = prefill) at line 86 and
**decode time** at line 97. Each loop iteration feeds one token and appends its K and V to the
cache (line 95). Line 98 matters for the next step: the reply's tokens are cached, so next time
they are part of A or C.

---

## 5. `kv.py`: the surgery

### 5.1 The math of moving a key (lines 15-25)

Qwen uses **RoPE** (rotary position embeddings): before a key is cached, pairs of its dimensions
are rotated by an angle `position x speed`. That rotation is how the model knows where a token is.
So a key cached at position 1,544 has been rotated by `1544 x speed`.

When B (12 tokens) is removed, C's first token should now be at position 1,532. Rotations add up:
rotating by `1544 x speed` and then by `-12 x speed` is exactly the same as rotating by
`1532 x speed`. So we do not need to recompute anything; we just rotate the cached keys back.

```python
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)              # split the 128 dims into two halves
    return torch.cat((-x2, x1), dim=-1)      # the "90 degree turn" RoPE uses


def shift_keys(keys, delta, inv_freq):
    angle = delta * inv_freq.float().to(keys.device)     # how far to turn each dimension pair
    emb = torch.cat((angle, angle))                      # same angle for both halves
    k = keys.float()                                     # do the math in fp32 for accuracy
    return (k * emb.cos() + rotate_half(k) * emb.sin()).to(keys.dtype)   # the 2D rotation formula
```

The last line is the standard rotation formula `(x cos a - y sin a, x sin a + y cos a)`, written
the same way Qwen applies RoPE. Values (V) have no position in them, so they are moved as they are.
Test T1 checks this identity; test T2 checks the whole splice against a fresh computation.

**What the rotation cannot change:** C's K and V were computed while B was still there, so they
carry some information from B. That leftover is the "subconscious" memory the drift experiment
measures.

### 5.2 `splice` (lines 28-41)

```python
def splice(cache, s: Split, inv_freq):
    if all(isinstance(layer, GrowingLayer) for layer in cache.layers):   # fast path (5.3)
        for layer in cache.layers:
            layer.splice(s, inv_freq)
        return
    keep = torch.cat((torch.arange(0, s.a), torch.arange(s.c0, s.c0 + s.c)))  # positions of A and C
    for layer in cache.layers:                                          # all 36 layers
        k = layer.keys.index_select(2, keep)                            # keep A and C, drop B
        v = layer.values.index_select(2, keep)
        if s.c and s.gap:
            k[:, :, s.a:] = shift_keys(k[:, :, s.a:], -s.gap, inv_freq) # rotate C's keys left by |B|
        layer.keys, layer.values = k, v
```

For every layer: select the A and C slices, rotate C's keys by `-len(B)`, done. The cache tensors
have shape `[batch, heads, tokens, 128]`, so dimension 2 is "tokens".

### 5.3 `GrowingLayer`: a faster cache (lines 44-92)

The standard Hugging Face cache appends each new token with `torch.cat`, which **copies the
entire cache every token** (about 1.5 GB per token at 10k context). `GrowingLayer` instead keeps a
bigger preallocated buffer and writes new tokens into the free space:

```python
def _reserve(self, need, like):
    cap = 0 if self.kbuf is None else self.kbuf.shape[-2]       # current capacity in tokens
    if need <= cap:
        return                                                  # still room: nothing to do
    new = max(need, int(cap * 1.25), cap + 2048)                # grow by 25% (or at least 2,048)
    kb = like.new_empty((*like.shape[:-2], new, like.shape[-1]))
    vb = torch.empty_like(kb)
    if self.n:
        kb[..., :self.n, :] = self.kbuf[..., :self.n, :]        # copy once when growing
        vb[..., :self.n, :] = self.vbuf[..., :self.n, :]
    self.kbuf, self.vbuf = kb, vb

def update(self, key_states, value_states, *args, **kwargs):   # called by the model for each layer
    ...
    self._reserve(self.n + add, key_states)
    self.kbuf[..., self.n:self.n + add, :] = key_states         # write in place
    self.vbuf[..., self.n:self.n + add, :] = value_states
    self.n += add
    self._view()                                                # keys/values = buffer[:n]
    return self.keys, self.values
```

Growth is 1.25x rather than the usual 2x so that a 32k context still fits next to the 8 GB of
weights on a 16 GB T4. `_view` hands the model a *view* of the first `n` tokens (no copy).

And the in-place splice only touches C:

```python
def splice(self, s: Split, inv_freq):
    if s.c:
        k = self.kbuf[..., s.c0:s.c0 + s.c, :]
        k = shift_keys(k, -s.gap, inv_freq) if s.gap else k.clone()   # rotated copy of C's keys
        v = self.vbuf[..., s.c0:s.c0 + s.c, :].clone()
        self.kbuf[..., s.a:s.a + s.c, :] = k       # write C right after A (over B's old slots)
        self.vbuf[..., s.a:s.a + s.c, :] = v
    self.n = s.a + s.c                             # everything after A·C is now free space
    self._view()
```

A never moves, so a prune costs work proportional to C only, not to the whole cache.

---

## 6. `attention.py`: keeping attention fast on a T4

Qwen3-4B uses **grouped-query attention**: 32 query heads share 8 key/value heads (4 queries per
KV head, `g = 4`). Hugging Face asks PyTorch for a special "GQA" mode, which the fast kernels do not
support on T4-class GPUs, so PyTorch silently fell back to a slow path that copied and expanded
the whole cache every token. Decode speed collapsed as the context grew.

```python
def attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    b, hq, q, d = query.shape          # batch, 32 query heads, query length, 128
    hkv = key.shape[1]                 # 8 KV heads
    g = hq // hkv                      # 4 query heads per KV head
    scaling = scaling if scaling is not None else d ** -0.5
    if q == 1 and attention_mask is None:                    # DECODE: one new token
        if decode_attention is not None and query.is_cuda and b == 1:
            out = decode_attention(query.reshape(hkv, g, d), key[0], value[0], scaling).reshape(1, hq, 1, d)
        else:
            out = F.scaled_dot_product_attention(query.reshape(b, hkv, g, d), key, value,
                                                 scale=scaling).reshape(b, hq, 1, d)
    else:                                                    # PREFILL: many tokens at once
        is_causal = attention_mask is None and q > 1 and q == key.shape[2]
        out = F.scaled_dot_product_attention(query, repeat_kv(key, g), repeat_kv(value, g),
                                             attn_mask=attention_mask, scale=scaling, is_causal=is_causal)
    return out.transpose(1, 2).contiguous(), None

AttentionInterface.register(NAME, attention)                 # plug it into transformers
AttentionMaskInterface.register(NAME, sdpa_mask)
```

- **Decode (one token):** reshape the 32 query heads into 8 groups of 4, so each group lines up
  with its own KV head. Then use our Triton kernel (section 7), or, without Triton, standard SDPA
  on that reshaped query. Either way the cache is read as it is, never copied.
- **Prefill (many tokens):** here the cost is dominated by the tokens being computed, so we expand
  the KV heads once (`repeat_kv`) and use PyTorch's memory-efficient kernel with the causal mask.
- The last two lines register this function under the name `"threadcut"`, which is what
  `engine.py` passes as `attn_implementation`.

---

## 7. `flash_decode.py`: the GPU kernel

**Why a custom kernel.** With one query token, each KV head has only 4 query rows to process.
Library kernels give each KV head one block of GPU threads: 8 blocks for a GPU that has 40
multiprocessors, and each block walks through the whole 10k-token cache alone. Most of the GPU sits
idle. **Flash-decoding** splits the cache into slices and processes all slices in parallel, then
merges the results.

```python
@triton.jit
def _partial(Q, K, V, O, L, n, scale, ...strides..., G, GP, D, BLOCK, SPLIT):
    h = tl.program_id(0)                 # which KV head this program handles
    s = tl.program_id(1)                 # which slice of the cache
    q = tl.load(...)                     # the 4 query rows for head h
    m = tl.full([GP], float("-inf"), tl.float32)   # running max score
    l = tl.zeros([GP], tl.float32)                 # running sum of exp(score)
    acc = tl.zeros([GP, D], tl.float32)            # running weighted sum of values
    for off in range(start, start + SPLIT, BLOCK): # walk this slice, 16 tokens at a time
        k = tl.load(...)                            # 16 keys
        v = tl.load(...)                            # 16 values
        qk = tl.sum(q[:, None, :] * k[None, :, :], axis=2) * scale   # attention scores
        m_new = tl.maximum(m, tl.max(qk, 1))                         # online softmax:
        p = tl.exp(qk - m_new)                                       #   rescale as the max grows
        alpha = tl.exp(m - m_new)
        l = l * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.sum(p[:, :, None] * v[None, :, :], axis=1)
        m = m_new
    out = acc / l                                  # this slice's attention output
    lse = m + tl.log(l)                            # this slice's log-sum-exp (its total weight)
    tl.store(O ..., out)
    tl.store(L ..., lse)
```

(Simplified above; the real file also guards against empty slices with `m_safe`.)

- The launch grid is `(8 KV heads, number of slices)`. At 10k tokens with 256-token slices that
  is 8 x 40 = 320 programs instead of 8, so the whole GPU works.
- Inside, it is the **online softmax** trick: keep a running maximum `m`, a running sum `l` and a
  running output `acc`, and rescale them whenever a bigger score appears. This gives the exact
  softmax without ever storing all 10k scores.
- Each slice writes its partial output and its **log-sum-exp**, a single number that says how
  much weight that slice carries.

```python
def decode_attention(q, k, v, scale, split=256, block=16):
    ...
    _partial[(hkv, splits)](...)                   # run all slices in parallel
    w = torch.softmax(lse, dim=0)[..., None]       # each slice's share of the total weight
    return (o * w).sum(0).to(q.dtype)              # exact merge of the slices
```

The merge is exact: weighting each slice's output by `softmax(lse)` reproduces the full softmax
over all tokens. Measured result: matches PyTorch to 3e-5, and 3-6x faster at long context.

---

## 8. How the tests map to this

| Test | What it proves | Code it covers |
|---|---|---|
| `test_split_rule` | A/B/C/D split on small hand-made lists | `match.py` |
| T1 `test_T1_shift_is_rotation_composition` | rotating a key by `-d` = RoPE at `p - d` | `kv.shift_keys` |
| T2 `test_T2_splice_matches_fresh_compute` | drop B + shift C gives the same logits as computing A·C from scratch | `engine.load`, `kv.splice` |
| `test_growing_layer_splice_matches_copy_splice` | the fast in-place splice = the simple splice | `kv.GrowingLayer` |
| T3 `test_T3_no_pruning_matches_hf_generate` | with no pruning, output = stock transformers | whole engine incl. `attention.py`, `flash_decode.py` |
| `test_prune_*`, `test_render_*`, `test_malformed_*` | pruning rule, byte-identical history, tool-call salvage | `chat.py` |
| `test_agent_loop_hits_suffix_cache_after_prune` | a real multi-step loop lands suffix hits after each prune | everything together |

## Suggested reading order

1. `match.py` (48 lines): the idea.
2. `kv.py` `shift_keys` and `splice`: the trick.
3. `engine.py` `load` and `generate`: where they are used.
4. `chat.py` `prune` and `Renderer`: how agent messages become cache-friendly tokens.
5. `server.py` `run`: how it all connects to the agent.
6. `attention.py` and `flash_decode.py`: the performance work (optional on first read).
