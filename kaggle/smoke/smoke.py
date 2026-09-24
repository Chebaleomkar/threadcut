import subprocess, time, sys
def sh(c): print(f"$ {c}\n" + subprocess.run(c, shell=True, capture_output=True, text=True).stdout[-3000:], flush=True)
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
sh("python -c 'import torch, transformers; print(torch.__version__, transformers.__version__, torch.version.cuda)'")
sh("node --version; npm --version")
sh("npm i -g @earendil-works/pi-coding-agent@0.84.3 2>&1 | tail -3; pi --version")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
M = "Qwen/Qwen3-4B-Instruct-2507"
t = time.time()
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float16, device_map="cuda:0")
print("load s", time.time() - t, "mem GB", torch.cuda.memory_allocated() / 1e9, flush=True)
ids = tok.apply_chat_template([{"role": "user", "content": "Write a python function that reverses a string."}], add_generation_prompt=True, return_tensors="pt").to("cuda:0")
if not torch.is_tensor(ids): ids = ids["input_ids"]
t = time.time(); out = model.generate(ids, max_new_tokens=128, do_sample=False); dt = time.time() - t
n = out.shape[1] - ids.shape[1]
print(tok.decode(out[0, ids.shape[1]:]), f"\n{n} tok in {dt:.1f}s = {n/dt:.1f} tok/s", flush=True)
x = torch.randint(0, 1000, (1, 4096), device="cuda:0")
torch.cuda.synchronize(); t = time.time()
with torch.no_grad(): lg = model(x).logits
torch.cuda.synchronize(); print("prefill 4096 tok s", time.time() - t, "finite", torch.isfinite(lg).all().item())
print(model.config.rope_parameters if hasattr(model.config, "rope_parameters") else model.config.rope_theta, model.config.head_dim, model.config.num_key_value_heads, model.config.num_hidden_layers)
