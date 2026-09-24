"""Kaggle 2xT4 job: correctness tests, the Pi agent benchmark on both GPUs, then the drift replay.

Pushed by kaggle/push.py. Source arrives as the private dataset omkarchebale/threadcut-src.
Everything the job produces lands in /kaggle/working/results.
"""
import json
import os
import queue
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request

WORK = "/kaggle/working"
SRC = "/kaggle/tmp/threadcut"
RES = f"{WORK}/results"
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_DIR = "/kaggle/tmp/Qwen3-4B"
SMALL_DIR = "/kaggle/tmp/Qwen3-0.6B"
NODE = "v22.20.0"
PI_TIMEOUT = 900
RESULTS_LOCK = threading.Lock()
MODES = [  # name, k (None = no pruning), suffix reuse
    ("full", None, True),
    ("prune_prefix", 1, False),
    ("prune_suffix", 1, True),
]


def sh(cmd, **kw):
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True, **kw)
    print((r.stdout + r.stderr)[-4000:], flush=True)
    return r


def setup():
    os.makedirs(RES, exist_ok=True)
    bundle = next(os.path.join(d, f) for d, _, fs in os.walk("/kaggle/input") for f in fs if f == "src.bundle")
    with tarfile.open(bundle, "r:gz") as t:
        t.extractall(SRC)
    print("source version:", open(f"{SRC}/VERSION").read().strip(), flush=True)
    sh("pip install -q transformers==5.12.1 fastapi uvicorn 2>&1 | tail -2")
    from huggingface_hub import snapshot_download
    pats = ["*.json", "*.safetensors", "*.txt", "*.jinja"]
    snapshot_download(MODEL_ID, local_dir=MODEL_DIR, allow_patterns=pats)
    snapshot_download("Qwen/Qwen3-0.6B", local_dir=SMALL_DIR, allow_patterns=pats)
    shutil.copytree(MODEL_DIR, f"{SRC}/models/Qwen3-4B-tok", ignore=shutil.ignore_patterns("*.safetensors"))
    tar = f"/kaggle/tmp/node.tar.xz"
    urllib.request.urlretrieve(f"https://nodejs.org/dist/{NODE}/node-{NODE}-linux-x64.tar.xz", tar)
    sh(f"tar -xf {tar} -C /kaggle/tmp")
    os.environ["PATH"] = f"/kaggle/tmp/node-{NODE}-linux-x64/bin:" + os.environ["PATH"]
    sh("npm i -g @earendil-works/pi-coding-agent@0.84.3 2>&1 | tail -1; pi --version")


def tests():
    r = sh(f"cd {SRC} && THREADCUT_TEST_MODEL={SMALL_DIR} python -m pytest -q -p no:cacheprovider tests 2>&1 | tail -15")
    open(f"{RES}/tests.txt", "w").write(r.stdout)


def start_server(gpu, port):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    log = open(f"{RES}/server{gpu}.log", "w")
    proc = subprocess.Popen([sys.executable, "-m", "threadcut.server", "--model", MODEL_DIR, "--port", str(port),
                             "--max-ctx", "32768"], cwd=SRC, env=env, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(300):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2)
            return proc
        except Exception:
            time.sleep(2)
    raise RuntimeError(f"server on gpu {gpu} did not start")


def post(port, path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def pi_config(port):
    d = f"/kaggle/tmp/pi-{port}"
    os.makedirs(d, exist_ok=True)
    models = json.load(open(f"{SRC}/pi-config/models.json"))
    models["providers"]["threadcut"]["baseUrl"] = f"http://127.0.0.1:{port}/v1"
    json.dump(models, open(f"{d}/models.json", "w"))
    shutil.copy(f"{SRC}/pi-config/settings.json", d)
    return d


def run_job(port, pidir, task, mode, k, suffix):
    name = f"{mode}__{task}"
    trace, dump = f"{RES}/traces/{name}.jsonl", f"{RES}/dumps/{name}.jsonl"
    post(port, "/admin/config", {"k": k, "suffix": suffix, "trace": trace, "dump": dump})
    wd = f"/kaggle/tmp/work/{name}"
    shutil.copytree(f"{SRC}/tasks/{task}", wd)
    prompt = open(f"{wd}/prompt.txt").read().strip()
    env = {**os.environ, "PI_CODING_AGENT_DIR": pidir, "PI_OFFLINE": "1"}
    t0 = time.time()
    with open(f"{RES}/pi/{name}.jsonl", "w") as out:
        try:
            p = subprocess.run(["pi", "--model", "threadcut/qwen3-4b", "--mode", "json", "-p", prompt], cwd=wd,
                               env=env, stdout=out, stderr=subprocess.STDOUT, timeout=PI_TIMEOUT)
            status = f"exit {p.returncode}"
        except subprocess.TimeoutExpired:
            status = "timeout"
    wall = time.time() - t0
    passed = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=wd,
                            capture_output=True).returncode == 0
    steps = [json.loads(line) for line in open(trace)] if os.path.exists(trace) else []
    rec = {"task": task, "mode": mode, "k": k, "suffix": suffix, "passed": passed, "pi_status": status,
           "wall_s": wall, "steps": len(steps),
           "prompt_tokens": sum(s["prompt_tokens"] for s in steps),
           "full_prompt_tokens": sum(s["full_prompt_tokens"] for s in steps),
           "computed_tokens": sum(s["computed_tokens"] for s in steps),
           "reused_tokens": sum(s["reused_tokens"] for s in steps),
           "ttft_s": sum(s["ttft_s"] for s in steps), "decode_s": sum(s["decode_s"] for s in steps),
           "completion_tokens": sum(s["completion_tokens"] for s in steps),
           "peak_kv_tokens": max((s["kv_tokens"] for s in steps), default=0),
           "peak_prompt_tokens": max((s["prompt_tokens"] for s in steps), default=0)}
    with open(f"{RES}/results.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def benchmark():
    for d in ("traces", "dumps", "pi"):
        os.makedirs(f"{RES}/{d}", exist_ok=True)
    cfg = json.load(open(f"{SRC}/run_config.json"))
    tasks = sorted(t for t in os.listdir(f"{SRC}/tasks") if os.path.exists(f"{SRC}/tasks/{t}/prompt.txt"))
    tasks = [t for t in tasks if t in cfg.get("tasks", tasks)]
    modes = [m for m in MODES if m[0] in cfg.get("modes", [m[0] for m in MODES])]
    print("tasks", tasks, "modes", [m[0] for m in modes], flush=True)
    jobs = queue.Queue()
    for task in tasks:
        for mode in modes:
            jobs.put((task, *mode))
    servers = [(g, 8000 + g, start_server(g, 8000 + g)) for g in (0, 1)]

    def worker(port):
        pidir = pi_config(port)
        while True:
            try:
                job = jobs.get_nowait()
            except queue.Empty:
                return
            try:
                run_job(port, pidir, *job)
            except Exception as e:  # keep the other jobs going; record the failure
                print("JOB FAILED", job, repr(e), flush=True)

    threads = [threading.Thread(target=worker, args=(port,)) for _, port, _ in servers]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for _, _, proc in servers:
        proc.terminate()
        proc.wait()


def drift():
    dumps = sorted(f"{RES}/dumps/{f}" for f in os.listdir(f"{RES}/dumps") if f.startswith("prune_suffix__"))
    if dumps:
        sh(f"cd {SRC} && python -m experiments.drift --model {MODEL_DIR} --k 1 --n 64 "
           f"--out {RES}/drift.jsonl {' '.join(dumps)} 2>&1 | tail -40")


if __name__ == "__main__":
    t0 = time.time()
    setup()
    tests()
    benchmark()
    drift()
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)
