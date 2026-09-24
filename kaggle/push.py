"""Ship the current source to Kaggle as a private dataset, then push and start a kernel.

Usage: python kaggle/push.py run      (kernel folder under kaggle/, e.g. run)
"""
import json
import pathlib
import subprocess
import sys
import tarfile
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
USER = "omkarchebale"
DATASET = f"{USER}/threadcut-src"
SHIP = ["threadcut", "experiments", "tasks", "tests", "pi-config/models.json", "pi-config/settings.json",
        "pyproject.toml", "run_config.json"]


def kaggle(*args):
    r = subprocess.run([sys.executable, "-m", "kaggle", *args], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    print((r.stdout + r.stderr).strip())
    return r


def build(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    version = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    (ROOT / "VERSION").write_text(version + ("-dirty" if dirty else "") + "\n")
    skip = lambda ti: None if ("__pycache__" in ti.name or ".pytest_cache" in ti.name) else ti
    with tarfile.open(out_dir / "src.bundle", "w:gz") as t:
        for p in SHIP + ["VERSION"]:
            t.add(ROOT / p, arcname=p, filter=skip)
    (ROOT / "VERSION").unlink()
    json.dump({"title": "threadcut-src", "id": DATASET, "licenses": [{"name": "other"}]},
              open(out_dir / "dataset-metadata.json", "w"))


def main():
    kernel = ROOT / "kaggle" / sys.argv[1]
    ds = ROOT / "kaggle" / "_dataset"
    build(ds)
    r = kaggle("datasets", "version", "-p", str(ds), "-m", "update", "-q")
    if r.returncode or "not found" in (r.stdout + r.stderr).lower():
        kaggle("datasets", "create", "-p", str(ds), "-q")
    for _ in range(60):
        if "ready" in kaggle("datasets", "status", DATASET).stdout.lower():
            break
        time.sleep(10)
    kaggle("kernels", "push", "-p", str(kernel))


if __name__ == "__main__":
    main()
