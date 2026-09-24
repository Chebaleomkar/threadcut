"""Build a self-contained HTML viewer for a benchmark results folder.

Pick any of the agent runs on the left and step through what the agent said, which tools it
called, what came back, and what the engine's cache did on that step (reused vs computed, what
was pruned). Each Pi turn is one model call, so turn i lines up with engine trace step i.

Usage: python -m experiments.viewer runs/full/results runs/full/viewer.html
"""
import html
import json
import pathlib
import sys

CLIP = 2500


def clip(s):
    s = s or ""
    return s if len(s) <= CLIP else s[:CLIP] + f"\n... [{len(s) - CLIP:,} more chars]"


def texts(content):
    if isinstance(content, str):
        return content
    return "\n".join(p.get("text", "") for p in content or [] if isinstance(p, dict) and p.get("type") == "text")


def load_run(res, rec):
    name = f"{rec['mode']}__{rec['task']}"
    trace = [json.loads(l) for l in open(res / "traces" / f"{name}.jsonl", encoding="utf-8")] \
        if (res / "traces" / f"{name}.jsonl").exists() else []
    turns = []
    pi = res / "pi" / f"{name}.jsonl"
    for line in open(pi, encoding="utf-8") if pi.exists() else []:
        if not line.startswith("{"):
            continue
        e = json.loads(line)
        if e.get("type") != "turn_end":
            continue
        msg = e.get("message") or {}
        content = msg.get("content") or []
        calls = [{"name": c.get("name"), "args": clip(json.dumps(c.get("arguments"), indent=1))}
                 for c in content if isinstance(c, dict) and c.get("type") == "toolCall"]
        results = [{"name": r.get("toolName"), "error": r.get("isError"), "text": clip(texts(r.get("content")))}
                   for r in e.get("toolResults") or []]
        turns.append({"text": clip(texts(content)), "calls": calls, "results": results})
    for i, t in enumerate(turns):
        s = trace[i] if i < len(trace) else {}
        t["stats"] = {k: s.get(k) for k in ("prompt_tokens", "full_prompt_tokens", "reused_prefix", "reused_suffix",
                                            "dropped_from_cache", "computed_tokens", "subtasks_pruned",
                                            "completion_tokens", "ttft_s", "decode_s", "malformed_calls")}
    return {"id": name, **rec, "turns": turns}


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>threadcut runs</title>
<style>
:root{--bg:#fcfcfb;--panel:#f3f2ef;--ink:#0b0b0b;--muted:#52514e;--line:#e4e3df;--pass:#008300;--fail:#e34948;
--full:#2a78d6;--prefix:#eb6834;--suffix:#1baf7a;--code:#f7f6f3}
@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--panel:#232322;--ink:#fff;--muted:#c3c2b7;--line:#34332f;
--pass:#2bb52b;--fail:#e66767;--full:#3987e5;--prefix:#d95926;--suffix:#199e70;--code:#121211}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif;display:flex;height:100vh}
nav{width:300px;min-width:240px;border-right:1px solid var(--line);overflow:auto;padding:12px;background:var(--panel)}
nav h1{font-size:15px;margin:0 0 4px}nav p{color:var(--muted);font-size:12px;margin:0 0 12px}
.task{margin:10px 0 4px;font-weight:600;font-size:13px}
.run{display:flex;justify-content:space-between;gap:6px;padding:6px 8px;border-radius:6px;cursor:pointer;border:1px solid transparent}
.run:hover{border-color:var(--line)}.run.sel{background:var(--bg);border-color:var(--muted)}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}
.pass{color:var(--pass)}.fail{color:var(--fail)}.meta{color:var(--muted);font-size:12px}
main{flex:1;overflow:auto;padding:16px 22px}
.cards{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0 14px}.card{background:var(--panel);border-radius:8px;padding:8px 12px}
.card b{display:block;font-size:17px}.card span{color:var(--muted);font-size:12px}
.step{border:1px solid var(--line);border-radius:8px;margin:10px 0;padding:10px 12px}
.step h3{margin:0 0 6px;font-size:13px;display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap}
.bar{height:8px;border-radius:4px;background:var(--line);overflow:hidden;display:flex;margin:4px 0 8px}
.bar i{display:block;height:100%}
pre{background:var(--code);border:1px solid var(--line);border-radius:6px;padding:8px;white-space:pre-wrap;word-break:break-word;
margin:6px 0;font:12px/1.4 ui-monospace,Consolas,monospace;max-height:320px;overflow:auto}
.lbl{font-size:12px;color:var(--muted);margin-top:6px}.err{border-color:var(--fail)}
.legend{font-size:12px;color:var(--muted)}.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 4px 0 10px;vertical-align:-1px}
@media (max-width:700px){body{flex-direction:column}nav{width:auto;max-height:40vh}}
</style></head><body>
<nav><h1>threadcut: agent runs</h1><p>Pi + Qwen3-4B on a Kaggle T4. Keys: j / k to switch runs.</p><div id="list"></div></nav>
<main id="main"></main>
<script>
const RUNS = __DATA__;
const MODE = {full:["No pruning","var(--full)"], prune_prefix:["Pruning + prefix cache","var(--prefix)"], prune_suffix:["Pruning + suffix reuse","var(--suffix)"]};
const esc = s => (s ?? "").toString().replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const fmt = n => n == null ? "–" : Number(n).toLocaleString();
let order = [], cur = 0;
function list(){
  const tasks = [...new Set(RUNS.map(r => r.task))].sort(); let h = "";
  for (const t of tasks){
    h += `<div class="task">${esc(t)}</div>`;
    for (const m of Object.keys(MODE)){
      const r = RUNS.find(x => x.task === t && x.mode === m); if (!r) continue;
      order.push(r.id);
      h += `<div class="run" data-id="${r.id}"><span><span class="dot" style="background:${MODE[m][1]}"></span>${MODE[m][0]}</span>
            <span class="${r.passed ? "pass" : "fail"}">${r.passed ? "pass" : (r.pi_status === "timeout" ? "timeout" : "fail")}</span></div>`;
    }
  }
  document.getElementById("list").innerHTML = h;
  document.querySelectorAll(".run").forEach(el => el.onclick = () => show(order.indexOf(el.dataset.id)));
}
function show(i){
  cur = Math.max(0, Math.min(order.length - 1, i)); const r = RUNS.find(x => x.id === order[cur]);
  document.querySelectorAll(".run").forEach(el => el.classList.toggle("sel", el.dataset.id === r.id));
  const reuse = r.prompt_tokens ? (100 * r.reused_tokens / r.prompt_tokens).toFixed(1) + "%" : "–";
  let h = `<h2 style="margin:0">${esc(r.task)} · <span style="color:${MODE[r.mode][1]}">${MODE[r.mode][0]}</span></h2>
    <div class="cards">
      <div class="card"><b class="${r.passed ? "pass" : "fail"}">${r.passed ? "Passed" : "Failed"}</b><span>${esc(r.pi_status)}</span></div>
      <div class="card"><b>${r.steps}</b><span>agent steps</span></div>
      <div class="card"><b>${fmt(Math.round(r.wall_s))} s</b><span>wall time</span></div>
      <div class="card"><b>${fmt(r.prompt_tokens)}</b><span>tokens in prompts</span></div>
      <div class="card"><b>${fmt(r.computed_tokens)}</b><span>tokens computed</span></div>
      <div class="card"><b>${reuse}</b><span>served from cache</span></div>
      <div class="card"><b>${fmt(r.peak_prompt_tokens)}</b><span>peak context</span></div></div>
    <div class="legend">Per step bar:<i style="background:var(--full)"></i>reused prefix<i style="background:var(--suffix)"></i>reused suffix<i style="background:var(--prefix)"></i>computed</div>`;
  r.turns.forEach((t, j) => {
    const s = t.stats || {}, P = s.prompt_tokens || 1;
    const w = x => (100 * (x || 0) / P).toFixed(2) + "%";
    h += `<div class="step"><h3><span>Step ${j + 1}</span><span class="meta">prompt ${fmt(s.prompt_tokens)}
      ${s.full_prompt_tokens && s.full_prompt_tokens !== s.prompt_tokens ? `(unpruned ${fmt(s.full_prompt_tokens)})` : ""}
      · computed ${fmt(s.computed_tokens)} · pruned subtasks ${fmt(s.subtasks_pruned)} · dropped from cache ${fmt(s.dropped_from_cache)}
      · prefill ${s.ttft_s != null ? s.ttft_s.toFixed(2) + " s" : "–"} · decode ${s.decode_s != null ? s.decode_s.toFixed(1) + " s" : "–"}
      ${s.malformed_calls ? " · <span class=fail>malformed tool call</span>" : ""}</span></h3>
      <div class="bar"><i style="width:${w(s.reused_prefix)};background:var(--full)"></i><i style="width:${w(s.reused_suffix)};background:var(--suffix)"></i><i style="width:${w(s.computed_tokens)};background:var(--prefix)"></i></div>
      ${t.text ? `<div class="lbl">Agent</div><pre>${esc(t.text)}</pre>` : ""}
      ${t.calls.map(c => `<div class="lbl">Tool call: ${esc(c.name)}</div><pre>${esc(c.args)}</pre>`).join("")}
      ${t.results.map(x => `<div class="lbl">Result: ${esc(x.name)}${x.error ? " (error)" : ""}</div><pre class="${x.error ? "err" : ""}">${esc(x.text)}</pre>`).join("")}
    </div>`;
  });
  const m = document.getElementById("main"); m.innerHTML = h; m.scrollTop = 0;
}
document.addEventListener("keydown", e => { if (e.key === "j") show(cur + 1); if (e.key === "k") show(cur - 1); });
list(); show(0);
</script></body></html>"""


def main():
    res, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    recs = [json.loads(l) for l in open(res / "results.jsonl", encoding="utf-8")]
    runs = [load_run(res, r) for r in recs]
    data = json.dumps(runs).replace("</", "<\\/")
    out.write_text(PAGE.replace("__DATA__", data), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 2**20:.1f} MB, {len(runs)} runs, {sum(len(r['turns']) for r in runs)} steps)")


if __name__ == "__main__":
    main()
