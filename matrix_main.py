"""The measurement matrix, run as one on-box job. Every row is a direct
measurement appended to /work/results_matrix.jsonl.

Per configuration: cold start (spawn -> first audio), lone-latency probes,
sustained closed loop (4 clients x 90 s), idle decay samples at +60/+300/
+600 s after last traffic. OV runs twice at the end to measure the
compile-cache restart.
"""
import json, os, signal, subprocess, time, urllib.request
from pathlib import Path

W = Path("/work")
RES = W / "results_matrix.jsonl"
PORT = 9200

def log(row):
    row["ts"] = round(time.time(), 1)
    with open(RES, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)

def cpu_seconds(pid):
    parts = open(f"/proc/{pid}/stat").read().split()
    return (int(parts[13]) + int(parts[14])) / 100.0

def rss_mb(pid):
    for line in open(f"/proc/{pid}/status"):
        if line.startswith("VmRSS"):
            return round(int(line.split()[1]) / 1024, 1)
    return None

def cgroup_mb():
    try:
        return round(int(open("/sys/fs/cgroup/memory.current").read()) / 1e6, 1)
    except OSError:
        return None

def post_tts(text, timeout=600):
    body = json.dumps({"text": text}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/tts", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        r.read()
        return {"wall_s": round(time.perf_counter() - t0, 3),
                "first_ms": float(r.headers["X-First-Audio-Ms"]),
                "audio_s": float(r.headers["X-Audio-S"])}

def start_server(name, env):
    e = dict(os.environ); e.update(env); e["PORT2"] = str(PORT)
    pidfile = W / f"serve_{env['BACKEND']}.pid"
    if pidfile.exists(): pidfile.unlink()
    proc = subprocess.Popen(["python", "serve_shim.py"], cwd=W, env=e,
                            stdout=open(W / f"mx_{name}.log", "w"), stderr=subprocess.STDOUT)
    t0 = time.perf_counter()
    ready = None
    for _ in range(1200):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
            ready = time.perf_counter() - t0
            break
        except Exception:
            time.sleep(0.25)
    first = post_tts("Welcome back.")
    return proc, {"ready_s": round(ready, 2) if ready else None,
                  "cold_first_audio_s": round(ready + first["first_ms"] / 1e3, 2) if ready else None}

TEXTS = [x["text"] for x in json.load(open(W / "corpus_texts.json"))]
PROBES = {"short": TEXTS[2], "medium": TEXTS[8], "long": TEXTS[14]}

def lone_probes(cfg):
    for band, text in PROBES.items():
        runs = [post_tts(text) for _ in range(3)]
        log({"kind": "lone", "cfg": cfg, "band": band,
             "audio_s": runs[0]["audio_s"],
             "first_ms": sorted(r["first_ms"] for r in runs)[1],
             "wall_s": sorted(r["wall_s"] for r in runs)[1]})

def sustained(cfg, pid, clients=4, secs=90, ramp=15):
    r = subprocess.run(["python", "loadgen_local.py", "--port", str(PORT), "--pid", str(pid),
                        "--clients", str(clients), "--secs", str(secs), "--ramp", str(ramp),
                        "--label", f"{cfg}-c{clients}"],
                       cwd=W, capture_output=True, text=True, timeout=secs + 240)
    line = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "{}"
    row = json.loads(line); row["kind"] = "sustained"; row["cfg"] = cfg
    log(row)

def idle_decay(cfg, pid):
    t0 = time.time()
    for mark in (60, 300, 600):
        time.sleep(max(0, mark - (time.time() - t0)))
        log({"kind": "idle", "cfg": cfg, "after_s": mark,
             "server_rss_mb": rss_mb(pid), "cgroup_mb": cgroup_mb()})

CONFIGS = [
    ("pt-default",  {"BACKEND": "pt"}, {}),
    ("pt-4t",       {"BACKEND": "pt", "TORCH_THREADS": "4"}, {}),
    ("ort-default", {"BACKEND": "ort"}, {}),
    ("ort-4t",      {"BACKEND": "ort", "ORT_THREADS": "4"}, {}),
    ("ov-default",  {"BACKEND": "ov"}, {}),
    ("ov-4t4s",     {"BACKEND": "ov", "OV_THREADS": "16", "OV_STREAMS": "4"}, {}),
    ("ov-cache",    {"BACKEND": "ov", "OV_THREADS": "16", "OV_STREAMS": "4",
                     "OV_CACHE_DIR": "/work/ovcache"}, {"repeat_start": True}),
]

for name, env, opts in CONFIGS:
    proc, cold = start_server(name, env)
    pid = int((W / f"serve_{env['BACKEND']}.pid").read_text())
    log({"kind": "cold", "cfg": name, **cold})
    if opts.get("repeat_start"):
        proc.send_signal(signal.SIGTERM); proc.wait(timeout=30)
        proc, cold2 = start_server(name, env)
        pid = int((W / f"serve_{env['BACKEND']}.pid").read_text())
        log({"kind": "cold-cached", "cfg": name, **cold2})
    lone_probes(name)
    sustained(name, pid, clients=4, secs=90)
    if name in ("ort-4t", "ov-4t4s"):
        for c in (1, 2, 8, 16):
            sustained(name, pid, clients=c, secs=60, ramp=12)
    if name in ("pt-default", "ort-default", "ov-default", "ov-4t4s"):
        idle_decay(name, pid)
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
    time.sleep(3)

log({"kind": "done"})
