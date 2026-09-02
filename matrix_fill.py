"""Fill pass for the three unmeasured table cells, same protocol as
matrix_main.py (cold start -> lone probes -> sustained c4 x 90 s -> idle
samples at +60/+300/+600 s), on one fresh container.

Targets:  ort-4t idle (Table 1), ov-cache idle (Table 2), ov-8s cold (Table 1),
ov-8s-cache cold-cached (Table 2, '+ 8 streams' after '+ compile cache').
Anchors (already reported, remeasured here for same-container comparison):
ort-default idle, ov-4t4s idle + cold, ov-cache cold/cold-cached.
Every row is printed as one JSON line prefixed with ROW.
"""
import json, os, signal, subprocess, time, urllib.request
from pathlib import Path

W = Path("/work")
RES = W / "results_fill.jsonl"
PORT = 9200
WANT_CPU = "8581C"

def log(row):
    row["ts"] = round(time.time(), 1)
    with open(RES, "a") as f:
        f.write(json.dumps(row) + "\n")
    print("ROW " + json.dumps(row), flush=True)

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

def rd(p):
    try: return open(p).read().strip()
    except OSError: return None

cpu = [l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
log({"kind": "host", "cpu": cpu, "cpus": os.cpu_count(),
     "cpu_max": rd("/sys/fs/cgroup/cpu.max"), "memory_max": rd("/sys/fs/cgroup/memory.max"),
     "kernel": os.uname().release})
if WANT_CPU not in cpu:
    print(f"WRONG-HOST {cpu}", flush=True)
    while True: time.sleep(3600)

def sh(cmd, timeout=1800):
    r = subprocess.run(cmd, shell=True, cwd=W, capture_output=True, text=True, timeout=timeout)
    print(f"$ {cmd} -> {r.returncode}", flush=True)
    if r.returncode != 0:
        print(r.stdout[-1500:]); print(r.stderr[-1500:], flush=True)
    return r.returncode

assert sh("python assets_probe.py") == 0
assert sh("python corpus_build.py") == 0
assert sh("python corpus_texts.py") == 0
print("SETUP-DONE", flush=True)

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
    ("ort-default", {"BACKEND": "ort"}, {}),
    ("ort-4t",      {"BACKEND": "ort", "ORT_THREADS": "4"}, {}),
    ("ov-4t4s",     {"BACKEND": "ov", "OV_THREADS": "16", "OV_STREAMS": "4"}, {}),
    ("ov-cache",    {"BACKEND": "ov", "OV_THREADS": "16", "OV_STREAMS": "4",
                     "OV_CACHE_DIR": "/work/ovcache"}, {"repeat_start": True}),
    ("ov-8s",       {"BACKEND": "ov", "OV_THREADS": "16", "OV_STREAMS": "8"}, {}),
    ("ov-8s-cache", {"BACKEND": "ov", "OV_THREADS": "16", "OV_STREAMS": "8",
                     "OV_CACHE_DIR": "/work/ovcache8"}, {"repeat_start": True}),
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
    idle_decay(name, pid)
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
    time.sleep(3)

log({"kind": "done-fill"})
print("FILL-DONE", flush=True)
for line in open(RES):
    print("FINAL " + line.rstrip(), flush=True)
while True: time.sleep(3600)
