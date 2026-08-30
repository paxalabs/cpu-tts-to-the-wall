"""The 'This work' configuration: tuned OV + compile cache + idle release.

Measures: cold start, lone latency, sustained c4, the idle decay THROUGH a
release (samples at 30/90/150/300/600 s after last traffic), the restore
latency (first request after a release), and post-restore sustained c4.
"""
import json, os, signal, subprocess, time, urllib.request
from pathlib import Path

W = Path("/work"); RES = W / "results_matrix.jsonl"; PORT = 9200
ENV = {"BACKEND": "ov", "OV_THREADS": "16", "OV_STREAMS": "4",
       "OV_CACHE_DIR": "/work/ovcache", "IDLE_RELEASE_SEC": "45"}
CFG = "ov-ours"

def log(row):
    row["ts"] = round(time.time(), 1)
    with open(RES, "a") as f: f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)

def rss_mb(pid):
    for line in open(f"/proc/{pid}/status"):
        if line.startswith("VmRSS"): return round(int(line.split()[1]) / 1024, 1)

def cgroup_mb():
    return round(int(open("/sys/fs/cgroup/memory.current").read()) / 1e6, 1)

def post_tts(text, timeout=600):
    body = json.dumps({"text": text}).encode()
    t0 = time.perf_counter()
    with urllib.request.urlopen(urllib.request.Request(
            f"http://127.0.0.1:{PORT}/tts", data=body,
            headers={"Content-Type": "application/json"}), timeout=timeout) as r:
        r.read()
        return {"wall_s": round(time.perf_counter() - t0, 3),
                "first_ms": float(r.headers["X-First-Audio-Ms"]),
                "audio_s": float(r.headers["X-Audio-S"])}

e = dict(os.environ); e.update(ENV); e["PORT2"] = str(PORT)
pidfile = W / "serve_ov.pid"
if pidfile.exists(): pidfile.unlink()
t0 = time.perf_counter()
proc = subprocess.Popen(["python", "serve_shim.py"], cwd=W, env=e,
                        stdout=open(W / "mx3.log", "w"), stderr=subprocess.STDOUT)
ready = None
for _ in range(1200):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
        ready = time.perf_counter() - t0; break
    except Exception: time.sleep(0.25)
first = post_tts("Welcome back.")
pid = int(pidfile.read_text())
log({"kind": "cold", "cfg": CFG, "ready_s": round(ready, 2),
     "cold_first_audio_s": round(ready + first["first_ms"] / 1e3, 2)})

TEXTS = [x["text"] for x in json.load(open(W / "corpus_texts.json"))]
for band, text in {"short": TEXTS[2], "medium": TEXTS[8], "long": TEXTS[14]}.items():
    runs = [post_tts(text) for _ in range(3)]
    log({"kind": "lone", "cfg": CFG, "band": band, "audio_s": runs[0]["audio_s"],
         "first_ms": sorted(r["first_ms"] for r in runs)[1]})

def sustained(tag, clients=4, secs=90, ramp=15):
    r = subprocess.run(["python", "loadgen_local.py", "--port", str(PORT), "--pid", str(pid),
                        "--clients", str(clients), "--secs", str(secs), "--ramp", str(ramp),
                        "--label", f"{CFG}-{tag}"],
                       cwd=W, capture_output=True, text=True, timeout=secs + 240)
    row = json.loads(r.stdout.strip().splitlines()[-1]); row["kind"] = "sustained"; row["cfg"] = CFG
    log(row)

sustained("c4a")
t_quiet = time.time()
for mark in (30, 90, 150, 300, 600):
    time.sleep(max(0, mark - (time.time() - t_quiet)))
    log({"kind": "idle", "cfg": CFG, "after_s": mark,
         "server_rss_mb": rss_mb(pid), "cgroup_mb": cgroup_mb()})
restore = post_tts(TEXTS[2])
log({"kind": "restore", "cfg": CFG, "first_ms": restore["first_ms"], "wall_s": restore["wall_s"]})
sustained("c4b")
proc.send_signal(signal.SIGTERM)
try: proc.wait(timeout=30)
except subprocess.TimeoutExpired: proc.kill()
log({"kind": "done-v3"})
