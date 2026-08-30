"""Second-architecture pass: full setup then a reduced matrix.
Run on a fresh container after the host rolls to an EPYC machine."""
import json, os, signal, subprocess, time, urllib.request
from pathlib import Path

W = Path("/work"); RES = W / "results_matrix_epyc.jsonl"; PORT = 9200

def log(row):
    row["ts"] = round(time.time(), 1)
    with open(RES, "a") as f: f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)

def sh(cmd, timeout=1800):
    r = subprocess.run(cmd, shell=True, cwd=W, capture_output=True, text=True, timeout=timeout)
    print(f"$ {cmd} -> {r.returncode}", flush=True)
    if r.returncode != 0:
        print(r.stdout[-1500:]); print(r.stderr[-1000:])
    return r.returncode

cpu = [l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
log({"kind": "host", "cpu": cpu, "cpus": os.cpu_count()})
assert sh("python assets_probe.py") == 0
assert sh("python corpus_build.py") == 0
assert sh("python corpus_texts.py") == 0

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
        return {"first_ms": float(r.headers["X-First-Audio-Ms"]),
                "audio_s": float(r.headers["X-Audio-S"])}

TEXTS = [x["text"] for x in json.load(open(W / "corpus_texts.json"))]

def run_cfg(name, env, sweeps, idle=False):
    e = dict(os.environ); e.update(env); e["PORT2"] = str(PORT)
    pidfile = W / f"serve_{env['BACKEND']}.pid"
    if pidfile.exists(): pidfile.unlink()
    t0 = time.perf_counter()
    proc = subprocess.Popen(["python", "serve_shim.py"], cwd=W, env=e,
                            stdout=open(W / f"ep_{name}.log", "w"), stderr=subprocess.STDOUT)
    ready = None
    for _ in range(1200):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
            ready = time.perf_counter() - t0; break
        except Exception: time.sleep(0.25)
    first = post_tts("Welcome back.")
    pid = int(pidfile.read_text())
    log({"kind": "cold", "cfg": name, "ready_s": round(ready, 2),
         "cold_first_audio_s": round(ready + first["first_ms"] / 1e3, 2)})
    for band, text in {"short": TEXTS[2], "medium": TEXTS[8], "long": TEXTS[14]}.items():
        runs = [post_tts(text) for _ in range(3)]
        log({"kind": "lone", "cfg": name, "band": band,
             "first_ms": sorted(r["first_ms"] for r in runs)[1], "audio_s": runs[0]["audio_s"]})
    for c in sweeps:
        r = subprocess.run(["python", "loadgen_local.py", "--port", str(PORT), "--pid", str(pid),
                            "--clients", str(c), "--secs", "90" if c == 4 else "60",
                            "--ramp", "15" if c == 4 else "12", "--label", f"ep-{name}-c{c}"],
                           cwd=W, capture_output=True, text=True, timeout=400)
        row = json.loads(r.stdout.strip().splitlines()[-1]); row["kind"] = "sustained"; row["cfg"] = name
        log(row)
    if idle:
        t_q = time.time()
        for mark in (30, 90, 300, 600):
            time.sleep(max(0, mark - (time.time() - t_q)))
            log({"kind": "idle", "cfg": name, "after_s": mark,
                 "server_rss_mb": rss_mb(pid), "cgroup_mb": cgroup_mb()})
        if "IDLE_RELEASE_SEC" in env:
            rst = post_tts(TEXTS[2])
            log({"kind": "restore", "cfg": name, "first_ms": rst["first_ms"]})
    proc.send_signal(signal.SIGTERM)
    try: proc.wait(timeout=30)
    except subprocess.TimeoutExpired: proc.kill()
    time.sleep(3)

run_cfg("pt-default", {"BACKEND": "pt"}, [4], idle=True)
run_cfg("ort-4t", {"BACKEND": "ort", "ORT_THREADS": "4"}, [4])
run_cfg("ov-8s", {"BACKEND": "ov", "OV_THREADS": "16", "OV_STREAMS": "8"}, [1, 4, 8, 16])
run_cfg("ov-ours8", {"BACKEND": "ov", "OV_THREADS": "16", "OV_STREAMS": "8",
                     "OV_CACHE_DIR": "/work/ovcache", "IDLE_RELEASE_SEC": "45"}, [4], idle=True)
log({"kind": "done-epyc"})
