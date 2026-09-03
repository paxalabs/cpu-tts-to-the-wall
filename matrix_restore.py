"""Repeat measurement of the restore-from-released-state latency for the
'This work' configuration (ov-ours8: OpenVINO, 16 threads, 8 streams,
compile cache, 45 s idle release), with same-container process cold
starts as comparators.

Same shim, texts, request path and release threshold as
matrix_lifecycle_8streams.py. Two deliberate differences: the number of
release-and-restore cycles (N_CYCLES instead of one), and the quiet
period per cycle (until the release is observed, plus SETTLE_S, instead
of the 600 s idle-decay window whose memory samples this run does not
need; the released state does not change after the release).

Protocol (one fresh container, host gated on WANT_CPU):
  host row
  setup: assets_probe, corpus_build, corpus_texts (as matrix_main.py)
  cold         x1  process cold start, empty compile cache
  cold-cached  x3  process cold start, populated cache, fresh process each
  on the last process:
    lone probes (3x short) as warm-up
    sustained c4 x 90 s, so the first release follows load as in the
    original run
    N_CYCLES x:
      wait for the release (the shim prints "released" when it drops the
      compiled model; the release thread fires 45 s after the last
      request, checked every 5 s), then SETTLE_S
      restore   one short request (TEXTS[2]) to the released server; the
                shim prints "restored" when the request rebuilt the model,
                and the row records that both lines were seen
      warm      the same request again, immediately
Every row is printed as one JSON line prefixed with ROW.
"""
import json, os, signal, subprocess, time, urllib.request
from pathlib import Path

W = Path("/work")
RES = W / "results_restore.jsonl"
PORT = 9200
WANT_CPU = os.environ.get("WANT_CPU", "8581C")
CFG = "ov-ours8"
ENV = {"BACKEND": "ov", "OV_THREADS": "16", "OV_STREAMS": "8",
       "OV_CACHE_DIR": "/work/ovcache8", "IDLE_RELEASE_SEC": "45"}
N_CYCLES = int(os.environ.get("N_CYCLES", "10"))
N_COLD_CACHED = 3
SETTLE_S = 10          # after the release is observed, before the restore request
RELEASE_WAIT_MAX_S = 180

def log(row):
    row["ts"] = round(time.time(), 1)
    with open(RES, "a") as f:
        f.write(json.dumps(row) + "\n")
    print("ROW " + json.dumps(row), flush=True)

def rd(p):
    try: return open(p).read().strip()
    except OSError: return None

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

cpu = [l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
log({"kind": "host", "cpu": cpu, "cpus": os.cpu_count(),
     "cpu_max": rd("/sys/fs/cgroup/cpu.max"), "memory_max": rd("/sys/fs/cgroup/memory.max"),
     "kernel": os.uname().release, "want_cpu": WANT_CPU})
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

def shim_events(logpath):
    """(#released, #restored) lines the shim has printed so far."""
    lines = logpath.read_text().splitlines() if logpath.exists() else []
    return lines.count("released"), lines.count("restored")

def start_server(tag):
    e = dict(os.environ); e.update(ENV); e["PORT2"] = str(PORT)
    pidfile = W / "serve_ov.pid"
    if pidfile.exists(): pidfile.unlink()
    logpath = W / f"mx_restore_{tag}.log"
    proc = subprocess.Popen(["python", "serve_shim.py"], cwd=W, env=e,
                            stdout=open(logpath, "w"), stderr=subprocess.STDOUT)
    proc.logpath = logpath
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
    pid = int(pidfile.read_text())
    return proc, pid, {"ready_s": round(ready, 2) if ready else None,
                       "cold_first_audio_s": round(ready + first["first_ms"] / 1e3, 2) if ready else None}

def stop_server(proc):
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
    time.sleep(3)

# process cold starts: one with an empty compile cache, then N_COLD_CACHED with it populated
assert not Path(ENV["OV_CACHE_DIR"]).exists(), "compile cache dir must not exist before the first start"
proc, pid, cold = start_server("cold")
log({"kind": "cold", "cfg": CFG, **cold})
stop_server(proc)
for i in range(N_COLD_CACHED):
    proc, pid, cold = start_server(f"cached{i}")
    log({"kind": "cold-cached", "cfg": CFG, "rep": i, **cold})
    if i < N_COLD_CACHED - 1:
        stop_server(proc)

# the last process stays up for the release/restore cycles
TEXTS = [x["text"] for x in json.load(open(W / "corpus_texts.json"))]
SHORT = TEXTS[2]
runs = [post_tts(SHORT) for _ in range(3)]
log({"kind": "lone", "cfg": CFG, "band": "short", "audio_s": runs[0]["audio_s"],
     "first_ms": sorted(r["first_ms"] for r in runs)[1],
     "wall_s": sorted(r["wall_s"] for r in runs)[1]})

r = subprocess.run(["python", "loadgen_local.py", "--port", str(PORT), "--pid", str(pid),
                    "--clients", "4", "--secs", "90", "--ramp", "15", "--label", f"{CFG}-c4"],
                   cwd=W, capture_output=True, text=True, timeout=90 + 240)
line = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "{}"
row = json.loads(line); row["kind"] = "sustained"; row["cfg"] = CFG
log(row)

for cycle in range(N_CYCLES):
    t_last = time.time()
    rel0, rst0 = shim_events(proc.logpath)
    released = False
    while time.time() - t_last < RELEASE_WAIT_MAX_S:
        time.sleep(2)
        if shim_events(proc.logpath)[0] > rel0:
            released = True
            break
    release_seen_s = round(time.time() - t_last, 1)
    time.sleep(SETTLE_S)
    before = {"rss_before_mb": rss_mb(pid), "cgroup_before_mb": cgroup_mb()}
    rst = post_tts(SHORT)
    restored = shim_events(proc.logpath)[1] > rst0
    log({"kind": "restore", "cfg": CFG, "cycle": cycle, "released": released, "restored": restored,
         "release_seen_s": release_seen_s, "quiet_s": round(time.time() - t_last - rst["wall_s"], 1),
         **before, "first_ms": rst["first_ms"], "wall_s": rst["wall_s"], "audio_s": rst["audio_s"]})
    warm = post_tts(SHORT)
    log({"kind": "warm-after-restore", "cfg": CFG, "cycle": cycle,
         "first_ms": warm["first_ms"], "wall_s": warm["wall_s"],
         "rss_after_mb": rss_mb(pid), "cgroup_after_mb": cgroup_mb()})

stop_server(proc)
log({"kind": "done-restore"})
print("RESTORE-DONE", flush=True)
for line in open(RES):
    print("FINAL " + line.rstrip(), flush=True)
while True: time.sleep(3600)
