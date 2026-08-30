"""Rerun of the OV sustained rows with the request-pool fix; everything
else from matrix 1 remains valid and untouched."""
import json, os, signal, subprocess, time, urllib.request
from pathlib import Path

W = Path("/work"); RES = W / "results_matrix.jsonl"; PORT = 9200

def log(row):
    row["ts"] = round(time.time(), 1)
    with open(RES, "a") as f: f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)

def start_server(name, env):
    e = dict(os.environ); e.update(env); e["PORT2"] = str(PORT)
    pidfile = W / f"serve_{env['BACKEND']}.pid"
    if pidfile.exists(): pidfile.unlink()
    proc = subprocess.Popen(["python", "serve_shim.py"], cwd=W, env=e,
                            stdout=open(W / f"mx2_{name}.log", "w"), stderr=subprocess.STDOUT)
    for _ in range(1200):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2); break
        except Exception: time.sleep(0.25)
    body = json.dumps({"text": "Welcome back."}).encode()
    urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/tts", data=body,
        headers={"Content-Type": "application/json"}), timeout=120).read()
    return proc

def sustained(cfg, pid, clients, secs=90, ramp=15):
    r = subprocess.run(["python", "loadgen_local.py", "--port", str(PORT), "--pid", str(pid),
                        "--clients", str(clients), "--secs", str(secs), "--ramp", str(ramp),
                        "--label", f"{cfg}-v2-c{clients}"],
                       cwd=W, capture_output=True, text=True, timeout=secs + 240)
    row = json.loads(r.stdout.strip().splitlines()[-1]); row["kind"] = "sustained"; row["cfg"] = cfg + "-v2"
    log(row)

for name, env, sweeps in [
    ("ov-default", {"BACKEND": "ov"}, [4]),
    ("ov-4t4s", {"BACKEND": "ov", "OV_THREADS": "16", "OV_STREAMS": "4"}, [1, 2, 4, 8, 16]),
    ("ov-8s", {"BACKEND": "ov", "OV_THREADS": "16", "OV_STREAMS": "8"}, [4, 8, 16]),
]:
    proc = start_server(name, env)
    pid = int((W / "serve_ov.pid").read_text())
    for c in sweeps:
        sustained(name, pid, c, secs=90 if c == 4 else 60, ramp=15 if c == 4 else 12)
    if name == "ov-4t4s":
        t0 = time.time()
        for mark in (60, 300, 600):
            time.sleep(max(0, mark - (time.time() - t0)))
            rss = None
            for line in open(f"/proc/{pid}/status"):
                if line.startswith("VmRSS"): rss = round(int(line.split()[1]) / 1024, 1)
            cg = round(int(open("/sys/fs/cgroup/memory.current").read()) / 1e6, 1)
            log({"kind": "idle", "cfg": name + "-v2", "after_s": mark, "server_rss_mb": rss, "cgroup_mb": cg})
    proc.send_signal(signal.SIGTERM)
    try: proc.wait(timeout=30)
    except subprocess.TimeoutExpired: proc.kill()
    time.sleep(3)
log({"kind": "done-v2"})
