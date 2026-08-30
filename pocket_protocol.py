"""The full transfer-check protocol for pocket-tts, one container.

Same lens as the main matrix: cold starts (fresh and warm weight cache),
warm lone latency across text lengths, sustained closed-loop load with
server-CPU bracketing, ten-minute idle decay, all under (alpha, beta).
Their server targets local single-user use; this is a billing-structure
check, not a baseline comparison, and the results file records everything
needed to say so precisely.
"""
import io, json, struct, subprocess, threading, time, urllib.request, uuid
from pathlib import Path

W = Path("/work"); OUT = {"host": None}
OUT["host"] = [l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
import os
OUT["vcpus"] = os.cpu_count()

TEXTS = {
    "short": "The meeting starts in five minutes.",
    "medium": "The library closes at nine on weekdays, so there is still time to return the books you borrowed last month.",
    "long": "The committee reviewed the proposal in detail during the afternoon session, weighing the projected costs against the expected benefits for each of the three districts involved. Several members raised questions about the maintenance schedule, and the engineers agreed to provide a revised estimate before the next meeting.",
}

def multipart(fields):
    b = uuid.uuid4().hex
    body = io.BytesIO()
    for k, v in fields.items():
        body.write(f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    body.write(f"--{b}--\r\n".encode())
    return body.getvalue(), f"multipart/form-data; boundary={b}"

def wav_seconds(data):
    """Parse the WAV header for sample rate and data size; fall back to
    treating the payload as headerless 24 kHz s16 mono."""
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        rate = struct.unpack("<I", data[24:28])[0]
        bits = struct.unpack("<H", data[34:36])[0]
        ch = struct.unpack("<H", data[22:24])[0]
        # find the data chunk
        i = 12
        while i < len(data) - 8:
            cid = data[i:i+4]; sz = struct.unpack("<I", data[i+4:i+8])[0]
            if cid == b"data":
                return sz / (rate * ch * bits // 8), rate
            i += 8 + sz
    return len(data) / 2 / 24000, None

def tts(text, timeout=600):
    data, ctype = multipart({"text": text})
    t0 = time.perf_counter()
    req = urllib.request.Request("http://127.0.0.1:8123/tts", data=data,
                                 headers={"Content-Type": ctype})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = r.read()
    wall = time.perf_counter() - t0
    secs, rate = wav_seconds(payload)
    return {"wall_s": round(wall, 3), "audio_s": round(secs, 3), "rate": rate,
            "bytes": len(payload)}

def server_pids():
    out = subprocess.run("pgrep -f 'pocket-tts serve'", shell=True,
                         capture_output=True, text=True).stdout.split()
    return [int(x) for x in out]

def cpu_seconds(pids):
    total = 0.0
    for pid in pids:
        try:
            parts = open(f"/proc/{pid}/stat").read().split()
            total += (int(parts[13]) + int(parts[14])) / 100.0
        except OSError:
            pass
    return total

def rss_mb(pids):
    best = 0
    for pid in pids:
        try:
            for line in open(f"/proc/{pid}/status"):
                if line.startswith("VmRSS"):
                    best = max(best, int(line.split()[1]) / 1024)
        except OSError:
            pass
    return round(best, 1)

def start_server():
    t0 = time.perf_counter()
    proc = subprocess.Popen("pocket-tts serve --host 127.0.0.1 --port 8123", shell=True,
                            cwd=W, stdout=open(W / "pf_serve.log", "a"), stderr=subprocess.STDOUT)
    for _ in range(2400):
        try:
            urllib.request.urlopen("http://127.0.0.1:8123/health", timeout=2)
            return proc, round(time.perf_counter() - t0, 2)
        except Exception:
            time.sleep(0.25)
    return proc, None

def stop_server(proc):
    subprocess.run("pkill -f 'pocket-tts serve'", shell=True)
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
    time.sleep(2)

# install (records whether weights were already cached: fresh container -> no)
t0 = time.time()
rc = subprocess.run("pip install -q pocket-tts", shell=True, capture_output=True, timeout=1200).returncode
OUT["pip_install_s"] = round(time.time() - t0, 1); OUT["pip_rc"] = rc

# 1. cold starts: first-ever (weight download) then warm-cache x3
colds = []
for i in range(4):
    proc, ready = start_server()
    first = tts(TEXTS["short"])
    colds.append({"ready_s": ready, "first_request_s": first["wall_s"],
                  "cold_to_audio_s": round(ready + first["wall_s"], 2)})
    if i < 3:
        stop_server(proc)
OUT["cold_first_ever"] = colds[0]
OUT["cold_warm_cache"] = colds[1:]

# 2. warm lone latency (median of 3 per band) + realtime factor
lone = {}
for band, text in TEXTS.items():
    runs = sorted([tts(text) for _ in range(3)], key=lambda r: r["wall_s"])
    mid = runs[1]
    lone[band] = {**mid, "rtf": round(mid["audio_s"] / mid["wall_s"], 2)}
OUT["lone"] = lone
OUT["wav_rate"] = lone["short"]["rate"]

# 3. sustained closed loop: c1 and c2 and c4, 60 s each, ramp 10
def sustained(clients, secs=60, ramp=10):
    pids = server_pids()
    lock, results, counter = threading.Lock(), [], [0]
    t0 = time.perf_counter(); deadline = t0 + secs
    texts = list(TEXTS.values())
    def client():
        while time.perf_counter() < deadline:
            with lock:
                i = counter[0]; counter[0] += 1
            started = time.perf_counter() - t0
            try:
                r = tts(texts[i % len(texts)])
            except Exception:
                r = {"audio_s": 0, "wall_s": 0}
            with lock:
                results.append((started, r))
    threads = [threading.Thread(target=client) for _ in range(clients)]
    for t in threads: t.start()
    while time.perf_counter() - t0 < ramp: time.sleep(0.05)
    c0 = cpu_seconds(pids); t_ramp = time.perf_counter() - t0
    for t in threads: t.join()
    c1 = cpu_seconds(pids); t_end = time.perf_counter() - t0
    win = [r for s, r in results if s >= t_ramp and r["audio_s"] > 0]
    bad = sum(1 for _, r in results if r["audio_s"] == 0)
    audio = sum(r["audio_s"] for r in win); wall = t_end - t_ramp
    lat = sorted(r["wall_s"] for r in win)
    return {"clients": clients, "reqs": len(win), "bad": bad,
            "audio_s": round(audio, 1), "window_s": round(wall, 1),
            "throughput_x": round(audio / wall, 2),
            "server_cpu_s": round(c1 - c0, 1),
            "audio_per_cpu_s": round(audio / (c1 - c0), 2) if c1 > c0 else None,
            "lat_p50_s": round(lat[len(lat)//2], 2) if lat else None,
            "lat_p90_s": round(lat[int(len(lat)*.9)], 2) if lat else None}

OUT["sustained"] = [sustained(1), sustained(2), sustained(4)]

# 4. idle decay: full ten minutes
t_q = time.time()
idle = {}
for mark in (60, 300, 600):
    time.sleep(max(0, mark - (time.time() - t_q)))
    idle[mark] = rss_mb(server_pids())
OUT["idle_rss_mb"] = idle

# 5. post-idle request (does anything need rewarming?)
OUT["post_idle_request"] = tts(TEXTS["short"])

json.dump(OUT, open(W / "results_pocket_full.json", "w"), indent=1)
print(json.dumps(OUT, indent=1))
stop_server(proc)
