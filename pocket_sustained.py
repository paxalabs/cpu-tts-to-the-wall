"""Sustained rerun with a correct duration measure: audio seconds derived
from payload bytes (16-bit mono at the rate the header declares)."""
import io, json, struct, subprocess, threading, time, urllib.request, uuid
from pathlib import Path

W = Path("/work")
TEXTS = [
    "The meeting starts in five minutes.",
    "The library closes at nine on weekdays, so there is still time to return the books you borrowed last month.",
    "The committee reviewed the proposal in detail during the afternoon session, weighing the projected costs against the expected benefits for each of the three districts involved. Several members raised questions about the maintenance schedule, and the engineers agreed to provide a revised estimate before the next meeting.",
]

def multipart(fields):
    b = uuid.uuid4().hex
    body = io.BytesIO()
    for k, v in fields.items():
        body.write(f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    body.write(f"--{b}--\r\n".encode())
    return body.getvalue(), f"multipart/form-data; boundary={b}"

def wav_rate(data):
    if data[:4] == b"RIFF":
        return struct.unpack("<I", data[24:28])[0]
    return 24000

def tts(text, timeout=600):
    data, ctype = multipart({"text": text})
    t0 = time.perf_counter()
    req = urllib.request.Request("http://127.0.0.1:8123/tts", data=data,
                                 headers={"Content-Type": ctype})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = r.read()
    rate = wav_rate(payload)
    return {"wall_s": round(time.perf_counter() - t0, 3),
            "audio_s": round(max(0, len(payload) - 44) / 2 / rate, 3)}

def server_pids():
    return [int(x) for x in subprocess.run("pgrep -f 'pocket-tts serve'", shell=True,
            capture_output=True, text=True).stdout.split()]

def cpu_seconds(pids):
    total = 0.0
    for pid in pids:
        try:
            parts = open(f"/proc/{pid}/stat").read().split()
            total += (int(parts[13]) + int(parts[14])) / 100.0
        except OSError:
            pass
    return total

proc = subprocess.Popen("pocket-tts serve --host 127.0.0.1 --port 8123", shell=True,
                        cwd=W, stdout=open(W / "ps_serve.log", "w"), stderr=subprocess.STDOUT)
for _ in range(2400):
    try:
        urllib.request.urlopen("http://127.0.0.1:8123/health", timeout=2); break
    except Exception:
        time.sleep(0.25)
tts(TEXTS[0])  # warm-up

def sustained(clients, secs=60, ramp=10):
    pids = server_pids()
    lock, results, counter = threading.Lock(), [], [0]
    t0 = time.perf_counter(); deadline = t0 + secs
    def client():
        while time.perf_counter() < deadline:
            with lock:
                i = counter[0]; counter[0] += 1
            started = time.perf_counter() - t0
            try:
                r = tts(TEXTS[i % len(TEXTS)])
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
    audio = sum(r["audio_s"] for r in win); wall = t_end - t_ramp
    lat = sorted(r["wall_s"] for r in win)
    return {"clients": clients, "reqs": len(win),
            "bad": sum(1 for _, r in results if r["audio_s"] == 0),
            "audio_s": round(audio, 1), "window_s": round(wall, 1),
            "throughput_x": round(audio / wall, 2),
            "server_cpu_s": round(c1 - c0, 1),
            "audio_per_cpu_s": round(audio / (c1 - c0), 2) if c1 > c0 else None,
            "lat_p50_s": round(lat[len(lat)//2], 2) if lat else None}

OUT = {"sustained": [sustained(1), sustained(2), sustained(4)]}
json.dump(OUT, open(W / "results_pocket_sus.json", "w"), indent=1)
print(json.dumps(OUT, indent=1))
subprocess.run("pkill -f 'pocket-tts'", shell=True)
