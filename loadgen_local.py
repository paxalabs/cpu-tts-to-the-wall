"""Closed-loop load against the local shim; sustained window; process CPU
of the SERVER pid bracketed around the window.

loadgen_local.py --port 9100 --pid SERVERPID --clients 4 --secs 60 --ramp 10
"""
import argparse, json, threading, time, urllib.request
from pathlib import Path

CORPUS = [x["text"] for x in json.load(open("/work/corpus_texts.json"))]

def cpu_seconds(pid):
    parts = open(f"/proc/{pid}/stat").read().split()
    return (int(parts[13]) + int(parts[14])) / 100.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--secs", type=float, default=60)
    ap.add_argument("--ramp", type=float, default=10)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    url = f"http://127.0.0.1:{a.port}/tts"
    lock, results, counter = threading.Lock(), [], [0]
    t0 = time.perf_counter(); deadline = t0 + a.secs
    def client():
        while time.perf_counter() < deadline:
            with lock:
                i = counter[0]; counter[0] += 1
            started = time.perf_counter() - t0
            body = json.dumps({"text": CORPUS[i % len(CORPUS)]}).encode()
            req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    r.read()
                    audio_s = float(r.headers["X-Audio-S"])
                    first_ms = float(r.headers["X-First-Audio-Ms"])
            except Exception as e:
                audio_s, first_ms = 0.0, -1.0
            with lock:
                results.append((started, time.perf_counter() - t0, audio_s, first_ms))
    threads = [threading.Thread(target=client) for _ in range(a.clients)]
    for t in threads: t.start()
    while time.perf_counter() - t0 < a.ramp: time.sleep(0.05)
    cpu0 = cpu_seconds(a.pid); t_ramp = time.perf_counter() - t0
    for t in threads: t.join()
    cpu1 = cpu_seconds(a.pid); t_end = time.perf_counter() - t0
    win = [r for r in results if r[0] >= t_ramp and r[2] > 0]
    audio = sum(r[2] for r in win); wall = t_end - t_ramp
    firsts = sorted(r[3] for r in win)
    lat = sorted(r[1] - r[0] for r in win)
    out = {"label": a.label, "clients": a.clients, "window_s": round(wall, 1),
           "reqs": len(win), "bad": sum(1 for r in results if r[2] == 0),
           "audio_s": round(audio, 1),
           "throughput_x": round(audio / wall, 2),
           "server_cpu_s": round(cpu1 - cpu0, 1),
           "audio_per_cpu_s": round(audio / (cpu1 - cpu0), 2) if cpu1 > cpu0 else None,
           "first_p50_ms": round(firsts[len(firsts)//2], 0) if firsts else None,
           "first_p90_ms": round(firsts[int(len(firsts)*.9)], 0) if firsts else None,
           "lat_p50_s": round(lat[len(lat)//2], 2) if lat else None}
    print(json.dumps(out))
    with open("/work/results_load.jsonl", "a") as f:
        f.write(json.dumps(out) + "\n")

if __name__ == "__main__":
    main()
