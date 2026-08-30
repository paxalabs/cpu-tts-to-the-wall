"""The serving shim: one HTTP server, three backends, identical text path.

BACKEND=pt|ort|ov selects the engine; the G2P front end (kokoro pipeline)
is shared, so every backend pays the same text cost and consumes identical
(ids, style) inputs. Chunked synthesis: first chunk's completion time is
recorded as first-audio.

POST /tts {"text": ...}   -> WAV bytes; X-First-Audio-Ms, X-Infer-Ms headers
GET  /mem                 -> rss, cgroup current (billed memory view)
GET  /health
Env: BACKEND, PORT, OMP/threads inherited as set by the experiment.
"""
import io, json, os, struct, time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np

W = Path("/work")
BACKEND = os.environ.get("BACKEND", "ort")
vocab = json.load(open(W / "assets/hexgrad/config.json"))["vocab"]
voices = np.fromfile(W / "assets/voices/af_heart.bin", dtype=np.float32).reshape(-1, 1, 256)
speed_arr = np.ones(1, dtype=np.float32)

from kokoro import KPipeline
if BACKEND == "pt" and os.environ.get("TORCH_THREADS"):
    import torch
    torch.set_num_threads(int(os.environ["TORCH_THREADS"]))
g2p_pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", model=(BACKEND == "pt"))

if BACKEND == "ort":
    import onnxruntime as ort
    _so = ort.SessionOptions()
    if os.environ.get("ORT_THREADS"):
        _so.intra_op_num_threads = int(os.environ["ORT_THREADS"])
    sess = ort.InferenceSession(str(W / "assets/onnx/model.onnx"), _so, providers=["CPUExecutionProvider"])
    def synth_ids(ids, style):
        return sess.run(None, {"input_ids": ids, "style": style, "speed": speed_arr})[0].ravel()
elif BACKEND == "ov":
    import ctypes, gc, threading
    import queue as _q
    import openvino as ov
    cfg = {}
    if os.environ.get("OV_THREADS"):
        cfg["INFERENCE_NUM_THREADS"] = int(os.environ["OV_THREADS"])
    if os.environ.get("OV_STREAMS"):
        cfg["NUM_STREAMS"] = int(os.environ["OV_STREAMS"])
    IDLE_SEC = float(os.environ.get("IDLE_RELEASE_SEC", "0") or 0)
    CACHE_DIR = os.environ.get("OV_CACHE_DIR")
    _state = {"comp": None, "pool": None, "last": time.time(), "gen": 0}
    _lock = threading.Lock()

    def _drop_cache_dir():
        if not CACHE_DIR:
            return
        for root, _, files in os.walk(CACHE_DIR):
            for name in files:
                fd = os.open(os.path.join(root, name), os.O_RDONLY)
                try:
                    os.fsync(fd)  # write-back first: DONTNEED skips dirty pages
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(fd)

    def _build():
        core = ov.Core()
        if CACHE_DIR:
            core.set_property({"CACHE_DIR": CACHE_DIR})
        comp = core.compile_model(str(W / "assets/onnx/model.onnx"), "CPU", cfg)
        pool = _q.Queue()
        for _ in range(max(int(os.environ.get("OV_STREAMS", "0") or 0), 8)):
            pool.put(comp.create_infer_request())
        _drop_cache_dir()
        return comp, pool

    _state["comp"], _state["pool"] = _build()

    def _release_if_idle():
        while True:
            time.sleep(5)
            with _lock:
                # revalidate at action time: the decision and the action
                # share the lock, and generation guards a racing request
                if _state["comp"] is not None and IDLE_SEC and time.time() - _state["last"] > IDLE_SEC:
                    _state["comp"] = None
                    _state["pool"] = None
                    gc.collect()
                    try:
                        ctypes.CDLL("libc.so.6").malloc_trim(0)
                    except OSError:
                        pass
                    _drop_cache_dir()
                    print("released", flush=True)
    if IDLE_SEC:
        threading.Thread(target=_release_if_idle, daemon=True).start()

    def synth_ids(ids, style):
        with _lock:
            _state["last"] = time.time()
            if _state["comp"] is None:
                _state["comp"], _state["pool"] = _build()
                print("restored", flush=True)
            pool = _state["pool"]
        req = pool.get()
        try:
            req.infer({"input_ids": ids, "style": style, "speed": speed_arr})
            return req.get_output_tensor(0).data.copy().ravel()
        finally:
            pool.put(req)
            _state["last"] = time.time()

def synth_text(text):
    """Returns (audio float32, first_chunk_wall_s, n_chunks)."""
    t0 = time.perf_counter()
    first = None
    parts = []
    for c in g2p_pipe(text, voice="af_heart", speed=1.0):
        if BACKEND == "pt":
            audio = np.asarray(c.audio, dtype=np.float32)
        else:
            ps = c.phonemes
            tokens = [vocab[p] for p in ps if p in vocab]
            ids = np.array([[0, *tokens, 0]], dtype=np.int64)
            style = voices[len(ps) - 1, :, :]
            audio = synth_ids(ids, style).astype(np.float32)
        if first is None:
            first = time.perf_counter() - t0
        parts.append(audio)
    return np.concatenate(parts), first, len(parts)

def wav_bytes(audio):
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
    hdr = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " + \
        struct.pack("<IHHIIHH", 16, 1, 1, 24000, 48000, 2, 16) + \
        b"data" + struct.pack("<I", len(pcm))
    return hdr + pcm

def billed():
    rss = 0
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                rss = int(line.split()[1]) * 1024
    cg = None
    try:
        cg = int(open("/sys/fs/cgroup/memory.current").read())
    except OSError:
        pass
    return {"rss_mb": round(rss / 1e6, 1), "cgroup_mb": round(cg / 1e6, 1) if cg else None}

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path == "/mem":
            b = json.dumps(billed()).encode()
        else:
            b = b'{"ok": true, "backend": "' + BACKEND.encode() + b'"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n))
        t0 = time.perf_counter()
        audio, first, chunks = synth_text(req["text"])
        infer_s = time.perf_counter() - t0
        body = wav_bytes(audio)
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-First-Audio-Ms", str(round(first * 1e3, 1)))
        self.send_header("X-Infer-Ms", str(round(infer_s * 1e3, 1)))
        self.send_header("X-Audio-S", str(round(len(audio) / 24000, 3)))
        self.send_header("X-Chunks", str(chunks))
        self.end_headers(); self.wfile.write(body)

if __name__ == "__main__":
    port = int(os.environ.get("PORT2", 9100))
    Path(f"/work/serve_{BACKEND}.pid").write_text(str(os.getpid()))
    print(f"serve backend={BACKEND} port={port} pid={os.getpid()}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
