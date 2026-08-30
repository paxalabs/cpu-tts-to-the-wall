"""Phase A: ground-truth assets on the box.

1. Download the community ONNX export of Kokoro-82M (fp32) + a voice.
2. Run the PyTorch pipeline (kokoro package) on a probe sentence.
3. Run the same phonemes through ONNX Runtime and OpenVINO.
4. Report durations, output lengths, and cross-backend SNR.

Everything it prints is a measurement or a file hash; no estimates.
"""
import json, time, hashlib
from pathlib import Path
import numpy as np

W = Path("/work"); OUT = {}

def sha(p, n=8):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]

# -- 1. assets ---------------------------------------------------------------
from huggingface_hub import hf_hub_download
t0 = time.time()
onnx_path = hf_hub_download("onnx-community/Kokoro-82M-v1.0-ONNX", "onnx/model.onnx", local_dir=W / "assets")
voice_path = hf_hub_download("onnx-community/Kokoro-82M-v1.0-ONNX", "voices/af_heart.bin", local_dir=W / "assets")
OUT["download_s"] = round(time.time() - t0, 1)
OUT["onnx_sha"] = sha(onnx_path); OUT["onnx_mb"] = round(Path(onnx_path).stat().st_size / 1e6, 1)

# -- 2. pytorch pipeline -----------------------------------------------------
PROBE = "The quick brown fox jumps over the lazy dog, and the afternoon light settles over the harbor."
from kokoro import KPipeline
t0 = time.time()
pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
OUT["kpipeline_init_s"] = round(time.time() - t0, 1)
t0 = time.time()
chunks = list(pipe(PROBE, voice="af_heart", speed=1.0))
OUT["pt_infer_s"] = round(time.time() - t0, 2)
pt_audio = np.concatenate([np.asarray(c.audio) for c in chunks]).astype(np.float32)
phonemes = "".join(c.phonemes for c in chunks)
OUT["pt_audio_s"] = round(len(pt_audio) / 24000, 2)
OUT["phoneme_len"] = len(phonemes)

# token ids from the ONNX repo's vocab (phoneme -> id), as documented
config = json.load(open(hf_hub_download("hexgrad/Kokoro-82M", "config.json", local_dir=W / "assets/hexgrad")))
vocab = config["vocab"]
ids = [0] + [vocab[p] for p in phonemes if p in vocab] + [0]
OUT["dropped_phonemes"] = sum(1 for p in phonemes if p not in vocab)
input_ids = np.array([ids], dtype=np.int64)
voices = np.fromfile(voice_path, dtype=np.float32).reshape(-1, 1, 256)
style = voices[len(ids) - 2]  # per-length style row, as in the reference impl
speed = np.array([1.0], dtype=np.float32)

# -- 3. onnxruntime ----------------------------------------------------------
import onnxruntime as ort
t0 = time.time()
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
OUT["ort_load_s"] = round(time.time() - t0, 1)
t0 = time.time()
ort_audio = sess.run(None, {"input_ids": input_ids, "style": style, "speed": speed})[0].ravel().astype(np.float32)
OUT["ort_infer_s"] = round(time.time() - t0, 2)
OUT["ort_audio_s"] = round(len(ort_audio) / 24000, 2)

# -- 4. openvino -------------------------------------------------------------
import openvino as ov
core = ov.Core()
t0 = time.time()
comp = core.compile_model(onnx_path, "CPU")
OUT["ov_compile_s"] = round(time.time() - t0, 1)
t0 = time.time()
ov_audio = comp({"input_ids": input_ids, "style": style, "speed": speed})[comp.output(0)].ravel().astype(np.float32)
OUT["ov_infer_s"] = round(time.time() - t0, 2)
OUT["ov_audio_s"] = round(len(ov_audio) / 24000, 2)

def snr(a, b):
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    return round(float(10 * np.log10((np.mean(a**2) + 1e-12) / (np.mean((a - b)**2) + 1e-12))), 1)

OUT["snr_ort_vs_ov_db"] = snr(ort_audio, ov_audio)
OUT["len_pt"] = len(pt_audio); OUT["len_ort"] = len(ort_audio); OUT["len_ov"] = len(ov_audio)
OUT["snr_pt_vs_ort_db"] = snr(pt_audio, ort_audio)

json.dump(OUT, open(W / "results_assets.json", "w"), indent=1)
print(json.dumps(OUT, indent=1))
