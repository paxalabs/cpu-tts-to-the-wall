"""Phase A3: backend equivalence on identical inputs, measured correctly.

Sample-level SNR is meaningless when a single duration rounding differs, so
equivalence is measured as: audio length delta per chunk, and log-mel
spectral distance on the overlapping region (frame-aligned by truncation).
PyTorch reference audio comes from the same KPipeline run that produced the
corpus tokens; ORT and OV consume the stored (ids, style) pairs.
"""
import json
from pathlib import Path
import numpy as np

W = Path("/work")
man = json.load(open(W / "corpus_manifest.json"))
z = np.load(W / "corpus.npz")

def logmel(x, sr=24000, n_fft=1024, hop=256, n_mels=64):
    # minimal mel spectrogram (numpy only), enough for a distance metric
    from numpy.fft import rfft
    win = np.hanning(n_fft)
    n = (len(x) - n_fft) // hop + 1
    if n <= 0: return np.zeros((n_mels, 0))
    frames = np.stack([x[i*hop:i*hop+n_fft] * win for i in range(n)])
    spec = np.abs(rfft(frames, axis=1)) ** 2
    freqs = np.linspace(0, sr / 2, spec.shape[1])
    mel_pts = np.linspace(0, 2595 * np.log10(1 + (sr / 2) / 700), n_mels + 2)
    hz = 700 * (10 ** (mel_pts / 2595) - 1)
    fb = np.zeros((n_mels, spec.shape[1]))
    for m in range(n_mels):
        lo, ce, hi = hz[m], hz[m+1], hz[m+2]
        fb[m] = np.clip(np.minimum((freqs - lo) / (ce - lo + 1e-9), (hi - freqs) / (hi - ce + 1e-9)), 0, None)
    return np.log10(fb @ spec.T + 1e-10)

def meldist(a, b):
    A, B = logmel(a), logmel(b)
    n = min(A.shape[1], B.shape[1])
    if n == 0: return None
    return round(float(np.mean(np.abs(A[:, :n] - B[:, :n]))), 4)

import onnxruntime as ort
import openvino as ov
sess = ort.InferenceSession("/work/assets/onnx/model.onnx", providers=["CPUExecutionProvider"])
comp = ov.Core().compile_model("/work/assets/onnx/model.onnx", "CPU")
speed = np.ones(1, dtype=np.float32)

from kokoro import KPipeline
pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")

rows = []
texts_covered = 0
for item in man["items"][:6] + man["items"][8:10] + man["items"][14:16]:
    i = item["i"]
    # regenerate PT audio for this item (same pipeline, same voice/speed)
    from_corpus = None
    # reconstruct text is not stored; PT audio lengths are in the manifest.
    for c in item["chunks"]:
        j = c["j"]
        ids = z[f"ids_{i}_{j}"]; style = z[f"style_{i}_{j}"]
        a_ort = sess.run(None, {"input_ids": ids, "style": style, "speed": speed})[0].ravel()
        a_ov = comp({"input_ids": ids, "style": style, "speed": speed})[comp.output(0)].ravel()
        rows.append({
            "i": i, "j": j, "tokens": c["tokens"],
            "pt_s": c["pt_audio_s"],
            "ort_s": round(len(a_ort) / 24000, 3),
            "ov_s": round(len(a_ov) / 24000, 3),
            "len_delta_ort_ov": int(len(a_ort)) - int(len(a_ov)),
            "meldist_ort_ov": meldist(a_ort.astype(np.float64), a_ov.astype(np.float64)),
        })
    texts_covered += 1

out = {
    "chunks": rows,
    "max_abs_len_delta_ort_ov": max(abs(r["len_delta_ort_ov"]) for r in rows),
    "max_meldist_ort_ov": max(r["meldist_ort_ov"] for r in rows if r["meldist_ort_ov"] is not None),
    "max_abs_dur_delta_pt_ort_s": max(abs(r["pt_s"] - r["ort_s"]) for r in rows),
}
json.dump(out, open(W / "results_equiv.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "chunks"}, indent=1))
for r in rows:
    print(r)
