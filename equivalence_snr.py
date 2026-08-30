"""Phase A3b: calibrated equivalence.

Reference points: (1) ORT run-to-run determinism (sample SNR, expect inf or
very high), (2) PT vs ORT sample SNR where durations already match exactly,
(3) ORT vs OV sample SNR on chunks whose lengths match. Every number is a
direct measurement on the stored corpus inputs.
"""
import json
from pathlib import Path
import numpy as np

W = Path("/work")
man = json.load(open(W / "corpus_manifest.json"))
z = np.load(W / "corpus.npz")
speed = np.ones(1, dtype=np.float32)

def snr(a, b):
    n = min(len(a), len(b)); a, b = a[:n].astype(np.float64), b[:n].astype(np.float64)
    err = np.mean((a - b) ** 2)
    if err == 0: return float("inf")
    return round(float(10 * np.log10((np.mean(a**2) + 1e-12) / err)), 1)

import onnxruntime as ort
import openvino as ov
sess = ort.InferenceSession("/work/assets/onnx/model.onnx", providers=["CPUExecutionProvider"])
comp = ov.Core().compile_model("/work/assets/onnx/model.onnx", "CPU")

from kokoro import KPipeline
pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
TEXTS = {0: "Welcome back.", 2: "The meeting starts in five minutes.",
         8: "The library closes at nine on weekdays, so there is still time to return the books you borrowed last month."}

rows = []
for i, text in TEXTS.items():
    pt_audio = np.concatenate([np.asarray(c.audio) for c in pipe(text, voice="af_heart", speed=1.0)]).astype(np.float32)
    ids = z[f"ids_{i}_0"]; style = z[f"style_{i}_0"]
    feed = {"input_ids": ids, "style": style, "speed": speed}
    a1 = sess.run(None, feed)[0].ravel()
    a2 = sess.run(None, feed)[0].ravel()
    a_ov = comp(feed)[comp.output(0)].ravel()
    a_ov2 = comp(feed)[comp.output(0)].ravel()
    rows.append({"i": i,
                 "ort_selfsnr_db": snr(a1, a2),
                 "ov_selfsnr_db": snr(a_ov, a_ov2),
                 "pt_vs_ort_snr_db": snr(pt_audio, a1),
                 "len_pt": len(pt_audio), "len_ort": len(a1), "len_ov": len(a_ov),
                 "ort_vs_ov_snr_db": snr(a1, a_ov) if len(a1) == len(a_ov) else None})
json.dump(rows, open(W / "results_equiv2.json", "w"), indent=1)
for r in rows: print(r)
