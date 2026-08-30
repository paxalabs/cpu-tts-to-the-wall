"""Phase A3c: mel-envelope distances with a calibrated floor, plus WAVs.

PT vs ORT share exact durations; their mel distance is the floor for
"same audio, different fine structure". ORT vs OV is compared against that
floor. Saves one probe WAV per backend for human listening.
"""
import json, struct
from pathlib import Path
import numpy as np

W = Path("/work")
z = np.load(W / "corpus.npz")
speed = np.ones(1, dtype=np.float32)

def logmel(x, sr=24000, n_fft=1024, hop=256, n_mels=64):
    from numpy.fft import rfft
    win = np.hanning(n_fft)
    n = (len(x) - n_fft) // hop + 1
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
    A, B = logmel(a.astype(np.float64)), logmel(b.astype(np.float64))
    n = min(A.shape[1], B.shape[1])
    return round(float(np.mean(np.abs(A[:, :n] - B[:, :n]))), 4)

def wav(path, audio):
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " +
                struct.pack("<IHHIIHH", 16, 1, 1, 24000, 48000, 2, 16) +
                b"data" + struct.pack("<I", len(pcm)) + pcm)

import onnxruntime as ort
import openvino as ov
sess = ort.InferenceSession("/work/assets/onnx/model.onnx", providers=["CPUExecutionProvider"])
comp = ov.Core().compile_model("/work/assets/onnx/model.onnx", "CPU")
from kokoro import KPipeline
pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")

TEXT = "The meeting starts in five minutes."
pt = np.concatenate([np.asarray(c.audio) for c in pipe(TEXT, voice="af_heart", speed=1.0)]).astype(np.float32)
ids, style = z["ids_2_0"], z["style_2_0"]
feed = {"input_ids": ids, "style": style, "speed": speed}
a_ort = sess.run(None, feed)[0].ravel().astype(np.float32)
a_ov = comp(feed)[comp.output(0)].ravel().astype(np.float32)

out = {"meldist_pt_ort": meldist(pt, a_ort),
       "meldist_pt_ov": meldist(pt, a_ov),
       "meldist_ort_ov": meldist(a_ort, a_ov),
       "rms_pt": round(float(np.sqrt(np.mean(pt**2))), 4),
       "rms_ort": round(float(np.sqrt(np.mean(a_ort**2))), 4),
       "rms_ov": round(float(np.sqrt(np.mean(a_ov**2))), 4)}
wav(W / "probe_pt.wav", pt); wav(W / "probe_ort.wav", a_ort); wav(W / "probe_ov.wav", a_ov)
json.dump(out, open(W / "results_equiv3.json", "w"), indent=1)
print(json.dumps(out, indent=1))
