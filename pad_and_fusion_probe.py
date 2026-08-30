"""G3 quantification + D1 feasibility, run only on an otherwise idle box.

1. Pad tolerance: does appending pad-id zeros to input_ids change the audio
   of the valid region? Decides whether token-bucketing needs masking.
2. Per-op-type kernel profile: the same real input executed on the
   dynamically compiled graph vs the statically reshaped one.
"""
import json
from pathlib import Path
import numpy as np
import openvino as ov

W = Path("/work")
z = np.load(W / "corpus.npz")
speed = np.ones(1, dtype=np.float32)
core = ov.Core()
R = {}

ids = z["ids_2_0"]; style = z["style_2_0"]  # 36 ids incl. BOS/EOS
n = ids.shape[1]

# --- 1. pad tolerance -------------------------------------------------------
m_dyn = core.read_model("/work/assets/onnx/model.onnx")
comp_dyn = core.compile_model(m_dyn, "CPU", {"INFERENCE_NUM_THREADS": 8})
a_ref = comp_dyn({"input_ids": ids, "style": style, "speed": speed})[comp_dyn.output(0)].ravel()
for pad_to in (64, 128):
    padded = np.zeros((1, pad_to), dtype=np.int64)
    padded[0, :n] = ids[0]
    a_pad = comp_dyn({"input_ids": padded, "style": style, "speed": speed})[comp_dyn.output(0)].ravel()
    k = min(len(a_ref), len(a_pad))
    err = np.mean((a_ref[:k] - a_pad[:k]) ** 2)
    snr = float(10 * np.log10((np.mean(a_ref[:k] ** 2) + 1e-12) / (err + 1e-12))) if err > 0 else float("inf")
    R[f"pad{pad_to}"] = {"len_ref": int(len(a_ref)), "len_pad": int(len(a_pad)),
                        "snr_valid_region_db": round(snr, 1)}

# --- 2. dynamic vs static per-op-type profile ------------------------------
def profile(model, static):
    m = core.read_model("/work/assets/onnx/model.onnx")
    if static:
        m.reshape({"input_ids": [1, n], "style": [1, 256], "speed": [1]})
    comp = core.compile_model(m, "CPU", {"INFERENCE_NUM_THREADS": 4, "NUM_STREAMS": 1, "PERF_COUNT": True})
    req = comp.create_infer_request()
    feed = {"input_ids": ids, "style": style, "speed": speed}
    req.infer(feed)
    import time
    calls, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < 12:
        req.infer(feed); calls += 1
    ms_call = 1e3 * (time.perf_counter() - t0) / calls
    by = {}
    for pi in req.profiling_info:
        if pi.status.name != "EXECUTED":
            continue
        c, ms = by.get(pi.node_type, (0, 0.0))
        by[pi.node_type] = (c + 1, ms + pi.real_time.total_seconds() * 1e3)
    top = sorted(by.items(), key=lambda kv: -kv[1][1])[:14]
    return round(ms_call, 1), [(k, v[0], round(v[1], 2)) for k, v in top]

R["dyn_ms"], R["dyn_top"] = profile(None, False)
R["static_ms"], R["static_top"] = profile(None, True)
json.dump(R, open(W / "results_g3.json", "w"), indent=1)
print(json.dumps(R, indent=1))
