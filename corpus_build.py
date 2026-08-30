"""Phase A2 (v2): the paper's fixed corpus, tokenized per chunk.

The model accepts at most 512 input ids, so serving is chunked; the corpus
therefore stores per-chunk token sequences exactly as the serving shim will
execute them. Style row convention: pack[len(ps)-1], matching the PyTorch
reference pipeline, applied identically to every backend.
"""
import json, hashlib
from pathlib import Path
import numpy as np

W = Path("/work")

SHORT = [
    "Welcome back.",
    "Your order has shipped.",
    "The meeting starts in five minutes.",
    "Please hold while I connect you.",
    "Thanks for calling, goodbye.",
    "The forecast calls for light rain.",
    "Your appointment is confirmed for Tuesday.",
    "Press one to continue.",
]
MEDIUM = [
    "The library closes at nine on weekdays, so there is still time to return the books you borrowed last month.",
    "After the update installs, the device restarts twice; do not unplug it until the light stops blinking.",
    "Our support team has received your request and will reply within two business days with the next steps.",
    "The trail follows the river for three miles before climbing steadily toward the ridge and the old fire tower.",
    "To change your delivery address, open the orders page, select the pending shipment, and choose the edit option.",
    "The orchestra returns this season with a program of early symphonies and a newly commissioned piece for strings.",
]
LONG = [
    "The committee reviewed the proposal in detail during the afternoon session, weighing the projected costs against the expected benefits for each of the three districts involved. Several members raised questions about the maintenance schedule, and the engineers agreed to provide a revised estimate before the next meeting. In the meantime, the temporary measures adopted last spring will remain in effect, and residents will continue to receive weekly updates through the usual channels.",
    "When the harbor was first dredged more than a century ago, the town's economy depended almost entirely on the fishing fleet, and every family kept at least one small boat along the shore. Over the decades the wharves gave way to warehouses, the warehouses to workshops, and the workshops to the quiet row of studios and cafes that visitors photograph today. What has not changed is the morning light on the water, which still arrives at the same angle it always has, and still empties the streets as everyone turns to look.",
]
CORPUS = SHORT + MEDIUM + LONG

from kokoro import KPipeline
pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
vocab = json.load(open(W / "assets/hexgrad/config.json"))["vocab"]
voices = np.fromfile(W / "assets/voices/af_heart.bin", dtype=np.float32).reshape(-1, 1, 256)

items = []
arrays = {}
for i, text in enumerate(CORPUS):
    chunks = list(pipe(text, voice="af_heart", speed=1.0))
    chunk_meta = []
    for j, c in enumerate(chunks):
        ps = c.phonemes
        tokens = [vocab[p] for p in ps if p in vocab]
        assert len(tokens) <= 510, (i, j, len(tokens))
        row = len(ps) - 1  # PyTorch reference convention
        arrays[f"ids_{i}_{j}"] = np.array([[0, *tokens, 0]], dtype=np.int64)
        arrays[f"style_{i}_{j}"] = voices[row, :, :]
        chunk_meta.append({"j": j, "phonemes": len(ps), "tokens": len(tokens),
                           "pt_audio_s": round(len(np.asarray(c.audio)) / 24000, 3),
                           "style_row": row})
    items.append({"i": i, "chars": len(text), "chunks": chunk_meta,
                  "band": "short" if i < len(SHORT) else "medium" if i < len(SHORT) + len(MEDIUM) else "long",
                  "sha": hashlib.sha256(text.encode()).hexdigest()[:8]})

np.savez(W / "corpus.npz", **arrays)
json.dump({"n": len(CORPUS), "voice": "af_heart", "speed": 1.0, "items": items},
          open(W / "corpus_manifest.json", "w"), indent=1)
print(json.dumps({"n_items": len(CORPUS),
                  "n_chunks": sum(len(x["chunks"]) for x in items),
                  "tokens_per_chunk": [c["tokens"] for x in items for c in x["chunks"]]}))
