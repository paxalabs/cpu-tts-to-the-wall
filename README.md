# Pushing CPU Speech Synthesis to the Wall: research artifact

Measurement scripts and raw measurement logs behind the
paper *Pushing CPU Speech Synthesis to the Wall: Extreme Inference Tuning
under Serverless Architecture and Billing* (Pakorn Nathong and Kunat
Pipatanakul, Paxa Labs).

Repository: https://github.com/paxalabs/cpu-tts-to-the-wall

**Provided as-is.** This is a record of a measurement campaign, not a
library and not production software. It is published so that every number
in the paper can be traced to a raw log and to the script that produced
it. No reproduction support is promised; the containers it ran on were
rented and have been deleted.

## What the paper's numbers come from

Every experimental value in the paper is computed from the files in
`results/`; nothing is hand-typed. The scripts that turn these rows into
the paper's macros, tables, and figures live with the paper source, not
here. This repository holds what ran on the measurement containers and
what they wrote.

## Layout

| file | role | reads / writes |
| --- | --- | --- |
| `serve_shim.py` | one HTTP server, three backends (`BACKEND=pt`, `ort`, `ov`) behind the same grapheme-to-phoneme front end; the OpenVINO backend carries the instance lifecycle (idle release, allocator trim, page-cache write-back and drop, restore from the compile cache) | env: `TORCH_THREADS`, `ORT_THREADS`, `OV_THREADS`, `OV_STREAMS`, `OV_CACHE_DIR`, `IDLE_RELEASE_SEC` |
| `loadgen_local.py` | closed-loop load generator run inside the serving container; measures a steady window and brackets the server process's CPU time around it | `corpus_texts.json` |
| `assets_probe.py` | downloads the public model assets, runs one probe utterance through all three backends, records hashes, timings, and cross-backend SNR | `results/results_assets.json` |
| `corpus_build.py`, `corpus_texts.py` | the fixed length-coverage corpus: phonemized once, token ids and style rows frozen | `results/corpus_manifest.json`, `corpus.npz`, `corpus_texts.json` |
| `equivalence_lengths.py`, `equivalence_snr.py`, `equivalence_mel.py` | cross-backend output equivalence: per-chunk audio-length deltas, determinism floors and SNR, log-mel distances | `results/results_equiv.json`, `results_equiv2.json`, `results_equiv3.json` |
| `matrix_main.py` | the main measurement matrix: cold start, lone-request latency, sustained closed-loop load, ten-minute idle decay, per configuration | `results/results_matrix.jsonl` |
| `matrix_rerun_pool_fix.py` | rerun of the OpenVINO sustained rows after the inference-request pool fix (rows suffixed `-v2`) | `results/results_matrix.jsonl` |
| `matrix_fill.py` | the fill pass: the four table cells the primary campaign did not sample (sized ONNX Runtime idle, `ov-cache` idle, `ov-8s` cold start, cached eight-stream cold start), plus same-container anchors, on a second container of the primary host's class; refuses to run on any other CPU model | `results/results_matrix_fill.jsonl` |
| `matrix_lifecycle.py`, `matrix_lifecycle_8streams.py` | the paper's configuration with four and with eight streams: cold start, lone latency, sustained load, idle decay through a release, restore latency | `results/results_matrix.jsonl` |
| `matrix_restore.py` | the restore repeats: ten release-and-restore cycles of `ov-ours8` after a sustained window, preceded by one fresh and three cached process cold starts of the same configuration, on a second container of each host class (`WANT_CPU` selects the host; refuses to run on any other CPU model) | `results/results_matrix_restore.jsonl`, `results_matrix_restore_epyc.jsonl` |
| `pad_and_fusion_probe.py` | pad-tolerance test and the dynamic-versus-static per-operator profile (run on an otherwise idle container) | `results/results_g3.json` |
| `matrix_second_arch.py` | the second-architecture pass: full setup, then a reduced matrix | `results/results_matrix_epyc.jsonl` |
| `pocket_protocol.py`, `pocket_sustained.py` | the transfer check on pocket-tts, served by its own HTTP server in its own container | `results/results_pocket_full.json`, `results_pocket_sus.json` |
| `box/Dockerfile`, `box/Dockerfile.pocket` | the container images the measurements ran in (pinned versions) | |
| `results/` | raw measurement outputs, committed verbatim | |

The scripts were renamed from their on-box working names for readability;
the bodies are unchanged except for the corresponding references between
scripts. On-box names: `c_serve.py` (serve_shim), `c_load.py`
(loadgen_local), `c_assets.py`, `c_corpus.py`, `c_texts.py`, `c_equiv.py`,
`c_equiv2.py` (equivalence_snr), `c_equiv3.py` (equivalence_mel),
`c_matrix.py`, `c_matrix2.py` (matrix_rerun_pool_fix), `c_matrix3.py`
(matrix_lifecycle), `c_matrix3b.py` (matrix_lifecycle_8streams),
`c_g3b.py` (pad_and_fusion_probe), `c_epyc.py` (matrix_second_arch),
`c_pocket_full.py` (pocket_protocol), `c_pocket_sus.py` (pocket_sustained).

## How the measurements were taken

**Model.** Kokoro-82M, the public fp32 community ONNX export
(`onnx-community/Kokoro-82M-v1.0-ONNX`), with the voice and tokenizer
configuration from `hexgrad/Kokoro-82M`; grapheme-to-phoneme conversion by
the reference `kokoro` pipeline for every backend. `assets_probe.py`
downloads these and records the ONNX file hash in
`results/results_assets.json`. Weights are not part of this repository.

**Hosts.** A 32-vCPU container on an Intel Xeon Platinum 8581C machine
(primary host) and a 48-vCPU container on an AMD EPYC 9655 machine, rented
from a container cloud with shared tenancy. Every comparison in the paper
is between rows measured in the same container; numbers are not compared
across containers. The transfer check ran in a third container of the
primary host's class.

**Software.** `box/Dockerfile`: `python:3.12-slim`, PyTorch 2.6.0 (CPU),
ONNX Runtime 1.20.1, OpenVINO 2026.3.0, `kokoro`, `misaki[en]`, `espeak-ng`.
OpenVINO 2026.3 is required: 2025.3 rejects the dynamic-shape `STFT` in
this export. `box/Dockerfile.pocket`: PyTorch 2.6.0 (CPU) and `pocket-tts`.
The images were driven by a small job runner (upload a script, run it,
fetch the results) that is not part of the artifact.

**Run order** (all scripts run from `/work` inside the container, in this
order):

1. `assets_probe.py`, `corpus_build.py`, `corpus_texts.py`
2. `equivalence_lengths.py`, `equivalence_snr.py`, `equivalence_mel.py`
3. `matrix_main.py`, then `matrix_rerun_pool_fix.py`
4. `matrix_lifecycle.py`, then `matrix_lifecycle_8streams.py`
5. `pad_and_fusion_probe.py` (with nothing else running)
6. `matrix_second_arch.py` on the second host (it performs step 1 itself)
7. `pocket_protocol.py`, then `pocket_sustained.py` in the pocket-tts container
8. `matrix_fill.py` on a fresh container of the primary host's class (it performs step 1 itself)
9. `matrix_restore.py` on a fresh container of each host class, `WANT_CPU=8581C` and `WANT_CPU=9655` (it performs step 1 itself)

**Configurations.** Each matrix script starts `serve_shim.py` with the
environment of one configuration, drives it with `loadgen_local.py`, and
appends one JSON row per measurement. The configuration names used in the
results (`pt-default`, `pt-4t`, `ort-default`, `ort-4t`, `ov-default`,
`ov-4t4s`, `ov-cache`, `ov-8s`, `ov-8s-cache`, `ov-ours`, `ov-ours8`) are defined, with
their exact environment, in the tables at the top of `matrix_main.py`,
`matrix_lifecycle.py`, `matrix_lifecycle_8streams.py`, `matrix_fill.py`,
`matrix_restore.py`, and `matrix_second_arch.py`. The fill pass and the
restore repeats ran in different containers from the primary campaign;
their cells carry a dagger in the paper's tables and their `host` rows
record the CPU model and cgroup limits they saw. `ov-ours8` is the paper's "This work".

## Raw result files

`results/results_matrix.jsonl`, `results/results_matrix_epyc.jsonl`,
`results/results_matrix_fill.jsonl`, `results/results_matrix_restore.jsonl`, and
`results/results_matrix_restore_epyc.jsonl` hold one JSON object per line, distinguished by `kind`:

| `kind` | fields |
| --- | --- |
| `host` | `cpu`, `cpus` |
| `cold`, `cold-cached` | `cfg`, `ready_s`, `cold_first_audio_s` (process start to first audio; `cold-cached` with a populated compile cache) |
| `lone` | `cfg`, `band` (`short`, `medium`, `long`), `first_ms`, `wall_s`, `audio_s` (one request on an idle server) |
| `sustained` | `cfg`, `clients`, `window_s`, `reqs`, `audio_s`, `server_cpu_s`, `audio_per_cpu_s`, `throughput_x`, `first_p50_ms`, `first_p90_ms`, `lat_p50_s`, `bad`, `label` |
| `idle` | `cfg`, `after_s` (seconds after the last response), `server_rss_mb`, `cgroup_mb` |
| `restore` | `cfg`, `first_ms`, `wall_s` (first request after an idle release); the restore repeats add `cycle`, `released`, `release_seen_s`, `quiet_s`, `rss_before_mb`, `cgroup_before_mb`, `audio_s` |
| `warm-after-restore` | `cfg`, `cycle`, `first_ms`, `wall_s`, `rss_after_mb`, `cgroup_after_mb` (the same request repeated immediately after a restore; restore repeats only) |
| `done*` | end-of-job markers with timestamps |

`audio_per_cpu_s` is the paper's sustained efficiency (audio-seconds
produced per CPU-second of the server process); `throughput_x` is audio
seconds per wall second. Rows with `bad > 0` are excluded by the analysis.

The remaining files: `corpus_manifest.json` (the corpus, per chunk:
characters, tokens, chunk index); `results_assets.json` (asset hashes and
probe timings); `results_equiv*.json` (equivalence checks);
`results_g3.json` (pad tolerance and per-operator profiles, dynamic and
static); `results_pocket_full.json` and `results_pocket_sus.json` (the
transfer check); `money_perday.json` (derived from the sustained and idle rows:
the bursty-day cost decomposition under the paper's worked-example rates
and under the alternative published rates).

## Citation

```bibtex
@misc{nathong2026wall,
  title  = {Pushing CPU Speech Synthesis to the Wall: Extreme Inference Tuning
            under Serverless Architecture and Billing},
  author = {Nathong, Pakorn and Pipatanakul, Kunat},
  year   = {2026},
  note   = {Paxa Labs},
  howpublished = {\url{https://github.com/paxalabs/cpu-tts-to-the-wall}}
}
```

## License

MIT, see `LICENSE`.
