# rtdetr-sportsmot

An [RT-DETRv2](https://huggingface.co/PekingU/rtdetr_v2_r50vd) object detector
fine-tuned to detect **basketball players** (single class: `player`), plus a
ByteTrack inference script that turns the per-frame detections into consistent
player tracks. This is the *detection* half of a tracking-by-detection pipeline
for basketball player tracking.

- **Hub model:** [`smallTech/rtdetr-sportsmot`](https://huggingface.co/smallTech/rtdetr-sportsmot)
- **Base model:** [`PekingU/rtdetr_v2_r50vd`](https://huggingface.co/PekingU/rtdetr_v2_r50vd) (pretrained on COCO)
- **Dataset:** [`Lekim89/sportsmot`](https://huggingface.co/datasets/Lekim89/sportsmot) — SportsMOT, MOTChallenge-format sports clips

## Files

This folder holds only `externals.json` (and this README): every operation —
data staging, training, and benchmarking — is delegated to **Kaggle** (free
tier) and lives in
[`external/kaggle/rtdetr-sportsmot/`](../../external/kaggle/rtdetr-sportsmot/):

| File | Purpose |
|---|---|
| `training/prepare-data.kaggle.ipynb` | CPU-only kernel (no GPU quota), run **once**: mirrors the SportsMOT train split inside Kaggle's network and outputs `sportsmot-train.tar` (6.6 GB) as its kernel output. |
| `evaluation/prepare-data.kaggle.ipynb` | Same, for the **val** split (~6 GB → `sportsmot-val.tar`) — the unseen, annotated data the benchmark runs on. |
| `training/index.kaggle.ipynb` | The trainer (free T4, fp16, batch 4 + grad-accum 4). Mounts the staged tar, trains, evaluates mAP on a held-out basketball sequence, and pushes model + card to the Hub. |
| `training/smoketest.kaggle.ipynb` | Few-minute environment check run before the trainer: GPU/torch kernel compatibility, deps, Hub auth, the staged-data mount + untar, and a mock training. |
| `evaluation/index.kaggle.ipynb` | The benchmark (free T4): pulls the trained model from the Hub and evaluates it on the **unseen val split** (45 annotated sequences) — overall + per-sequence COCO mAP/mAR, latency percentiles/fps — logging everything and writing `benchmarks.json` to the kernel output. |
| `evaluation/smoketest.kaggle.ipynb` | Few-minute check run before the benchmark: GPU/torch, deps, token, val-tar mount, gt parsing, model download, mini eval. |
| `testing/prepare-data-1.kaggle.ipynb` + `-2` | CPU kernels: stage the FULL `test/` split as two mutually exclusive halves (even-/odd-indexed sequences — the ~21 GB split exceeds Kaggle's 20 GB per-kernel output cap). |
| `testing/index.kaggle.ipynb` | Performance-only benchmark (free T4) over the whole split: batched fp16 pipeline for max GPU utilization — throughput, single-image latency percentiles, detections/frame, confidence stats. No accuracy metrics (test has no public ground truth). Writes `benchmarks.json` and publishes the section to the model card. |
| `testing/smoketest.kaggle.ipynb` | Few-minute check before the performance benchmark. |
| `*.config.json` (next to each notebook) | Kernel options (accelerator `machine_shape`, internet, `dataset_sources`, `kernel_sources` — including the `external-secrets` token dataset and the staging kernels' outputs) consumed by the kaggle service runner (`external/kaggle/service.py`). |

## The model

- **Task:** object detection, single class `player`.
- **Approach:** the COCO detection head of RT-DETRv2 is replaced with a one-class
  head (`ignore_mismatched_sizes=True`) while reusing the pretrained backbone.
- **Data:** only the `train/` split of SportsMOT carries ground-truth boxes
  (45 sequences, ~28.5k frames / 6.6 GB, across basketball / soccer / volleyball).
  One **basketball** sequence (`v_-6Os86HzwCs_c009`) is held out for validation,
  so metrics reflect the target domain; the rest are training. Annotations are
  MOTChallenge `gt.txt` (`frame, track_id, x, y, w, h, conf, class, visibility`);
  only per-frame boxes are used for detection (track_id is ignored here, but is
  what ByteTrack rebuilds at inference time).
- **Data staging:** bulk-fetching ~28k small files from the Hub proved
  unreliable everywhere it was tried (in-session downloads ate a whole GPU
  session; residential IPs trip the Hub CDN's burst protection). So the data is
  staged **once, inside Kaggle's own network** by the `training/prepare-data` CPU kernel,
  whose output tar training kernels mount (`kernel_sources`) and untar locally
  in ~2 minutes. The trainer auto-detects staged data and only falls back to
  the Hub download (with a loud warning) when none is mounted.

## Training configuration (Kaggle T4)

| | value |
|---|---|
| GPU | T4 (16 GB; one of the session's two — DataParallel corrupts detection labels) |
| Precision | fp16 (T4 has no native bf16; emulated bf16 is several times slower) |
| Batch / grad-accum | 4 / 4 (effective 16) |
| Epochs | 20 |
| Image size | 640×640 |
| LR / schedule | 5e-5 / cosine, 300 warmup |
| Frame stride | 2 (train), 5 (val) |
| Augmentation | hflip, brightness/contrast, hue/sat, gauss noise |
| Model selection | best `eval_loss` |

The trainer pushes the model, image processor, and a metrics-filled model card
to the Hub when training completes.

## Usage

All from the workspace root via the launcher (`run.sh`; Windows: `run.ps1`);
see the [workspace README](../../README.md) for the one-time `.env` setup.

**Train on Kaggle (free):**

```bash
./run.sh smallTech/rtdetr-sportsmot/training/prepare-data      # once: stage train data (CPU)
./run.sh smallTech/rtdetr-sportsmot/training/smoketest    # minutes: env check
./run.sh smallTech/rtdetr-sportsmot/training/index        # the real training run
```

**Benchmark the trained model on unseen data (val split):**

```bash
./run.sh smallTech/rtdetr-sportsmot/evaluation/prepare-data    # once: stage val data (CPU)
./run.sh smallTech/rtdetr-sportsmot/evaluation/smoketest  # minutes: env check
./run.sh smallTech/rtdetr-sportsmot/evaluation/index      # the full benchmark
```

The benchmark logs overall + per-sequence mAP/mAR and latency, saves
`benchmarks.json` to the kernel's Output tab, and publishes the section to
the Hub model card. The val split is the accuracy benchmark set because
training used only `train/` sequences and SportsMOT's `test/` split has no
public ground truth.

**Performance-benchmark on the test split** (no accuracy — no ground truth):

```bash
./run.sh smallTech/rtdetr-sportsmot/testing/prepare-data-1 # once: stage test half 1 (CPU)
./run.sh smallTech/rtdetr-sportsmot/testing/prepare-data-2 # once: stage test half 2 (CPU)
./run.sh smallTech/rtdetr-sportsmot/testing/smoketest      # minutes: env check
./run.sh smallTech/rtdetr-sportsmot/testing/index          # the performance benchmark
```

Also writes `benchmarks.json` and updates its own model-card section.
