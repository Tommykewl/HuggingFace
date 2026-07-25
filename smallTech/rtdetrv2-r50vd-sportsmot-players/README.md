# rtdetrv2-r50vd-sportsmot-players

An [RT-DETRv2](https://huggingface.co/PekingU/rtdetr_v2_r50vd) object detector
fine-tuned to detect **basketball players** (single class: `player`), plus a
ByteTrack inference script that turns the per-frame detections into consistent
player tracks. This is the *detection* half of a tracking-by-detection pipeline
for basketball player tracking.

- **Hub model:** [`smallTech/rtdetrv2-r50vd-sportsmot-players`](https://huggingface.co/smallTech/rtdetrv2-r50vd-sportsmot-players)
- **Base model:** [`PekingU/rtdetr_v2_r50vd`](https://huggingface.co/PekingU/rtdetr_v2_r50vd) (pretrained on COCO)
- **Dataset:** [`Lekim89/sportsmot`](https://huggingface.co/datasets/Lekim89/sportsmot) — SportsMOT, MOTChallenge-format sports clips

## Files

This folder (the model's first-party pipeline):

| File | Purpose |
|---|---|
| `inferenceandtesting/inference.py` | Loads the trained model from the Hub, runs it on a video or a folder of frames, applies **ByteTrack**, and writes an annotated MP4 with persistent player IDs. Runs locally (CUDA / MPS / CPU). |
| `inferenceandtesting/inference.config.json` | How to run it, plus the collated `hf jobs uv run` option reference for a cloud-GPU alternative. |
| `externals.json` | Declares the external services this model uses — currently the Kaggle external below. |
| `README.md` | This file. |

Training and data staging run on **Kaggle** (free tier) and live in
[`external/kaggle/rtdetrv2-r50vd-sportsmot-players/`](../../external/kaggle/rtdetrv2-r50vd-sportsmot-players/):

| File | Purpose |
|---|---|
| `data-preparation/prepare-data.kaggle.ipynb` | CPU-only kernel (no GPU quota), run **once**: mirrors the SportsMOT train split inside Kaggle's network and outputs `sportsmot-train.tar` (6.6 GB) as its kernel output. |
| `trainingandevaluation/train.kaggle.ipynb` | The trainer (free T4, fp16, batch 4 + grad-accum 4). Mounts the staged tar, trains 20 epochs, evaluates mAP on a held-out basketball sequence, and pushes model + card to the Hub. |
| `trainingandevaluation/smoketest.kaggle.ipynb` | Few-minute environment check run before the trainer: GPU/torch kernel compatibility, deps, Hub auth, the staged-data mount + untar, and a 25-step mock training. |
| `*.config.json` (next to each notebook) | Kernel options (accelerator `machine_shape`, internet, `dataset_sources`, `kernel_sources` — including the `external-secrets` token dataset and the staging kernel's output) consumed by the kaggle service runner (`external/kaggle/service.py`). |

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
  staged **once, inside Kaggle's own network** by the `prepare-data` CPU kernel,
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

**Train on Kaggle (free)** — from the workspace root via the launcher
(`run.sh`; Windows: `run.ps1`); see the [workspace README](../../README.md) for
one-time Kaggle CLI + `external-secrets` setup:

```bash
./run.sh smallTech/rtdetrv2-r50vd-sportsmot-players/data-preparation/prepare-data   # once: stage data (CPU)
./run.sh smallTech/rtdetrv2-r50vd-sportsmot-players/trainingandevaluation/smoketest # minutes: env check
./run.sh smallTech/rtdetrv2-r50vd-sportsmot-players/trainingandevaluation/train     # the real training run
```

**Run tracking on the trained model:**

```bash
# a video file
uv run inferenceandtesting/inference.py --source game_clip.mp4

# a folder of frames (e.g. a SportsMOT test sequence)
uv run inferenceandtesting/inference.py --source path/to/v_XXXX/img1 --fps 25
```

`inference.py` auto-selects CUDA / Apple MPS / CPU, so it runs locally on modest
hardware — only *training* needs a dedicated GPU.
