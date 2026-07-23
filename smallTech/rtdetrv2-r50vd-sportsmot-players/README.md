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

| File | Purpose |
|---|---|
| `trainer.py` | Training script for **Hugging Face Jobs** (`hf jobs uv run`). Self-contained uv script; targets an A10G GPU (bf16, batch 8). |
| `trainer.kaggle.ipynb` | The same training run as a **Kaggle** notebook, adapted for the free T4 GPU (fp16 instead of bf16, smaller batch + grad-accum, `pip install` of extras). Pushes to the Hub when an `HF_TOKEN` secret is attached, otherwise saves the model to the kernel output. |
| `inference.py` | Loads the trained model from the Hub, runs it on a video or a folder of frames, applies **ByteTrack**, and writes an annotated MP4 with persistent player IDs. |
| `README.md` | This file. |

The Kaggle launcher is the shared `kaggle_train.sh` at the **workspace root** — see
the [workspace README](../../README.md). It is generic and takes this model's path
as its argument; it does not live in this folder.

## The model

- **Task:** object detection, single class `player`.
- **Approach:** the COCO detection head of RT-DETRv2 is replaced with a one-class
  head (`ignore_mismatched_sizes=True`) while reusing the pretrained backbone.
- **Data:** only the `train/` split of SportsMOT carries ground-truth boxes
  (10 sequences across basketball / soccer / volleyball). We train on 9 of them
  and hold out one **basketball** sequence (`v_-6Os86HzwCs_c009`) for validation,
  so metrics reflect the target domain. Annotations are MOTChallenge `gt.txt`
  (`frame, track_id, x, y, w, h, conf, class, visibility`); only per-frame boxes
  are used for detection (track_id is ignored here, but is what ByteTrack rebuilds
  at inference time).

## Training configuration

| | `trainer.py` (HF Jobs) | `trainer.kaggle.ipynb` (Kaggle T4) |
|---|---|---|
| GPU | A10G (24 GB) | T4 (16 GB) |
| Precision | bf16 | fp16 (T4 has no bf16) |
| Batch / grad-accum | 8 / 2 | 4 / 4 |
| Effective batch | 16 | 16 |
| Epochs | 20 | 20 |
| Image size | 640×640 | 640×640 |
| LR / schedule | 5e-5 / cosine, 300 warmup | same |
| Frame stride | 2 (train), 5 (val) | same |
| Augmentation | hflip, brightness/contrast, hue/sat, gauss noise | same |
| Model selection | best `eval_loss` | same |

Both scripts push the model, image processor, and a metrics-filled model card to
the Hub when training completes.

## Usage

**Train on Hugging Face Jobs** (needs prepaid Jobs credits):

```bash
hf jobs uv run trainer.py --flavor a10g-large --timeout 6h --detach \
  --name rtdetrv2-sportsmot --secrets "HF_TOKEN=$(hf auth token)"
```

**Train on Kaggle (free):** either open `trainer.kaggle.ipynb` in a Kaggle notebook
(GPU T4 + Internet + **Add Data → your `external-secrets` dataset**, then Run All),
or drive it headless from the workspace root:

```bash
./kaggle_train.sh smallTech/rtdetrv2-r50vd-sportsmot-players
```

See the [workspace README](../../README.md) for Kaggle CLI setup and how the HF
token is supplied via the private `external-secrets` dataset.

**Run tracking on the trained model:**

```bash
# a video file
uv run inference.py --source game_clip.mp4

# a folder of frames (e.g. a SportsMOT test sequence)
uv run inference.py --source path/to/v_XXXX/img1 --fps 25
```

`inference.py` auto-selects CUDA / Apple MPS / CPU, so it runs locally on modest
hardware — only *training* needs a dedicated GPU.
