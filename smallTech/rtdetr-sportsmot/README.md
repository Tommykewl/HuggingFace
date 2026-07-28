---
license: apache-2.0
base_model: PekingU/rtdetr_v2_r50vd
datasets:
- Lekim89/sportsmot
pipeline_tag: object-detection
tags:
- rt-detr-v2
- sports
- basketball
- player-detection
- tracking
model-index:
- name: rtdetr-sportsmot
  results:
  - task:
      type: object-detection
      name: Player detection
    dataset:
      name: SportsMOT (val split, unseen by training)
      type: Lekim89/sportsmot
      split: validation
    metrics:
    - type: map
      name: mAP@[.5:.95]
      value: 0.7928
    - type: map50
      name: mAP@50
      value: 0.9383
    - type: map75
      name: mAP@75
      value: 0.882
    - type: mar100
      name: mAR@100
      value: 0.8269
    source:
      name: Kaggle evaluation kernel (benchmarks.json)
      url: https://www.kaggle.com/code/tamobiswas/rtdetr-sportsmot-evaluation
co2_eq_emissions:
  emissions: 300
  source: 'post-hoc estimate, ML CO2 Impact methodology: 70W Tesla T4 x ~9.5h training x ~1.1
    PUE x ~430 gCO2eq/kWh (measured tracking via codecarbon added for future runs)'
  training_type: fine-tuning
  geographical_location: Google Cloud via Kaggle free tier (region not disclosed, US assumed)
  hardware_used: 1x NVIDIA Tesla T4 (16 GB)
---

# rtdetr-sportsmot — RT-DETRv2 (r50vd) fine-tuned on SportsMOT for player detection

Single-class (`player`) detector fine-tuned from `PekingU/rtdetr_v2_r50vd` on the annotated
train sequences of [Lekim89/sportsmot](https://huggingface.co/datasets/Lekim89/sportsmot)
(basketball, soccer, and volleyball broadcast clips in MOTChallenge format).
Intended as the detection stage of a tracking-by-detection pipeline (e.g. with
ByteTrack via the `supervision` library) for basketball player tracking.

## Try it in one click

The repo ships a ready-to-run inference notebook (`notebook.ipynb`): open it
directly in [Colab](https://huggingface.co/smallTech/rtdetr-sportsmot/colab)
or [Kaggle](https://huggingface.co/smallTech/rtdetr-sportsmot/kaggle), or via
the "Use this model" button above — install → load → detect → visualize, no
token needed, CPU or GPU.

## Validation (held-out basketball sequence `v_-6Os86HzwCs_c009`)

| metric | value |
|---|---|
| mAP@[.5:.95] | 0.7809 |
| mAP@50 | 0.9597 |
| mAP@75 | 0.8700 |
| mAR@100 | 0.8081 |

## Usage

```python
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

processor = AutoImageProcessor.from_pretrained("smallTech/rtdetr-sportsmot")
model = AutoModelForObjectDetection.from_pretrained("smallTech/rtdetr-sportsmot")

image = Image.open("frame.jpg")
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
results = processor.post_process_object_detection(
    outputs, target_sizes=torch.tensor([image.size[::-1]]), threshold=0.5
)
```

## Training details

- **Approach:** RT-DETRv2's COCO detection head is replaced with a one-class
  (`player`) head while reusing the pretrained backbone.
- **Data:** the annotated `train/` split of SportsMOT — 44 of its 45
  sequences for training (14,049 frames at stride 2, frames without boxes
  skipped), with one basketball sequence (`v_-6Os86HzwCs_c009`, 100 frames at
  stride 5) held out for validation. Annotations are MOTChallenge `gt.txt`
  boxes; `conf==0` ignore-boxes are skipped, track ids are unused for
  detection. The `val/` and `test/` splits were never touched by training —
  they are the benchmark sets below.
- **Training** ran entirely on Kaggle's free tier (single T4, fp16): batch 4
  with 4-step gradient accumulation (effective 16), 20 epochs at 640×640,
  lr 5e-5 cosine with 300-step warmup, augmentation (hflip, color jitter,
  gaussian noise).
- **Model selection:** checkpoint with the best held-out `eval_loss` —
  epoch 11 of 20 (later epochs overfit).

## Benchmark — unseen val split (45 sequences)

Evaluated on **every annotated frame of the SportsMOT `val` split** —
26,970 frames / 295,573 ground-truth boxes across
45 sequences (basketball, soccer, volleyball) **never used in
training** (training consumed only the `train/` split; `test/` has no public
ground truth). Frame stride 1; same metric implementation as
the training-time validation (torchmetrics COCO mAP, detection threshold 0.01).

| metric | value |
|---|---|
| mAP@[.5:.95] | **0.7928** |
| mAP@50 | 0.9383 |
| mAP@75 | 0.8820 |
| mAR@100 | 0.8269 |

**Latency** (single image, batch 1, fp32, Tesla T4, torch 2.10.0+cu128):
mean 61.6 ms · p50 61.4 ms · p90 62.8 ms ·
p99 66.3 ms → **16.2 fps**. Model load:
5.4 s.

<details>
<summary>Per-sequence results (sorted by mAP)</summary>

| sequence | mAP@[.5:.95] | mAP@50 | mAP@75 |
|---|---|---|---|
| `v_4-EmEtrturE_c009` | 0.899 | 0.980 | 0.959 |
| `v_cC2mHWqMcjk_c009` | 0.868 | 0.970 | 0.937 |
| `v_cC2mHWqMcjk_c007` | 0.859 | 0.970 | 0.937 |
| `v_00HRwkvvjtQ_c003` | 0.852 | 0.980 | 0.934 |
| `v_cC2mHWqMcjk_c008` | 0.839 | 0.959 | 0.906 |
| `v_4r8QL_wglzQ_c001` | 0.835 | 0.970 | 0.915 |
| `v_5ekaksddqrc_c005` | 0.832 | 0.969 | 0.913 |
| `v_G-vNjfx1GGc_c008` | 0.825 | 0.976 | 0.943 |
| `v_dw7LOz17Omg_c067` | 0.823 | 0.954 | 0.923 |
| `v_5ekaksddqrc_c002` | 0.822 | 0.979 | 0.908 |
| `v_5ekaksddqrc_c004` | 0.821 | 0.979 | 0.908 |
| `v_G-vNjfx1GGc_c004` | 0.819 | 0.964 | 0.931 |
| `v_dw7LOz17Omg_c053` | 0.816 | 0.958 | 0.913 |
| `v_5ekaksddqrc_c003` | 0.814 | 0.970 | 0.900 |
| `v_2QhNRucNC7E_c017` | 0.814 | 0.923 | 0.902 |
| `v_00HRwkvvjtQ_c001` | 0.812 | 0.960 | 0.890 |
| `v_0kUtTtmLaJA_c004` | 0.810 | 0.950 | 0.895 |
| `v_0kUtTtmLaJA_c008` | 0.809 | 0.969 | 0.882 |
| `v_0kUtTtmLaJA_c010` | 0.807 | 0.970 | 0.893 |
| `v_ITo3sCnpw_k_c010` | 0.804 | 0.914 | 0.894 |
| `v_0kUtTtmLaJA_c007` | 0.801 | 0.960 | 0.868 |
| `v_i2_L4qquVg0_c010` | 0.796 | 0.936 | 0.905 |
| `v_BgwzTUxJaeU_c014` | 0.791 | 0.948 | 0.878 |
| `v_i2_L4qquVg0_c007` | 0.791 | 0.921 | 0.898 |
| `v_ITo3sCnpw_k_c012` | 0.790 | 0.928 | 0.906 |
| `v_00HRwkvvjtQ_c005` | 0.789 | 0.950 | 0.870 |
| `v_00HRwkvvjtQ_c008` | 0.788 | 0.959 | 0.864 |
| `v_BgwzTUxJaeU_c008` | 0.781 | 0.959 | 0.845 |
| `v_5ekaksddqrc_c001` | 0.777 | 0.960 | 0.876 |
| `v_0kUtTtmLaJA_c005` | 0.776 | 0.940 | 0.862 |
| `v_i2_L4qquVg0_c009` | 0.774 | 0.916 | 0.873 |
| `v_00HRwkvvjtQ_c011` | 0.773 | 0.949 | 0.855 |
| `v_ITo3sCnpw_k_c011` | 0.773 | 0.915 | 0.872 |
| `v_0kUtTtmLaJA_c006` | 0.771 | 0.938 | 0.862 |
| `v_i2_L4qquVg0_c006` | 0.766 | 0.911 | 0.879 |
| `v_9MHDmAMxO5I_c009` | 0.760 | 0.911 | 0.844 |
| `v_9MHDmAMxO5I_c002` | 0.752 | 0.940 | 0.849 |
| `v_00HRwkvvjtQ_c007` | 0.752 | 0.938 | 0.846 |
| `v_BgwzTUxJaeU_c012` | 0.746 | 0.938 | 0.841 |
| `v_9MHDmAMxO5I_c003` | 0.743 | 0.910 | 0.819 |
| `v_ITo3sCnpw_k_c007` | 0.734 | 0.878 | 0.835 |
| `v_9MHDmAMxO5I_c006` | 0.727 | 0.910 | 0.791 |
| `v_G-vNjfx1GGc_c600` | 0.723 | 0.952 | 0.869 |
| `v_G-vNjfx1GGc_c601` | 0.649 | 0.878 | 0.724 |
| `v_9MHDmAMxO5I_c004` | 0.633 | 0.782 | 0.704 |

</details>

<!-- benchmark:test:start -->
## Benchmark — test split (performance only)

Measured on the **full SportsMOT `test` split** (150
sequences, 94,835 frames), staged as two mutually exclusive
halves to fit Kaggle's kernel-output cap. The test split has **no public
ground truth** (withheld for the SportsMOT challenge server), so no accuracy
metrics are possible here — see the val-split benchmark above for those.

| metric (Tesla T4) | value |
|---|---|
| throughput (batched x8, fp16 autocast, 4-thread prefetch) | **28.0 fps** (GPU-timed) · 26.5 fps wall-clock |
| single-image latency (batch 1, fp16, 300-frame probe) | mean 74.4 ms · p50 62.5 · p90 68.1 · p99 90.1 |
| detections per frame (conf ≥ 0.5) | mean 10.2 · p50 10 · max 23 |
| mean confidence of detections | 0.920 |
| model load time | 5.2 s |

**Environment & method** — the assumptions these numbers were measured under:

| | |
|---|---|
| GPU | Tesla T4 (sm_75, 15.6 GB) |
| Software | torch 2.10.0+cu128 · CUDA 12.8 · transformers 5.0.0 · Python 3.12.13 |
| CPU | 4 vCPUs (image decode / prefetch) |
| Batching & precision | 8 images per forward, fp16 autocast (fp32 weights) |
| Parallelism | 4-thread image prefetch; part-2 tar extracted in a background thread during part-1 compute |
| Timing scope | CUDA-synchronized: preprocess + forward + postprocess per batch |
| Latency probe | separate batch-1 pass over the first 300 frames |
| Inputs | frame stride 1, confidence ≥ 0.5 |
<!-- benchmark:test:end -->

## Pipeline & reproducibility

This model was trained, evaluated, and benchmarked end-to-end on Kaggle's
free tier, orchestrated from the
[HuggingFace workspace repository](https://github.com/Tommykewl/HuggingFace)
(staging → smoketest → run for each stage). The benchmark sections of this
card are written by the evaluation/testing kernels themselves and update in
place on re-runs.
