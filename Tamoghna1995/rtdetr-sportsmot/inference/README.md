---
title: RT-DETRv2 SportsMOT Player Detection
emoji: 🏀
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
python_version: "3.12"
short_description: Detect sports players with RT-DETRv2 (ZeroGPU)
models:
- smallTech/rtdetr-sportsmot
datasets:
- Lekim89/sportsmot
license: apache-2.0
---

# RT-DETRv2 · SportsMOT player detection

Live demo for
[`smallTech/rtdetr-sportsmot`](https://huggingface.co/smallTech/rtdetr-sportsmot) —
a single-class **player** detector fine-tuned from
[`PekingU/rtdetr_v2_r50vd`](https://huggingface.co/PekingU/rtdetr_v2_r50vd) on the
[SportsMOT](https://huggingface.co/datasets/Lekim89/sportsmot) train split, built
as the detection stage of a ByteTrack tracking-by-detection pipeline.

Drop a sports frame (broadcast footage works best), tune the confidence
threshold, and get annotated player boxes. Inference runs on **ZeroGPU** (a
per-request slice of an NVIDIA GPU). Benchmarks — **mAP@50 0.938** on 45 unseen
val sequences — live on the
[model card](https://huggingface.co/smallTech/rtdetr-sportsmot).

The model repo is gated; this Space reads a `HF_TOKEN` secret to download the
weights. Example images are frames from the SportsMOT `val` split.
