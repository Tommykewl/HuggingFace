---
title: RT-DETR SportsMOT Player Detection
emoji: 🏀
colorFrom: green
colorTo: blue
sdk: static
app_file: index.html
short_description: In-browser sports player detection (RT-DETRv2)
models:
- smallTech/rtdetr-sportsmot
datasets:
- Lekim89/sportsmot
license: apache-2.0
---

# RT-DETRv2 · SportsMOT player detection — in-browser

Live demo for
[`smallTech/rtdetr-sportsmot`](https://huggingface.co/smallTech/rtdetr-sportsmot) —
a single-class **player** detector fine-tuned from
[`PekingU/rtdetr_v2_r50vd`](https://huggingface.co/PekingU/rtdetr_v2_r50vd) on the
[SportsMOT](https://huggingface.co/datasets/Lekim89/sportsmot) train split, built
as the detection stage of a ByteTrack tracking-by-detection pipeline.

Inference runs **entirely in the browser** via
[transformers.js](https://huggingface.co/docs/transformers.js) and an ONNX
q8-quantized export of the model (~45 MB, cached after first load) — this is a
free Static Space with no server-side compute, and uploaded images never leave
the page.

Benchmarks — **mAP@50 0.938** on 45 unseen val sequences, 28 fps batched-fp16
on a T4 — live on the
[model card](https://huggingface.co/smallTech/rtdetr-sportsmot). The q8 export
was parity-validated against the PyTorch reference (same 7 detections on the
validation frame, scores within ±0.04).

Example images are frames from the SportsMOT `val` split.
