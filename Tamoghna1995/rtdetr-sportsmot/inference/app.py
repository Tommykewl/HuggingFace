# ZeroGPU Gradio demo for smallTech/rtdetr-sportsmot — sports player detection.
# `import spaces` MUST precede torch: it monkey-patches torch.cuda so the
# module-scope `.to("cuda")` below is intercepted and the weights are streamed
# into the GPU worker on the first @spaces.GPU call.
import spaces
import glob
import os
import random
import time
from io import BytesIO

import torch
import gradio as gr
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

MODEL_ID = "smallTech/rtdetr-sportsmot"
TOKEN = os.environ.get("HF_TOKEN")          # gated model repo — Space secret

# Example frames (SportsMOT test split) live in an HF Storage Bucket, mounted
# read-only at /data. A public bucket URL is the fallback if the mount isn't
# ready when the app first reads it.
BUCKET_ID = "Tamoghna1995/rtdetr-sportsmot-examples"
MOUNT_DIR = "/data/examples"
BUCKET_URL = f"https://huggingface.co/buckets/{BUCKET_ID}/resolve/examples"

# Load once at module scope (ZeroGPU rule 2). String "cuda", never an int id.
processor = AutoImageProcessor.from_pretrained(MODEL_ID, token=TOKEN)
model = AutoModelForObjectDetection.from_pretrained(MODEL_ID, token=TOKEN).to("cuda").eval()


def _list_examples():
    """Bucket example refs: local mount paths, else public bucket URLs."""
    if os.path.isdir(MOUNT_DIR):
        paths = sorted(glob.glob(f"{MOUNT_DIR}/*.jpg"))
        if paths:
            return paths
    try:                                     # mount not ready -> list via the API
        from huggingface_hub import HfApi
        names = [os.path.basename(str(getattr(t, "path", t)))
                 for t in HfApi().list_bucket_tree(BUCKET_ID, token=TOKEN)
                 if str(getattr(t, "path", t)).endswith(".jpg")]
        return [f"{BUCKET_URL}/{n}" for n in names]
    except Exception as exc:
        print("bucket listing failed:", exc)
        return []


_examples = None


def _examples_list():
    global _examples
    if not _examples:                        # lazy: mount may lag module import
        _examples = _list_examples()
    return _examples


def _load(ref):
    if ref.startswith("http"):
        import requests
        return Image.open(BytesIO(requests.get(ref, timeout=30).content)).convert("RGB")
    return Image.open(ref).convert("RGB")


def pick_random():
    """CPU-only: pick a random bucket frame and show it in the input."""
    refs = _examples_list()
    return _load(random.choice(refs)) if refs else None


@spaces.GPU(duration=15)
def detect(image, threshold: float = 0.5):
    """Detect sports players in an image with RT-DETRv2 fine-tuned on SportsMOT.

    Args:
        image: Input image — a sports-broadcast frame works best.
        threshold: Confidence threshold; detections below it are dropped.

    Returns:
        The image annotated with player boxes, plus a one-line summary.
    """
    if image is None:
        return None, "Click **🎲 Random test image**, or upload your own frame."
    t0 = time.perf_counter()
    inputs = processor(images=image, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    target = torch.tensor([image.size[::-1]]).to("cuda")
    (res,) = processor.post_process_object_detection(
        outputs, target_sizes=target, threshold=threshold)
    ms = (time.perf_counter() - t0) * 1000

    scores = res["scores"].tolist()
    annotations = [
        (tuple(int(v) for v in box), f"player {score:.2f}")
        for score, box in zip(scores, res["boxes"].tolist())
    ]
    if annotations:
        summary = (f"**{len(annotations)} players** at conf ≥ {threshold:g} "
                   f"(scores {min(scores):.2f}–{max(scores):.2f}) · {ms:.0f} ms on ZeroGPU.")
    else:
        summary = (f"No players at conf ≥ {threshold:g} — lower the threshold "
                   "or try another frame.")
    return (image, annotations), summary


TITLE = "# 🏀 RT-DETRv2 · SportsMOT player detection"
DESCRIPTION = """\
Single-class **player** detector —
[`smallTech/rtdetr-sportsmot`](https://huggingface.co/smallTech/rtdetr-sportsmot),
an RT-DETRv2 (r50vd) fine-tune on
[SportsMOT](https://huggingface.co/datasets/Lekim89/sportsmot).
**mAP@50 0.938** on 45 unseen val sequences · full benchmarks on the model card.
Running on **ZeroGPU**. Hit **🎲 Random test image** to detect on a random
frame from the SportsMOT **test** split (served from an HF Storage Bucket), or
upload your own sports frame.
"""

with gr.Blocks(title="RT-DETRv2 · SportsMOT player detection") as demo:
    gr.Markdown(TITLE)
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            image_in = gr.Image(type="pil", label="Sports frame")
            threshold = gr.Slider(0.05, 0.95, value=0.5, step=0.05,
                                  label="Confidence threshold")
            with gr.Row():
                random_btn = gr.Button("🎲 Random test image", variant="primary")
                detect_btn = gr.Button("Detect")
        with gr.Column():
            annotated = gr.AnnotatedImage(label="Detected players")
            summary = gr.Markdown()

    # Random: load a bucket frame (CPU) into the input, then run the single
    # GPU detect on it. detect_btn binds `detect` directly so ZeroGPU's
    # startup scan finds the @spaces.GPU function.
    random_btn.click(pick_random, outputs=image_in).then(
        detect, [image_in, threshold], [annotated, summary])
    detect_btn.click(detect, [image_in, threshold], [annotated, summary])

if __name__ == "__main__":
    demo.launch(mcp_server=True)
