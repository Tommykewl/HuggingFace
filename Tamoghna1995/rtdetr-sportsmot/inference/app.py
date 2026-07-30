# ZeroGPU Gradio demo for smallTech/rtdetr-sportsmot — sports player detection.
# `import spaces` MUST precede torch: it monkey-patches torch.cuda so the
# module-scope `.to("cuda")` below is intercepted and the weights are streamed
# into the GPU worker on the first @spaces.GPU call.
import spaces
import os
import time

import torch
import gradio as gr
from transformers import AutoImageProcessor, AutoModelForObjectDetection

MODEL_ID = "smallTech/rtdetr-sportsmot"
TOKEN = os.environ.get("HF_TOKEN")      # gated model repo — provided as a Space secret

# Load once at module scope (ZeroGPU rule 2). String "cuda", never an int id.
processor = AutoImageProcessor.from_pretrained(MODEL_ID, token=TOKEN)
model = AutoModelForObjectDetection.from_pretrained(MODEL_ID, token=TOKEN).to("cuda").eval()


@spaces.GPU(duration=15)
def detect(image, threshold: float = 0.5):
    """Detect sports players in an image with RT-DETRv2 fine-tuned on SportsMOT.

    Args:
        image: Input image — a sports-broadcast frame works best.
        threshold: Confidence threshold; detections scoring below it are dropped.

    Returns:
        The image annotated with player boxes, plus a one-line summary.
    """
    if image is None:
        return None, "Upload an image first."
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
                   "or try a sports-broadcast frame.")
    return (image, annotations), summary


DESCRIPTION = """\
Single-class **player** detector —
[`smallTech/rtdetr-sportsmot`](https://huggingface.co/smallTech/rtdetr-sportsmot),
an RT-DETRv2 (r50vd) fine-tune on
[SportsMOT](https://huggingface.co/datasets/Lekim89/sportsmot).
**mAP@50 0.938** on 45 unseen val sequences · full benchmarks on the model card.
Built as the detection stage of a ByteTrack tracking-by-detection pipeline.
Running on **ZeroGPU** — drop a sports frame and tune the confidence threshold.
"""

demo = gr.Interface(
    fn=detect,
    inputs=[
        gr.Image(type="pil", label="Sports frame"),
        gr.Slider(0.05, 0.95, value=0.5, step=0.05, label="Confidence threshold"),
    ],
    outputs=[
        gr.AnnotatedImage(label="Detected players"),
        gr.Markdown(),
    ],
    title="🏀 RT-DETRv2 · SportsMOT player detection",
    description=DESCRIPTION,
    examples=[
        ["examples/basketball.jpg", 0.5],
        ["examples/frame2.jpg", 0.5],
        ["examples/frame3.jpg", 0.5],
    ],
    cache_examples=True,
    cache_mode="lazy",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(mcp_server=True)
