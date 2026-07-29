# Gradio demo for smallTech/rtdetr-sportsmot — player detection on sports
# frames. Runs on the free cpu-basic tier: the model is 42M params and takes
# a few seconds per image on 2 vCPUs. The model repo is gated, so the Space
# reads HF_TOKEN from its secrets to download the weights.
import os

import gradio as gr
from transformers import pipeline

MODEL_ID = "smallTech/rtdetr-sportsmot"

# Module-scope load: one download + init at startup, shared by all requests.
pipe = pipeline(
    "object-detection",
    model=MODEL_ID,
    device="cpu",
    token=os.environ.get("HF_TOKEN"),      # gated repo — Space secret
)


def detect_players(image, threshold: float = 0.5):
    """Detect sports players in an image with RT-DETRv2 fine-tuned on SportsMOT.

    Args:
        image: Input image (a frame from a sports broadcast works best).
        threshold: Confidence threshold — detections scoring below it are dropped.

    Returns:
        The image with player bounding boxes, and a text summary of the
        detections (count and confidence range).
    """
    if image is None:
        return None, "Upload an image first."
    detections = pipe(image, threshold=threshold)
    sections = []
    for det in detections:
        box = det["box"]
        sections.append(
            ((box["xmin"], box["ymin"], box["xmax"], box["ymax"]),
             f"player {det['score']:.2f}")
        )
    if detections:
        scores = [d["score"] for d in detections]
        summary = (f"**{len(detections)} players** detected at conf ≥ {threshold:g} "
                   f"(scores {min(scores):.2f}–{max(scores):.2f}).")
    else:
        summary = (f"No players at conf ≥ {threshold:g} — try lowering the "
                   "threshold, or use a sports-broadcast style frame.")
    return (image, sections), summary


EXAMPLES = [
    ["examples/basketball.jpg", 0.5],
    ["examples/frame2.jpg", 0.5],
    ["examples/frame3.jpg", 0.5],
]

demo = gr.Interface(
    fn=detect_players,
    inputs=[
        gr.Image(type="pil", label="Sports frame"),
        gr.Slider(0.05, 0.95, value=0.5, step=0.05, label="Confidence threshold"),
    ],
    outputs=[
        gr.AnnotatedImage(label="Detected players"),
        gr.Markdown(label="Summary"),
    ],
    title="RT-DETRv2 · SportsMOT player detection",
    description=(
        "Single-class **player** detector — "
        "[`smallTech/rtdetr-sportsmot`](https://huggingface.co/smallTech/rtdetr-sportsmot), "
        "an RT-DETRv2 (r50vd) fine-tune on "
        "[SportsMOT](https://huggingface.co/datasets/Lekim89/sportsmot). "
        "mAP@50 **0.938** on 45 unseen val sequences · full benchmarks on the model card. "
        "Built as the detection stage of a ByteTrack tracking-by-detection pipeline. "
        "Running on the free CPU tier — allow a few seconds per image."
    ),
    examples=EXAMPLES,
    cache_examples=True,
    cache_mode="lazy",
    flagging_mode="never",
)

demo.launch(mcp_server=True)
