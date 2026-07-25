# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "torchvision",
#   "transformers>=4.48",
#   "supervision>=0.22",
#   "opencv-python",
#   "pillow",
#   "numpy",
# ]
# ///
"""Track basketball players with the fine-tuned RT-DETRv2 detector + ByteTrack.

The RT-DETRv2 model detects players in each frame (boxes); ByteTrack links those
per-frame detections into consistent player tracks (stable IDs across the clip).
Output is an annotated MP4 with a labeled box per player.

Usage
-----
    # A video file
    uv run track_players.py --source game_clip.mp4

    # A folder of frames (MOTChallenge img1/ style, e.g. a SportsMOT sequence)
    uv run track_players.py --source path/to/v_XXXX/img1 --fps 25

    # Options
    uv run track_players.py --source clip.mp4 \
        --model smallTech/rtdetrv2-r50vd-sportsmot-players \
        --conf 0.5 --output tracked.mp4 --device cpu
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection

DEFAULT_MODEL = "smallTech/rtdetrv2-r50vd-sportsmot-players"


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Detector:
    """Wraps the fine-tuned RT-DETRv2 model as a per-frame player detector."""

    def __init__(self, model_id: str, device: str, conf: float):
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForObjectDetection.from_pretrained(model_id).to(device).eval()
        self.device = device
        self.conf = conf

    @torch.no_grad()
    def __call__(self, frame_bgr: np.ndarray) -> sv.Detections:
        # transformers expects RGB; OpenCV gives BGR
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        h, w = frame_bgr.shape[:2]
        (result,) = self.processor.post_process_object_detection(
            outputs,
            target_sizes=torch.tensor([[h, w]]).to(self.device),
            threshold=self.conf,
        )
        result = {k: v.cpu() for k, v in result.items()}
        return sv.Detections.from_transformers(result)


def frame_source(source: str, fps: float):
    """Yield (frame_bgr, video_info) from a video file OR a directory of images."""
    path = Path(source)
    if path.is_dir():
        frames = sorted(
            p for p in path.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not frames:
            raise SystemExit(f"No image frames found in {source}")
        first = cv2.imread(str(frames[0]))
        h, w = first.shape[:2]
        info = sv.VideoInfo(width=w, height=h, fps=int(round(fps)), total_frames=len(frames))

        def gen():
            for fp in frames:
                img = cv2.imread(str(fp))
                if img is not None:
                    yield img

        return gen(), info

    info = sv.VideoInfo.from_video_path(source)
    return sv.get_video_frames_generator(source), info


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, help="video file or directory of frames")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Hub model id or local path")
    ap.add_argument("--output", default=None, help="output mp4 path")
    ap.add_argument("--conf", type=float, default=0.5, help="detection confidence threshold")
    ap.add_argument("--fps", type=float, default=25.0, help="fps when source is a frame folder")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    args = ap.parse_args()

    device = pick_device(args.device)
    output = args.output or f"{Path(args.source).stem}_tracked.mp4"
    print(f"device={device} model={args.model} conf={args.conf}")

    detector = Detector(args.model, device, args.conf)
    frames, info = frame_source(args.source, args.fps)
    print(f"video: {info.width}x{info.height} @ {info.fps}fps, "
          f"{info.total_frames or '?'} frames -> {output}")

    # ByteTrack turns per-frame detections into consistent player tracks.
    tracker = sv.ByteTrack(frame_rate=info.fps)
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    n = 0
    with sv.VideoSink(output, video_info=info) as sink:
        for frame in frames:
            detections = detector(frame)
            detections = tracker.update_with_detections(detections)
            labels = [f"#{tid}" for tid in detections.tracker_id]
            annotated = box_annotator.annotate(frame.copy(), detections)
            annotated = label_annotator.annotate(annotated, detections, labels)
            sink.write_frame(annotated)
            n += 1
            if n % 50 == 0:
                print(f"  processed {n} frames...")

    print(f"Done. {n} frames written to {output}")


if __name__ == "__main__":
    main()
