# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "torchvision",
#   "transformers[torch]>=4.48",
#   "accelerate",
#   "albumentations>=1.4.10",
#   "huggingface_hub",
#   "pillow",
#   "torchmetrics",
#   "pycocotools",
# ]
# ///
"""Fine-tune RT-DETRv2 (r50vd) on Lekim89/sportsmot for sports player detection.

The dataset is MOTChallenge-style (per-sequence JPEG frames + gt.txt). Only the
10 `train/` sequences carry annotations (basketball, soccer, volleyball clips).
We train a single-class "player" detector on 9 sequences and validate on one
held-out basketball sequence, since the target use case is basketball tracking.
"""

import os
import random
from collections import defaultdict
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from huggingface_hub import HfApi, snapshot_download
from PIL import Image
from torch.utils.data import Dataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    Trainer,
    TrainingArguments,
)

DATASET_ID = "Lekim89/sportsmot"
CHECKPOINT = "PekingU/rtdetr_v2_r50vd"
HUB_MODEL_ID = "smallTech/rtdetrv2-r50vd-sportsmot-players"
VAL_SEQ = "v_-6Os86HzwCs_c009"  # held-out basketball sequence
TRAIN_STRIDE = 2  # adjacent frames are near-duplicates
VAL_STRIDE = 5
IMAGE_SIZE = 640
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)


def parse_gt(gt_path: Path) -> dict[int, list[list[float]]]:
    """MOT gt.txt -> {frame: [[x, y, w, h], ...]} (pixel COCO boxes)."""
    frames = defaultdict(list)
    for line in gt_path.read_text().strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        frame, _tid = int(parts[0]), int(parts[1])
        x, y, w, h = (float(v) for v in parts[2:6])
        conf = int(parts[6])
        if conf == 0 or w <= 0 or h <= 0:
            continue
        frames[frame].append([x, y, w, h])
    return frames


def load_samples(root: Path):
    train_samples, val_samples = [], []
    for seq_dir in sorted((root / "train").iterdir()):
        if not seq_dir.is_dir():
            continue
        gt = parse_gt(seq_dir / "gt" / "gt.txt")
        img_dir = seq_dir / "img1"
        is_val = seq_dir.name == VAL_SEQ
        stride = VAL_STRIDE if is_val else TRAIN_STRIDE
        for i, img_path in enumerate(sorted(img_dir.glob("*.jpg"))):
            if i % stride != 0:
                continue
            frame = int(img_path.stem)
            boxes = gt.get(frame, [])
            if not boxes:
                continue
            (val_samples if is_val else train_samples).append((img_path, boxes))
    return train_samples, val_samples


train_aug = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.4),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
        A.GaussNoise(p=0.2),
    ],
    bbox_params=A.BboxParams(format="coco", label_fields=["labels"], clip=True, min_area=4),
)


class MotDataset(Dataset):
    def __init__(self, samples, processor, augment):
        self.samples = samples
        self.processor = processor
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, boxes = self.samples[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        labels = [0] * len(boxes)
        if self.augment:
            out = train_aug(image=image, bboxes=boxes, labels=labels)
            image, boxes, labels = out["image"], out["bboxes"], out["labels"]
        annotations = {
            "image_id": idx,
            "annotations": [
                {"bbox": list(b), "category_id": c, "area": b[2] * b[3], "iscrowd": 0}
                for b, c in zip(boxes, labels)
            ],
        }
        encoding = self.processor(images=image, annotations=annotations, return_tensors="pt")
        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "labels": encoding["labels"][0],
        }


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": [b["labels"] for b in batch],
    }


def main():
    print("Downloading annotated train split...")
    root = Path(
        snapshot_download(DATASET_ID, repo_type="dataset", allow_patterns=["train/*"])
    )
    train_samples, val_samples = load_samples(root)
    print(f"train images: {len(train_samples)}, val images (basketball): {len(val_samples)}")

    processor = AutoImageProcessor.from_pretrained(
        CHECKPOINT,
        do_resize=True,
        size={"height": IMAGE_SIZE, "width": IMAGE_SIZE},
        use_fast=True,
    )
    model = AutoModelForObjectDetection.from_pretrained(
        CHECKPOINT,
        id2label={0: "player"},
        label2id={"player": 0},
        anchor_image_size=None,
        ignore_mismatched_sizes=True,
    )

    train_ds = MotDataset(train_samples, processor, augment=True)
    val_ds = MotDataset(val_samples, processor, augment=False)

    args = TrainingArguments(
        output_dir="rtdetrv2-sportsmot",
        num_train_epochs=20,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        weight_decay=1e-4,
        max_grad_norm=0.1,
        warmup_steps=300,
        lr_scheduler_type="cosine",
        bf16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=25,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none",
        seed=SEED,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
    )
    trainer.train()

    # Final mAP on the held-out basketball sequence
    print("Computing mAP on held-out basketball sequence...")
    device = model.device
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    with torch.no_grad():
        for img_path, boxes in val_samples:
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            outputs = model(**inputs)
            (result,) = processor.post_process_object_detection(
                outputs,
                target_sizes=torch.tensor([image.size[::-1]]),
                threshold=0.01,
            )
            gt_xyxy = torch.tensor(
                [[x, y, x + w, y + h] for x, y, w, h in boxes], dtype=torch.float32
            )
            metric.update(
                [{k: result[k].cpu() for k in ("boxes", "scores", "labels")}],
                [{"boxes": gt_xyxy, "labels": torch.zeros(len(boxes), dtype=torch.long)}],
            )
    metrics = {k: float(v) for k, v in metric.compute().items() if v.numel() == 1}
    print("Validation metrics:", metrics)

    print(f"Pushing to hub: {HUB_MODEL_ID}")
    model.push_to_hub(HUB_MODEL_ID, private=False)
    processor.push_to_hub(HUB_MODEL_ID)

    card = f"""---
license: apache-2.0
base_model: {CHECKPOINT}
datasets:
- {DATASET_ID}
pipeline_tag: object-detection
tags:
- rt-detr-v2
- sports
- basketball
- player-detection
- tracking
---

# RT-DETRv2 (r50vd) fine-tuned on SportsMOT for player detection

Single-class (`player`) detector fine-tuned from `{CHECKPOINT}` on the annotated
train sequences of [{DATASET_ID}](https://huggingface.co/datasets/{DATASET_ID})
(basketball, soccer, and volleyball broadcast clips in MOTChallenge format).
Intended as the detection stage of a tracking-by-detection pipeline (e.g. with
ByteTrack via the `supervision` library) for basketball player tracking.

## Validation (held-out basketball sequence `{VAL_SEQ}`)

| metric | value |
|---|---|
| mAP@[.5:.95] | {metrics.get('map', float('nan')):.4f} |
| mAP@50 | {metrics.get('map_50', float('nan')):.4f} |
| mAP@75 | {metrics.get('map_75', float('nan')):.4f} |
| mAR@100 | {metrics.get('mar_100', float('nan')):.4f} |

## Usage

```python
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

processor = AutoImageProcessor.from_pretrained("{HUB_MODEL_ID}")
model = AutoModelForObjectDetection.from_pretrained("{HUB_MODEL_ID}")

image = Image.open("frame.jpg")
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
results = processor.post_process_object_detection(
    outputs, target_sizes=torch.tensor([image.size[::-1]]), threshold=0.5
)
```

For tracking, feed the per-frame detections into ByteTrack:

```python
import supervision as sv
tracker = sv.ByteTrack(frame_rate=25)
detections = sv.Detections.from_transformers(results[0])
tracked = tracker.update_with_detections(detections)
```

Training: {len(train_samples)} frames (stride {TRAIN_STRIDE}) from 9 sequences,
20 epochs, 640x640, lr 5e-5 cosine, bf16, augmentation (hflip, color jitter, noise).
"""
    HfApi().upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=HUB_MODEL_ID,
    )
    print("Done.")


if __name__ == "__main__":
    main()
