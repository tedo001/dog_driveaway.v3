"""
evaluate/eval_ssd.py — Evaluate SSD300 dog+person detector.
Reports mAP, precision, recall per class, and inference speed.
"""

import sys
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from torch.utils.data import DataLoader

from config import (
    SSD_MODEL_PATH,
    SSD_NUM_CLASSES,
    SSD_CONF_THRESHOLD,
    SSD_NMS_THRESHOLD,
    SSD_CLASS_NAMES,
    DATASET_DIR,
)
from models.ssd_model import SSDDogDetector
from data.ssd_dataset import SSDDogDataset, ssd_collate_fn


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


def compute_ap(precision_list, recall_list):
    """Compute Average Precision using 11-point interpolation."""
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        precisions_at_recall = [p for p, r in zip(precision_list, recall_list) if r >= t]
        if precisions_at_recall:
            ap += max(precisions_at_recall) / 11.0
    return ap


def eval_ssd():
    """Evaluate SSD on validation set."""

    val_images = DATASET_DIR / "valid" / "images"
    val_labels = DATASET_DIR / "valid" / "labels"

    if not val_images.exists():
        print(f"ERROR: Validation images not found at {val_images}")
        sys.exit(1)

    if not SSD_MODEL_PATH.exists():
        print(f"ERROR: Trained SSD model not found at {SSD_MODEL_PATH}")
        print("Run 'python train/train_ssd.py' first")
        sys.exit(1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  SSD300 Evaluation — Dog + Person Detector")
    print("=" * 60)
    print(f"  Model   : {SSD_MODEL_PATH}")
    print(f"  Dataset : {val_images}")
    print(f"  Device  : {device}")
    print("=" * 60)

    # Load model
    model = SSDDogDetector(num_classes=SSD_NUM_CLASSES, pretrained_backbone=False)
    model.load_state_dict(torch.load(str(SSD_MODEL_PATH), map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # Load data
    val_dataset = SSDDogDataset(val_images, val_labels, augment=False)
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=2, collate_fn=ssd_collate_fn,
    )

    print(f"  Samples : {len(val_dataset)}")

    # Collect all predictions and ground truths
    all_predictions = defaultdict(list)  # class_id → list of (confidence, is_tp)
    total_gt = defaultdict(int)          # class_id → count of ground truth boxes
    total_time = 0.0
    total_frames = 0

    for images, targets in val_loader:
        images = [img.to(device) for img in images]

        start = time.time()
        with torch.no_grad():
            predictions = model(images)
        total_time += time.time() - start
        total_frames += 1

        pred = predictions[0]
        gt = targets[0]

        pred_boxes = pred["boxes"].cpu().numpy()
        pred_labels = pred["labels"].cpu().numpy()
        pred_scores = pred["scores"].cpu().numpy()

        gt_boxes = gt["boxes"].numpy()
        gt_labels = gt["labels"].numpy()

        # Count ground truths per class
        for gl in gt_labels:
            total_gt[int(gl)] += 1

        # Match predictions to ground truths
        matched_gt = set()
        for i in range(len(pred_boxes)):
            if pred_scores[i] < SSD_CONF_THRESHOLD:
                continue

            cls_id = int(pred_labels[i])
            best_iou = 0
            best_gt_idx = -1

            for j in range(len(gt_boxes)):
                if int(gt_labels[j]) != cls_id:
                    continue
                if j in matched_gt:
                    continue
                iou = compute_iou(pred_boxes[i], gt_boxes[j])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            is_tp = best_iou >= 0.5 and best_gt_idx >= 0
            if is_tp:
                matched_gt.add(best_gt_idx)

            all_predictions[cls_id].append((pred_scores[i], is_tp))

    # Compute metrics per class
    print("\n  Per-Class Results (IoU=0.50):")
    print("-" * 60)

    class_aps = []
    for cls_id in range(1, SSD_NUM_CLASSES):  # skip background (0)
        cls_name = SSD_CLASS_NAMES[cls_id] if cls_id < len(SSD_CLASS_NAMES) else f"class_{cls_id}"
        preds = all_predictions[cls_id]
        n_gt = total_gt[cls_id]

        if n_gt == 0:
            print(f"  {cls_name:>10}: no ground truth")
            continue

        # Sort by confidence descending
        preds.sort(key=lambda x: x[0], reverse=True)

        tp_cumsum = 0
        fp_cumsum = 0
        precisions = []
        recalls = []

        for score, is_tp in preds:
            if is_tp:
                tp_cumsum += 1
            else:
                fp_cumsum += 1
            precision = tp_cumsum / (tp_cumsum + fp_cumsum)
            recall = tp_cumsum / n_gt
            precisions.append(precision)
            recalls.append(recall)

        ap = compute_ap(precisions, recalls)
        class_aps.append(ap)

        final_precision = precisions[-1] if precisions else 0
        final_recall = recalls[-1] if recalls else 0

        print(f"  {cls_name:>10}: AP={ap:.4f}  Precision={final_precision:.4f}  "
              f"Recall={final_recall:.4f}  GT={n_gt}")

    # Overall mAP
    mAP = np.mean(class_aps) if class_aps else 0
    avg_time = total_time / max(total_frames, 1)
    fps = 1.0 / avg_time if avg_time > 0 else 0

    print("-" * 60)
    print(f"\n  mAP@0.50    : {mAP:.4f}")
    print(f"  Avg speed   : {avg_time * 1000:.1f}ms/frame ({fps:.1f} FPS)")
    print("=" * 60)

    return mAP


if __name__ == "__main__":
    eval_ssd()
