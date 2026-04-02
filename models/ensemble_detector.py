"""
models/ensemble_detector.py — High-accuracy ensemble detector.
Fuses YOLO + SSD predictions using weighted box fusion for maximum detection accuracy.

Architecture:
  Frame → YOLO (fast, anchor-free) ──┐
                                      ├──→ Weighted Box Fusion → Final Detections
  Frame → SSD (anchor-based)     ────┘

Why ensemble is better than single detector:
  - YOLO is fast but can miss small/occluded objects
  - SSD uses multi-scale features, catches different-sized objects
  - Combining both: higher recall (fewer missed detections)
  - Weighted Box Fusion: merges overlapping boxes from both models
  - Result: ~5-15% higher mAP than either model alone

Fusion methods:
  1. NMS (Non-Maximum Suppression) — picks highest confidence, simple
  2. Soft-NMS — reduces confidence instead of removing, better
  3. WBF (Weighted Box Fusion) — averages overlapping boxes, BEST
"""

import numpy as np
from config import YOLO_DEVICE


class EnsembleDetector:
    """
    Ensemble detector that combines YOLO + SSD for maximum accuracy.
    Uses Weighted Box Fusion to merge predictions from both models.
    """

    def __init__(self, use_yolo=True, use_ssd=True, device=None):
        self.detectors = []
        self.weights = []

        if use_yolo:
            from models.yolo_model import DualYOLODetector
            print("[ENSEMBLE] Loading YOLO detector...")
            self.yolo = DualYOLODetector(device=device)
            self.detectors.append(("yolo", self.yolo))
            self.weights.append(1.0)  # YOLO weight
        else:
            self.yolo = None

        if use_ssd:
            from models.ssd_model import SSDDetector
            print("[ENSEMBLE] Loading SSD detector...")
            self.ssd = SSDDetector(device=device)
            self.detectors.append(("ssd", self.ssd))
            self.weights.append(0.8)  # SSD slightly lower weight
        else:
            self.ssd = None

        # Fusion parameters
        self.iou_threshold = 0.5     # IoU threshold for matching boxes
        self.conf_threshold = 0.4    # minimum confidence after fusion
        self.skip_box_threshold = 0.01

        print(f"[ENSEMBLE] Active detectors: {[name for name, _ in self.detectors]}")
        print(f"[ENSEMBLE] Weights: {self.weights}")

    def detect(self, frame):
        """
        Run all detectors and fuse results.
        Returns same format as individual detectors.
        """
        if len(self.detectors) == 1:
            # Single detector — no fusion needed
            return self.detectors[0][1].detect(frame)

        # Collect predictions from all detectors
        all_predictions = []
        for name, detector in self.detectors:
            preds = detector.detect(frame)
            all_predictions.append(preds)

        # Fuse predictions
        fused = self._weighted_box_fusion(all_predictions, frame.shape[:2])
        return fused

    def _weighted_box_fusion(self, all_predictions, frame_shape):
        """
        Weighted Box Fusion — merges overlapping boxes from multiple detectors.

        Steps:
          1. Normalize all boxes to [0, 1]
          2. Cluster overlapping boxes (IoU > threshold)
          3. For each cluster: weighted average of box coordinates
          4. Confidence = weighted average of confidences
          5. Scale back to pixel coordinates
        """
        h, w = frame_shape

        # Separate predictions by class
        class_predictions = {}  # class_name → list of (boxes, scores, detector_idx)

        for det_idx, preds in enumerate(all_predictions):
            for pred in preds:
                cls = pred["class"]
                if cls not in class_predictions:
                    class_predictions[cls] = []

                # Normalize bbox to [0, 1]
                x1, y1, x2, y2 = pred["bbox"]
                norm_box = [x1 / w, y1 / h, x2 / w, y2 / h]
                class_predictions[cls].append({
                    "box": norm_box,
                    "score": pred["confidence"],
                    "det_idx": det_idx,
                })

        # Fuse each class separately
        fused_detections = []

        for cls_name, preds in class_predictions.items():
            if not preds:
                continue

            # Sort by confidence descending
            preds.sort(key=lambda x: x["score"], reverse=True)

            # Cluster overlapping boxes
            used = [False] * len(preds)
            clusters = []

            for i in range(len(preds)):
                if used[i]:
                    continue

                cluster = [preds[i]]
                used[i] = True

                for j in range(i + 1, len(preds)):
                    if used[j]:
                        continue
                    if self._iou(preds[i]["box"], preds[j]["box"]) > self.iou_threshold:
                        cluster.append(preds[j])
                        used[j] = True

                clusters.append(cluster)

            # Merge each cluster into one detection
            for cluster in clusters:
                if not cluster:
                    continue

                # Weighted average of boxes and scores
                total_weight = 0
                fused_box = [0, 0, 0, 0]
                fused_score = 0

                for pred in cluster:
                    weight = self.weights[pred["det_idx"]] * pred["score"]
                    total_weight += weight
                    fused_score += weight * pred["score"]
                    for k in range(4):
                        fused_box[k] += weight * pred["box"][k]

                if total_weight < self.skip_box_threshold:
                    continue

                for k in range(4):
                    fused_box[k] /= total_weight
                fused_score /= total_weight

                # Boost confidence when multiple detectors agree
                num_detectors_agreed = len(set(p["det_idx"] for p in cluster))
                if num_detectors_agreed > 1:
                    fused_score = min(fused_score * 1.15, 1.0)

                if fused_score < self.conf_threshold:
                    continue

                # Scale back to pixel coordinates
                fused_detections.append({
                    "bbox": (
                        fused_box[0] * w,
                        fused_box[1] * h,
                        fused_box[2] * w,
                        fused_box[3] * h,
                    ),
                    "class": cls_name,
                    "confidence": fused_score,
                    "sources": num_detectors_agreed,
                })

        return fused_detections

    def get_dog_crops(self, frame, detections=None):
        """Extract dog crops — same interface as single detectors."""
        if detections is None:
            detections = self.detect(frame)

        crops = []
        h, w = frame.shape[:2]
        for det in detections:
            if det["class"] != "dog":
                continue
            x1, y1, x2, y2 = [int(c) for c in det["bbox"]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 > 10 and y2 - y1 > 10:
                crop = frame[y1:y2, x1:x2]
                crops.append((crop, det["bbox"]))
        return crops

    @staticmethod
    def _iou(box1, box2):
        """Compute IoU between two normalized boxes [x1, y1, x2, y2]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0
