"""
models/yolo_model.py — Dual YOLO detection: custom dog detector + COCO person detector.
Uses fine-tuned YOLOv8n for dogs and pretrained YOLOv8n for humans.
Optimized for RTX 4060 CUDA inference.
"""

import torch
from ultralytics import YOLO
from pathlib import Path

from config import (
    YOLO_MODEL_PATH,
    YOLO_BASE_MODEL,
    YOLO_CONF_THRESHOLD,
    YOLO_IOU_THRESHOLD,
    YOLO_DEVICE,
    YOLO_IMGSZ,
)


class DualYOLODetector:
    """
    Runs two YOLO models:
      1. Custom fine-tuned model for dog detection
      2. Pretrained COCO model for person detection (class 0 in COCO)
    Returns unified detection list with bounding boxes and classes.
    """

    COCO_PERSON_CLASS = 0  # COCO class index for 'person'

    def __init__(self, dog_model_path=None, device=None):
        self.device = device or YOLO_DEVICE
        dog_path = Path(dog_model_path) if dog_model_path else YOLO_MODEL_PATH

        # Load custom dog detector
        if dog_path.exists():
            print(f"[YOLO] Loading custom dog detector: {dog_path}")
            self.dog_model = YOLO(str(dog_path))
        else:
            print(f"[YOLO] Custom model not found at {dog_path}, using base model")
            self.dog_model = YOLO(YOLO_BASE_MODEL)

        # Load pretrained COCO model for person detection
        print(f"[YOLO] Loading COCO person detector: {YOLO_BASE_MODEL}")
        self.person_model = YOLO(YOLO_BASE_MODEL)

        # Warm up models on GPU
        self._warmup()

    def _warmup(self):
        """Run dummy inference to warm up GPU."""
        dummy = torch.zeros(1, 3, YOLO_IMGSZ, YOLO_IMGSZ).to(self.device)
        try:
            self.dog_model.predict(
                source=dummy, device=self.device, verbose=False,
            )
            self.person_model.predict(
                source=dummy, device=self.device, verbose=False,
            )
            print("[YOLO] GPU warmup complete")
        except Exception:
            print("[YOLO] Warmup skipped (non-critical)")

    def detect(self, frame):
        """
        Run detection on a single BGR frame.

        Args:
            frame: numpy array (H, W, 3) BGR image.

        Returns:
            list of dicts: [
                {
                    'bbox': (x1, y1, x2, y2),
                    'class': 'dog' or 'person',
                    'confidence': float,
                },
                ...
            ]
        """
        detections = []

        # Detect dogs with custom model
        dog_results = self.dog_model.predict(
            source=frame,
            conf=YOLO_CONF_THRESHOLD,
            iou=YOLO_IOU_THRESHOLD,
            device=self.device,
            imgsz=YOLO_IMGSZ,
            verbose=False,
        )
        for result in dog_results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "class": "dog",
                    "confidence": conf,
                })

        # Detect persons with COCO model
        person_results = self.person_model.predict(
            source=frame,
            conf=YOLO_CONF_THRESHOLD,
            iou=YOLO_IOU_THRESHOLD,
            device=self.device,
            imgsz=YOLO_IMGSZ,
            classes=[self.COCO_PERSON_CLASS],
            verbose=False,
        )
        for result in person_results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "class": "person",
                    "confidence": conf,
                })

        return detections

    def get_dog_crops(self, frame, detections=None):
        """
        Extract cropped dog regions from frame.

        Args:
            frame: BGR numpy array.
            detections: Optional pre-computed detections. If None, runs detect().

        Returns:
            list of (crop, bbox) tuples where crop is a BGR numpy array.
        """
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

    @property
    def dog_count(self):
        """Return count from last detection (for convenience)."""
        return self._last_dog_count if hasattr(self, "_last_dog_count") else 0

    @property
    def person_count(self):
        """Return count from last detection (for convenience)."""
        return self._last_person_count if hasattr(self, "_last_person_count") else 0
