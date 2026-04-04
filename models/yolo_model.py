"""
models/yolo_model.py — YOLO detector for dog + person.

When custom model exists: uses it (classes 0=dog, 1=person)
When no custom model: uses COCO pretrained (class 16=dog, 0=person)
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

COCO_DOG_CLASS = 16
COCO_PERSON_CLASS = 0


class DualYOLODetector:
    """
    Smart YOLO detector:
      - If custom model trained: single model (0=dog, 1=person)
      - If no custom model: COCO pretrained filtering for dog(16) + person(0)
    """

    def __init__(self, dog_model_path=None, device=None):
        self.device = device or YOLO_DEVICE
        dog_path = Path(dog_model_path) if dog_model_path else YOLO_MODEL_PATH

        if dog_path.exists():
            print(f"[YOLO] Loading custom model: {dog_path}")
            self.model = YOLO(str(dog_path))
            self.custom_model = True
        else:
            print(f"[YOLO] No custom model found, using COCO pretrained")
            self.model = YOLO(YOLO_BASE_MODEL)
            self.custom_model = False

        self._warmup()

    def _warmup(self):
        try:
            dummy = torch.zeros(1, 3, 320, 320)
            self.model.predict(source=dummy, device=self.device, verbose=False)
            print("[YOLO] Warmup complete")
        except Exception:
            pass

    def detect(self, frame):
        if self.custom_model:
            return self._detect_custom(frame)
        else:
            return self._detect_coco(frame)

    def _detect_custom(self, frame):
        """Custom trained model: 0=dog, 1=person."""
        detections = []
        results = self.model.predict(
            source=frame, conf=YOLO_CONF_THRESHOLD, iou=YOLO_IOU_THRESHOLD,
            device=self.device, imgsz=YOLO_IMGSZ, verbose=False,
        )
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = "dog" if cls_id == 0 else "person" if cls_id == 1 else None
                if cls_name:
                    detections.append({
                        "bbox": (x1, y1, x2, y2),
                        "class": cls_name,
                        "confidence": conf,
                    })
        return detections

    def _detect_coco(self, frame):
        """COCO pretrained: dog=16, person=0."""
        detections = []
        results = self.model.predict(
            source=frame, conf=YOLO_CONF_THRESHOLD, iou=YOLO_IOU_THRESHOLD,
            device=self.device, imgsz=YOLO_IMGSZ, verbose=False,
            classes=[COCO_DOG_CLASS, COCO_PERSON_CLASS],
        )
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                if cls_id == COCO_DOG_CLASS:
                    cls_name = "dog"
                elif cls_id == COCO_PERSON_CLASS:
                    cls_name = "person"
                else:
                    continue
                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "class": cls_name,
                    "confidence": conf,
                })
        return detections

    def get_dog_crops(self, frame, detections=None):
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
