"""
models/yolo_model.py — YOLOv8 (ultralytics) detector for dog + person.

Logic:
  - Custom model exists  → uses it  (class 0=dog, 1=person)
  - No custom model      → COCO pretrained (dog=16, person=0)

Proximity check:
  - dog_near_human() returns True when a DANGER dog bbox centre is within
    ULTRASONIC_TRIGGER_DIST pixels of any human bbox → used to fire repeller.
"""

import math
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
    ULTRASONIC_TRIGGER_DIST,
)

COCO_DOG_CLASS    = 16
COCO_PERSON_CLASS = 0


def _bbox_centre(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _pixel_dist(c1, c2):
    return math.hypot(c1[0] - c2[0], c1[1] - c2[1])


class DualYOLODetector:
    """
    Smart YOLO detector:
      - Custom model  → single model detecting dog(0) + person(1)
      - No custom     → COCO pretrained filtering dog(16) + person(0)

    All public methods return detections as list of dicts:
        {
            "bbox":       (x1, y1, x2, y2),   # floats
            "class":      "dog" | "person",
            "confidence": float,
        }
    """

    def __init__(self, dog_model_path=None, device=None):
        self.device = device or YOLO_DEVICE
        dog_path = Path(dog_model_path) if dog_model_path else YOLO_MODEL_PATH

        if dog_path.exists():
            print(f"[YOLO] Loading custom model: {dog_path}")
            self.model        = YOLO(str(dog_path))
            self.custom_model = True
        else:
            print(f"[YOLO] Custom model not found at {dog_path}")
            print(f"[YOLO] Falling back to COCO pretrained: {YOLO_BASE_MODEL}")
            self.model        = YOLO(YOLO_BASE_MODEL)
            self.custom_model = False

        self._warmup()

    def _warmup(self):
        try:
            dummy = torch.zeros(1, 3, 320, 320)
            self.model.predict(source=dummy, device=self.device, verbose=False)
            print("[YOLO] Warmup complete")
        except Exception:
            pass

    # ── Main detection entry point ────────────────────────────────────────────

    def detect(self, frame):
        """Run YOLO on frame, return list of detection dicts."""
        if self.custom_model:
            return self._detect_custom(frame)
        else:
            return self._detect_coco(frame)

    # ── Internal detection methods ────────────────────────────────────────────

    def _detect_custom(self, frame):
        """Custom trained model: class 0=dog, class 1=person."""
        detections = []
        results = self.model.predict(
            source=frame,
            conf=YOLO_CONF_THRESHOLD,
            iou=YOLO_IOU_THRESHOLD,
            device=self.device,
            imgsz=YOLO_IMGSZ,
            verbose=False,
        )
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf   = float(box.conf[0])
                cls_id = int(box.cls[0])
                if cls_id == 0:
                    cls_name = "dog"
                elif cls_id == 1:
                    cls_name = "person"
                else:
                    continue
                detections.append({
                    "bbox":       (float(x1), float(y1), float(x2), float(y2)),
                    "class":      cls_name,
                    "confidence": conf,
                })
        return detections

    def _detect_coco(self, frame):
        """COCO pretrained: dog=16, person=0. Filters all other classes."""
        detections = []
        results = self.model.predict(
            source=frame,
            conf=YOLO_CONF_THRESHOLD,
            iou=YOLO_IOU_THRESHOLD,
            device=self.device,
            imgsz=YOLO_IMGSZ,
            verbose=False,
            classes=[COCO_DOG_CLASS, COCO_PERSON_CLASS],
        )
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf   = float(box.conf[0])
                cls_id = int(box.cls[0])
                if cls_id == COCO_DOG_CLASS:
                    cls_name = "dog"
                elif cls_id == COCO_PERSON_CLASS:
                    cls_name = "person"
                else:
                    continue
                detections.append({
                    "bbox":       (float(x1), float(y1), float(x2), float(y2)),
                    "class":      cls_name,
                    "confidence": conf,
                })
        return detections

    # ── Crop helpers ──────────────────────────────────────────────────────────

    def get_dog_crops(self, frame, detections=None):
        """
        Extract DOG image crops from frame for CNN behavior classification.
        Returns list of (crop_image, bbox) tuples.
        """
        if detections is None:
            detections = self.detect(frame)
        crops = []
        h, w = frame.shape[:2]
        for det in detections:
            if det["class"] != "dog":   # ← FIXED: crop dogs, not persons
                continue
            x1, y1, x2, y2 = [int(c) for c in det["bbox"]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if (x2 - x1) > 10 and (y2 - y1) > 10:
                crop = frame[y1:y2, x1:x2]
                crops.append((crop, det["bbox"]))
        return crops

    def get_counts(self, detections):
        """Return (num_dogs, num_humans) from a detections list."""
        dogs   = sum(1 for d in detections if d["class"] == "dog")
        humans = sum(1 for d in detections if d["class"] == "person")
        return dogs, humans

    # ── Proximity check for ultrasonic trigger ────────────────────────────────

    def dog_near_human(self, dog_bbox, detections):
        """
        Returns True if the centre of dog_bbox is within
        ULTRASONIC_TRIGGER_DIST pixels of any human bbox centre.

        Use this after CNN classifies a dog as DANGER to decide
        whether to fire the ultrasonic repeller.

        Args:
            dog_bbox   : (x1,y1,x2,y2) of the dog
            detections : full detection list from detect()
        """
        dog_centre = _bbox_centre(dog_bbox)
        for det in detections:
            if det["class"] != "person":
                continue
            human_centre = _bbox_centre(det["bbox"])
            dist = _pixel_dist(dog_centre, human_centre)
            if dist <= ULTRASONIC_TRIGGER_DIST:
                return True
        return False

    def get_all_dog_human_pairs(self, detections):
        """
        Returns list of (dog_det, human_det, pixel_dist) for every
        dog-human pair within ULTRASONIC_TRIGGER_DIST.
        Useful for drawing proximity warnings on frame.
        """
        pairs = []
        dogs   = [d for d in detections if d["class"] == "dog"]
        humans = [d for d in detections if d["class"] == "person"]
        for dog in dogs:
            dc = _bbox_centre(dog["bbox"])
            for human in humans:
                hc  = _bbox_centre(human["bbox"])
                dist = _pixel_dist(dc, hc)
                if dist <= ULTRASONIC_TRIGGER_DIST:
                    pairs.append((dog, human, dist))
        return pairs