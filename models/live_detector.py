"""
models/live_detector.py — Connects YOLOv8 + BehaviorNetV2 CNN in one pipeline.

Pipeline per frame:
    1. YOLO  → detects all dogs and humans in frame
    2. CNN   → classifies behavior of each detected dog crop
    3. Engine→ fuses CNN + audio → final threat + ultrasonic decision
    4. Proximity check → dog must be near a human to trigger ultrasonic

Usage (already called by app.py):
    detector = LiveDetector()
    result   = detector.process(frame, audio_state)
    # result keys: detections, cnn_results, threat, trigger_ultrasonic,
    #              num_dogs, num_humans, reason
"""

import math
import time
from pathlib import Path

from config import (
    YOLO_MODEL_PATH,
    YOLO_BASE_MODEL,
    CNN_MODEL_PATH,
    ULTRASONIC_TRIGGER_DIST,
    ULTRASONIC_COOLDOWN,
    ULTRASONIC_TRIGGER_ON_CLASSES,
    THREAT_CLASSES,
)


def _centre(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class LiveDetector:
    """
    Single object that owns the full YOLO → CNN → ThreatEngine pipeline.

    app.py already has this logic inline; this class makes it reusable,
    testable, and adds the proximity check before ultrasonic trigger.
    """

    def __init__(self):
        # ── Load YOLO ──────────────────────────────────────────────────────
        from models.yolo_model import DualYOLODetector
        self.yolo = DualYOLODetector()

        # ── Load CNN (only if model file exists) ───────────────────────────
        self.cnn = None
        if CNN_MODEL_PATH.exists():
            try:
                from models.behavior_net_v2 import BehaviorClassifierV2
                self.cnn = BehaviorClassifierV2()
                print(f"[LiveDetector] CNN loaded: {CNN_MODEL_PATH.name}")
            except Exception as e:
                print(f"[LiveDetector] CNN load failed: {e}")
        else:
            print(f"[LiveDetector] WARNING: CNN model not found at {CNN_MODEL_PATH}")
            print(f"[LiveDetector] Go to Training → Train CNN first.")

        # ── Load ThreatEngine ──────────────────────────────────────────────
        from models.threat_engine import ThreatEngine
        self.engine = ThreatEngine()

        # Ultrasonic cooldown tracker
        self._last_trigger_time = 0.0

        print("[LiveDetector] Ready. Pipeline: YOLO → CNN → ThreatEngine → Proximity → Ultrasonic")

    # ── Main per-frame method ──────────────────────────────────────────────

    def process(self, frame, audio_state=None):
        """
        Run the full pipeline on one camera frame.

        Args:
            frame       : BGR numpy array from cv2.VideoCapture
            audio_state : dict with bark/growl/scream/level keys
                          (pass {} or None if audio is not used)

        Returns:
            dict with:
                detections         - list of YOLO detection dicts (class/bbox/conf)
                cnn_results        - list of (class_idx, label, confidence) per dog
                num_dogs           - int
                num_humans         - int
                threat_label       - str  e.g. "DANGER"
                threat_class       - int
                confidence         - float
                trigger_ultrasonic - bool  (True only if proximity check also passes)
                reason             - str   explanation from ThreatEngine
                dog_near_human     - bool  proximity check result
        """
        if audio_state is None:
            audio_state = {"bark": False, "growl": False, "scream": False, "level": 0.0}

        # ── Step 1: YOLO detection ─────────────────────────────────────────
        detections = self.yolo.detect(frame)
        num_dogs, num_humans = self.yolo.get_counts(detections)

        # ── Step 2: CNN classification of every dog crop ───────────────────
        cnn_results = []
        dog_bboxes  = []

        if num_dogs > 0 and self.cnn is not None:
            dog_crops = self.yolo.get_dog_crops(frame, detections)
            for crop, bbox in dog_crops:
                tc, tl, conf = self.cnn.classify(crop)
                cnn_results.append((tc, tl, conf))
                dog_bboxes.append(bbox)
                print(f"[CNN] Dog crop → {tl} ({conf:.1%})")

        elif num_dogs > 0 and self.cnn is None:
            # CNN not loaded — treat all dogs as IDLE, warn user
            print("[LiveDetector] CNN not loaded — all dogs treated as IDLE. Train CNN first.")

        # ── Step 3: ThreatEngine fusion ────────────────────────────────────
        result = self.engine.evaluate(
            cnn_results=cnn_results,
            num_dogs=num_dogs,
            num_humans=num_humans,
            audio_state=audio_state,
        )

        threat_label = result["threat_label"]
        trigger      = result["trigger_ultrasonic"]

        # ── Step 4: Proximity check ────────────────────────────────────────
        # Only trigger ultrasonic if a DANGER dog is physically close to a human.
        # This prevents false triggers when dog and human are far apart in frame.
        near = False
        if trigger and dog_bboxes:
            for bbox in dog_bboxes:
                if self.yolo.dog_near_human(bbox, detections):
                    near = True
                    break
            if not near:
                trigger = False
                result["reason"] += " (proximity check failed — dog not near human)"

        # ── Step 5: Ultrasonic cooldown ────────────────────────────────────
        if trigger:
            now = time.time()
            if now - self._last_trigger_time < ULTRASONIC_COOLDOWN:
                trigger = False
                result["reason"] += " (cooldown active)"
            else:
                self._last_trigger_time = now

        return {
            "detections":         detections,
            "cnn_results":        cnn_results,
            "num_dogs":           num_dogs,
            "num_humans":         num_humans,
            "threat_label":       threat_label,
            "threat_class":       result["threat_class"],
            "confidence":         result["confidence"],
            "trigger_ultrasonic": trigger,
            "reason":             result["reason"],
            "dog_near_human":     near,
        }

    # ── Status helpers ─────────────────────────────────────────────────────

    def is_cnn_loaded(self):
        return self.cnn is not None

    def is_yolo_custom(self):
        return self.yolo.custom_model

    def status(self):
        return {
            "yolo_model":   "custom" if self.is_yolo_custom() else "COCO pretrained (fallback)",
            "cnn_loaded":   self.is_cnn_loaded(),
            "cnn_classes":  list(THREAT_CLASSES.values()) if self.is_cnn_loaded() else [],
        }