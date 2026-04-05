"""
models/live_detector.py — Clean YOLO + CNN pipeline.

Logic (nothing more, nothing less):
    1. YOLO  → detect all dogs and humans in frame
    2. CNN   → classify each dog crop: DANGER or IDLE
    3. Decision:
         • DANGER dog + human in frame  → 🔊 fire ultrasonic  (5s cooldown)
         • DANGER dog + NO human        → no trigger
         • dog vs dog only              → no trigger
         • IDLE dog (any situation)     → no trigger
"""

import time
from pathlib import Path

from config import (
    CNN_MODEL_PATH,
    ULTRASONIC_COOLDOWN,
    ULTRASONIC_TRIGGER_DIST,
    THREAT_CLASSES,
)


def _bbox_centre(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _pixel_dist(a, b):
    import math
    return math.hypot(a[0] - b[0], a[1] - b[1])


class LiveDetector:
    """
    Owns the complete YOLO → CNN → decision pipeline.
    Call process(frame) once per camera frame.
    """

    def __init__(self):
        # ── YOLO ──────────────────────────────────────────────────────────
        from models.yolo_model import DualYOLODetector
        self.yolo = DualYOLODetector()
        print(f"[LiveDetector] YOLO: {'custom model' if self.yolo.custom_model else 'COCO pretrained (fallback)'}")

        # ── CNN (optional — runs without it, all dogs show IDLE) ──────────
        self.cnn = None
        if CNN_MODEL_PATH.exists():
            try:
                from models.behavior_net_v2 import BehaviorClassifierV2
                self.cnn = BehaviorClassifierV2()
                print(f"[LiveDetector] CNN loaded — classes: {self.cnn.class_names}")
            except Exception as e:
                print(f"[LiveDetector] CNN load failed: {e}")
        else:
            print(f"[LiveDetector] CNN not found at {CNN_MODEL_PATH} — train CNN first")

        # Cooldown tracker
        self._last_trigger_time = 0.0

        print("[LiveDetector] Ready. Pipeline: YOLO → CNN → Decision")

    # ── Main per-frame method ──────────────────────────────────────────────

    def process(self, frame):
        """
        Run full pipeline on one frame.

        Returns dict:
            detections         - list of YOLO dicts (class/bbox/confidence)
            cnn_results        - list of (label, confidence) per detected dog
            num_dogs           - int
            num_humans         - int
            threat_label       - "DANGER" | "IDLE"
            confidence         - float
            trigger_ultrasonic - bool
            reason             - str
        """

        # ── Step 1: YOLO ──────────────────────────────────────────────────
        detections = self.yolo.detect(frame)
        num_dogs, num_humans = self.yolo.get_counts(detections)

        # ── Step 2: CNN — classify each dog crop ──────────────────────────
        cnn_results = []   # list of (label, confidence)
        if num_dogs > 0 and self.cnn is not None:
            dog_crops = self.yolo.get_dog_crops(frame, detections)
            for crop, bbox in dog_crops:
                _, label, conf = self.cnn.classify(crop)
                cnn_results.append((label, conf))
                print(f"[CNN] Dog → {label} ({conf:.1%})")

        # ── Step 3: Decision ──────────────────────────────────────────────
        threat_label, confidence, trigger, reason = self._decide(
            cnn_results, num_dogs, num_humans
        )

        return {
            "detections":         detections,
            "cnn_results":        cnn_results,
            "num_dogs":           num_dogs,
            "num_humans":         num_humans,
            "threat_label":       threat_label,
            "confidence":         confidence,
            "trigger_ultrasonic": trigger,
            "reason":             reason,
        }

    # ── Decision logic ────────────────────────────────────────────────────

    def _decide(self, cnn_results, num_dogs, num_humans):
        """
        Pure decision function. No audio. No spatial math. Just these rules:

        Rule 1: No dogs detected            → IDLE, no trigger
        Rule 2: DANGER dog + NO human       → IDLE, no trigger
                (dog vs dog — no human at risk)
        Rule 3: DANGER dog + human present  → DANGER, trigger (with cooldown)
        Rule 4: IDLE dog (any situation)    → IDLE, no trigger
        """

        # Rule 1: no dogs
        if not cnn_results:
            return "IDLE", 0.0, False, "No dogs in frame"

        # Find highest-confidence DANGER result
        danger_conf = 0.0
        idle_conf   = 0.0
        for label, conf in cnn_results:
            if label == "DANGER":
                danger_conf = max(danger_conf, conf)
            else:
                idle_conf = max(idle_conf, conf)

        any_danger = danger_conf > 0.0

        # Rule 4: all dogs IDLE
        if not any_danger:
            return "IDLE", idle_conf, False, "Dog is IDLE — no threat"

        # Rule 2: DANGER dog but no human in frame
        if num_humans == 0:
            return "IDLE", danger_conf, False, \
                "DANGER dog detected but NO human in frame — no trigger (dog vs dog or alone)"

        # Rule 3: DANGER dog + human present → trigger with cooldown
        now = time.time()
        if now - self._last_trigger_time < ULTRASONIC_COOLDOWN:
            remaining = ULTRASONIC_COOLDOWN - (now - self._last_trigger_time)
            return "DANGER", danger_conf, False, \
                f"DANGER + human — cooldown active ({remaining:.1f}s remaining)"

        # Fire!
        self._last_trigger_time = now
        return "DANGER", danger_conf, True, \
            f"Rule 3: DANGER dog + {num_humans} human(s) → TRIGGER ultrasonic"

    # ── Status helpers ─────────────────────────────────────────────────────

    def is_cnn_loaded(self):
        return self.cnn is not None

    def is_yolo_custom(self):
        return self.yolo.custom_model

    def status(self):
        return {
            "yolo_model":  "custom" if self.is_yolo_custom() else "COCO pretrained (fallback)",
            "cnn_loaded":  self.is_cnn_loaded(),
            "cnn_classes": self.cnn.class_names if self.is_cnn_loaded() else [],
        }