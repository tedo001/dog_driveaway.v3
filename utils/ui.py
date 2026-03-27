"""
utils/ui.py — OpenCV display renderer for real-time threat detection UI.
Draws bounding boxes, threat labels, audio meters, and FPS counter.
"""

import time
import cv2
import numpy as np

from config import (
    THREAT_COLORS,
    WINDOW_NAME,
    SHOW_FPS,
    SHOW_AUDIO_METER,
)


class UIRenderer:
    """Renders detection results on OpenCV frames."""

    def __init__(self):
        self._fps_history = []
        self._last_time = time.time()

    def draw_detections(self, frame, detections, threat_label, threat_conf):
        """
        Draw bounding boxes and labels on frame.

        Args:
            frame: BGR numpy array from camera.
            detections: List of dicts with keys:
                'bbox' (x1, y1, x2, y2), 'class' ('dog'/'person'),
                'confidence' (float).
            threat_label: Current threat label string.
            threat_conf: Confidence of threat classification.
        """
        color = THREAT_COLORS.get(threat_label, (255, 255, 255))

        for det in detections:
            x1, y1, x2, y2 = [int(c) for c in det["bbox"]]
            cls = det["class"]
            conf = det["confidence"]

            box_color = (0, 255, 0) if cls == "person" else color
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            label = f"{cls} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), box_color, -1)
            cv2.putText(
                frame, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1,
            )

        return frame

    def draw_threat_banner(self, frame, threat_label, threat_conf):
        """Draw threat status banner at top of frame."""
        h, w = frame.shape[:2]
        color = THREAT_COLORS.get(threat_label, (255, 255, 255))

        cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
        text = f"THREAT: {threat_label} ({threat_conf:.1%})"
        cv2.putText(
            frame, text, (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
        )

        if threat_label == "DANGER":
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

        return frame

    def draw_audio_meter(self, frame, audio_state):
        """
        Draw audio level indicators.

        Args:
            audio_state: dict with keys 'bark', 'growl', 'scream' (bool),
                         and 'level' (float 0-1).
        """
        if not SHOW_AUDIO_METER:
            return frame

        h, w = frame.shape[:2]
        x_start = w - 160
        y_start = 50

        cv2.rectangle(frame, (x_start - 5, y_start - 5),
                       (w - 5, y_start + 100), (0, 0, 0), -1)

        cv2.putText(frame, "AUDIO", (x_start, y_start + 15),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        level = audio_state.get("level", 0.0)
        bar_width = int(140 * min(level, 1.0))
        bar_color = (0, 255, 0) if level < 0.5 else (0, 255, 255) if level < 0.8 else (0, 0, 255)
        cv2.rectangle(frame, (x_start, y_start + 25),
                       (x_start + bar_width, y_start + 40), bar_color, -1)

        indicators = [
            ("BARK", audio_state.get("bark", False)),
            ("GROWL", audio_state.get("growl", False)),
            ("SCREAM", audio_state.get("scream", False)),
        ]
        for i, (name, active) in enumerate(indicators):
            y = y_start + 55 + i * 15
            color = (0, 0, 255) if active else (100, 100, 100)
            cv2.putText(frame, name, (x_start, y),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        return frame

    def draw_fps(self, frame):
        """Draw FPS counter on frame."""
        if not SHOW_FPS:
            return frame

        now = time.time()
        dt = now - self._last_time
        self._last_time = now

        if dt > 0:
            self._fps_history.append(1.0 / dt)
        if len(self._fps_history) > 30:
            self._fps_history.pop(0)

        fps = sum(self._fps_history) / max(len(self._fps_history), 1)
        h, w = frame.shape[:2]
        cv2.putText(
            frame, f"FPS: {fps:.1f}", (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )
        return frame

    def render(self, frame, detections, threat_label, threat_conf, audio_state):
        """Full render pipeline — call this from main loop."""
        frame = self.draw_detections(frame, detections, threat_label, threat_conf)
        frame = self.draw_threat_banner(frame, threat_label, threat_conf)
        frame = self.draw_audio_meter(frame, audio_state)
        frame = self.draw_fps(frame)
        return frame

    @staticmethod
    def show(frame):
        cv2.imshow(WINDOW_NAME, frame)
        return cv2.waitKey(1) & 0xFF

    @staticmethod
    def cleanup():
        cv2.destroyAllWindows()
