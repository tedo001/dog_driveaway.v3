"""
models/spatial_analyzer.py — Distance-based threat scoring.
Measures real-time spatial relationships between dogs and humans.

This is CRITICAL for accuracy — the CNN alone can't reliably determine
if a dog is "approaching" or "attacking" from a single crop.
The spatial analyzer tracks:
  - Dog-to-human distance (pixels → estimated meters)
  - Approach speed (distance change per frame)
  - Approach angle (head-on is more threatening)
  - Dog size relative to frame (closer = bigger = more danger)
"""

import math
import numpy as np
from collections import deque


class SpatialAnalyzer:
    """
    Tracks spatial relationships between detected dogs and humans.
    Computes distance, speed, and threat proximity scores.
    """

    def __init__(self, alert_distance=200, danger_distance=100, history_frames=10):
        """
        Args:
            alert_distance: pixel distance for ALERT level.
            danger_distance: pixel distance for DANGER level.
            history_frames: number of frames to track for speed estimation.
        """
        self.alert_distance = alert_distance
        self.danger_distance = danger_distance

        # Track dog positions over time: dog_id → deque of (cx, cy, frame_num)
        self._dog_tracks = {}
        self._frame_count = 0
        self._history_len = history_frames

    def analyze(self, detections):
        """
        Analyze spatial relationships in current frame.

        Args:
            detections: list of dicts from detector
                {'bbox': (x1,y1,x2,y2), 'class': 'dog'/'person', 'confidence': float}

        Returns:
            dict: {
                'dog_human_pairs': [
                    {
                        'dog_bbox': (x1,y1,x2,y2),
                        'human_bbox': (x1,y1,x2,y2),
                        'distance': float (pixels),
                        'approach_speed': float (pixels/frame, negative = approaching),
                        'dog_size_ratio': float (bbox area / frame area estimate),
                        'threat_score': float (0.0 to 1.0),
                    },
                    ...
                ],
                'min_distance': float,
                'max_threat_score': float,
                'num_dogs': int,
                'num_humans': int,
                'any_approaching': bool,
            }
        """
        self._frame_count += 1

        dogs = [d for d in detections if d["class"] == "dog"]
        humans = [d for d in detections if d["class"] == "person"]

        # Update dog tracks
        self._update_tracks(dogs)

        pairs = []
        min_distance = float("inf")
        max_threat = 0.0
        any_approaching = False

        for i, dog in enumerate(dogs):
            dog_center = self._center(dog["bbox"])
            dog_area = self._area(dog["bbox"])

            for human in humans:
                human_center = self._center(human["bbox"])
                distance = math.hypot(
                    dog_center[0] - human_center[0],
                    dog_center[1] - human_center[1],
                )

                # Estimate approach speed from track history
                approach_speed = self._get_approach_speed(i, human_center)

                # Dog size ratio (bigger in frame = closer to camera)
                dog_size_ratio = dog_area / max(1, 640 * 480)  # normalize to standard frame

                # Compute threat score
                threat_score = self._compute_threat_score(
                    distance, approach_speed, dog_size_ratio,
                )

                if approach_speed < -2.0:  # moving toward human
                    any_approaching = True

                pair = {
                    "dog_bbox": dog["bbox"],
                    "human_bbox": human["bbox"],
                    "distance": distance,
                    "approach_speed": approach_speed,
                    "dog_size_ratio": dog_size_ratio,
                    "threat_score": threat_score,
                }
                pairs.append(pair)

                min_distance = min(min_distance, distance)
                max_threat = max(max_threat, threat_score)

        if not pairs:
            min_distance = 0

        return {
            "dog_human_pairs": pairs,
            "min_distance": min_distance,
            "max_threat_score": max_threat,
            "num_dogs": len(dogs),
            "num_humans": len(humans),
            "any_approaching": any_approaching,
        }

    def _compute_threat_score(self, distance, approach_speed, size_ratio):
        """
        Compute threat score (0.0 to 1.0) from spatial features.

        Factors:
          - Distance: closer = higher threat
          - Speed: faster approach = higher threat
          - Size: bigger dog in frame = higher threat
        """
        # Distance score: 1.0 at 0px, 0.0 at alert_distance*2
        max_dist = self.alert_distance * 2
        dist_score = max(0, 1.0 - (distance / max_dist))

        # Speed score: faster approach = higher (negative speed = approaching)
        speed_score = 0.0
        if approach_speed < 0:
            speed_score = min(1.0, abs(approach_speed) / 10.0)

        # Size score: larger dog in frame = closer = more threat
        size_score = min(1.0, size_ratio * 10)

        # Weighted combination
        threat = (
            0.50 * dist_score +
            0.30 * speed_score +
            0.20 * size_score
        )

        return min(1.0, max(0.0, threat))

    def _update_tracks(self, dogs):
        """Update dog position history for speed tracking."""
        for i, dog in enumerate(dogs):
            center = self._center(dog["bbox"])
            if i not in self._dog_tracks:
                self._dog_tracks[i] = deque(maxlen=self._history_len)
            self._dog_tracks[i].append((center[0], center[1], self._frame_count))

        # Clean old tracks
        stale = [k for k in self._dog_tracks if k >= len(dogs)]
        for k in stale:
            del self._dog_tracks[k]

    def _get_approach_speed(self, dog_idx, human_center):
        """
        Compute approach speed toward a human.
        Negative = approaching, Positive = moving away.
        """
        track = self._dog_tracks.get(dog_idx)
        if not track or len(track) < 2:
            return 0.0

        # Current and previous positions
        curr = track[-1]
        prev = track[0]

        frames_elapsed = curr[2] - prev[2]
        if frames_elapsed < 1:
            return 0.0

        # Distance to human: now vs then
        dist_now = math.hypot(curr[0] - human_center[0], curr[1] - human_center[1])
        dist_prev = math.hypot(prev[0] - human_center[0], prev[1] - human_center[1])

        # Negative = getting closer
        return (dist_now - dist_prev) / frames_elapsed

    @staticmethod
    def _center(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @staticmethod
    def _area(bbox):
        x1, y1, x2, y2 = bbox
        return abs(x2 - x1) * abs(y2 - y1)
