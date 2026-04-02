"""
simulation/simulator.py — Full visual simulation of dog threat detection.
Animates dogs and humans on a 2D canvas. No camera, no models needed.q
Tests the complete threat logic pipeline with controllable scenarios.

Controls:
  1-6  : Switch scenario
  SPACE : Pause/Resume
  R     : Reset current scenario
  Q     : Quit
"""

import math
import time
import random
import cv2
import numpy as np

from config import (
    SIM_WIDTH,
    SIM_HEIGHT,
    SIM_FPS,
    SIM_DOG_SPEED,
    SIM_HUMAN_SPEED,
    SIM_APPROACH_DISTANCE,
    SIM_ATTACK_DISTANCE,
    SIM_DOG_FIGHT_DISTANCE,
    THREAT_COLORS,
    ULTRASONIC_COOLDOWN,
)
from models.threat_engine import ThreatEngine


class Entity:
    """A simulated entity (dog or human) on the canvas."""

    def __init__(self, etype, x, y, color, size=30, name=""):
        self.type = etype          # "dog" or "person"
        self.x = float(x)
        self.y = float(y)
        self.color = color
        self.size = size
        self.name = name
        self.vx = 0.0
        self.vy = 0.0
        self.aggressive = False    # only for dogs
        self.target = None         # entity this one moves toward

    def move(self):
        self.x += self.vx
        self.y += self.vy
        # Clamp to canvas
        self.x = max(self.size, min(SIM_WIDTH - self.size, self.x))
        self.y = max(self.size, min(SIM_HEIGHT - self.size, self.y))

    def move_toward(self, target, speed):
        dx = target.x - self.x
        dy = target.y - self.y
        dist = math.hypot(dx, dy)
        if dist > 2:
            self.vx = (dx / dist) * speed
            self.vy = (dy / dist) * speed
        else:
            self.vx = 0
            self.vy = 0

    def move_away_from(self, target, speed):
        dx = self.x - target.x
        dy = self.y - target.y
        dist = math.hypot(dx, dy)
        if dist > 2:
            self.vx = (dx / dist) * speed
            self.vy = (dy / dist) * speed
        else:
            self.vx = random.uniform(-1, 1) * speed
            self.vy = random.uniform(-1, 1) * speed

    def wander(self, speed):
        if random.random() < 0.02:
            angle = random.uniform(0, 2 * math.pi)
            self.vx = math.cos(angle) * speed * 0.5
            self.vy = math.sin(angle) * speed * 0.5

    def distance_to(self, other):
        return math.hypot(self.x - other.x, self.y - other.y)

    def bbox(self):
        s = self.size
        return (self.x - s, self.y - s, self.x + s, self.y + s)


class Simulator:
    """
    Visual simulation engine. Renders animated dogs + humans,
    computes distances, classifies threats, triggers ultrasonic indicator.
    """

    def __init__(self):
        self.entities = []
        self.engine = ThreatEngine()
        self.paused = False
        self.frame_count = 0
        self.ultrasonic_active = False
        self.ultrasonic_timer = 0.0
        self.ultrasonic_cooldown_timer = 0.0
        self.ultrasonic_total = 0
        self.current_result = None
        self.event_log = []

    def clear(self):
        self.entities.clear()
        self.ultrasonic_active = False
        self.ultrasonic_timer = 0.0
        self.frame_count = 0
        self.event_log.clear()

    def add_dog(self, x, y, aggressive=False, name="Dog"):
        color = (0, 100, 255) if aggressive else (200, 150, 50)
        dog = Entity("dog", x, y, color, size=25, name=name)
        dog.aggressive = aggressive
        self.entities.append(dog)
        return dog

    def add_human(self, x, y, name="Human"):
        human = Entity("person", x, y, (255, 200, 150), size=30, name=name)
        self.entities.append(human)
        return human

    def get_dogs(self):
        return [e for e in self.entities if e.type == "dog"]

    def get_humans(self):
        return [e for e in self.entities if e.type == "person"]

    def classify_threats(self):
        """
        Classify each dog's threat level based on distance to humans/other dogs.
        Returns CNN-like results for the threat engine.
        """
        dogs = self.get_dogs()
        humans = self.get_humans()
        num_dogs = len(dogs)
        num_humans = len(humans)

        cnn_results = []

        for dog in dogs:
            # Check distance to nearest human
            min_human_dist = float("inf")
            for human in humans:
                d = dog.distance_to(human)
                if d < min_human_dist:
                    min_human_dist = d

            # Check distance to nearest other dog
            min_dog_dist = float("inf")
            for other_dog in dogs:
                if other_dog is dog:
                    continue
                d = dog.distance_to(other_dog)
                if d < min_dog_dist:
                    min_dog_dist = d

            # Classification logic
            if dog.aggressive and num_dogs >= 2 and min_dog_dist < SIM_DOG_FIGHT_DISTANCE:
                # Two aggressive dogs close = DOG_FIGHT
                threat_class = 3
                confidence = 0.90
            elif dog.aggressive and min_human_dist < SIM_ATTACK_DISTANCE:
                # Aggressive dog very close to human = DANGER
                threat_class = 2
                confidence = 0.95
            elif dog.aggressive and min_human_dist < SIM_APPROACH_DISTANCE:
                # Aggressive dog approaching human = ALERT
                threat_class = 1
                confidence = 0.80
            elif dog.aggressive:
                # Aggressive dog alone (far from humans) = IDLE
                # KEY: aggressive but alone → NO ultrasonic
                threat_class = 0
                confidence = 0.70
            else:
                # Calm dog = IDLE
                threat_class = 0
                confidence = 0.95

            from config import THREAT_CLASSES
            threat_label = THREAT_CLASSES[threat_class]
            cnn_results.append((threat_class, threat_label, confidence))

        # Simulated audio state
        audio_state = {
            "bark": any(d.aggressive for d in dogs),
            "growl": any(
                d.aggressive and d.distance_to(h) < SIM_APPROACH_DISTANCE
                for d in dogs for h in humans
            ) if humans else False,
            "scream": any(
                d.aggressive and d.distance_to(h) < SIM_ATTACK_DISTANCE
                for d in dogs for h in humans
            ) if humans else False,
            "level": 0.7 if any(d.aggressive for d in dogs) else 0.1,
        }

        # Run through the REAL threat engine (enforces all safety rules)
        result = self.engine.evaluate(
            cnn_results=cnn_results,
            num_dogs=num_dogs,
            num_humans=num_humans,
            audio_state=audio_state,
        )

        self.current_result = result
        return result, cnn_results, audio_state

    def update_ultrasonic(self, result, dt):
        """Handle ultrasonic trigger state with cooldown."""
        if result["trigger_ultrasonic"]:
            if self.ultrasonic_cooldown_timer <= 0:
                self.ultrasonic_active = True
                self.ultrasonic_timer = 2.0  # 2 second burst
                self.ultrasonic_cooldown_timer = ULTRASONIC_COOLDOWN
                self.ultrasonic_total += 1
                self.event_log.append(
                    f"[{self.frame_count:05d}] ULTRASONIC TRIGGERED! "
                    f"Reason: {result['reason']}"
                )

        if self.ultrasonic_timer > 0:
            self.ultrasonic_timer -= dt
            if self.ultrasonic_timer <= 0:
                self.ultrasonic_active = False

        if self.ultrasonic_cooldown_timer > 0:
            self.ultrasonic_cooldown_timer -= dt

    def render(self):
        """Render the simulation to an OpenCV frame."""
        canvas = np.zeros((SIM_HEIGHT, SIM_WIDTH, 3), dtype=np.uint8)
        canvas[:] = (30, 30, 30)  # dark background

        # Draw grid
        for x in range(0, SIM_WIDTH, 80):
            cv2.line(canvas, (x, 0), (x, SIM_HEIGHT), (50, 50, 50), 1)
        for y in range(0, SIM_HEIGHT, 80):
            cv2.line(canvas, (0, y), (SIM_WIDTH, y), (50, 50, 50), 1)

        # Draw distance rings around humans (approach + attack zones)
        for human in self.get_humans():
            hx, hy = int(human.x), int(human.y)
            cv2.circle(canvas, (hx, hy), SIM_APPROACH_DISTANCE, (40, 80, 40), 1)
            cv2.circle(canvas, (hx, hy), SIM_ATTACK_DISTANCE, (40, 40, 100), 1)
            cv2.putText(canvas, "ALERT ZONE", (hx - 45, hy - SIM_APPROACH_DISTANCE - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (40, 80, 40), 1)
            cv2.putText(canvas, "DANGER ZONE", (hx - 50, hy - SIM_ATTACK_DISTANCE - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (40, 40, 100), 1)

        # Draw connection lines between aggressive dogs and nearest human
        for dog in self.get_dogs():
            if dog.aggressive:
                nearest = None
                min_dist = float("inf")
                for h in self.get_humans():
                    d = dog.distance_to(h)
                    if d < min_dist:
                        min_dist = d
                        nearest = h
                if nearest and min_dist < SIM_APPROACH_DISTANCE * 2:
                    line_color = (0, 0, 255) if min_dist < SIM_ATTACK_DISTANCE else (0, 255, 255)
                    cv2.line(canvas, (int(dog.x), int(dog.y)),
                             (int(nearest.x), int(nearest.y)), line_color, 1)
                    mid_x = int((dog.x + nearest.x) / 2)
                    mid_y = int((dog.y + nearest.y) / 2)
                    cv2.putText(canvas, f"{min_dist:.0f}px",
                                (mid_x, mid_y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, line_color, 1)

        # Draw entities
        for entity in self.entities:
            x, y = int(entity.x), int(entity.y)
            s = entity.size

            if entity.type == "dog":
                # Draw dog as diamond
                pts = np.array([
                    [x, y - s], [x + s, y], [x, y + s], [x - s, y]
                ], np.int32)
                color = (0, 0, 255) if entity.aggressive else entity.color
                cv2.fillPoly(canvas, [pts], color)
                cv2.polylines(canvas, [pts], True, (255, 255, 255), 1)

                label = f"{entity.name}"
                if entity.aggressive:
                    label += " [AGGRESSIVE]"
                cv2.putText(canvas, label, (x - 40, y - s - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            elif entity.type == "person":
                # Draw human as circle
                cv2.circle(canvas, (x, y), s, entity.color, -1)
                cv2.circle(canvas, (x, y), s, (255, 255, 255), 1)
                cv2.putText(canvas, entity.name, (x - 20, y - s - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return canvas

    def draw_hud(self, canvas, result, audio_state, scenario_name):
        """Draw heads-up display with threat info, ultrasonic state, controls."""
        h, w = canvas.shape[:2]

        # ── Top banner: Threat level ──
        threat_label = result["threat_label"]
        threat_conf = result["confidence"]
        color = THREAT_COLORS.get(threat_label, (255, 255, 255))

        cv2.rectangle(canvas, (0, 0), (w, 45), (0, 0, 0), -1)
        cv2.putText(canvas, f"THREAT: {threat_label} ({threat_conf:.0%})",
                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Scenario name
        cv2.putText(canvas, f"Scenario: {scenario_name}",
                     (w - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # ── DANGER red flash overlay ──
        if threat_label == "DANGER":
            overlay = canvas.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
            alpha = 0.12 + 0.08 * math.sin(self.frame_count * 0.3)
            cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)

        # ── Ultrasonic indicator ──
        us_y = 55
        cv2.rectangle(canvas, (0, us_y), (w, us_y + 35), (0, 0, 0), -1)

        if self.ultrasonic_active:
            # Pulsing ultrasonic bar
            pulse = int(128 + 127 * math.sin(self.frame_count * 0.5))
            us_color = (pulse, 0, 255)
            cv2.rectangle(canvas, (0, us_y), (w, us_y + 35), us_color, -1)
            cv2.putText(canvas, ">>> ULTRASONIC EMITTING <<<",
                         (w // 2 - 160, us_y + 25),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            no_trigger = result.get("reason", "")
            if "no ultrasonic" in no_trigger.lower() or "no humans" in no_trigger.lower() \
                    or "downgrade" in no_trigger.lower():
                cv2.putText(canvas, f"ULTRASONIC: OFF — {no_trigger}",
                             (10, us_y + 22),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
            elif self.ultrasonic_cooldown_timer > 0:
                cv2.putText(canvas,
                             f"ULTRASONIC: COOLDOWN ({self.ultrasonic_cooldown_timer:.1f}s)",
                             (10, us_y + 22),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1)
            else:
                cv2.putText(canvas, "ULTRASONIC: STANDBY",
                             (10, us_y + 22),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        # Trigger count
        cv2.putText(canvas, f"Triggers: {self.ultrasonic_total}",
                     (w - 130, us_y + 22),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # ── Right panel: Entity info ──
        panel_x = w - 220
        panel_y = 100
        cv2.rectangle(canvas, (panel_x - 5, panel_y - 5),
                       (w - 5, panel_y + 30 * len(self.entities) + 60), (20, 20, 20), -1)

        cv2.putText(canvas, "ENTITIES", (panel_x, panel_y + 15),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(canvas, f"Dogs: {len(self.get_dogs())}  Humans: {len(self.get_humans())}",
                     (panel_x, panel_y + 35),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        for i, e in enumerate(self.entities):
            y_pos = panel_y + 55 + i * 20
            dot_color = (0, 0, 255) if (e.type == "dog" and e.aggressive) else e.color
            cv2.circle(canvas, (panel_x + 8, y_pos - 4), 5, dot_color, -1)
            cv2.putText(canvas, f"{e.name} ({e.type})", (panel_x + 20, y_pos),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        # ── Bottom: Audio state + reason ──
        cv2.rectangle(canvas, (0, h - 65), (w, h), (0, 0, 0), -1)

        # Audio indicators
        audio_items = [
            ("BARK", audio_state.get("bark", False)),
            ("GROWL", audio_state.get("growl", False)),
            ("SCREAM", audio_state.get("scream", False)),
        ]
        for i, (name, active) in enumerate(audio_items):
            ax = 10 + i * 90
            acolor = (0, 0, 255) if active else (80, 80, 80)
            cv2.putText(canvas, name, (ax, h - 42),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, acolor, 1)

        # Reason
        cv2.putText(canvas, f"Engine: {result.get('reason', '')}",
                     (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        # Controls
        cv2.putText(canvas, "[1-6] Scenario  [SPACE] Pause  [R] Reset  [Q] Quit",
                     (10, h - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)

        # Paused overlay
        if self.paused:
            cv2.putText(canvas, "PAUSED", (w // 2 - 60, h // 2),
                         cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        return canvas
