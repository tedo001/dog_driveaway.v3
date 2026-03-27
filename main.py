"""
main.py — Entry point for Smart Dog Threat Detection System.
Real-time pipeline: Camera → YOLO → CNN → Threat Engine → Ultrasonic.
Combines video (webcam) + audio (mic) for threat assessment.
"""

import sys
import time
import cv2

from config import (
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    WINDOW_NAME,
    THREAT_CLASSES,
)
from models.yolo_model import DualYOLODetector
from models.cnn_model import BehaviorClassifier
from models.threat_engine import ThreatEngine
from audio.audio_detector import AudioDetector
from audio.audio_combiner import AudioCombiner
from audio.ultrasonic_trigger import UltrasonicTrigger
from utils.ui import UIRenderer
from utils.logger import EventLogger


def main():
    print("=" * 60)
    print("  Smart Dog Threat Detection System v1.4")
    print("=" * 60)
    print()

    # ── Initialize all components ────────────────────────────────
    print("[INIT] Loading models...")
    yolo = DualYOLODetector()
    cnn = BehaviorClassifier()
    engine = ThreatEngine()

    print("[INIT] Starting audio system...")
    audio_detector = AudioDetector()
    audio_combiner = AudioCombiner()
    ultrasonic = UltrasonicTrigger()
    audio_detector.start()

    print("[INIT] Setting up UI and logger...")
    ui = UIRenderer()
    logger = EventLogger()

    # ── Open camera ──────────────────────────────────────────────
    print(f"[INIT] Opening camera {CAMERA_INDEX}...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        print("ERROR: Could not open camera!")
        print("  - Check if webcam is connected")
        print("  - Try changing CAMERA_INDEX in config.py")
        sys.exit(1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INIT] Camera ready: {actual_w}x{actual_h}")
    print()
    print("[RUNNING] Press 'q' to quit")
    print("=" * 60)

    # ── Main detection loop ──────────────────────────────────────
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame capture failed, retrying...")
                time.sleep(0.1)
                continue

            # Step 1: YOLO detection (dogs + humans)
            detections = yolo.detect(frame)
            num_dogs = sum(1 for d in detections if d["class"] == "dog")
            num_humans = sum(1 for d in detections if d["class"] == "person")

            # Step 2: CNN behavior classification on dog crops
            cnn_results = []
            if num_dogs > 0:
                dog_crops = yolo.get_dog_crops(frame, detections)
                for crop, bbox in dog_crops:
                    tc, tl, conf = cnn.classify(crop)
                    cnn_results.append((tc, tl, conf))

            # Step 3: Get audio state
            raw_audio = audio_detector.get_state()
            audio_combiner.update(raw_audio)
            audio_state = audio_combiner.get_combined_state(num_dogs, num_humans)

            # Step 4: Threat engine decision (enforces all safety rules)
            result = engine.evaluate(
                cnn_results=cnn_results,
                num_dogs=num_dogs,
                num_humans=num_humans,
                audio_state=audio_state,
            )

            threat_label = result["threat_label"]
            threat_conf = result["confidence"]
            trigger = result["trigger_ultrasonic"]

            # Step 5: Ultrasonic trigger (ONLY for DANGER + humans)
            ultrasonic_fired = False
            if trigger:
                ultrasonic_fired = ultrasonic.trigger()

            # Step 6: Render UI
            frame = ui.render(
                frame, detections, threat_label, threat_conf, audio_state,
            )

            # Step 7: Log event
            logger.log(
                num_dogs=num_dogs,
                num_humans=num_humans,
                threat_class=result["threat_class"],
                threat_label=threat_label,
                confidence=threat_conf,
                audio_bark=audio_state.get("bark", False),
                audio_growl=audio_state.get("growl", False),
                audio_scream=audio_state.get("scream", False),
                ultrasonic_triggered=ultrasonic_fired,
                notes=result["reason"],
            )

            # Step 8: Display and check for quit
            key = ui.show(frame)
            if key == ord("q"):
                print("\n[EXIT] Quit requested by user")
                break

    except KeyboardInterrupt:
        print("\n[EXIT] Interrupted by user")

    finally:
        # Cleanup
        print("[CLEANUP] Shutting down...")
        audio_detector.stop()
        cap.release()
        ui.cleanup()
        logger.close()
        print("[CLEANUP] Done. Goodbye!")


if __name__ == "__main__":
    main()
