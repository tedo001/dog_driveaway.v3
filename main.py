"""
main.py — Entry point for Smart Dog Threat Detection System.

Three modes (set SYSTEM_MODE in .env or config.py):
  simulation : Visual sim with animated dogs/humans — no hardware needed
  live       : Real webcam + laptop speaker ultrasonic
  hardware   : Real webcam + Arduino + external ultrasonic sensor

Usage:
  python main.py                  → uses SYSTEM_MODE from config
  python main.py --mode simulation
  python main.py --mode live
  python main.py --mode hardware
"""

import sys
import time
import argparse
import cv2

from config import (
    SYSTEM_MODE,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    WINDOW_NAME,
)


def run_simulation():
    """Launch visual simulation mode."""
    from simulation_main import run_simulation as sim_run
    sim_run()


def run_live():
    """Live mode: webcam + YOLO + CNN + laptop speaker ultrasonic."""
    from models.yolo_model import DualYOLODetector
    from models.cnn_model import BehaviorClassifier
    from models.threat_engine import ThreatEngine
    from audio.audio_detector import AudioDetector
    from audio.audio_combiner import AudioCombiner
    from audio.ultrasonic_trigger import UltrasonicTrigger
    from utils.ui import UIRenderer
    from utils.logger import EventLogger

    print("=" * 60)
    print("  LIVE MODE — Smart Dog Threat Detection System")
    print("=" * 60)
    print()

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

    print(f"[INIT] Opening camera {CAMERA_INDEX}...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        print("ERROR: Could not open camera!")
        sys.exit(1)

    print(f"[INIT] Camera ready: "
          f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print("[RUNNING] Press 'q' to quit")
    print("=" * 60)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            detections = yolo.detect(frame)
            num_dogs = sum(1 for d in detections if d["class"] == "dog")
            num_humans = sum(1 for d in detections if d["class"] == "person")

            cnn_results = []
            if num_dogs > 0:
                dog_crops = yolo.get_dog_crops(frame, detections)
                for crop, bbox in dog_crops:
                    cnn_results.append(cnn.classify(crop))

            raw_audio = audio_detector.get_state()
            audio_combiner.update(raw_audio)
            audio_state = audio_combiner.get_combined_state(num_dogs, num_humans)

            result = engine.evaluate(
                cnn_results=cnn_results,
                num_dogs=num_dogs,
                num_humans=num_humans,
                audio_state=audio_state,
            )

            ultrasonic_fired = False
            if result["trigger_ultrasonic"]:
                ultrasonic_fired = ultrasonic.trigger()

            frame = ui.render(
                frame, detections, result["threat_label"],
                result["confidence"], audio_state,
            )

            logger.log(
                num_dogs=num_dogs,
                num_humans=num_humans,
                threat_class=result["threat_class"],
                threat_label=result["threat_label"],
                confidence=result["confidence"],
                audio_bark=audio_state.get("bark", False),
                audio_growl=audio_state.get("growl", False),
                audio_scream=audio_state.get("scream", False),
                ultrasonic_triggered=ultrasonic_fired,
                notes=result["reason"],
            )

            key = ui.show(frame)
            if key == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        audio_detector.stop()
        cap.release()
        ui.cleanup()
        logger.close()
        print("[LIVE] Done!")


def run_hardware():
    """Hardware mode: webcam + YOLO + CNN + Arduino ultrasonic sensor."""
    from models.yolo_model import DualYOLODetector
    from models.cnn_model import BehaviorClassifier
    from models.threat_engine import ThreatEngine
    from audio.audio_detector import AudioDetector
    from audio.audio_combiner import AudioCombiner
    from hardware.ultrasonic_hw import UltrasonicHardware
    from utils.ui import UIRenderer
    from utils.logger import EventLogger

    print("=" * 60)
    print("  HARDWARE MODE — Arduino + Ultrasonic Sensor")
    print("=" * 60)
    print()

    print("[INIT] Loading models...")
    yolo = DualYOLODetector()
    cnn = BehaviorClassifier()
    engine = ThreatEngine()

    print("[INIT] Starting audio system...")
    audio_detector = AudioDetector()
    audio_combiner = AudioCombiner()

    print("[INIT] Connecting to Arduino...")
    ultrasonic = UltrasonicHardware()

    print("[INIT] Setting up UI and logger...")
    ui = UIRenderer()
    logger = EventLogger()

    print(f"[INIT] Opening camera {CAMERA_INDEX}...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        print("ERROR: Could not open camera!")
        sys.exit(1)

    print(f"[INIT] Camera ready: "
          f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print("[RUNNING] Press 'q' to quit")
    print("=" * 60)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            detections = yolo.detect(frame)
            num_dogs = sum(1 for d in detections if d["class"] == "dog")
            num_humans = sum(1 for d in detections if d["class"] == "person")

            cnn_results = []
            if num_dogs > 0:
                dog_crops = yolo.get_dog_crops(frame, detections)
                for crop, bbox in dog_crops:
                    cnn_results.append(cnn.classify(crop))

            raw_audio = audio_detector.get_state()
            audio_combiner.update(raw_audio)
            audio_state = audio_combiner.get_combined_state(num_dogs, num_humans)

            result = engine.evaluate(
                cnn_results=cnn_results,
                num_dogs=num_dogs,
                num_humans=num_humans,
                audio_state=audio_state,
            )

            # Hardware ultrasonic trigger (Arduino)
            ultrasonic_fired = False
            if result["trigger_ultrasonic"]:
                ultrasonic_fired = ultrasonic.trigger()

            frame = ui.render(
                frame, detections, result["threat_label"],
                result["confidence"], audio_state,
            )

            logger.log(
                num_dogs=num_dogs,
                num_humans=num_humans,
                threat_class=result["threat_class"],
                threat_label=result["threat_label"],
                confidence=result["confidence"],
                audio_bark=audio_state.get("bark", False),
                audio_growl=audio_state.get("growl", False),
                audio_scream=audio_state.get("scream", False),
                ultrasonic_triggered=ultrasonic_fired,
                notes=f"[HW] {result['reason']}",
            )

            key = ui.show(frame)
            if key == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        audio_detector.stop()
        ultrasonic.cleanup()
        cap.release()
        ui.cleanup()
        logger.close()
        print("[HARDWARE] Done!")


def main():
    parser = argparse.ArgumentParser(description="Smart Dog Threat Detection System")
    parser.add_argument(
        "--mode",
        choices=["simulation", "live", "hardware"],
        default=SYSTEM_MODE,
        help="Run mode: simulation (default), live (webcam+speaker), hardware (webcam+Arduino)",
    )
    args = parser.parse_args()

    mode = args.mode.lower()

    print()
    print("  ===== SMART DOG THREAT DETECTION SYSTEM =====")
    print(f"  Mode: {mode.upper()}")
    print()
    print("  CORE OBJECTIVE:")
    print("    Aggressive dog ALONE       → NO ultrasonic")
    print("    Aggressive dog + HUMAN     → EMIT ultrasonic")
    print("    Dog fight (dog vs dog)     → NEVER ultrasonic")
    print()

    if mode == "simulation":
        run_simulation()
    elif mode == "live":
        run_live()
    elif mode == "hardware":
        run_hardware()
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
