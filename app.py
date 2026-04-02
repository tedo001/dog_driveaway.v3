"""
app.py — Smart Dog Threat Detection System — Unified Application.

ONE entry point for EVERYTHING:
  - Run detection (simulation, live webcam, hardware)
  - MLOps pipeline (load data, preprocess, train models)
  - System status and setup check

Usage:
  python app.py                          # Interactive menu (RECOMMENDED)
  python app.py --run simulation         # Direct: run simulation
  python app.py --run live               # Direct: run live detection
  python app.py --run live --detector ssd
  python app.py --run live --detector ensemble
  python app.py --run hardware
  python app.py --load "D:\\dog_cnn"     # Direct: load dataset
  python app.py --train all              # Direct: train all models
  python app.py --train yolo --epochs 30
  python app.py --status                 # Direct: show status
  python app.py --setup                  # Check/install dependencies
"""

import sys
import os
import time
import subprocess
import importlib
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Dependency Check ───────────────────────────────────────────────────────

REQUIRED_PACKAGES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "cv2": "opencv-python",
    "ultralytics": "ultralytics",
    "numpy": "numpy",
    "PIL": "Pillow",
    "tqdm": "tqdm",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "dotenv": "python-dotenv",
    "matplotlib": "matplotlib",
    "serial": "pyserial",
}

OPTIONAL_PACKAGES = {
    "sounddevice": "sounddevice",
    "roboflow": "roboflow",
    "onnx": "onnx",
    "onnxruntime": "onnxruntime-gpu",
}


def check_dependencies(verbose=True):
    """Check which required packages are installed."""
    missing = []
    installed = []

    for module_name, pip_name in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(module_name)
            installed.append(pip_name)
        except ImportError:
            missing.append(pip_name)

    optional_missing = []
    optional_installed = []
    for module_name, pip_name in OPTIONAL_PACKAGES.items():
        try:
            importlib.import_module(module_name)
            optional_installed.append(pip_name)
        except ImportError:
            optional_missing.append(pip_name)

    if verbose:
        print(f"\n  Required packages: {len(installed)}/{len(REQUIRED_PACKAGES)} installed")
        if missing:
            print(f"  MISSING: {', '.join(missing)}")
        print(f"  Optional packages: {len(optional_installed)}/{len(OPTIONAL_PACKAGES)} installed")
        if optional_missing:
            print(f"  Optional missing: {', '.join(optional_missing)}")

    return {
        "all_required_ok": len(missing) == 0,
        "missing_required": missing,
        "missing_optional": optional_missing,
        "installed": installed + optional_installed,
    }


def run_setup():
    """Interactive setup: check and install dependencies."""
    print("\n" + "=" * 60)
    print("  SETUP — Dependency Check & Install")
    print("=" * 60)

    result = check_dependencies(verbose=True)

    # Check CUDA
    print()
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = round(torch.cuda.get_device_properties(0).total_mem / (1024**3), 1)
            print(f"  GPU: {gpu_name} ({vram} GB VRAM)")
            print(f"  CUDA: {torch.version.cuda}")
            print(f"  PyTorch: {torch.__version__}")
        else:
            print("  GPU: Not available (CPU mode)")
            print(f"  PyTorch: {torch.__version__}")
    except ImportError:
        print("  PyTorch: NOT INSTALLED")

    if result["all_required_ok"] and not result["missing_optional"]:
        print("\n  All dependencies are installed!")
        return True

    # Offer to install missing
    all_missing = result["missing_required"] + result["missing_optional"]
    if all_missing:
        print(f"\n  Missing packages: {', '.join(all_missing)}")
        print()
        print("  Install options:")
        print("    [1] Install all missing packages (pip install)")
        print("    [2] Install required only (skip optional)")
        print("    [3] Show install command (manual)")
        print("    [0] Skip")
        print()

        choice = input("  Choice: ").strip()

        if choice == "1":
            _pip_install(all_missing)
        elif choice == "2" and result["missing_required"]:
            _pip_install(result["missing_required"])
        elif choice == "3":
            if result["missing_required"]:
                print(f"\n  pip install {' '.join(result['missing_required'])}")
            if result["missing_optional"]:
                print(f"  pip install {' '.join(result['missing_optional'])}")
            print()
            print("  For CUDA GPU support:")
            print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        elif choice == "0":
            pass

    return result["all_required_ok"]


def _pip_install(packages):
    """Install packages via pip."""
    cmd = [sys.executable, "-m", "pip", "install"] + packages
    print(f"\n  Running: pip install {' '.join(packages)}")
    print("  " + "-" * 50)
    try:
        subprocess.check_call(cmd)
        print("\n  Installation complete!")
    except subprocess.CalledProcessError as e:
        print(f"\n  Installation failed: {e}")
        print("  Try running manually:")
        print(f"    pip install {' '.join(packages)}")


# ── UI Helpers ─────────────────────────────────────────────────────────────

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_banner():
    print()
    print("=" * 60)
    print()
    print("     SMART DOG THREAT DETECTION SYSTEM")
    print("     ─────────────────────────────────")
    print("     Detect aggressive dogs • Protect humans")
    print()
    print("=" * 60)

    # Device info
    try:
        from config import DEVICE_NAME, DEVICE_VRAM_GB, DEVICE
        if DEVICE != "cpu":
            print(f"  GPU: {DEVICE_NAME} ({DEVICE_VRAM_GB} GB)")
        else:
            print(f"  Device: CPU mode")
    except Exception:
        print("  Device: Unknown")

    # Model status
    try:
        from config import YOLO_MODEL_PATH, SSD_MODEL_PATH, CNN_MODEL_PATH
        models = []
        if YOLO_MODEL_PATH.exists():
            models.append("YOLO")
        if SSD_MODEL_PATH.exists():
            models.append("SSD")
        if CNN_MODEL_PATH.exists():
            models.append("CNN")
        if models:
            print(f"  Models ready: {', '.join(models)}")
        else:
            print("  Models: None trained yet (use MLOps Pipeline first)")
    except Exception:
        pass

    print("=" * 60)


def prompt_menu(options, prompt_text="Choose"):
    """Show numbered menu and get user choice."""
    print()
    for i, (label, _) in enumerate(options):
        print(f"    [{i + 1}] {label}")
    print(f"    [0] Exit")
    print()
    while True:
        try:
            choice = input(f"  {prompt_text} (0-{len(options)}): ").strip()
            if choice == "0" or choice.lower() == "q":
                return None
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx]
        except (ValueError, IndexError):
            pass
        print(f"    Invalid. Enter 0-{len(options)}")


# ── Detection Functions ────────────────────────────────────────────────────

def _load_detector(backend):
    """Load the selected detector backend."""
    if backend == "ensemble":
        from models.ensemble_detector import EnsembleDetector
        print("[INIT] Loading ENSEMBLE detector (YOLO + SSD fusion)...")
        return EnsembleDetector(use_yolo=True, use_ssd=True)
    elif backend == "ssd":
        from models.ssd_model import SSDDetector
        print("[INIT] Loading SSD300-VGG16 detector...")
        return SSDDetector()
    else:
        from models.yolo_model import DualYOLODetector
        print("[INIT] Loading YOLOv8 detector...")
        return DualYOLODetector()


def run_simulation():
    """Launch visual simulation mode with animated dogs/humans."""
    import cv2
    from config import SIM_FPS, WINDOW_NAME
    from simulation.simulator import Simulator
    from simulation.scenarios import ScenarioManager, update_scenario_behavior
    from utils.logger import EventLogger

    print("=" * 60)
    print("  SIMULATION MODE — Smart Dog Threat Detection")
    print("=" * 60)
    print()
    print("  No camera or models needed!")
    print("  Testing threat logic with animated scenarios.")
    print()
    print("  Controls:")
    print("    1-6   : Switch scenario")
    print("    SPACE  : Pause / Resume")
    print("    R      : Reset scenario")
    print("    Q      : Quit")
    print()
    for sid, name in ScenarioManager.SCENARIOS.items():
        print(f"    [{sid}] {name}")
    print()
    print("=" * 60)

    sim = Simulator()
    logger = EventLogger()

    current_scenario = 1
    scenario_name = ScenarioManager.load(sim, current_scenario)
    print(f"\n[SIM] Loaded: {scenario_name}")

    frame_delay = 1.0 / SIM_FPS

    try:
        while True:
            start_time = time.time()

            if not sim.paused:
                update_scenario_behavior(sim, current_scenario)
                sim.frame_count += 1

            result, cnn_results, audio_state = sim.classify_threats()
            dt = frame_delay if not sim.paused else 0
            sim.update_ultrasonic(result, dt)

            canvas = sim.render()
            canvas = sim.draw_hud(canvas, result, audio_state, scenario_name)

            logger.log(
                num_dogs=len(sim.get_dogs()),
                num_humans=len(sim.get_humans()),
                threat_class=result["threat_class"],
                threat_label=result["threat_label"],
                confidence=result["confidence"],
                audio_bark=audio_state.get("bark", False),
                audio_growl=audio_state.get("growl", False),
                audio_scream=audio_state.get("scream", False),
                ultrasonic_triggered=result["trigger_ultrasonic"],
                notes=f"[SIM-{current_scenario}] {result['reason']}",
            )

            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord(" "):
                sim.paused = not sim.paused
                print(f"[SIM] {'Paused' if sim.paused else 'Resumed'}")
            elif key == ord("r"):
                scenario_name = ScenarioManager.load(sim, current_scenario)
                print(f"[SIM] Reset: {scenario_name}")
            elif ord("1") <= key <= ord("6"):
                current_scenario = key - ord("0")
                scenario_name = ScenarioManager.load(sim, current_scenario)
                print(f"\n[SIM] Switched to Scenario {current_scenario}: {scenario_name}")

            elapsed = time.time() - start_time
            sleep_time = frame_delay - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[SIM] Interrupted")
    finally:
        cv2.destroyAllWindows()
        logger.close()
        if sim.event_log:
            print("\n  Event Log:")
            for event in sim.event_log:
                print(f"    {event}")
        print(f"\n  Total ultrasonic triggers: {sim.ultrasonic_total}")
        print("[SIM] Done!")


def run_live(detector_backend):
    """Live mode: webcam + detector + CNN + laptop speaker ultrasonic."""
    import cv2
    from models.behavior_net_v2 import BehaviorClassifierV2
    from models.threat_engine import ThreatEngine
    from audio.audio_detector import AudioDetector
    from audio.audio_combiner import AudioCombiner
    from audio.ultrasonic_trigger import UltrasonicTrigger
    from utils.ui import UIRenderer
    from utils.logger import EventLogger
    from config import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT

    print("=" * 60)
    print("  LIVE MODE — Smart Dog Threat Detection System")
    print(f"  Detector: {detector_backend.upper()}")
    print("=" * 60)
    print()

    detector = _load_detector(detector_backend)
    print("[INIT] Loading BehaviorNetV2 CNN...")
    cnn = BehaviorClassifierV2()
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
        return

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

            detections = detector.detect(frame)
            num_dogs = sum(1 for d in detections if d["class"] == "dog")
            num_humans = sum(1 for d in detections if d["class"] == "person")

            cnn_results = []
            if num_dogs > 0:
                dog_crops = detector.get_dog_crops(frame, detections)
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
                notes=f"[{detector_backend.upper()}] {result['reason']}",
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


def run_hardware(detector_backend):
    """Hardware mode: webcam + detector + CNN + Arduino ultrasonic sensor."""
    import cv2
    from models.behavior_net_v2 import BehaviorClassifierV2
    from models.threat_engine import ThreatEngine
    from audio.audio_detector import AudioDetector
    from audio.audio_combiner import AudioCombiner
    from hardware.ultrasonic_hw import UltrasonicHardware
    from utils.ui import UIRenderer
    from utils.logger import EventLogger
    from config import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT

    print("=" * 60)
    print("  HARDWARE MODE — Arduino + Ultrasonic Sensor")
    print(f"  Detector: {detector_backend.upper()}")
    print("=" * 60)
    print()

    detector = _load_detector(detector_backend)
    print("[INIT] Loading BehaviorNetV2 CNN...")
    cnn = BehaviorClassifierV2()
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
        return

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

            detections = detector.detect(frame)
            num_dogs = sum(1 for d in detections if d["class"] == "dog")
            num_humans = sum(1 for d in detections if d["class"] == "person")

            cnn_results = []
            if num_dogs > 0:
                dog_crops = detector.get_dog_crops(frame, detections)
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
                notes=f"[HW-{detector_backend.upper()}] {result['reason']}",
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


# ── Run Detection Menu ─────────────────────────────────────────────────────

def menu_run_detection():
    """Submenu: Run detection system."""
    options = [
        ("Simulation  (no camera needed — test logic with animated scenarios)", "simulation"),
        ("Live        (webcam + laptop speaker ultrasonic)", "live"),
        ("Hardware    (webcam + Arduino + external ultrasonic)", "hardware"),
    ]

    choice = prompt_menu(options, "Mode")
    if not choice:
        return

    mode = choice[1]
    detector = "yolo"  # default

    if mode in ("live", "hardware"):
        det_options = [
            ("YOLO    (YOLOv8n — fast, recommended)", "yolo"),
            ("SSD     (SSD300-VGG16 — anchor-based)", "ssd"),
            ("Ensemble (YOLO + SSD fusion — highest accuracy)", "ensemble"),
        ]
        det_choice = prompt_menu(det_options, "Detector")
        if not det_choice:
            return
        detector = det_choice[1]

    print(f"\n  Starting {mode.upper()} mode with {detector.upper()} detector...")
    print("  " + "-" * 50)
    print()

    if mode == "simulation":
        run_simulation()
    elif mode == "live":
        run_live(detector)
    elif mode == "hardware":
        run_hardware(detector)


# ── MLOps Pipeline ─────────────────────────────────────────────────────────

def menu_mlops():
    """Submenu: MLOps pipeline for data + training."""
    try:
        from mlops.state import load_state, save_state, log_event, update_dataset, update_model, get_status_summary
        from mlops.data_loader import load_dataset_from_path, download_coco, count_images
        from mlops.preprocessor import auto_select_mapping, create_custom_mapping, extract_behavior_crops, MAPPING_PRESETS
        from mlops.trainer import get_device_info, clear_old_models, train_yolo, train_ssd, train_cnn, train_all
    except ImportError as e:
        print(f"\n  ERROR: Missing dependency — {e}")
        print("  Run Setup first to install required packages.")
        input("\n  Press Enter to continue...")
        return

    state = load_state()

    while True:
        clear_screen()
        print()
        print("=" * 60)
        print("     MLOps Pipeline — Data & Training")
        print("=" * 60)

        device = get_device_info()
        print(f"  Device: {device['display']}")

        # Quick status
        coco = state["datasets"]["coco"]
        beh = state["datasets"]["behavior"]
        print(f"\n  Data:   COCO={'LOADED' if coco['loaded'] else 'NO'}  |  "
              f"Behavior={'LOADED' if beh['loaded'] else 'NO'}")
        models_status = []
        for name, info in state["models"].items():
            models_status.append(f"{name.upper()}={'OK' if info['trained'] else 'NO'}")
        print(f"  Models: {' | '.join(models_status)}")

        print("=" * 60)

        options = [
            ("Load Data        (ZIP, folder, or COCO download)", "load"),
            ("Train Models     (YOLO, SSD, CNN, or ALL)", "train"),
            ("Quick Pipeline   (data → preprocess → train ALL)", "quick"),
            ("Clear Models     (remove old weights)", "clear"),
            ("View Status      (datasets, models, history)", "status"),
            ("Back to Main Menu", "back"),
        ]

        choice = prompt_menu(options, "Choose")

        if choice is None or choice[1] == "back":
            break

        # Delegate to mlops/app.py handlers
        from mlops.app import (
            menu_load_data, menu_train, menu_quick_pipeline,
            menu_clear_models, menu_status,
        )

        if choice[1] == "load":
            menu_load_data(state)
        elif choice[1] == "train":
            menu_train(state)
        elif choice[1] == "quick":
            menu_quick_pipeline(state)
        elif choice[1] == "clear":
            menu_clear_models(state)
        elif choice[1] == "status":
            menu_status(state)

        # Reload state
        state = load_state()


# ── System Info ────────────────────────────────────────────────────────────

def menu_system_info():
    """Show full system info."""
    print("\n" + "=" * 60)
    print("  SYSTEM INFORMATION")
    print("=" * 60)

    # Python
    print(f"\n  Python: {sys.version.split()[0]}")
    print(f"  Path:   {sys.executable}")

    # Dependencies
    check_dependencies(verbose=True)

    # GPU
    try:
        import torch
        print(f"\n  PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  CUDA:    {torch.version.cuda}")
            print(f"  GPU:     {torch.cuda.get_device_name(0)}")
            vram = round(torch.cuda.get_device_properties(0).total_mem / (1024**3), 1)
            print(f"  VRAM:    {vram} GB")
        else:
            print("  CUDA:    Not available")
    except ImportError:
        print("\n  PyTorch: NOT INSTALLED")

    # Models
    print()
    try:
        from config import YOLO_MODEL_PATH, SSD_MODEL_PATH, CNN_MODEL_PATH, CNN_SCRIPTED_PATH, YOLO_ONNX_PATH
        model_files = [
            ("YOLO detector", YOLO_MODEL_PATH),
            ("SSD detector", SSD_MODEL_PATH),
            ("CNN BehaviorNet", CNN_MODEL_PATH),
            ("CNN Scripted", CNN_SCRIPTED_PATH),
            ("YOLO ONNX", YOLO_ONNX_PATH),
        ]
        for name, path in model_files:
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"  {name:20s}: {path.name} ({size_mb:.1f} MB)")
            else:
                print(f"  {name:20s}: not found")
    except Exception as e:
        print(f"  Model check error: {e}")

    # MLOps state
    try:
        from mlops.state import load_state, get_status_summary
        state = load_state()
        print(get_status_summary(state))
    except Exception:
        pass

    input("\n  Press Enter to continue...")


# ── Main Menu Loop ─────────────────────────────────────────────────────────

def main_menu():
    """Main interactive menu — the ONE entry point."""
    while True:
        clear_screen()
        print_banner()

        options = [
            ("Run Detection    (simulation / live / hardware)", "run"),
            ("MLOps Pipeline   (load data, preprocess, train models)", "mlops"),
            ("System Info      (GPU, models, dependencies)", "info"),
            ("Setup            (check & install dependencies)", "setup"),
        ]

        choice = prompt_menu(options, "Choose")

        if choice is None:
            print("\n  Goodbye!\n")
            break
        elif choice[1] == "run":
            menu_run_detection()
        elif choice[1] == "mlops":
            menu_mlops()
        elif choice[1] == "info":
            menu_system_info()
        elif choice[1] == "setup":
            run_setup()
            input("\n  Press Enter to continue...")


# ── CLI Entry Point ────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Smart Dog Threat Detection System — Unified Application",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python app.py                          Interactive menu
  python app.py --run simulation         Run simulation mode
  python app.py --run live               Run live detection (YOLO)
  python app.py --run live --detector ensemble   Live with YOLO+SSD fusion
  python app.py --load "/path/to/data"   Load dataset for training
  python app.py --train all              Train all models
  python app.py --train cnn --epochs 30  Train CNN with 30 epochs
  python app.py --status                 Show pipeline status
  python app.py --setup                  Check/install dependencies
""",
    )
    parser.add_argument("--run", type=str, choices=["simulation", "live", "hardware"],
                        help="Run detection mode")
    parser.add_argument("--detector", type=str, choices=["yolo", "ssd", "ensemble"],
                        default="yolo", help="Detector backend (default: yolo)")
    parser.add_argument("--load", type=str, metavar="PATH",
                        help="Load dataset from path (ZIP or folder)")
    parser.add_argument("--train", type=str, choices=["yolo", "ssd", "cnn", "all"],
                        help="Train model(s)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training epochs")
    parser.add_argument("--status", action="store_true",
                        help="Show pipeline status")
    parser.add_argument("--setup", action="store_true",
                        help="Check and install dependencies")
    parser.add_argument("--clear", action="store_true",
                        help="Clear all trained models")
    parser.add_argument("--coco", type=int, metavar="N",
                        help="Download COCO dog+human dataset (N images)")

    args = parser.parse_args()

    # If no arguments → interactive menu
    has_args = any([args.run, args.load, args.train, args.status, args.setup, args.clear, args.coco])
    if not has_args:
        main_menu()
        return

    # ── CLI direct commands ──

    if args.setup:
        run_setup()
        return

    if args.status:
        from mlops.state import load_state, get_status_summary
        state = load_state()
        print(get_status_summary(state))
        return

    if args.run:
        print()
        print("  ===== SMART DOG THREAT DETECTION SYSTEM =====")
        print(f"  Mode     : {args.run.upper()}")
        print(f"  Detector : {args.detector.upper()}")
        print()
        if args.run == "simulation":
            run_simulation()
        elif args.run == "live":
            run_live(args.detector)
        elif args.run == "hardware":
            run_hardware(args.detector)
        return

    if args.clear:
        from mlops.trainer import clear_old_models
        from mlops.state import load_state, log_event
        clear_old_models()
        state = load_state()
        log_event(state, "Models Cleared", "All models deleted")
        print("  All models cleared.")
        return

    if args.coco:
        from mlops.data_loader import download_coco
        from mlops.state import load_state, update_dataset, log_event
        result = download_coco(max_images=args.coco)
        if result["success"]:
            state = load_state()
            update_dataset(state, "coco", loaded=True, images=args.coco)
            log_event(state, "COCO Download", f"{args.coco} images")
            print(f"  COCO dataset ready: {args.coco} images")
        else:
            print(f"  ERROR: {result.get('error')}")
        return

    if args.load:
        from mlops.data_loader import load_dataset_from_path
        from mlops.preprocessor import auto_select_mapping, extract_behavior_crops
        from mlops.state import load_state, update_dataset, log_event

        info = load_dataset_from_path(args.load)
        if info["success"]:
            print(f"\n  Dataset loaded: {info['format']}")
            print(f"  Classes: {info.get('classes', [])}")

            classes = info.get("classes", [])
            if classes:
                preset_name, mapping = auto_select_mapping(classes)
                if mapping:
                    print(f"  Auto-mapping: {preset_name}")
                    result = extract_behavior_crops(
                        img_dir=info["img_dir"],
                        lbl_dir=info["lbl_dir"],
                        class_names=classes,
                        class_mapping=mapping,
                        clear_old=True,
                    )
                    if result["success"]:
                        state = load_state()
                        update_dataset(state, "behavior", loaded=True,
                                       crops=result["stats"].get("train", {}))
                        log_event(state, "Data Loaded", f"{result['total']} crops")
        else:
            print(f"  ERROR: {info.get('error')}")
        return

    if args.train:
        from mlops.trainer import train_yolo, train_ssd, train_cnn, train_all
        from mlops.state import load_state, update_model, log_event

        state = load_state()
        epochs = args.epochs

        if args.train == "all":
            results = train_all(epochs=epochs)
            for name, res in results.items():
                if res.get("success"):
                    update_model(state, name, trained=True,
                                 epochs=res.get("epochs", epochs or 60),
                                 best_metric=res.get("best_val_acc") or res.get("best_val_loss"),
                                 path=res.get("model_path", ""))
                    log_event(state, f"{name.upper()} Trained", f"epochs={res.get('epochs')}")
        elif args.train == "yolo":
            res = train_yolo(epochs=epochs)
            if res.get("success"):
                update_model(state, "yolo", trained=True, epochs=epochs or 60,
                             path=res.get("model_path", ""))
                log_event(state, "YOLO Trained", f"epochs={epochs or 60}")
        elif args.train == "ssd":
            res = train_ssd(epochs=epochs)
            if res.get("success"):
                update_model(state, "ssd", trained=True, epochs=res.get("epochs"),
                             best_metric=res.get("best_val_loss"),
                             path=res.get("model_path", ""))
                log_event(state, "SSD Trained", f"epochs={res.get('epochs')}")
        elif args.train == "cnn":
            res = train_cnn(epochs=epochs)
            if res.get("success"):
                update_model(state, "cnn", trained=True, epochs=res.get("epochs"),
                             best_metric=res.get("best_val_acc"),
                             path=res.get("model_path", ""))
                log_event(state, "CNN Trained", f"epochs={res.get('epochs')}")
        return


if __name__ == "__main__":
    main()
