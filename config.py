"""
config.py — Central configuration for Smart Dog Threat Detection System.
All settings in one place. Override via .env where noted.

MODES:
  - simulation : Visual simulation with animated dogs/humans (no camera needed)
  - live       : Real webcam + laptop speaker ultrasonic
  - hardware   : Real webcam + Arduino + external ultrasonic sensor
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── System Mode ──────────────────────────────────────────────────────────────
# "simulation" | "live" | "hardware"
SYSTEM_MODE = os.getenv("SYSTEM_MODE", "simulation")

# ── Detector Backend ─────────────────────────────────────────────────────────
# "yolo" | "ssd"  — which object detector to use in live/hardware mode
DETECTOR_BACKEND = os.getenv("DETECTOR_BACKEND", "yolo")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
EXPORT_DIR = BASE_DIR / "export"
LOG_DIR = BASE_DIR / "logs"

YOLO_MODEL_PATH = EXPORT_DIR / "dog_detector.pt"
CNN_MODEL_PATH = EXPORT_DIR / "behavior_net.pt"
CNN_SCRIPTED_PATH = EXPORT_DIR / "behavior_net_scripted.pt"
YOLO_ONNX_PATH = EXPORT_DIR / "dog_detector.onnx"
SSD_MODEL_PATH = EXPORT_DIR / "ssd_dog_detector.pt"

# ── Dataset ──────────────────────────────────────────────────────────────────
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
ROBOFLOW_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "durgamanis-workspace-on48g")
ROBOFLOW_PROJECT = os.getenv("ROBOFLOW_PROJECT", "dog-behaviour-1")
ROBOFLOW_VERSION = int(os.getenv("ROBOFLOW_VERSION", "8"))
DATASET_DIR = DATA_DIR / "dataset"
CROPS_DIR = DATA_DIR / "crops"

# ── Device Auto-Detection ────────────────────────────────────────────────────
import torch as _torch
if _torch.cuda.is_available():
    DEVICE = "0"                   # CUDA GPU
    DEVICE_NAME = _torch.cuda.get_device_name(0)
    DEVICE_VRAM_GB = round(_torch.cuda.get_device_properties(0).total_mem / (1024**3), 1)
else:
    DEVICE = "cpu"                 # CPU fallback
    DEVICE_NAME = "CPU"
    DEVICE_VRAM_GB = 0

# ── YOLO Settings ────────────────────────────────────────────────────────────
YOLO_BASE_MODEL = "yolov8n.pt"
YOLO_IMGSZ = 640 if DEVICE != "cpu" else 416   # smaller on CPU = faster
YOLO_EPOCHS = 60
YOLO_BATCH = 16 if DEVICE != "cpu" else 4      # GPU: 16, CPU: 4
YOLO_CONF_THRESHOLD = 0.5
YOLO_IOU_THRESHOLD = 0.45
YOLO_DEVICE = DEVICE

# ── SSD (Single Shot Detector) Settings ──────────────────────────────────────
SSD_IMGSZ = 300                # SSD300 fixed input size
SSD_NUM_CLASSES = 3            # 0=background, 1=dog, 2=person
SSD_CLASS_NAMES = ["__background__", "dog", "person"]
SSD_CONF_THRESHOLD = 0.5
SSD_NMS_THRESHOLD = 0.45
SSD_BATCH_SIZE = 4 if DEVICE == "cpu" else 16   # GPU: 16, CPU: 4
SSD_EPOCHS = 60
SSD_LR = 0.005                 # SGD learning rate
SSD_PATIENCE = 12              # early stopping patience

# ── CNN (BehaviorNet) Settings ───────────────────────────────────────────────
CNN_INPUT_SIZE = 128
CNN_NUM_CLASSES = 4
CNN_BATCH_SIZE = 16 if DEVICE == "cpu" else 64  # GPU: 64, CPU: 16
CNN_EPOCHS = 60
CNN_LR = 0.001
CNN_PATIENCE = 10  # early stopping patience

# ── MLOps Pipeline Settings ─────────────────────────────────────────────────
MLOPS_STATE_FILE = BASE_DIR / "mlops" / "pipeline_state.json"
MLOPS_DEFAULT_EPOCHS = 60

# ── Threat Classes ───────────────────────────────────────────────────────────
THREAT_CLASSES = {
    0: "IDLE",
    1: "ALERT",
    2: "DANGER",
    3: "DOG_FIGHT",
}
THREAT_COLORS = {
    "IDLE": (0, 255, 0),       # green
    "ALERT": (0, 255, 255),    # yellow
    "DANGER": (0, 0, 255),     # red
    "DOG_FIGHT": (255, 0, 255),  # magenta
}

# ── Audio Settings ───────────────────────────────────────────────────────────
AUDIO_SAMPLE_RATE = 44100
AUDIO_CHANNELS = 1
AUDIO_BLOCK_SIZE = 2048
AUDIO_BARK_THRESHOLD = float(os.getenv("BARK_THRESHOLD", "0.15"))
AUDIO_GROWL_THRESHOLD = float(os.getenv("GROWL_THRESHOLD", "0.10"))
AUDIO_SCREAM_THRESHOLD = float(os.getenv("SCREAM_THRESHOLD", "0.20"))

# ── Ultrasonic Settings ─────────────────────────────────────────────────────
ULTRASONIC_FREQ_MIN = 20000   # Hz
ULTRASONIC_FREQ_MAX = 25000   # Hz
ULTRASONIC_DURATION = 2.0     # seconds
ULTRASONIC_COOLDOWN = 5.0     # seconds between triggers
ULTRASONIC_VOLUME = 0.8       # 0.0 to 1.0

# ── Camera Settings ──────────────────────────────────────────────────────────
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# ── Simulation Settings ─────────────────────────────────────────────────────
SIM_WIDTH = 800
SIM_HEIGHT = 600
SIM_FPS = 30
SIM_DOG_SPEED = 3.0          # pixels per frame base speed
SIM_HUMAN_SPEED = 1.5        # pixels per frame
SIM_APPROACH_DISTANCE = 120  # pixels — dog within this = ALERT
SIM_ATTACK_DISTANCE = 50     # pixels — dog within this = DANGER
SIM_DOG_FIGHT_DISTANCE = 60  # pixels — dog-to-dog within this = DOG_FIGHT

# ── Arduino / Hardware Settings ──────────────────────────────────────────────
ARDUINO_PORT = os.getenv("ARDUINO_PORT", "COM3")  # Windows COM port
ARDUINO_BAUD = 9600
ARDUINO_TIMEOUT = 1.0  # seconds
# Commands sent to Arduino over serial:
ARDUINO_CMD_TRIGGER = "ULTRASONIC_ON"
ARDUINO_CMD_STOP = "ULTRASONIC_OFF"
ARDUINO_CMD_STATUS = "STATUS"

# ── Safety Rules (hardcoded — NEVER bypass) ──────────────────────────────────
# CORE OBJECTIVE:
#   - Aggressive dog ALONE       → NEVER emit ultrasonic
#   - Aggressive dog + HUMAN     → EMIT ultrasonic
#
# Rule 1: DOG_FIGHT (dog vs dog) → NEVER trigger ultrasonic
# Rule 2: No humans in frame     → NEVER trigger ultrasonic (even if DANGER)
# Rule 3: DANGER + humans nearby → TRIGGER ultrasonic
# Rule 4: Audio scream detected  → escalate to DANGER
# Rule 5: Audio growl + ALERT    → upgrade to DANGER

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_EVENTS = True
LOG_FILE_PREFIX = "threat_log"

# ── Display ──────────────────────────────────────────────────────────────────
WINDOW_NAME = "Dog Threat Detection System"
SHOW_FPS = True
SHOW_AUDIO_METER = True
