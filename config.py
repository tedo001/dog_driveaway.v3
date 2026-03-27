"""
config.py — Central configuration for Smart Dog Threat Detection System.
All settings in one place. Override via .env where noted.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
EXPORT_DIR = BASE_DIR / "export"
LOG_DIR = BASE_DIR / "logs"
TRAIN_DIR = BASE_DIR / "train"

YOLO_MODEL_PATH = EXPORT_DIR / "dog_detector.pt"
CNN_MODEL_PATH = EXPORT_DIR / "behavior_net.pt"
CNN_SCRIPTED_PATH = EXPORT_DIR / "behavior_net_scripted.pt"
YOLO_ONNX_PATH = EXPORT_DIR / "dog_detector.onnx"

# ── Dataset ──────────────────────────────────────────────────────────────────
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
ROBOFLOW_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "dog-behaviour-1")
ROBOFLOW_PROJECT = os.getenv("ROBOFLOW_PROJECT", "dog-behaviour-1")
ROBOFLOW_VERSION = int(os.getenv("ROBOFLOW_VERSION", "8"))
DATASET_DIR = DATA_DIR / "dataset"
CROPS_DIR = DATA_DIR / "crops"

# ── YOLO Settings ────────────────────────────────────────────────────────────
YOLO_BASE_MODEL = "yolov8n.pt"
YOLO_IMGSZ = 640
YOLO_EPOCHS = 100
YOLO_BATCH = 16
YOLO_CONF_THRESHOLD = 0.5
YOLO_IOU_THRESHOLD = 0.45
YOLO_DEVICE = "0"  # CUDA GPU 0 (RTX 4060)

# ── CNN (BehaviorNet) Settings ───────────────────────────────────────────────
CNN_INPUT_SIZE = 128
CNN_NUM_CLASSES = 4
CNN_BATCH_SIZE = 32
CNN_EPOCHS = 60
CNN_LR = 0.001
CNN_PATIENCE = 10  # early stopping patience

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

# ── Safety Rules (hardcoded — NEVER bypass) ──────────────────────────────────
# Rule 1: DOG_FIGHT → NEVER trigger ultrasonic
# Rule 2: No humans in frame → NEVER classify DANGER
# Rule 3: Only DANGER + humans present → trigger ultrasonic
# Rule 4: Audio scream detected → always escalate to DANGER
# Rule 5: Audio growl + visual ALERT → upgrade to DANGER

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_EVENTS = True
LOG_FILE_PREFIX = "threat_log"

# ── Display ──────────────────────────────────────────────────────────────────
WINDOW_NAME = "Dog Threat Detection System"
SHOW_FPS = True
SHOW_AUDIO_METER = True
