"""
config.py — Central configuration for Smart Dog Threat Detection System.
All settings in one place. Override via .env where noted.

Detection: YOLO only (dog + person)
Classification: BehaviorNetV2 CNN (IDLE / ALERT / DANGER / DOG_FIGHT)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── System Mode ──────────────────────────────────────────────────────────────
# "simulation" | "live" | "hardware"
SYSTEM_MODE = os.getenv("SYSTEM_MODE", "simulation")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
EXPORT_DIR = BASE_DIR / "export"
LOG_DIR = BASE_DIR / "logs"

YOLO_MODEL_PATH = EXPORT_DIR / "dog_detector.pt"
CNN_MODEL_PATH = EXPORT_DIR / "behavior_net_v2.pt"

# ── Dataset ──────────────────────────────────────────────────────────────────
DATASET_DIR = DATA_DIR / "dataset"
CROPS_DIR = DATA_DIR / "crops"

# ── Device Auto-Detection ────────────────────────────────────────────────────
import torch as _torch
if _torch.cuda.is_available():
    DEVICE = "cuda"
    DEVICE_NAME = _torch.cuda.get_device_name(0)
    DEVICE_VRAM_GB = round(_torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
else:
    DEVICE = "cpu"
    DEVICE_NAME = "CPU"
    DEVICE_VRAM_GB = 0

# ── YOLO Settings ────────────────────────────────────────────────────────────
YOLO_BASE_MODEL = "yolov8n.pt"
YOLO_IMGSZ = 640 if DEVICE != "cpu" else 416
YOLO_EPOCHS = 60
YOLO_BATCH = 16 if DEVICE != "cpu" else 4
YOLO_CONF_THRESHOLD = 0.5
YOLO_IOU_THRESHOLD = 0.45
YOLO_DEVICE = DEVICE

# ── CNN (BehaviorNetV2) Settings ─────────────────────────────────────────────
CNN_INPUT_SIZE = 128
CNN_NUM_CLASSES = 4
CNN_BATCH_SIZE = 16 if DEVICE == "cpu" else 64
CNN_EPOCHS = 60
CNN_LR = 0.001
CNN_PATIENCE = 10

# ── Threat Classes ───────────────────────────────────────────────────────────
THREAT_CLASSES = {
    0: "IDLE",
    1: "ALERT",
    2: "DANGER",
    3: "DOG_FIGHT",
}
THREAT_COLORS = {
    "IDLE": (0, 255, 0),
    "ALERT": (0, 255, 255),
    "DANGER": (0, 0, 255),
    "DOG_FIGHT": (255, 0, 255),
}

# ── Audio Settings ───────────────────────────────────────────────────────────
AUDIO_SAMPLE_RATE = 44100
AUDIO_CHANNELS = 1
AUDIO_BLOCK_SIZE = 2048
AUDIO_BARK_THRESHOLD = float(os.getenv("BARK_THRESHOLD", "0.15"))
AUDIO_GROWL_THRESHOLD = float(os.getenv("GROWL_THRESHOLD", "0.10"))
AUDIO_SCREAM_THRESHOLD = float(os.getenv("SCREAM_THRESHOLD", "0.20"))

# ── Ultrasonic Settings ─────────────────────────────────────────────────────
ULTRASONIC_FREQ_MIN = 20000
ULTRASONIC_FREQ_MAX = 25000
ULTRASONIC_DURATION = 2.0
ULTRASONIC_COOLDOWN = 5.0
ULTRASONIC_VOLUME = 0.8

# ── Camera Settings ──────────────────────────────────────────────────────────
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# ── Simulation Settings ─────────────────────────────────────────────────────
SIM_WIDTH = 800
SIM_HEIGHT = 600
SIM_FPS = 30
SIM_DOG_SPEED = 3.0
SIM_HUMAN_SPEED = 1.5
SIM_APPROACH_DISTANCE = 120
SIM_ATTACK_DISTANCE = 50
SIM_DOG_FIGHT_DISTANCE = 60

# ── Arduino / Hardware Settings ──────────────────────────────────────────────
ARDUINO_PORT = os.getenv("ARDUINO_PORT", "COM3")
ARDUINO_BAUD = 9600
ARDUINO_TIMEOUT = 1.0
ARDUINO_CMD_TRIGGER = "ULTRASONIC_ON"
ARDUINO_CMD_STOP = "ULTRASONIC_OFF"
ARDUINO_CMD_STATUS = "STATUS"

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_EVENTS = True
LOG_FILE_PREFIX = "threat_log"

# ── Display ──────────────────────────────────────────────────────────────────
WINDOW_NAME = "Dog Threat Detection System"
SHOW_FPS = True
SHOW_AUDIO_METER = True
