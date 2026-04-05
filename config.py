"""
config.py — Central configuration for Smart Dog Threat Detection System.
All settings in one place. Override via .env where noted.

Detection:      YOLOv8 (ultralytics) — detects dogs + humans
Classification: BehaviorNetV2 CNN — classifies dog behavior
Trigger:        Ultrasonic repeller fires when DANGER dog is near a human
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── System Mode ───────────────────────────────────────────────────────────────
# "simulation" | "live" | "hardware"
SYSTEM_MODE = os.getenv("SYSTEM_MODE", "simulation")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "data"
EXPORT_DIR = BASE_DIR / "export"
LOG_DIR    = BASE_DIR / "logs"

# ── YOLO: use the already-trained weights; do NOT retrain ─────────────────────
# Trained via coco128.yaml — weights live at runs/detect/train6/weights/best.pt
YOLO_MODEL_PATH = BASE_DIR / "runs" / "detect" / "train6" / "weights" / "best.pt"

CNN_MODEL_PATH  = EXPORT_DIR / "behavior_net_v2.pt"

# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET_DIR = DATA_DIR / "roboflow_download"
CROPS_DIR   = DATA_DIR / "crops"

# ── Device Auto-Detection ─────────────────────────────────────────────────────
import torch as _torch
if _torch.cuda.is_available():
    DEVICE        = "cuda"
    DEVICE_NAME   = _torch.cuda.get_device_name(0)
    DEVICE_VRAM_GB = round(_torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
else:
    DEVICE        = "cpu"
    DEVICE_NAME   = "CPU"
    DEVICE_VRAM_GB = 0

# ── YOLOv8 (ultralytics) Settings ────────────────────────────────────────────
# The model is LOADED (not trained) from YOLO_MODEL_PATH above.
# YOLO_BASE_MODEL is only used as a fallback if YOLO_MODEL_PATH is missing.
YOLO_BASE_MODEL     = "yolov8n.pt"   # nano=fast  |  yolov8s/m/l = more accurate
YOLO_IMGSZ          = 640 if DEVICE != "cpu" else 416
YOLO_EPOCHS         = 60
YOLO_BATCH          = 16 if DEVICE != "cpu" else 4
YOLO_CONF_THRESHOLD = 0.5
YOLO_IOU_THRESHOLD  = 0.45
YOLO_DEVICE         = DEVICE

# COCO class indices (best.pt was trained on coco128 which uses standard COCO ids)
YOLO_CLASS_PERSON = 0    # COCO "person"
YOLO_CLASS_DOG    = 16   # COCO "dog"
YOLO_CLASSES      = [YOLO_CLASS_PERSON, YOLO_CLASS_DOG]   # only detect these two

# Proximity detection — dog is considered "near a human" if:
#   pixel distance between bbox centres < YOLO_PROXIMITY_DIST
YOLO_PROXIMITY_DIST = 150   # pixels

# ── CNN (BehaviorNetV2) Settings ──────────────────────────────────────────────
CNN_INPUT_SIZE  = 128
CNN_BATCH_SIZE  = 16 if DEVICE == "cpu" else 64
CNN_EPOCHS      = 60
CNN_LR          = 0.001
CNN_PATIENCE    = 10

# ── Threat Classes ────────────────────────────────────────────────────────────
# CURRENT: 2 classes (DANGER + IDLE)
# ImageFolder sorts folder names A→Z → DANGER=0, IDLE=1
#
# TO EXPAND TO 4 CLASSES LATER:
#   1. Collect ALERT and DOG_FIGHT image crops
#   2. Create data/crops/train/ALERT and data/crops/train/DOG_FIGHT
#   3. Set CNN_NUM_CLASSES = 4
#   4. Uncomment entries 2 and 3 in THREAT_CLASSES below
#   5. Delete export/behavior_net_v2.pt and retrain
CNN_NUM_CLASSES = 2

THREAT_CLASSES = {
    0: "DANGER",     # A→Z index 0  (folder: DANGER)
    1: "IDLE",       # A→Z index 1  (folder: IDLE)
    # 2: "ALERT",    # future — uncomment when data is ready
    # 3: "DOG_FIGHT",
}

THREAT_COLORS = {
    "IDLE":      (0, 255, 0),      # green
    "ALERT":     (0, 255, 255),    # yellow
    "DANGER":    (0, 0, 255),      # red
    "DOG_FIGHT": (255, 0, 255),    # magenta
}

# CNN confidence threshold — results below this are shown as UNKNOWN
CNN_CONF_THRESHOLD = 0.55

# ── Ultrasonic Repeller Trigger Logic ─────────────────────────────────────────
# Repeller fires when ALL conditions are true:
#   1. CNN classifies dog as a class in ULTRASONIC_TRIGGER_ON_CLASSES
#   2. Dog bbox centre is within ULTRASONIC_TRIGGER_DIST pixels of a human bbox
#   3. ULTRASONIC_COOLDOWN seconds have passed since last trigger
ULTRASONIC_TRIGGER_ON_CLASSES = ["DANGER", "DOG_FIGHT"]
ULTRASONIC_TRIGGER_DIST       = 200     # pixels
ULTRASONIC_FREQ_MIN           = 20000   # Hz
ULTRASONIC_FREQ_MAX           = 25000   # Hz
ULTRASONIC_DURATION           = 2.0     # seconds per burst
ULTRASONIC_COOLDOWN           = 5.0     # seconds between triggers
ULTRASONIC_VOLUME             = 0.8     # 0.0 – 1.0

# ── Audio Settings ────────────────────────────────────────────────────────────
AUDIO_SAMPLE_RATE      = 44100
AUDIO_CHANNELS         = 1
AUDIO_BLOCK_SIZE       = 2048
AUDIO_BARK_THRESHOLD   = float(os.getenv("BARK_THRESHOLD",   "0.15"))
AUDIO_GROWL_THRESHOLD  = float(os.getenv("GROWL_THRESHOLD",  "0.10"))
AUDIO_SCREAM_THRESHOLD = float(os.getenv("SCREAM_THRESHOLD", "0.20"))

# ── Camera Settings ───────────────────────────────────────────────────────────
CAMERA_INDEX  = 0
CAMERA_WIDTH  = 640
CAMERA_HEIGHT = 480
CAMERA_FPS    = 30

# ── Simulation Settings ───────────────────────────────────────────────────────
SIM_WIDTH              = 800
SIM_HEIGHT             = 600
SIM_FPS                = 30
SIM_DOG_SPEED          = 3.0
SIM_HUMAN_SPEED        = 1.5
SIM_APPROACH_DISTANCE  = 120
SIM_ATTACK_DISTANCE    = 50
SIM_DOG_FIGHT_DISTANCE = 60

# ── Arduino / Hardware Settings ───────────────────────────────────────────────
ARDUINO_PORT        = os.getenv("ARDUINO_PORT", "COM3")
ARDUINO_BAUD        = 9600
ARDUINO_TIMEOUT     = 1.0
ARDUINO_CMD_TRIGGER = "ULTRASONIC_ON"
ARDUINO_CMD_STOP    = "ULTRASONIC_OFF"
ARDUINO_CMD_STATUS  = "STATUS"

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_EVENTS      = True
LOG_FILE_PREFIX = "threat_log"

# ── Display ───────────────────────────────────────────────────────────────────
WINDOW_NAME      = "Dog Threat Detection System"
SHOW_FPS         = True
SHOW_AUDIO_METER = True