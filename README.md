# Smart Dog Threat Detection System

Real-time AI system that detects aggressive dogs approaching humans and emits ultrasonic sound to deter them. Uses computer vision (YOLO + SSD + CNN) and audio analysis.

## Core Objective

| Scenario | Action |
|----------|--------|
| Aggressive dog **ALONE** | NO ultrasonic |
| Aggressive dog **approaching HUMAN** | EMIT ultrasonic |
| Dog fight (dog vs dog) | NEVER ultrasonic |

## System Modes

| Mode | Command | What It Does |
|------|---------|--------------|
| Simulation | `python main.py --mode simulation` | Animated test — no camera needed |
| Live | `python main.py --mode live` | Webcam + laptop speaker |
| Hardware | `python main.py --mode hardware` | Webcam + Arduino + ultrasonic sensor |

## Detector Backends

| Backend | Command | Architecture |
|---------|---------|--------------|
| YOLO | `--detector yolo` | YOLOv8n — grid-based, anchor-free, ~0.9ms/frame |
| SSD | `--detector ssd` | SSD300-VGG16 — anchor-based, multi-scale, ~3-5ms/frame |

## Quick Start (5 Minutes)

```bash
# 1. Clone and setup
git clone https://github.com/tedo001/dog_driveaway.v3.git
cd dog_driveaway.v3
git checkout claude/smart-dog-threat-detection-bLGM2
pip install -r requirements.txt

# 2. Run simulation immediately (no data needed)
python main.py --mode simulation
```

Controls: `1-6` switch scenarios, `SPACE` pause, `R` reset, `Q` quit.

## Full Training Pipeline

### Option A: COCO Dataset (Recommended — No Manual Work)

```bash
# Downloads dog+person images from COCO 2017 automatically
python data/download_coco_dogs.py --max 5000

# Create behavior crops for CNN
python data/prepare_crops.py

# Train all 3 models
python train/train_yolo.py          # YOLO dog detector (~1-3 hours)
python train/train_ssd.py           # SSD dog detector (~2-4 hours)
python train/train_cnn.py           # Behavior classifier (~20-60 min)

# Evaluate
python evaluate/eval_yolo.py
python evaluate/eval_ssd.py
python evaluate/eval_cnn.py

# Run live
python main.py --mode live --detector yolo
```

### Option B: Your Own Images

```bash
# 1. Put your images in data/raw_images/ (.jpg or .png)
#    Collect from Google Images, phone camera, or YouTube

# 2. Auto-label with COCO-pretrained YOLO (detects dogs + humans instantly)
python data/prepare_coco_dog_human.py

# 3. Train all models
python train/train_yolo.py
python train/train_ssd.py
python train/train_cnn.py

# 4. Run
python main.py --mode live --detector yolo
```

### Option C: Roboflow Dataset

```bash
# 1. Setup .env with your Private API key
copy .env.example .env
# Edit .env → ROBOFLOW_API_KEY=your_private_key

# 2. Download
python data/download_dataset.py

# 3. Continue with training (same as above)
```

## Project Structure

```
dog_driveaway.v3/
├── main.py                      ← Entry point (3 modes + 2 detectors)
├── simulation_main.py           ← Simulation entry point
├── config.py                    ← All settings
├── requirements.txt
├── .env.example                 ← Environment template (copy to .env)
│
├── models/
│   ├── yolo_model.py            ← YOLOv8 dual detector (dog + person)
│   ├── ssd_model.py             ← SSD300-VGG16 detector
│   ├── cnn_model.py             ← BehaviorNet CNN (4 conv blocks)
│   └── threat_engine.py         ← Decision logic + 5 safety rules
│
├── simulation/
│   ├── simulator.py             ← 2D animated simulation engine
│   └── scenarios.py             ← 6 pre-built test scenarios
│
├── audio/
│   ├── audio_detector.py        ← Mic: bark/growl/scream detection
│   ├── audio_combiner.py        ← Fuses audio + visual signals
│   └── ultrasonic_trigger.py    ← 20-25kHz speaker output
│
├── hardware/
│   ├── arduino_bridge.py        ← Serial communication to Arduino
│   ├── ultrasonic_hw.py         ← Hardware ultrasonic control
│   └── arduino_sketch.ino       ← Arduino firmware (25kHz PWM)
│
├── data/
│   ├── download_coco_dogs.py    ← Download COCO dog+person subset
│   ├── download_dataset.py      ← Download from Roboflow
│   ├── prepare_coco_dog_human.py← Auto-label your images with COCO YOLO
│   ├── prepare_crops.py         ← Create CNN training crops
│   ├── ssd_dataset.py           ← SSD PyTorch Dataset loader
│   ├── auto_label.py            ← YOLO auto-labeling
│   └── collect_youtube.py       ← YouTube frame extractor
│
├── train/
│   ├── train_yolo.py            ← Fine-tune YOLOv8n
│   ├── train_ssd.py             ← Train SSD300-VGG16
│   └── train_cnn.py             ← Train BehaviorNet CNN
│
├── evaluate/
│   ├── eval_yolo.py             ← YOLO metrics (mAP, precision, recall)
│   ├── eval_ssd.py              ← SSD metrics
│   └── eval_cnn.py              ← CNN metrics + confusion matrix
│
├── export/
│   └── export_models.py         ← Export to ONNX + TorchScript
│
├── utils/
│   ├── ui.py                    ← OpenCV display renderer
│   └── logger.py                ← CSV event logger
│
└── logs/                        ← Auto-generated event logs
```

## ML Architecture

### Detection Pipeline
```
Camera Frame (640x480)
    │
    ├──→ YOLO/SSD ──→ Dog bounding boxes + Person bounding boxes
    │
    ├──→ Crop dogs ──→ BehaviorNet CNN ──→ IDLE/ALERT/DANGER/DOG_FIGHT
    │
    ├──→ Microphone ──→ Bark/Growl/Scream detection
    │
    └──→ Threat Engine (fuses visual + audio)
              │
              ├── IDLE      → green box, no action
              ├── ALERT     → yellow box, monitoring
              ├── DANGER    → red box + ULTRASONIC TRIGGERED
              └── DOG_FIGHT → magenta box, NEVER trigger
```

### Safety Rules (Hardcoded — Never Bypassed)
1. **DOG_FIGHT → NEVER trigger ultrasonic** (dog vs dog is not our problem)
2. **No humans in frame → NEVER classify DANGER** (no one to protect)
3. **DANGER + humans → TRIGGER ultrasonic** (protect the human)
4. **Audio scream → escalate to DANGER** (human in distress)
5. **Audio growl + ALERT → upgrade to DANGER** (attack imminent)

## Hardware Setup (Future)

### Arduino Wiring
```
Arduino Uno/Nano
  Pin 9  → Ultrasonic transducer (+)
  GND    → Ultrasonic transducer (-)
  USB    → Laptop (serial COM port)
```

Upload `hardware/arduino_sketch.ino` to Arduino, then:
```bash
# Set COM port in .env
ARDUINO_PORT=COM3

# Run
python main.py --mode hardware
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Detection (YOLO) | Ultralytics YOLOv8n |
| Detection (SSD) | torchvision SSD300-VGG16 |
| Classification | Custom BehaviorNet CNN (2.7M params) |
| Training | PyTorch 2.5+ / CUDA 12.1 |
| Audio | sounddevice + scipy |
| Display | OpenCV |
| Hardware | Arduino + pyserial |
| Dataset | COCO 2017 / Roboflow |

## Requirements

- **GPU**: NVIDIA RTX 4060 (8GB VRAM) or similar
- **Python**: 3.10+
- **OS**: Windows 11 / Linux
- **Camera**: Any USB webcam
- **Optional**: Arduino Uno/Nano + ultrasonic transducer

## Current Status

- [x] Project structure and config
- [x] YOLO detector (dog + person)
- [x] SSD detector (dog + person)
- [x] BehaviorNet CNN (4 classes)
- [x] Threat engine with safety rules
- [x] Audio detection (bark/growl/scream)
- [x] Ultrasonic trigger (software)
- [x] Visual simulation (6 scenarios)
- [x] Arduino hardware bridge
- [x] Training + evaluation pipeline
- [x] COCO dataset downloader
- [ ] Collecting training data ← **YOU ARE HERE**
- [ ] Training models on collected data
- [ ] Live camera testing
- [ ] Arduino hardware integration
- [ ] Field testing with real dogs
