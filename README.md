# Smart Dog Threat Detection System

Real-time AI system that detects aggressive dogs approaching humans and emits ultrasonic sound to deter them. Uses computer vision (YOLO + SSD + CNN) and audio analysis.

## Core Objective

| Scenario | Action |
|----------|--------|
| Aggressive dog **ALONE** | NO ultrasonic |
| Aggressive dog **approaching HUMAN** | EMIT ultrasonic |
| Dog fight (dog vs dog) | NEVER ultrasonic |

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/tedo001/dog_driveaway.v3.git
cd dog_driveaway.v3

# 2. Run the app (checks deps, installs if needed)
python app.py
```

The app gives you an interactive menu:
```
[1] Run Detection    (simulation / live / hardware)
[2] MLOps Pipeline   (load data, preprocess, train models)
[3] System Info      (GPU, models, dependencies)
[4] Setup            (check & install dependencies)
```

### CLI Shortcuts

```bash
python app.py --setup                          # Check/install dependencies
python app.py --run simulation                 # Run simulation (no camera needed)
python app.py --run live --detector yolo       # Run live detection
python app.py --run live --detector ensemble   # Live with YOLO+SSD fusion (best)
python app.py --run hardware                   # Arduino + ultrasonic hardware
python app.py --load "/path/to/dataset"        # Load dataset (ZIP or folder)
python app.py --train all --epochs 60          # Train all models
python app.py --coco 5000                      # Download COCO dog+human dataset
python app.py --status                         # View pipeline status
```

## Training Pipeline

### Option A: Interactive (Recommended)

```bash
python app.py
# Choose [2] MLOps Pipeline → [3] Quick Pipeline
# Point to your dataset → auto-detect → auto-preprocess → train all
```

### Option B: CLI

```bash
# Load Roboflow dataset + auto-map classes + train everything
python app.py --load "/path/to/dataset" && python app.py --train all

# Or download COCO + train
python app.py --coco 5000 && python app.py --train all
```

## Project Structure

```
dog_driveaway.v3/
├── app.py                       ← UNIFIED APP — start here!
├── config.py                    ← All settings (GPU auto-detect)
├── requirements.txt
├── run.bat / run.sh             ← Double-click launchers
│
├── models/
│   ├── yolo_model.py            ← YOLOv8n dual detector (dog + person)
│   ├── ssd_model.py             ← SSD300-VGG16 detector
│   ├── ensemble_detector.py     ← YOLO + SSD fusion (best accuracy)
│   ├── behavior_net_v2.py       ← BehaviorNetV2 CNN (residual + SE attention)
│   ├── spatial_analyzer.py      ← Dog-human distance tracking
│   ├── threat_engine.py         ← Decision logic + 5 safety rules
│   ├── pipeline.py              ← 5-stage unified detection pipeline
│   └── ssd_dataset.py           ← SSD PyTorch Dataset loader
│
├── mlops/
│   ├── app.py                   ← MLOps pipeline dashboard
│   ├── data_loader.py           ← Unified data loading (ZIP/folder/COCO)
│   ├── preprocessor.py          ← Auto class mapping + crop extraction
│   ├── trainer.py               ← Unified training (YOLO/SSD/CNN)
│   └── state.py                 ← Pipeline state tracking (JSON)
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
├── utils/
│   ├── ui.py                    ← OpenCV display renderer
│   └── logger.py                ← CSV event logger
│
├── data/                        ← Dataset storage (auto-created)
├── export/                      ← Trained model weights
└── logs/                        ← Event logs
```

## Detection Pipeline

```
Camera Frame (640x480)
    │
    ├──→ YOLO/SSD/Ensemble ──→ Dog + Person bounding boxes
    │
    ├──→ Spatial Analyzer ──→ Distance + approach speed
    │
    ├──→ Crop dogs ──→ BehaviorNetV2 ──→ IDLE/ALERT/DANGER/DOG_FIGHT
    │
    ├──→ Microphone ──→ Bark/Growl/Scream detection
    │
    └──→ Threat Engine (fuses all signals)
              │
              ├── IDLE      → green, no action
              ├── ALERT     → yellow, monitoring
              ├── DANGER    → red + ULTRASONIC TRIGGERED
              └── DOG_FIGHT → magenta, NEVER trigger
```

### Safety Rules (Hardcoded)
1. **DOG_FIGHT → NEVER trigger ultrasonic**
2. **No humans in frame → NEVER classify DANGER**
3. **DANGER + humans → TRIGGER ultrasonic**
4. **Audio scream → escalate to DANGER**
5. **Audio growl + ALERT → upgrade to DANGER**

## Hardware Setup

```
Arduino Uno/Nano
  Pin 9  → Ultrasonic transducer (+)
  GND    → Ultrasonic transducer (-)
  USB    → Laptop (serial COM port)
```

Upload `hardware/arduino_sketch.ino`, set `ARDUINO_PORT=COM3` in `.env`, then:
```bash
python app.py --run hardware
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Detection | YOLOv8n + SSD300-VGG16 + Ensemble fusion |
| Classification | BehaviorNetV2 (residual + SE attention, ~4.2M params) |
| Training | PyTorch 2.5+ / CUDA 12.1 |
| Audio | sounddevice + scipy |
| Display | OpenCV |
| Hardware | Arduino + pyserial |
| Dataset | COCO 2017 / Roboflow |

## Requirements

- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **Python**: 3.10+
- **OS**: Windows / Linux
- **Camera**: Any USB webcam (for live/hardware mode)
- **Optional**: Arduino Uno/Nano + ultrasonic transducer
