# Smart Dog Threat Detection System

AI system that detects aggressive dogs approaching humans and emits ultrasonic sound. Uses YOLO detection + CNN behavior classification.

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

Double-click `run.bat` (Windows) or `./run.sh` (Linux).

## How It Works

```
Camera Frame → YOLO (dog+person) → Spatial Analysis → CNN Behavior → Threat Engine
                                                                          ↓
                                                          DANGER + human → ULTRASONIC
```

**Safety Rules:**
- Aggressive dog ALONE → NO ultrasonic
- Aggressive dog + HUMAN → EMIT ultrasonic  
- Dog fight → NEVER ultrasonic

## The App

GUI application with 4 tabs:

| Tab | What It Does |
|-----|-------------|
| **Dashboard** | GPU info, model status, data status |
| **Data Pipeline** | Browse dataset → auto-detect → preprocess → ready |
| **Training** | Train YOLO / CNN with progress bar |
| **Run Detection** | Simulation, live camera, or hardware mode |

## Project Structure

```
app.py                  ← GUI Application (start here)
config.py               ← Settings (GPU auto-detect)

models/
  yolo_model.py         ← YOLOv8n dog+person detector
  behavior_net_v2.py    ← BehaviorNetV2 CNN (residual + SE)
  spatial_analyzer.py   ← Dog-human distance tracking
  threat_engine.py      ← 5 safety rules → final decision
  pipeline.py           ← Connects all models

mlops/
  data_loader.py        ← Load any dataset (ZIP/folder/COCO)
  preprocessor.py       ← Auto class mapping + crop extraction
  trainer.py            ← Train YOLO + CNN
  state.py              ← Pipeline state tracking

simulation/             ← Animated test scenarios (no camera needed)
audio/                  ← Bark/growl detection + ultrasonic output
hardware/               ← Arduino serial bridge
utils/                  ← UI renderer + CSV logger
```

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA (recommended)
- Webcam (for live/hardware mode)
- Optional: Arduino + ultrasonic transducer
