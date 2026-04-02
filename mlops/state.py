"""
mlops/state.py — Pipeline state and history tracker.

Tracks:
  - Which datasets are loaded and their stats
  - Which models are trained and their metrics
  - Pipeline run history with timestamps
  - Current system status

State is saved to mlops/pipeline_state.json so it persists between sessions.
"""

import json
import time
from pathlib import Path
from datetime import datetime


STATE_FILE = Path(__file__).resolve().parent / "pipeline_state.json"

DEFAULT_STATE = {
    "datasets": {
        "coco": {"loaded": False, "images": 0, "path": ""},
        "behavior": {"loaded": False, "crops": {"DANGER": 0, "IDLE": 0}, "path": ""},
    },
    "models": {
        "yolo": {"trained": False, "epochs": 0, "best_metric": None, "path": ""},
        "ssd": {"trained": False, "epochs": 0, "best_metric": None, "path": ""},
        "cnn": {"trained": False, "epochs": 0, "best_metric": None, "path": ""},
    },
    "history": [],
}


def load_state():
    """Load pipeline state from disk."""
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except (json.JSONDecodeError, KeyError):
            pass
    return DEFAULT_STATE.copy()


def save_state(state):
    """Save pipeline state to disk."""
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def log_event(state, action, details=""):
    """Add an event to pipeline history."""
    state["history"].append({
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "action": action,
        "details": details,
    })
    # Keep last 100 events
    state["history"] = state["history"][-100:]
    save_state(state)


def update_dataset(state, name, **kwargs):
    """Update dataset status."""
    if name in state["datasets"]:
        state["datasets"][name].update(kwargs)
        save_state(state)


def update_model(state, name, **kwargs):
    """Update model training status."""
    if name in state["models"]:
        state["models"][name].update(kwargs)
        save_state(state)


def get_status_summary(state):
    """Get a formatted status summary string."""
    lines = []
    lines.append("=" * 60)
    lines.append("  PIPELINE STATUS")
    lines.append("=" * 60)

    # Datasets
    lines.append("\n  DATASETS:")
    coco = state["datasets"]["coco"]
    if coco["loaded"]:
        lines.append(f"    COCO Dog+Human  : LOADED ({coco['images']} images)")
    else:
        lines.append(f"    COCO Dog+Human  : NOT LOADED")

    beh = state["datasets"]["behavior"]
    if beh["loaded"]:
        d = beh["crops"].get("DANGER", 0)
        i = beh["crops"].get("IDLE", 0)
        lines.append(f"    Behavior Crops  : LOADED (DANGER={d}, IDLE={i})")
    else:
        lines.append(f"    Behavior Crops  : NOT LOADED")

    # Models
    lines.append("\n  MODELS:")
    for name, info in state["models"].items():
        if info["trained"]:
            metric = info.get("best_metric", "?")
            lines.append(f"    {name.upper():5s} : TRAINED (epochs={info['epochs']}, metric={metric})")
        else:
            lines.append(f"    {name.upper():5s} : NOT TRAINED")

    # Recent history
    if state["history"]:
        lines.append("\n  RECENT ACTIVITY:")
        for event in state["history"][-5:]:
            lines.append(f"    [{event['time']}] {event['action']}")
            if event.get("details"):
                lines.append(f"      {event['details']}")

    lines.append("=" * 60)
    return "\n".join(lines)
