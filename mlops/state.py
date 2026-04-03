"""
mlops/state.py — Pipeline state tracker. Persists to pipeline_state.json.
"""

import json
from pathlib import Path
from datetime import datetime

STATE_FILE = Path(__file__).resolve().parent / "pipeline_state.json"

DEFAULT_STATE = {
    "datasets": {
        "detection": {"loaded": False, "images": 0, "path": ""},
        "behavior": {"loaded": False, "crops": {"DANGER": 0, "IDLE": 0}, "path": ""},
    },
    "models": {
        "yolo": {"trained": False, "epochs": 0, "best_metric": None, "path": ""},
        "cnn": {"trained": False, "epochs": 0, "best_metric": None, "path": ""},
    },
    "history": [],
}


def load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except (json.JSONDecodeError, KeyError):
            pass
    return json.loads(json.dumps(DEFAULT_STATE))


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def log_event(state, action, details=""):
    state["history"].append({
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "action": action,
        "details": details,
    })
    state["history"] = state["history"][-100:]
    save_state(state)


def update_dataset(state, name, **kwargs):
    if name in state["datasets"]:
        state["datasets"][name].update(kwargs)
        save_state(state)


def update_model(state, name, **kwargs):
    if name in state["models"]:
        state["models"][name].update(kwargs)
        save_state(state)
