"""
evaluate/eval_yolo.py — Evaluate YOLO dog detector performance.
Reports mAP50, mAP50-95, Precision, Recall, and per-class metrics.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO
from config import YOLO_MODEL_PATH, YOLO_BASE_MODEL, DATASET_DIR, YOLO_DEVICE, YOLO_IMGSZ


def eval_yolo():
    """Run YOLO validation on the test/valid split."""
    data_yaml = DATASET_DIR / "data.yaml"
    if not data_yaml.exists():
        print(f"ERROR: data.yaml not found at {data_yaml}")
        print("Run 'python data/download_dataset.py' first")
        sys.exit(1)

    model_path = YOLO_MODEL_PATH if YOLO_MODEL_PATH.exists() else YOLO_BASE_MODEL
    print("=" * 60)
    print("  YOLO Evaluation — Dog Detector")
    print("=" * 60)
    print(f"  Model   : {model_path}")
    print(f"  Dataset : {data_yaml}")
    print(f"  Device  : cuda:{YOLO_DEVICE}")
    print("=" * 60)

    model = YOLO(str(model_path))

    results = model.val(
        data=str(data_yaml),
        imgsz=YOLO_IMGSZ,
        device=YOLO_DEVICE,
        batch=16,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("  Results Summary")
    print("=" * 60)
    print(f"  mAP50      : {results.box.map50:.4f}")
    print(f"  mAP50-95   : {results.box.map:.4f}")
    print(f"  Precision  : {results.box.mp:.4f}")
    print(f"  Recall     : {results.box.mr:.4f}")
    print(f"  Speed      : {results.speed}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    eval_yolo()
