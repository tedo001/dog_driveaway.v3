"""
train/train_yolo.py — Fine-tune YOLOv8n on dog behaviour dataset.
Auto-detects GPU/CPU and adjusts settings accordingly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO
from config import (
    YOLO_BASE_MODEL,
    YOLO_EPOCHS,
    YOLO_BATCH,
    YOLO_IMGSZ,
    YOLO_DEVICE,
    DEVICE_NAME,
    DATASET_DIR,
    EXPORT_DIR,
)


def train_yolo():
    """Fine-tune YOLOv8n on the dog behaviour dataset."""
    data_yaml = DATASET_DIR / "data.yaml"
    if not data_yaml.exists():
        print(f"ERROR: data.yaml not found at {data_yaml}")
        print("Run 'python data/download_dataset.py' first")
        sys.exit(1)

    is_cpu = YOLO_DEVICE == "cpu"

    print("=" * 60)
    print("  YOLO Training — Dog Detector")
    print("=" * 60)
    print(f"  Base model : {YOLO_BASE_MODEL}")
    print(f"  Dataset    : {data_yaml}")
    print(f"  Epochs     : {YOLO_EPOCHS}")
    print(f"  Batch size : {YOLO_BATCH}")
    print(f"  Image size : {YOLO_IMGSZ}")
    print(f"  Device     : {DEVICE_NAME} ({'CPU' if is_cpu else 'GPU'})")
    if is_cpu:
        print()
        print("  NOTE: Training on CPU — this will be slow (~5-15 hours)")
        print("  Reduce epochs with: YOLO_EPOCHS in config.py")
    print("=" * 60)

    model = YOLO(YOLO_BASE_MODEL)

    # Adjust settings for CPU vs GPU
    workers = 2 if is_cpu else 4

    results = model.train(
        data=str(data_yaml),
        epochs=YOLO_EPOCHS,
        batch=YOLO_BATCH,
        imgsz=YOLO_IMGSZ,
        device=YOLO_DEVICE,
        project=str(EXPORT_DIR.parent / "runs" / "yolo"),
        name="dog_detector",
        exist_ok=True,
        patience=20,
        save=True,
        save_period=10,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        close_mosaic=10,
        amp=not is_cpu,  # Mixed precision only on GPU
        workers=workers,
        seed=42,
        verbose=True,
    )

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    best_path = Path(results.save_dir) / "weights" / "best.pt"
    if best_path.exists():
        import shutil
        dest = EXPORT_DIR / "dog_detector.pt"
        shutil.copy2(best_path, dest)
        print(f"\n[TRAIN] Best model saved to: {dest}")
    else:
        print("\n[TRAIN] WARNING: best.pt not found in training output")

    print("\n[TRAIN] Training complete!")
    print(f"  Results: {results.save_dir}")

    return results


if __name__ == "__main__":
    train_yolo()
