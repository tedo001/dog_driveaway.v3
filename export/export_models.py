"""
export/export_models.py — Export trained models to deployment formats.
Exports YOLO to ONNX and BehaviorNet to TorchScript.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from ultralytics import YOLO

from config import (
    YOLO_MODEL_PATH,
    YOLO_ONNX_PATH,
    CNN_MODEL_PATH,
    CNN_SCRIPTED_PATH,
    CNN_INPUT_SIZE,
    CNN_NUM_CLASSES,
    EXPORT_DIR,
    YOLO_IMGSZ,
)
from models.cnn_model import BehaviorNet


def export_yolo_onnx():
    """Export YOLO model to ONNX format."""
    if not YOLO_MODEL_PATH.exists():
        print(f"[EXPORT] YOLO model not found at {YOLO_MODEL_PATH}")
        return False

    print(f"[EXPORT] Exporting YOLO to ONNX...")
    model = YOLO(str(YOLO_MODEL_PATH))
    model.export(
        format="onnx",
        imgsz=YOLO_IMGSZ,
        simplify=True,
        opset=12,
    )

    # Move to export directory
    exported = YOLO_MODEL_PATH.with_suffix(".onnx")
    if exported.exists():
        import shutil
        shutil.move(str(exported), str(YOLO_ONNX_PATH))
        print(f"[EXPORT] YOLO ONNX saved: {YOLO_ONNX_PATH}")
        return True

    print("[EXPORT] YOLO ONNX export failed")
    return False


def export_cnn_torchscript():
    """Export BehaviorNet to TorchScript format."""
    if not CNN_MODEL_PATH.exists():
        print(f"[EXPORT] CNN model not found at {CNN_MODEL_PATH}")
        return False

    print(f"[EXPORT] Exporting BehaviorNet to TorchScript...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = BehaviorNet(num_classes=CNN_NUM_CLASSES)
    model.load_state_dict(torch.load(str(CNN_MODEL_PATH), map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # Trace model
    dummy_input = torch.randn(1, 3, CNN_INPUT_SIZE, CNN_INPUT_SIZE).to(device)
    scripted = torch.jit.trace(model, dummy_input)
    scripted.save(str(CNN_SCRIPTED_PATH))

    print(f"[EXPORT] BehaviorNet TorchScript saved: {CNN_SCRIPTED_PATH}")
    return True


def export_all():
    """Export all models."""
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Model Export")
    print("=" * 60)

    yolo_ok = export_yolo_onnx()
    cnn_ok = export_cnn_torchscript()

    print("\n" + "=" * 60)
    print("  Export Summary")
    print("=" * 60)
    print(f"  YOLO → ONNX        : {'OK' if yolo_ok else 'SKIPPED'}")
    print(f"  CNN → TorchScript   : {'OK' if cnn_ok else 'SKIPPED'}")
    print("=" * 60)


if __name__ == "__main__":
    export_all()
