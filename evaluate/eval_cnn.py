"""
evaluate/eval_cnn.py — Evaluate BehaviorNet CNN on validation set.
Reports accuracy, per-class precision/recall/F1, and confusion matrix.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix

from config import (
    CNN_INPUT_SIZE,
    CNN_NUM_CLASSES,
    CNN_BATCH_SIZE,
    CNN_MODEL_PATH,
    CROPS_DIR,
    THREAT_CLASSES,
)
from models.cnn_model import BehaviorNet


def eval_cnn():
    """Evaluate BehaviorNet on validation crops."""
    val_dir = CROPS_DIR / "val"
    if not val_dir.exists():
        print(f"ERROR: Validation crops not found at {val_dir}")
        print("Run 'python data/prepare_crops.py' first")
        sys.exit(1)

    if not CNN_MODEL_PATH.exists():
        print(f"ERROR: Trained model not found at {CNN_MODEL_PATH}")
        print("Run 'python train/train_cnn.py' first")
        sys.exit(1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  BehaviorNet CNN Evaluation")
    print("=" * 60)
    print(f"  Model   : {CNN_MODEL_PATH}")
    print(f"  Val dir : {val_dir}")
    print(f"  Device  : {device}")
    print("=" * 60)

    # Load model
    model = BehaviorNet(num_classes=CNN_NUM_CLASSES)
    model.load_state_dict(torch.load(str(CNN_MODEL_PATH), map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # Load validation data
    val_transform = transforms.Compose([
        transforms.Resize((CNN_INPUT_SIZE, CNN_INPUT_SIZE)),
        transforms.ToTensor(),
    ])
    val_dataset = datasets.ImageFolder(str(val_dir), transform=val_transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=CNN_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"  Samples : {len(val_dataset)}")
    print(f"  Classes : {val_dataset.classes}")

    # Run evaluation
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Overall accuracy
    accuracy = (all_preds == all_labels).mean()
    print(f"\n  Overall Accuracy: {accuracy:.4f} ({accuracy:.2%})")

    # Classification report
    class_names = [THREAT_CLASSES[i] for i in range(CNN_NUM_CLASSES)]
    # Use dataset class names if available (folder order may differ)
    if val_dataset.classes:
        class_names = val_dataset.classes

    print("\n  Classification Report:")
    print("-" * 60)
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print(report)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("  Confusion Matrix:")
    print("-" * 60)

    # Header
    header = "          " + "  ".join(f"{name:>10}" for name in class_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{val:>10}" for val in row)
        print(f"  {class_names[i]:>8} {row_str}")

    print("=" * 60)

    return accuracy, report, cm


if __name__ == "__main__":
    eval_cnn()
