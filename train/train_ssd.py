"""
train/train_ssd.py — Train SSD300-VGG16 dog+person detector.
Fine-tunes pretrained COCO SSD on your custom dog dataset.
Optimized for RTX 4060 Laptop GPU (8GB VRAM).

What SSD Does (step by step):
  1. Input image resized to 300x300
  2. VGG16 backbone extracts feature maps at 6 different scales
  3. At each scale, SSD predicts:
     - Class probabilities for each anchor box
     - Bounding box offsets (dx, dy, dw, dh) for each anchor box
  4. Total: 8732 anchor boxes across all scales
  5. Non-Maximum Suppression removes overlapping detections
  6. Output: list of (class, confidence, x1, y1, x2, y2) per detection

Loss function:
  - Classification: Cross-Entropy on hard-negatives (3:1 negative:positive ratio)
  - Regression: Smooth L1 on positive anchors only
  - Total = classification_loss + regression_loss
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from config import (
    SSD_EPOCHS,
    SSD_BATCH_SIZE,
    SSD_LR,
    SSD_PATIENCE,
    SSD_NUM_CLASSES,
    SSD_IMGSZ,
    SSD_MODEL_PATH,
    SSD_CLASS_NAMES,
    DATASET_DIR,
    EXPORT_DIR,
)
from models.ssd_model import SSDDogDetector
from data.ssd_dataset import SSDDogDataset, ssd_collate_fn


def train_ssd():
    """Train SSD300 on dog detection dataset."""

    # ── Check dataset ────────────────────────────────────────────
    train_images = DATASET_DIR / "train" / "images"
    train_labels = DATASET_DIR / "train" / "labels"
    val_images = DATASET_DIR / "valid" / "images"
    val_labels = DATASET_DIR / "valid" / "labels"

    if not train_images.exists():
        print(f"ERROR: Training images not found at {train_images}")
        print("Run 'python data/download_dataset.py' first")
        sys.exit(1)

    # ── Setup device ─────────────────────────────────────────────
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  SSD300 Training — Dog + Person Detector")
    print("=" * 60)
    print(f"  Device      : {device}")
    print(f"  Input size  : {SSD_IMGSZ}x{SSD_IMGSZ}")
    print(f"  Classes     : {SSD_NUM_CLASSES} ({SSD_CLASS_NAMES})")
    print(f"  Batch size  : {SSD_BATCH_SIZE}")
    print(f"  Epochs      : {SSD_EPOCHS}")
    print(f"  LR          : {SSD_LR}")
    print(f"  Patience    : {SSD_PATIENCE}")
    print("=" * 60)

    # ── Data loading ─────────────────────────────────────────────
    train_dataset = SSDDogDataset(train_images, train_labels, augment=True)
    val_dataset = SSDDogDataset(val_images, val_labels, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=SSD_BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=ssd_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=SSD_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=ssd_collate_fn,
    )

    print(f"  Train samples : {len(train_dataset)}")
    print(f"  Val samples   : {len(val_dataset)}")

    # ── Model ────────────────────────────────────────────────────
    model = SSDDogDetector(num_classes=SSD_NUM_CLASSES, pretrained_backbone=True)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params     : {total_params:,}")
    print(f"  Trainable params : {trainable_params:,}")

    # ── Optimizer + Scheduler ────────────────────────────────────
    optimizer = optim.SGD(
        model.parameters(),
        lr=SSD_LR,
        momentum=0.9,
        weight_decay=5e-4,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True,
    )

    # ── Training loop ────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n  Training started...\n")

    for epoch in range(1, SSD_EPOCHS + 1):
        start_time = time.time()

        # ── Train phase ──
        model.train()
        train_loss = 0.0
        train_batches = 0

        for images, targets in train_loader:
            # Move to device
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Skip batches with no positive targets
            has_boxes = any(t["boxes"].shape[0] > 0 for t in targets)
            if not has_boxes:
                continue

            # Forward pass — SSD returns loss dict during training
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            train_loss += losses.item()
            train_batches += 1

        avg_train_loss = train_loss / max(train_batches, 1)

        # ── Validation phase ──
        model.train()  # SSD needs train mode to compute losses
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                has_boxes = any(t["boxes"].shape[0] > 0 for t in targets)
                if not has_boxes:
                    continue

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)

        # Scheduler step
        scheduler.step(avg_val_loss)

        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch:3d}/{SSD_EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"LR: {current_lr:.6f} | {elapsed:.1f}s"
        )

        # ── Early stopping ──
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), str(SSD_MODEL_PATH))
            print(f"  >>> Best model saved (val_loss={avg_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= SSD_PATIENCE:
                print(f"\n  Early stopping at epoch {epoch} (patience={SSD_PATIENCE})")
                break

    print("\n" + "=" * 60)
    print("  SSD Training Complete!")
    print("=" * 60)
    print(f"  Best epoch    : {best_epoch}")
    print(f"  Best val loss : {best_val_loss:.4f}")
    print(f"  Model saved   : {SSD_MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    train_ssd()
