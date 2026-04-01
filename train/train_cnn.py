"""
train/train_cnn.py — Train BehaviorNet CNN on dog behavior crops.
4 classes: IDLE, ALERT, DANGER, DOG_FIGHT.
Includes early stopping, learning rate scheduling, and CUDA optimization.
"""

import sys
import os
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from config import (
    CNN_INPUT_SIZE,
    CNN_NUM_CLASSES,
    CNN_BATCH_SIZE,
    CNN_EPOCHS,
    CNN_LR,
    CNN_PATIENCE,
    CNN_MODEL_PATH,
    CROPS_DIR,
    EXPORT_DIR,
    THREAT_CLASSES,
)
from models.cnn_model import BehaviorNet


def get_transforms():
    """Get training and validation transforms."""
    train_transform = transforms.Compose([
        transforms.Resize((CNN_INPUT_SIZE, CNN_INPUT_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((CNN_INPUT_SIZE, CNN_INPUT_SIZE)),
        transforms.ToTensor(),
    ])

    return train_transform, val_transform


def train_cnn():
    """Train BehaviorNet CNN."""
    # Check dataset
    train_dir = CROPS_DIR / "train"
    val_dir = CROPS_DIR / "val"

    if not train_dir.exists():
        print(f"ERROR: Training crops not found at {train_dir}")
        print("Run 'python data/prepare_crops.py' first")
        sys.exit(1)

    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("  BehaviorNet CNN Training")
    print("=" * 60)
    print(f"  Device     : {device}")
    print(f"  Input size : {CNN_INPUT_SIZE}x{CNN_INPUT_SIZE}")
    print(f"  Classes    : {CNN_NUM_CLASSES} ({', '.join(THREAT_CLASSES.values())})")
    print(f"  Batch size : {CNN_BATCH_SIZE}")
    print(f"  Epochs     : {CNN_EPOCHS}")
    print(f"  LR         : {CNN_LR}")
    print(f"  Patience   : {CNN_PATIENCE}")
    print("=" * 60)

    # Data loading
    train_transform, val_transform = get_transforms()

    train_dataset = datasets.ImageFolder(str(train_dir), transform=train_transform)
    val_dataset = datasets.ImageFolder(str(val_dir), transform=val_transform)

    print(f"\n  Training samples   : {len(train_dataset)}")
    print(f"  Validation samples : {len(val_dataset)}")
    print(f"  Classes found      : {train_dataset.classes}")

    is_cpu = not torch.cuda.is_available()
    num_workers = 2 if is_cpu else 4

    if is_cpu:
        print(f"\n  NOTE: Training on CPU — ~1-3 hours")

    train_loader = DataLoader(
        train_dataset,
        batch_size=CNN_BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=not is_cpu,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CNN_BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=not is_cpu,
    )

    # Model
    model = BehaviorNet(num_classes=CNN_NUM_CLASSES).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters : {total_params:,}")

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CNN_LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True,
    )

    # Early stopping
    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0
    best_epoch = 0

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n  Training started...\n")

    for epoch in range(1, CNN_EPOCHS + 1):
        start_time = time.time()

        # ── Training phase ──
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in tqdm(train_loader, desc=f"  Epoch {epoch}/{CNN_EPOCHS} [Train]", leave=False):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # ── Validation phase ──
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"  Epoch {epoch}/{CNN_EPOCHS} [Val]", leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        # Scheduler step
        scheduler.step(val_loss)

        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch:3d}/{CNN_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.6f} | {elapsed:.1f}s"
        )

        # ── Early stopping check ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0

            # Save best model
            torch.save(model.state_dict(), str(CNN_MODEL_PATH))
            print(f"  ✓ Best model saved (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= CNN_PATIENCE:
                print(f"\n  Early stopping at epoch {epoch} (patience={CNN_PATIENCE})")
                break

    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"  Best epoch     : {best_epoch}")
    print(f"  Best val loss  : {best_val_loss:.4f}")
    print(f"  Best val acc   : {best_val_acc:.4f}")
    print(f"  Model saved to : {CNN_MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    train_cnn()
