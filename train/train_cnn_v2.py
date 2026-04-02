"""
train/train_cnn_v2.py — Train BehaviorNetV2 (residual CNN with SE attention).
Auto-detects number of classes from your data folders.
Works with 2 classes (IDLE, DANGER) or 4 classes (IDLE, ALERT, DANGER, DOG_FIGHT).
"""

import sys
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
    CNN_BATCH_SIZE,
    CNN_EPOCHS,
    CNN_LR,
    CNN_PATIENCE,
    CROPS_DIR,
    EXPORT_DIR,
)
from models.behavior_net_v2 import BehaviorNetV2


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((CNN_INPUT_SIZE, CNN_INPUT_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.1),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((CNN_INPUT_SIZE, CNN_INPUT_SIZE)),
        transforms.ToTensor(),
    ])

    return train_transform, val_transform


def train_cnn_v2():
    """Train BehaviorNetV2 with auto-detected classes."""
    train_dir = CROPS_DIR / "train"
    val_dir = CROPS_DIR / "val"

    if not train_dir.exists():
        print(f"ERROR: Training crops not found at {train_dir}")
        print("Run 'python data/load_custom_dataset.py --idle ... --danger ...' first")
        sys.exit(1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    is_cpu = not torch.cuda.is_available()
    num_workers = 2 if is_cpu else 4

    # Load data and auto-detect classes
    train_transform, val_transform = get_transforms()
    train_dataset = datasets.ImageFolder(str(train_dir), transform=train_transform)
    val_dataset = datasets.ImageFolder(str(val_dir), transform=val_transform)

    num_classes = len(train_dataset.classes)
    class_names = train_dataset.classes

    print("=" * 60)
    print("  BehaviorNetV2 Training (Residual + SE Attention)")
    print("=" * 60)
    print(f"  Device      : {device} {'(CPU — will be slow)' if is_cpu else ''}")
    print(f"  Input size  : {CNN_INPUT_SIZE}x{CNN_INPUT_SIZE}")
    print(f"  Classes     : {num_classes} → {class_names}")
    print(f"  Train       : {len(train_dataset)} images")
    print(f"  Validation  : {len(val_dataset)} images")
    print(f"  Batch size  : {CNN_BATCH_SIZE}")
    print(f"  Epochs      : {CNN_EPOCHS}")
    print(f"  LR          : {CNN_LR}")
    print("=" * 60)

    # Check class distribution
    print("\n  Class distribution:")
    for cls_idx, cls_name in enumerate(class_names):
        count = sum(1 for _, label in train_dataset.samples if label == cls_idx)
        print(f"    {cls_name}: {count} images")
    print()

    train_loader = DataLoader(
        train_dataset, batch_size=CNN_BATCH_SIZE, shuffle=True,
        num_workers=num_workers, pin_memory=not is_cpu,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CNN_BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=not is_cpu,
    )

    # Model
    model = BehaviorNetV2(num_classes=num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters : {total_params:,}")

    # Loss with class weighting (handles imbalanced IDLE/DANGER)
    class_counts = [0] * num_classes
    for _, label in train_dataset.samples:
        class_counts[label] += 1
    max_count = max(class_counts)
    class_weights = torch.FloatTensor([max_count / c for c in class_counts]).to(device)
    print(f"  Class weights: {[f'{w:.2f}' for w in class_weights.tolist()]}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=CNN_LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CNN_EPOCHS)

    # Training loop
    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0
    best_epoch = 0

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = EXPORT_DIR / "behavior_net_v2.pt"

    print("\n  Training started...\n")

    for epoch in range(1, CNN_EPOCHS + 1):
        start = time.time()

        # Train
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

        # Validate
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

        scheduler.step()
        elapsed = time.time() - start
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"  Epoch {epoch:3d}/{CNN_EPOCHS} | "
            f"Train: {train_loss:.4f} / {train_acc:.4f} | "
            f"Val: {val_loss:.4f} / {val_acc:.4f} | "
            f"LR: {lr:.6f} | {elapsed:.1f}s"
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), str(save_path))
            print(f"  >>> Best model saved (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= CNN_PATIENCE:
                print(f"\n  Early stopping at epoch {epoch}")
                break

    print("\n" + "=" * 60)
    print("  BehaviorNetV2 Training Complete!")
    print("=" * 60)
    print(f"  Best epoch    : {best_epoch}")
    print(f"  Best val acc  : {best_val_acc:.4f} ({best_val_acc:.2%})")
    print(f"  Classes       : {class_names}")
    print(f"  Model saved   : {save_path}")
    print("=" * 60)


if __name__ == "__main__":
    train_cnn_v2()
