"""
mlops/trainer.py — Unified training orchestration for all models.

Handles:
  - YOLO training (dog+person detection)
  - SSD training (dog+person detection)
  - CNN BehaviorNetV2 training (behavior classification)
  - Training progress tracking
  - Model cleanup (remove old weights)
  - GPU memory management

All training uses config.py settings but can be overridden per-run.
"""

import sys
import shutil
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def get_device_info():
    """Get current device information."""
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        return {
            "type": "cuda",
            "name": name,
            "memory_gb": round(mem, 1),
            "display": f"{name} ({mem:.1f} GB VRAM)",
        }
    return {
        "type": "cpu",
        "name": "CPU",
        "memory_gb": 0,
        "display": "CPU (no GPU detected — training will be slow)",
    }


def clear_old_models(models_to_clear=None):
    """
    Remove old trained model weights.

    Args:
        models_to_clear: list of model names ["yolo", "ssd", "cnn"] or None for all
    """
    from config import EXPORT_DIR, YOLO_MODEL_PATH, SSD_MODEL_PATH, CNN_MODEL_PATH

    model_paths = {
        "yolo": YOLO_MODEL_PATH,
        "ssd": SSD_MODEL_PATH,
        "cnn": CNN_MODEL_PATH,
        "cnn_v1": EXPORT_DIR / "behavior_net.pt",
        "cnn_v2": EXPORT_DIR / "behavior_net_v2.pt",
        "onnx": EXPORT_DIR / "dog_detector.onnx",
        "scripted": EXPORT_DIR / "behavior_net_scripted.pt",
    }

    if models_to_clear is None:
        models_to_clear = list(model_paths.keys())

    cleared = []
    for name in models_to_clear:
        if name in model_paths:
            path = model_paths[name]
            if path.exists():
                path.unlink()
                cleared.append(str(path))
                print(f"  Deleted: {path}")

    # Also clear YOLO training runs
    runs_dir = EXPORT_DIR.parent / "runs"
    if runs_dir.exists() and ("yolo" in models_to_clear):
        shutil.rmtree(runs_dir, ignore_errors=True)
        cleared.append(str(runs_dir))
        print(f"  Deleted: {runs_dir}")

    if cleared:
        print(f"\n  Cleared {len(cleared)} old model files")
    else:
        print("  No old model files found to clear")

    return cleared


def train_yolo(epochs=None, batch=None, imgsz=None):
    """
    Train YOLO detector on COCO dog+person data.

    Args:
        epochs: override config epochs
        batch: override config batch size
        imgsz: override config image size

    Returns:
        dict with training results
    """
    from config import (
        YOLO_BASE_MODEL, YOLO_EPOCHS, YOLO_BATCH, YOLO_IMGSZ,
        YOLO_DEVICE, DATASET_DIR, EXPORT_DIR,
    )
    from ultralytics import YOLO

    epochs = epochs or YOLO_EPOCHS
    batch = batch or YOLO_BATCH
    imgsz = imgsz or YOLO_IMGSZ

    data_yaml = DATASET_DIR / "data.yaml"
    if not data_yaml.exists():
        return {"success": False, "error": f"data.yaml not found at {data_yaml}"}

    is_cpu = YOLO_DEVICE == "cpu"

    print("=" * 60)
    print("  YOLO Training — Dog + Person Detector")
    print("=" * 60)
    print(f"  Device     : {get_device_info()['display']}")
    print(f"  Epochs     : {epochs}")
    print(f"  Batch size : {batch}")
    print(f"  Image size : {imgsz}")
    print(f"  Dataset    : {data_yaml}")
    print("=" * 60)

    model = YOLO(YOLO_BASE_MODEL)

    start = time.time()
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
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
        amp=not is_cpu,
        workers=2 if is_cpu else 4,
        seed=42,
        verbose=True,
    )
    elapsed = time.time() - start

    # Copy best model
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    best_path = Path(results.save_dir) / "weights" / "best.pt"
    dest = EXPORT_DIR / "dog_detector.pt"
    if best_path.exists():
        shutil.copy2(best_path, dest)
        print(f"\n  Best model saved: {dest}")

    return {
        "success": True,
        "epochs": epochs,
        "time_seconds": round(elapsed),
        "model_path": str(dest),
        "results_dir": str(results.save_dir),
    }


def train_ssd(epochs=None, batch=None, lr=None):
    """
    Train SSD300 detector on COCO dog+person data.

    Returns:
        dict with training results
    """
    from config import (
        SSD_EPOCHS, SSD_BATCH_SIZE, SSD_LR, SSD_PATIENCE,
        SSD_NUM_CLASSES, SSD_IMGSZ, SSD_MODEL_PATH,
        SSD_CLASS_NAMES, DATASET_DIR, EXPORT_DIR,
    )
    import torch
    import torch.optim as optim
    from models.ssd_model import SSDDogDetector
    from models.ssd_dataset import SSDDogDataset, ssd_collate_fn
    from torch.utils.data import DataLoader

    epochs = epochs or SSD_EPOCHS
    batch = batch or SSD_BATCH_SIZE
    lr = lr or SSD_LR

    train_images = DATASET_DIR / "train" / "images"
    train_labels = DATASET_DIR / "train" / "labels"
    val_images = DATASET_DIR / "valid" / "images"
    val_labels = DATASET_DIR / "valid" / "labels"

    if not train_images.exists():
        return {"success": False, "error": f"Training images not found at {train_images}"}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    is_cpu = not torch.cuda.is_available()

    print("=" * 60)
    print("  SSD300 Training — Dog + Person Detector")
    print("=" * 60)
    print(f"  Device     : {get_device_info()['display']}")
    print(f"  Epochs     : {epochs}")
    print(f"  Batch size : {batch}")
    print(f"  LR         : {lr}")
    print("=" * 60)

    # Data
    train_dataset = SSDDogDataset(train_images, train_labels, augment=True)
    val_dataset = SSDDogDataset(val_images, val_labels, augment=False)

    num_workers = 2 if is_cpu else 4
    train_loader = DataLoader(
        train_dataset, batch_size=batch, shuffle=True,
        num_workers=num_workers, pin_memory=not is_cpu,
        collate_fn=ssd_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch, shuffle=False,
        num_workers=num_workers, pin_memory=not is_cpu,
        collate_fn=ssd_collate_fn,
    )

    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # Model
    model = SSDDogDetector(num_classes=SSD_NUM_CLASSES, pretrained_backbone=True)
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    start = time.time()

    for epoch in range(1, epochs + 1):
        ep_start = time.time()

        # Train
        model.train()
        train_loss = 0.0
        train_batches = 0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            if not any(t["boxes"].shape[0] > 0 for t in targets):
                continue
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            train_loss += losses.item()
            train_batches += 1

        avg_train = train_loss / max(train_batches, 1)

        # Validate
        model.train()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                if not any(t["boxes"].shape[0] > 0 for t in targets):
                    continue
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
                val_batches += 1

        avg_val = val_loss / max(val_batches, 1)
        scheduler.step(avg_val)

        ep_time = time.time() - ep_start
        print(f"  Epoch {epoch:3d}/{epochs} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | {ep_time:.1f}s")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), str(SSD_MODEL_PATH))
            print(f"  >>> Best model saved (val_loss={avg_val:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= SSD_PATIENCE:
                print(f"\n  Early stopping at epoch {epoch}")
                break

    elapsed = time.time() - start
    return {
        "success": True,
        "epochs": best_epoch,
        "best_val_loss": round(best_val_loss, 4),
        "time_seconds": round(elapsed),
        "model_path": str(SSD_MODEL_PATH),
    }


def train_cnn(epochs=None, batch=None, lr=None):
    """
    Train BehaviorNetV2 on behavior crops (DANGER/IDLE).

    Returns:
        dict with training results
    """
    from config import (
        CNN_INPUT_SIZE, CNN_BATCH_SIZE, CNN_EPOCHS,
        CNN_LR, CNN_PATIENCE, CROPS_DIR, EXPORT_DIR,
    )
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from models.behavior_net_v2 import BehaviorNetV2

    epochs = epochs or CNN_EPOCHS
    batch = batch or CNN_BATCH_SIZE
    lr = lr or CNN_LR

    train_dir = CROPS_DIR / "train"
    val_dir = CROPS_DIR / "val"

    if not train_dir.exists():
        return {"success": False, "error": f"Training crops not found at {train_dir}"}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    is_cpu = not torch.cuda.is_available()

    # Transforms
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

    train_dataset = datasets.ImageFolder(str(train_dir), transform=train_transform)
    val_dataset = datasets.ImageFolder(str(val_dir), transform=val_transform)

    num_classes = len(train_dataset.classes)
    class_names = train_dataset.classes

    print("=" * 60)
    print("  BehaviorNetV2 Training — Dog Behavior Classifier")
    print("=" * 60)
    print(f"  Device     : {get_device_info()['display']}")
    print(f"  Classes    : {num_classes} → {class_names}")
    print(f"  Train      : {len(train_dataset)} | Val: {len(val_dataset)}")
    print(f"  Epochs     : {epochs}")
    print(f"  Batch size : {batch}")
    print("=" * 60)

    # Class distribution
    for cls_idx, cls_name in enumerate(class_names):
        count = sum(1 for _, label in train_dataset.samples if label == cls_idx)
        print(f"    {cls_name}: {count} images")

    num_workers = 2 if is_cpu else 4
    train_loader = DataLoader(
        train_dataset, batch_size=batch, shuffle=True,
        num_workers=num_workers, pin_memory=not is_cpu,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch, shuffle=False,
        num_workers=num_workers, pin_memory=not is_cpu,
    )

    # Model + weighted loss
    model = BehaviorNetV2(num_classes=num_classes).to(device)

    class_counts = [0] * num_classes
    for _, label in train_dataset.samples:
        class_counts[label] += 1
    max_count = max(class_counts)
    class_weights = torch.FloatTensor([max_count / c for c in class_counts]).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0
    best_epoch = 0

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = EXPORT_DIR / "behavior_net_v2.pt"

    start = time.time()

    for epoch in range(1, epochs + 1):
        ep_start = time.time()

        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for images, labels in train_loader:
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
            for images, labels in val_loader:
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

        ep_time = time.time() - ep_start
        print(
            f"  Epoch {epoch:3d}/{epochs} | "
            f"Train: {train_loss:.4f}/{train_acc:.4f} | "
            f"Val: {val_loss:.4f}/{val_acc:.4f} | {ep_time:.1f}s"
        )

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

    elapsed = time.time() - start
    return {
        "success": True,
        "epochs": best_epoch,
        "best_val_acc": round(best_val_acc, 4),
        "best_val_loss": round(best_val_loss, 4),
        "time_seconds": round(elapsed),
        "model_path": str(save_path),
        "classes": class_names,
    }


def train_all(epochs=None):
    """
    Train all models in sequence: YOLO → SSD → CNN.

    Args:
        epochs: override epochs for all models (or None to use config defaults)
    """
    print("\n" + "=" * 60)
    print("  FULL TRAINING PIPELINE")
    print("  YOLO → SSD → CNN (BehaviorNetV2)")
    print("=" * 60)

    results = {}

    # Step 1: YOLO
    print("\n  [1/3] Training YOLO...")
    results["yolo"] = train_yolo(epochs=epochs)
    if not results["yolo"]["success"]:
        print(f"  YOLO FAILED: {results['yolo'].get('error')}")
        return results

    # Step 2: SSD
    print("\n  [2/3] Training SSD...")
    results["ssd"] = train_ssd(epochs=epochs)
    if not results["ssd"]["success"]:
        print(f"  SSD FAILED: {results['ssd'].get('error')}")
        return results

    # Step 3: CNN
    print("\n  [3/3] Training CNN BehaviorNetV2...")
    results["cnn"] = train_cnn(epochs=epochs)
    if not results["cnn"]["success"]:
        print(f"  CNN FAILED: {results['cnn'].get('error')}")
        return results

    print("\n" + "=" * 60)
    print("  ALL TRAINING COMPLETE!")
    print("=" * 60)
    for name, res in results.items():
        t = res.get("time_seconds", 0)
        minutes = t // 60
        print(f"    {name.upper():5s}: {minutes}m — {res.get('model_path', 'N/A')}")
    print("=" * 60)

    return results
