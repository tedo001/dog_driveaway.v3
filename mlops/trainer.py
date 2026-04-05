"""
mlops/trainer.py — Training for YOLO detector + CNN behavior classifier.
"""

import sys
import shutil
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def get_device_info():
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return {"type": "cuda", "name": name, "memory_gb": round(mem, 1),
                "display": f"{name} ({mem:.1f} GB VRAM)"}
    return {"type": "cpu", "name": "CPU", "memory_gb": 0,
            "display": "CPU (no GPU — training will be slow)"}


def clear_old_models(models_to_clear=None):
    from config import EXPORT_DIR, YOLO_MODEL_PATH, CNN_MODEL_PATH

    model_paths = {
        "yolo": YOLO_MODEL_PATH,
        "cnn": CNN_MODEL_PATH,
    }

    if models_to_clear is None:
        models_to_clear = list(model_paths.keys())

    cleared = []
    for name in models_to_clear:
        if name in model_paths and model_paths[name].exists():
            model_paths[name].unlink()
            cleared.append(str(model_paths[name]))

    runs_dir = EXPORT_DIR.parent / "runs"
    if runs_dir.exists() and "yolo" in models_to_clear:
        shutil.rmtree(runs_dir, ignore_errors=True)
        cleared.append(str(runs_dir))

    return cleared


def train_yolo(epochs=None, batch=None, imgsz=None, progress_callback=None):
    """Train YOLO detector. progress_callback(epoch, total, loss) for GUI updates."""
    from config import (
        YOLO_BASE_MODEL, YOLO_EPOCHS, YOLO_BATCH, YOLO_IMGSZ,
        YOLO_DEVICE, DATASET_DIR, EXPORT_DIR,
    )
    from ultralytics import YOLO

    epochs = epochs or YOLO_EPOCHS
    batch = batch or YOLO_BATCH
    imgsz = imgsz or YOLO_IMGSZ

    data_yaml = DATASET_DIR / "coco128.yaml"
    if not data_yaml.exists():
        return {"success": False, "error": f"coco128.yaml not found at {data_yaml}"}

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
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        close_mosaic=10,
        amp=YOLO_DEVICE != "cpu",
        workers=2 if YOLO_DEVICE == "cpu" else 4,
        seed=42,
        verbose=True,
    )
    elapsed = time.time() - start

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    best_path = Path(results.save_dir) / "weights" / "best.pt"
    dest = EXPORT_DIR / "dog_detector.pt"
    if best_path.exists():
        shutil.copy2(best_path, dest)

    return {
        "success": True,
        "epochs": epochs,
        "time_seconds": round(elapsed),
        "model_path": str(dest),
    }


def train_cnn(epochs=None, batch=None, lr=None, progress_callback=None):
    """Train BehaviorNetV2. progress_callback(epoch, total, train_acc, val_acc) for GUI."""
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
    val_dir   = CROPS_DIR / "val"

    # ── DEBUG: verify paths and per-class image counts before loading ────────
    _IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    print("\n" + "="*60)
    print("[CNN DEBUG] Data path verification")
    print("="*60)
    print(f"[CNN DEBUG] CROPS_DIR = {CROPS_DIR}")
    print(f"[CNN DEBUG] train_dir = {train_dir}  (exists={train_dir.exists()})")
    print(f"[CNN DEBUG] val_dir   = {val_dir}  (exists={val_dir.exists()})")

    for split_name, split_dir in [("TRAIN", train_dir), ("VAL", val_dir)]:
        if not split_dir.exists():
            print(f"[CNN DEBUG] ❌ {split_name} dir MISSING: {split_dir}")
            continue
        cls_dirs = sorted(d for d in split_dir.iterdir() if d.is_dir())
        print(f"[CNN DEBUG] {split_name} class folders found: {[d.name for d in cls_dirs]}")
        for cls_dir in cls_dirs:
            n = sum(1 for f in cls_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in _IMG_EXTS)
            print(f"[CNN DEBUG]   {split_name}/{cls_dir.name}: {n} images")

    print("="*60 + "\n")
    # ── END DEBUG ─────────────────────────────────────────────────────────────

    if not train_dir.exists():
        return {"success": False, "error": f"Training crops not found at {train_dir}"}
    if not val_dir.exists():
        return {"success": False, "error": f"Validation crops not found at {val_dir}"}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([
        transforms.Resize((CNN_INPUT_SIZE, CNN_INPUT_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.1),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((CNN_INPUT_SIZE, CNN_INPUT_SIZE)),
        transforms.ToTensor(),
    ])

    # ── FIX: case-insensitive is_valid_file ───────────────────────────────────
    # Older torchvision versions only match lowercase extensions (.jpg not .JPG).
    # This custom checker fixes "Found no valid file for class IDLE/DANGER"
    # when crops were saved with uppercase extensions (.JPG, .PNG, etc.).
    _VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    def _is_valid_file(path: str) -> bool:
        return Path(path).suffix.lower() in _VALID_EXTS

    train_dataset = datasets.ImageFolder(
        str(train_dir),
        transform=train_transform,
        is_valid_file=_is_valid_file,
    )
    val_dataset = datasets.ImageFolder(
        str(val_dir),
        transform=val_transform,
        is_valid_file=_is_valid_file,
    )
    # ── END FIX ───────────────────────────────────────────────────────────────

    # ── DEBUG: confirm what ImageFolder actually loaded ───────────────────────
    print("\n" + "="*60)
    print("[CNN DEBUG] ImageFolder load results")
    print("="*60)
    print(f"[CNN DEBUG] Loaded classes  : {train_dataset.classes}")
    print(f"[CNN DEBUG] class_to_idx    : {train_dataset.class_to_idx}")
    print(f"[CNN DEBUG] Train total     : {len(train_dataset)} samples")
    print(f"[CNN DEBUG] Val   total     : {len(val_dataset)} samples")
    for cls_name, cls_idx in train_dataset.class_to_idx.items():
        n_train = sum(1 for _, lbl in train_dataset.samples if lbl == cls_idx)
        n_val   = sum(1 for _, lbl in val_dataset.samples   if lbl == cls_idx)
        print(f"[CNN DEBUG]   {cls_name:10s} → train={n_train:5d}, val={n_val:5d}")
    print("="*60 + "\n")
    # ── END DEBUG ─────────────────────────────────────────────────────────────

    # Guard: both expected classes must be present
    if set(train_dataset.classes) != {"DANGER", "IDLE"}:
        return {
            "success": False,
            "error": (
                f"Expected classes ['DANGER', 'IDLE'], "
                f"but ImageFolder found: {train_dataset.classes}. "
                f"Check folder names inside {train_dir}"
            ),
        }

    num_classes = len(train_dataset.classes)
    class_names = train_dataset.classes

    num_workers = 2 if not torch.cuda.is_available() else 4
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True,
                              num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False,
                            num_workers=num_workers, pin_memory=torch.cuda.is_available())

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

        train_acc = train_correct / train_total

        model.eval()
        val_correct = 0
        val_total = 0
        val_loss_sum = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss_sum += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total
        scheduler.step()

        if progress_callback:
            progress_callback(epoch, epochs, train_acc, val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), str(save_path))
        else:
            patience_counter += 1
            if patience_counter >= CNN_PATIENCE:
                break

    elapsed = time.time() - start
    return {
        "success": True,
        "epochs": best_epoch,
        "best_val_acc": round(best_val_acc, 4),
        "time_seconds": round(elapsed),
        "model_path": str(save_path),
        "classes": class_names,
    }


def train_all(epochs=None, progress_callback=None):
    """Train YOLO → CNN."""
    results = {}
    results["yolo"] = train_yolo(epochs=epochs, progress_callback=progress_callback)
    if results["yolo"]["success"]:
        results["cnn"] = train_cnn(epochs=epochs, progress_callback=progress_callback)
    return results