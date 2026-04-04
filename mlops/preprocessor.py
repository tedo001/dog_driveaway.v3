"""
mlops/preprocessor.py — Data preprocessing for CNN behavior training.

Handles:
  1. Auto-detect class mapping from any dataset format
  2. Roboflow emotion → behavior mapping (angry→DANGER, happy/relaxed/sad→IDLE)
  3. Custom class mapping (user-defined)
  4. Bounding box crop extraction from YOLO labels
  5. Train/val split with stratification
  6. Image resizing to CNN_INPUT_SIZE

The user just points to a folder — this handles everything else.
"""

import sys
import os
import random
import cv2
from pathlib import Path
from collections import Counter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CROPS_DIR, CNN_INPUT_SIZE


# ── Predefined Class Mappings ───────────────────────────────────────────────

MAPPING_PRESETS = {
    "roboflow_emotion": {
        "description": "Roboflow dog emotion (angry/happy/relaxed/sad) → DANGER/IDLE",
        "mapping": {
            "angry": "DANGER",
            "happy": "IDLE",
            "relaxed": "IDLE",
            "sad": "IDLE",
        },
    },
    "roboflow_emotion_3class": {
        "description": "Roboflow dog emotion → DANGER/ALERT/IDLE",
        "mapping": {
            "angry": "DANGER",
            "sad": "ALERT",
            "happy": "IDLE",
            "relaxed": "IDLE",
        },
    },
    "binary_aggression": {
        "description": "Binary: aggressive vs non-aggressive",
        "mapping": {
            "aggressive": "DANGER",
            "angry": "DANGER",
            "attack": "DANGER",
            "danger": "DANGER",
            "non-aggressive": "IDLE",
            "calm": "IDLE",
            "friendly": "IDLE",
            "idle": "IDLE",
            "happy": "IDLE",
            "relaxed": "IDLE",
            "sad": "IDLE",
        },
    },
}


def auto_select_mapping(class_names):
    """
    Auto-select the best mapping preset based on detected class names.

    Args:
        class_names: list of class name strings from dataset

    Returns:
        (preset_name, mapping_dict) or (None, None) if no match
    """
    class_set = set(c.lower() for c in class_names)

    # Check Roboflow emotion format
    if class_set == {"angry", "happy", "relaxed", "sad"}:
        return "roboflow_emotion", MAPPING_PRESETS["roboflow_emotion"]["mapping"]

    # Check if classes match any preset
    for name, preset in MAPPING_PRESETS.items():
        preset_classes = set(preset["mapping"].keys())
        if class_set.issubset(preset_classes):
            filtered = {k: v for k, v in preset["mapping"].items() if k in class_set}
            return name, filtered

    return None, None


def create_custom_mapping(class_names, behavior_classes=None):
    """
    Interactive: let user map each class to a behavior.

    Args:
        class_names: list of source class names
        behavior_classes: target classes (default: DANGER, IDLE)

    Returns:
        dict mapping source → target
    """
    if behavior_classes is None:
        behavior_classes = ["DANGER", "IDLE"]

    print("\n  Custom Class Mapping")
    print("  " + "-" * 40)
    print(f"  Target behaviors: {behavior_classes}")
    print()

    mapping = {}
    for cls in class_names:
        print(f"  Map '{cls}' to:")
        for i, beh in enumerate(behavior_classes):
            print(f"    [{i + 1}] {beh}")

        while True:
            try:
                choice = input(f"  Choice for '{cls}' (1-{len(behavior_classes)}): ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(behavior_classes):
                    mapping[cls] = behavior_classes[idx]
                    print(f"    → {cls} → {behavior_classes[idx]}")
                    break
            except (ValueError, IndexError):
                pass
            print(f"    Invalid. Enter 1-{len(behavior_classes)}")

    return mapping


# ── Crop Extraction ─────────────────────────────────────────────────────────

def extract_behavior_crops(
    img_dir,
    lbl_dir,
    class_names,
    class_mapping,
    train_split=0.8,
    clear_old=False,
    all_splits=None,
):
    """
    Extract CNN behavior crops from YOLO-labeled images.

    Args:
        img_dir: Path to images directory
        lbl_dir: Path to labels directory (YOLO format .txt)
        class_names: list of class names (index = class ID in labels)
        class_mapping: dict mapping class_name → behavior (DANGER/IDLE)
        train_split: train/val split ratio
        clear_old: clear existing crops first

    Returns:
        dict with crop counts per class and split
    """
    img_dir = Path(img_dir)
    lbl_dir = Path(lbl_dir)

    if clear_old and CROPS_DIR.exists():
        print(f"  Clearing old crops at {CROPS_DIR}...")
        import shutil
        shutil.rmtree(CROPS_DIR)

    # Build class_id → behavior mapping
    id_to_behavior = {}
    for idx, name in enumerate(class_names):
        name_lower = name.lower()
        if name_lower in class_mapping:
            id_to_behavior[idx] = class_mapping[name_lower]
        elif name in class_mapping:
            id_to_behavior[idx] = class_mapping[name]

    # Get unique behavior classes
    behaviors = sorted(set(id_to_behavior.values()))

    # Create output directories
    for split in ["train", "val"]:
        for beh in behaviors:
            (CROPS_DIR / split / beh).mkdir(parents=True, exist_ok=True)

    # Collect images from all splits or single directory
    image_label_pairs = []
    if all_splits:
        # Use pre-existing splits from dataset (e.g. Roboflow train/valid)
        for split_info in all_splits:
            s_img_dir = Path(split_info["img_dir"])
            s_lbl_dir = Path(split_info["lbl_dir"])
            s_name = split_info["split"]
            # Map valid/val to "val", train to "train"
            target_split = "val" if s_name in ("valid", "val", "test") else "train"
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                for img_path in s_img_dir.glob(ext):
                    lbl_path = s_lbl_dir / f"{img_path.stem}.txt"
                    if lbl_path.exists():
                        image_label_pairs.append((img_path, lbl_path, target_split))
    else:
        # Single directory — do random train/val split
        images = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            images.extend(list(img_dir.glob(ext)))
        random.seed(42)
        random.shuffle(images)
        split_idx = int(len(images) * train_split)
        for idx, img_path in enumerate(images):
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            target_split = "train" if idx < split_idx else "val"
            image_label_pairs.append((img_path, lbl_path, target_split))

    if not image_label_pairs:
        return {"success": False, "error": f"No images found in {img_dir}"}

    print(f"\n  Extracting behavior crops from {len(image_label_pairs)} images...")
    print(f"  Class mapping: {class_mapping}")
    print(f"  Output: {CROPS_DIR}")
    print()

    stats = {split: {beh: 0 for beh in behaviors} for split in ["train", "val"]}
    source_stats = Counter()
    skipped = 0

    for idx, (img_path, lbl_path, split_name) in enumerate(tqdm(image_label_pairs, desc="  Cropping")):
        if not lbl_path.exists():
            skipped += 1
            continue

        # Read image
        frame = cv2.imread(str(img_path))
        if frame is None:
            skipped += 1
            continue

        h, w = frame.shape[:2]

        # Parse YOLO labels
        lines = lbl_path.read_text().strip().split("\n")
        for ci, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            if class_id not in id_to_behavior:
                continue

            behavior = id_to_behavior[class_id]
            source_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"

            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

            # Convert normalized → pixel
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Skip tiny boxes
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue

            # Crop and resize
            crop = frame[y1:y2, x1:x2]
            crop = cv2.resize(crop, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))

            # Save
            crop_name = f"{source_name}_{idx:06d}_c{ci}.jpg"
            cv2.imwrite(str(CROPS_DIR / split_name / behavior / crop_name), crop)

            stats[split_name][behavior] += 1
            source_stats[f"{source_name} → {behavior}"] += 1

    # Summary
    total = sum(stats[s][b] for s in stats for b in stats[s])

    print(f"\n  Crop Extraction Complete!")
    print(f"  " + "-" * 40)
    print(f"  Total crops: {total}")
    print(f"  Skipped: {skipped}")
    print()
    print(f"  Per-split breakdown:")
    for split in ["train", "val"]:
        for beh in behaviors:
            print(f"    {split}/{beh}: {stats[split][beh]}")

    print(f"\n  Source class distribution:")
    for key, count in source_stats.most_common():
        print(f"    {key}: {count}")

    return {
        "success": True,
        "total": total,
        "stats": stats,
        "source_stats": dict(source_stats),
        "behaviors": behaviors,
    }
