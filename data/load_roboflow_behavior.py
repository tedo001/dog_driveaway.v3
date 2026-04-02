"""
data/load_roboflow_behavior.py — Load Roboflow dog emotion dataset (ZIP) for CNN training.

Maps Roboflow 4-class emotion labels to our 2-class behavior system:
  angry  (class 0) → DANGER
  happy  (class 1) → IDLE
  relaxed(class 2) → IDLE
  sad    (class 3) → IDLE

Expected input structure (extracted ZIP):
  D:\\dog_cnn\\
  ├── train/
  │   ├── images/   ← angry_1475_jpg.rf.xxx.jpg, happy_3222_jpg.rf.xxx.jpg, ...
  │   └── labels/   ← angry_1475_jpg.rf.xxx.txt, happy_3222_jpg.rf.xxx.txt, ...
  └── data.yaml

Usage:
  python data/load_roboflow_behavior.py --src "D:\\dog_cnn"
  python data/load_roboflow_behavior.py --src "D:\\dog_cnn" --split 0.85
"""

import sys
import random
import cv2
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CROPS_DIR, CNN_INPUT_SIZE

# ── Class Mapping ───────────────────────────────────────────────────────────
# Roboflow data.yaml: names: ['angry', 'happy', 'relaxed', 'sad']
ROBOFLOW_TO_BEHAVIOR = {
    0: "DANGER",    # angry  → DANGER
    1: "IDLE",      # happy  → IDLE
    2: "IDLE",      # relaxed → IDLE
    3: "IDLE",      # sad    → IDLE
}

ROBOFLOW_CLASS_NAMES = {0: "angry", 1: "happy", 2: "relaxed", 3: "sad"}


def load_roboflow_behavior(src_dir, train_split=0.8):
    """
    Load Roboflow dog emotion dataset and create CNN behavior crops.

    Args:
        src_dir: Path to extracted Roboflow ZIP (contains train/images + train/labels)
        train_split: Train/val split ratio
    """
    src = Path(src_dir)

    # ── Find images and labels ──────────────────────────────────
    # Check common structures: train/images or just images
    img_dir = None
    lbl_dir = None

    for candidate in [src / "train" / "images", src / "images"]:
        if candidate.exists():
            img_dir = candidate
            break
    for candidate in [src / "train" / "labels", src / "labels"]:
        if candidate.exists():
            lbl_dir = candidate
            break

    if not img_dir or not lbl_dir:
        print(f"ERROR: Could not find images/ and labels/ folders in {src}")
        print(f"Expected: {src}/train/images/ and {src}/train/labels/")
        sys.exit(1)

    # Collect all image files
    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        image_files.extend(list(img_dir.glob(ext)))

    if not image_files:
        print(f"ERROR: No images found in {img_dir}")
        sys.exit(1)

    print("=" * 60)
    print("  Roboflow Behavior Dataset Loader")
    print("=" * 60)
    print(f"  Source      : {src}")
    print(f"  Images dir  : {img_dir}")
    print(f"  Labels dir  : {lbl_dir}")
    print(f"  Total images: {len(image_files)}")
    print(f"  Split       : {train_split:.0%} train / {1-train_split:.0%} val")
    print()
    print("  Class mapping:")
    print("    angry (0)   → DANGER")
    print("    happy (1)   → IDLE")
    print("    relaxed (2) → IDLE")
    print("    sad (3)     → IDLE")
    print("=" * 60)

    # ── Parse all labels and create crop entries ────────────────
    # Each entry: (image_path, bbox, behavior_class, original_class_id)
    crop_entries = []
    skipped_no_label = 0
    skipped_no_bbox = 0
    class_stats = {0: 0, 1: 0, 2: 0, 3: 0}

    print("\n  Scanning labels...")
    for img_path in tqdm(image_files, desc="  Parsing"):
        lbl_path = lbl_dir / f"{img_path.stem}.txt"

        if not lbl_path.exists():
            skipped_no_label += 1
            continue

        lines = lbl_path.read_text().strip().split("\n")
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            if class_id not in ROBOFLOW_TO_BEHAVIOR:
                continue

            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            behavior = ROBOFLOW_TO_BEHAVIOR[class_id]

            crop_entries.append({
                "img_path": img_path,
                "cx": cx, "cy": cy, "bw": bw, "bh": bh,
                "behavior": behavior,
                "class_id": class_id,
            })
            class_stats[class_id] += 1

    if not crop_entries:
        print("\n  ERROR: No valid bounding boxes found!")
        sys.exit(1)

    behavior_stats = {"DANGER": 0, "IDLE": 0}
    for entry in crop_entries:
        behavior_stats[entry["behavior"]] += 1

    print(f"\n  Roboflow class distribution:")
    for cid, name in ROBOFLOW_CLASS_NAMES.items():
        print(f"    {name:10s} (class {cid}): {class_stats[cid]:6d} boxes")

    print(f"\n  Mapped behavior distribution:")
    print(f"    DANGER (angry)              : {behavior_stats['DANGER']:6d} crops")
    print(f"    IDLE   (happy+relaxed+sad)  : {behavior_stats['IDLE']:6d} crops")

    # ── Shuffle and split ───────────────────────────────────────
    random.seed(42)
    random.shuffle(crop_entries)

    split_idx = int(len(crop_entries) * train_split)
    splits = {
        "train": crop_entries[:split_idx],
        "val": crop_entries[split_idx:],
    }

    # ── Create output directories ───────────────────────────────
    for split in ["train", "val"]:
        for cls in ["DANGER", "IDLE"]:
            (CROPS_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    # ── Extract and save crops ──────────────────────────────────
    saved_counts = {"train": {"DANGER": 0, "IDLE": 0}, "val": {"DANGER": 0, "IDLE": 0}}
    failed = 0

    for split_name, entries in splits.items():
        print(f"\n  Processing {split_name}: {len(entries)} crops")

        for idx, entry in enumerate(tqdm(entries, desc=f"  {split_name}")):
            img = cv2.imread(str(entry["img_path"]))
            if img is None:
                failed += 1
                continue

            h, w = img.shape[:2]
            cx, cy, bw, bh = entry["cx"], entry["cy"], entry["bw"], entry["bh"]

            # Convert normalized YOLO coords to pixel coords
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)

            # Clamp to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Skip tiny boxes
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue

            # Crop and resize
            crop = img[y1:y2, x1:x2]
            crop = cv2.resize(crop, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))

            # Save
            behavior = entry["behavior"]
            orig_class = ROBOFLOW_CLASS_NAMES[entry["class_id"]]
            crop_name = f"rf_{orig_class}_{idx:06d}.jpg"
            save_path = CROPS_DIR / split_name / behavior / crop_name

            cv2.imwrite(str(save_path), crop)
            saved_counts[split_name][behavior] += 1

    # ── Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Roboflow Behavior Dataset Ready!")
    print("=" * 60)
    print()
    print("  CNN Crops saved to:")
    total_saved = 0
    for split in ["train", "val"]:
        for cls in ["DANGER", "IDLE"]:
            count = saved_counts[split][cls]
            total_saved += count
            print(f"    {CROPS_DIR / split / cls}: {count} crops")
    print(f"\n  Total saved : {total_saved}")

    if failed > 0:
        print(f"  Failed reads: {failed}")
    if skipped_no_label > 0:
        print(f"  No label file: {skipped_no_label}")

    print()
    print("  NEXT STEP:")
    print("    python train/train_cnn_v2.py    # train BehaviorNetV2")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Load Roboflow dog emotion dataset for CNN behavior training",
    )
    parser.add_argument(
        "--src", type=str, required=True,
        help='Path to extracted Roboflow ZIP folder (e.g. "D:\\dog_cnn")',
    )
    parser.add_argument(
        "--split", type=float, default=0.8,
        help="Train/val split ratio (default: 0.8)",
    )
    args = parser.parse_args()

    load_roboflow_behavior(args.src, train_split=args.split)
