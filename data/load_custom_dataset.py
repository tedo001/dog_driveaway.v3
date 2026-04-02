"""
data/load_custom_dataset.py — Load YOUR labeled dataset from any folder structure.

Supports messy/nested folder layouts like:
    idle_dog/
    ├── idle dog/
    │   ├── image/
    │   └── label/
    ├── idle dog 2/
    │   ├── image/
    │   └── label/
    └── idle dog 4/
        ├── idle dog 1/
        │   ├── image/
        │   └── label/
        ├── image/
        └── label/

Usage:
  python data/load_custom_dataset.py --idle "C:/Users/.../idle_dog" --danger "C:/Users/.../danger_dog"
  python data/load_custom_dataset.py --idle "C:/path/idle" --danger "C:/path/danger" --alert "C:/path/alert"

The script will:
  1. Recursively scan all subfolders for image/ + label/ pairs
  2. Copy and rename all images+labels (no duplicates)
  3. Remap class IDs to our standard: 0=dog
  4. Create train/valid split (80/20)
  5. Create data.yaml for YOLO training
  6. Create CNN behavior crop folders
"""

import sys
import os
import cv2
import shutil
import random
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATASET_DIR, CROPS_DIR, CNN_INPUT_SIZE


def find_image_label_pairs(root_dir):
    """
    Recursively find all (image_dir, label_dir) pairs in a folder.
    Looks for folders named 'image' or 'images' paired with 'label' or 'labels'.
    """
    root = Path(root_dir)
    pairs = []

    for dirpath, dirnames, filenames in os.walk(root):
        dirpath = Path(dirpath)
        subdirs = {d.lower(): d for d in dirnames}

        img_dir = None
        lbl_dir = None

        for name in ["image", "images", "img"]:
            if name in subdirs:
                img_dir = dirpath / subdirs[name]
        for name in ["label", "labels", "lbl"]:
            if name in subdirs:
                lbl_dir = dirpath / subdirs[name]

        if img_dir and lbl_dir:
            pairs.append((img_dir, lbl_dir))

    return pairs


def collect_files(root_dir, class_name):
    """
    Collect all image+label files from a root directory.
    Returns list of (image_path, label_path, class_name).
    """
    pairs = find_image_label_pairs(root_dir)

    if not pairs:
        print(f"  WARNING: No image/label folders found in {root_dir}")
        print(f"  Expected structure: some_folder/image/ + some_folder/label/")
        return []

    print(f"  [{class_name}] Found {len(pairs)} image/label folder pairs in {root_dir}")

    all_files = []
    for img_dir, lbl_dir in pairs:
        # Find images
        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            image_files.extend(list(img_dir.glob(ext)))

        for img_path in image_files:
            # Find matching label
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if lbl_path.exists():
                all_files.append((img_path, lbl_path, class_name))
            else:
                # Try xml format (some tools export XML)
                xml_path = lbl_dir / f"{img_path.stem}.xml"
                if xml_path.exists():
                    all_files.append((img_path, xml_path, class_name))

        print(f"    {img_dir.parent.name}/{img_dir.name}: "
              f"{len(image_files)} images")

    return all_files


def load_custom_dataset(class_dirs, train_split=0.8):
    """
    Load dataset from multiple class directories.

    Args:
        class_dirs: dict of {class_name: directory_path}
                    e.g. {"IDLE": "C:/data/idle_dog", "DANGER": "C:/data/danger_dog"}
        train_split: fraction for training (rest = validation)
    """
    print("=" * 60)
    print("  Custom Dataset Loader")
    print("=" * 60)

    # ── Step 1: Collect all files ────────────────────────────────
    all_files = []  # list of (img_path, lbl_path, class_name)

    for class_name, dir_path in class_dirs.items():
        dir_path = Path(dir_path)
        if not dir_path.exists():
            print(f"\n  ERROR: Directory not found: {dir_path}")
            continue
        files = collect_files(dir_path, class_name)
        all_files.extend(files)
        print(f"  [{class_name}] Total: {len(files)} labeled images\n")

    if not all_files:
        print("\n  ERROR: No images found!")
        print("  Make sure your folders have this structure:")
        print("    your_folder/")
        print("      ├── image/    ← .jpg files here")
        print("      └── label/    ← .txt files here (same names)")
        sys.exit(1)

    print(f"\n  TOTAL: {len(all_files)} labeled images across "
          f"{len(class_dirs)} classes")

    # ── Step 2: Shuffle and split ────────────────────────────────
    random.shuffle(all_files)
    split_idx = int(len(all_files) * train_split)

    splits = {
        "train": all_files[:split_idx],
        "valid": all_files[split_idx:],
    }

    # ── Step 3: Create output directories ────────────────────────
    for split in ["train", "valid"]:
        (DATASET_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    # CNN crop dirs
    for class_name in class_dirs.keys():
        for split in ["train", "val"]:
            (CROPS_DIR / split / class_name).mkdir(parents=True, exist_ok=True)

    # ── Step 4: Copy files with unique names ─────────────────────
    class_counts = {name: 0 for name in class_dirs}
    crop_counts = {name: 0 for name in class_dirs}

    for split_name, files in splits.items():
        crop_split = "train" if split_name == "train" else "val"
        print(f"\n  Processing {split_name}: {len(files)} images")

        for idx, (img_path, lbl_path, class_name) in enumerate(tqdm(files, desc=f"  {split_name}")):
            # Unique filename to avoid collisions
            safe_class = class_name.lower().replace(" ", "_")
            unique_name = f"{safe_class}_{idx:06d}"

            # Copy image
            img_ext = img_path.suffix
            dst_img = DATASET_DIR / split_name / "images" / f"{unique_name}{img_ext}"
            shutil.copy2(img_path, dst_img)

            # Process label file
            dst_lbl = DATASET_DIR / split_name / "labels" / f"{unique_name}.txt"

            if lbl_path.suffix == ".txt":
                # YOLO format label — remap class IDs
                # All detections become class 0 (dog) since YOLO just detects dogs
                # The behavior class (IDLE/DANGER) is handled by CNN, not YOLO
                lines = lbl_path.read_text().strip().split("\n")
                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # Remap all classes to 0 (dog) for YOLO
                        new_lines.append(f"0 {parts[1]} {parts[2]} {parts[3]} {parts[4]}")
                if new_lines:
                    dst_lbl.write_text("\n".join(new_lines))
            else:
                # Just copy as-is
                shutil.copy2(lbl_path, dst_lbl)

            class_counts[class_name] += 1

            # ── Create CNN crop ──────────────────────────────────
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            h, w = frame.shape[:2]

            # Read bounding boxes from label
            if lbl_path.suffix == ".txt":
                lines = lbl_path.read_text().strip().split("\n")
                for ci, line in enumerate(lines):
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

                    # Convert normalized to pixel coordinates
                    x1 = int((cx - bw / 2) * w)
                    y1 = int((cy - bh / 2) * h)
                    x2 = int((cx + bw / 2) * w)
                    y2 = int((cy + bh / 2) * h)

                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    if x2 - x1 < 20 or y2 - y1 < 20:
                        continue

                    crop = frame[y1:y2, x1:x2]
                    crop = cv2.resize(crop, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))

                    crop_name = f"{unique_name}_crop{ci}.jpg"
                    cv2.imwrite(str(CROPS_DIR / crop_split / class_name / crop_name), crop)
                    crop_counts[class_name] += 1

    # ── Step 5: Create data.yaml ─────────────────────────────────
    # For YOLO: single class "dog" (behavior is classified by CNN separately)
    data_yaml = DATASET_DIR / "data.yaml"
    data_yaml.write_text(
        f"train: {DATASET_DIR / 'train' / 'images'}\n"
        f"val: {DATASET_DIR / 'valid' / 'images'}\n"
        f"\n"
        f"nc: 1\n"
        f"names: ['dog']\n"
    )

    # ── Step 6: Print summary ────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Dataset Ready!")
    print("=" * 60)
    print()
    print("  YOLO labels (dog detection):")
    for split in ["train", "valid"]:
        n_img = len(list((DATASET_DIR / split / "images").glob("*")))
        n_lbl = len(list((DATASET_DIR / split / "labels").glob("*")))
        print(f"    {split}: {n_img} images, {n_lbl} labels")
    print(f"    data.yaml: {data_yaml}")
    print()
    print("  CNN crops (behavior classification):")
    for class_name, count in crop_counts.items():
        print(f"    {class_name}: {count} crops")
    print()
    print("  Images per class:")
    for class_name, count in class_counts.items():
        print(f"    {class_name}: {count}")
    print()
    print("  NEXT STEPS:")
    print("    python train/train_yolo.py     # train dog detector")
    print("    python train/train_ssd.py      # train SSD detector")
    print("    python train/train_cnn_v2.py   # train behavior classifier (BehaviorNetV2)")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Load custom labeled dataset from any folder structure",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python data/load_custom_dataset.py --idle "C:/Users/you/idle_dog"
  python data/load_custom_dataset.py --idle "C:/idle" --danger "C:/danger"
  python data/load_custom_dataset.py --idle "C:/idle" --danger "C:/danger" --alert "C:/alert" --dogfight "C:/fight"
        """,
    )
    parser.add_argument("--idle", type=str, default=None,
                        help="Path to IDLE dog images folder")
    parser.add_argument("--danger", type=str, default=None,
                        help="Path to DANGER dog images folder")
    parser.add_argument("--alert", type=str, default=None,
                        help="Path to ALERT dog images folder")
    parser.add_argument("--dogfight", type=str, default=None,
                        help="Path to DOG_FIGHT images folder")
    parser.add_argument("--split", type=float, default=0.8,
                        help="Train/val split ratio (default: 0.8)")
    args = parser.parse_args()

    # Build class directories from arguments
    class_dirs = {}
    if args.idle:
        class_dirs["IDLE"] = args.idle
    if args.danger:
        class_dirs["DANGER"] = args.danger
    if args.alert:
        class_dirs["ALERT"] = args.alert
    if args.dogfight:
        class_dirs["DOG_FIGHT"] = args.dogfight

    if not class_dirs:
        print("ERROR: Provide at least one class directory")
        print()
        print("Example:")
        print('  python data/load_custom_dataset.py --idle "C:/Users/Sri Mathialagan/Downloads/idel_dog/idel_dog" --danger "C:/path/to/danger_dog"')
        sys.exit(1)

    load_custom_dataset(class_dirs, train_split=args.split)
