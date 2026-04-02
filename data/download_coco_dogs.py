"""
data/download_coco_dogs.py — Download COCO dataset subset with only dogs + humans.
Uses the fiftyone library to download specific COCO classes without manual work.

If fiftyone is not installed, falls back to direct COCO API download.
Downloads only images containing dogs and/or people — saves storage and time.
"""

import sys
import os
import json
import shutil
import random
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATASET_DIR, DATA_DIR


# COCO 2017 URLs
COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_TRAIN_URL = "http://images.cocodataset.org/zips/train2017.zip"
COCO_VAL_URL = "http://images.cocodataset.org/zips/val2017.zip"

# COCO class IDs we need
COCO_PERSON_ID = 1
COCO_DOG_ID = 18

# Our remapped classes for YOLO
# 0 = dog, 1 = person
REMAP = {COCO_DOG_ID: 0, COCO_PERSON_ID: 1}


def download_coco_subset(
    max_images=5000,
    train_split=0.8,
    include_person_only=True,
):
    """
    Download COCO 2017 annotations, filter dog+person images,
    and download only those images.

    Args:
        max_images: Maximum total images to download.
        train_split: Fraction for training set.
        include_person_only: Also include images with only people (for person detection).
    """
    coco_dir = DATA_DIR / "coco_cache"
    coco_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Download annotations ─────────────────────────────
    ann_zip = coco_dir / "annotations_trainval2017.zip"
    ann_file = coco_dir / "annotations" / "instances_train2017.json"

    if not ann_file.exists():
        print("[COCO] Downloading annotations (252MB, one time only)...")
        _download_with_progress(COCO_ANNOTATIONS_URL, str(ann_zip))

        print("[COCO] Extracting annotations...")
        import zipfile
        with zipfile.ZipFile(str(ann_zip), "r") as z:
            z.extractall(str(coco_dir))
        print("[COCO] Annotations ready")
    else:
        print("[COCO] Annotations already cached")

    # ── Step 2: Parse annotations and filter dog/person images ───
    print("[COCO] Parsing annotations...")
    with open(str(ann_file), "r") as f:
        coco_data = json.load(f)

    # Build image ID → image info mapping
    id_to_image = {img["id"]: img for img in coco_data["images"]}

    # Find all annotations for dog and person
    dog_image_ids = set()
    person_image_ids = set()
    image_annotations = {}  # image_id → list of annotations

    for ann in coco_data["annotations"]:
        cat_id = ann["category_id"]
        img_id = ann["image_id"]

        if cat_id not in REMAP:
            continue

        if cat_id == COCO_DOG_ID:
            dog_image_ids.add(img_id)
        elif cat_id == COCO_PERSON_ID:
            person_image_ids.add(img_id)

        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].append(ann)

    # Select images: prioritize images with dogs
    selected_ids = list(dog_image_ids)
    if include_person_only:
        # Add some person-only images
        person_only = list(person_image_ids - dog_image_ids)
        random.shuffle(person_only)
        person_limit = min(len(person_only), max_images // 4)
        selected_ids.extend(person_only[:person_limit])

    random.shuffle(selected_ids)
    selected_ids = selected_ids[:max_images]

    print(f"[COCO] Found {len(dog_image_ids)} images with dogs")
    print(f"[COCO] Found {len(person_image_ids)} images with people")
    print(f"[COCO] Selected {len(selected_ids)} images to download")

    # ── Step 3: Download images and create YOLO labels ───────────
    split_idx = int(len(selected_ids) * train_split)
    splits = {
        "train": selected_ids[:split_idx],
        "valid": selected_ids[split_idx:],
    }

    for split in ["train", "valid"]:
        (DATASET_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    total_downloaded = 0
    total_failed = 0

    for split_name, img_ids in splits.items():
        print(f"\n[COCO] Downloading {split_name}: {len(img_ids)} images")

        for i, img_id in enumerate(img_ids):
            img_info = id_to_image.get(img_id)
            if not img_info:
                continue

            url = img_info["coco_url"]
            filename = img_info["file_name"]
            img_w = img_info["width"]
            img_h = img_info["height"]

            dst_img = DATASET_DIR / split_name / "images" / filename
            dst_lbl = DATASET_DIR / split_name / "labels" / f"{Path(filename).stem}.txt"

            # Download image
            if not dst_img.exists():
                try:
                    urlretrieve(url, str(dst_img))
                    total_downloaded += 1
                except (URLError, Exception):
                    total_failed += 1
                    continue

            # Create YOLO label
            anns = image_annotations.get(img_id, [])
            label_lines = []

            for ann in anns:
                cat_id = ann["category_id"]
                if cat_id not in REMAP:
                    continue

                our_cls = REMAP[cat_id]
                bx, by, bw, bh = ann["bbox"]  # COCO format: x, y, w, h (absolute)

                # Convert to YOLO format: cx, cy, w, h (normalized)
                cx = (bx + bw / 2) / img_w
                cy = (by + bh / 2) / img_h
                nw = bw / img_w
                nh = bh / img_h

                # Clamp
                cx = max(0, min(1, cx))
                cy = max(0, min(1, cy))
                nw = max(0, min(1, nw))
                nh = max(0, min(1, nh))

                if nw > 0.01 and nh > 0.01:
                    label_lines.append(f"{our_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            if label_lines:
                dst_lbl.write_text("\n".join(label_lines))

            # Progress
            if (i + 1) % 100 == 0:
                print(f"    {split_name}: {i + 1}/{len(img_ids)} downloaded")

    # ── Step 4: Create data.yaml ─────────────────────────────────
    data_yaml = DATASET_DIR / "data.yaml"
    data_yaml.write_text(
        f"train: {DATASET_DIR / 'train' / 'images'}\n"
        f"val: {DATASET_DIR / 'valid' / 'images'}\n"
        f"\n"
        f"nc: 2\n"
        f"names: ['dog', 'person']\n"
    )

    # Print stats
    print("\n" + "=" * 60)
    print("  COCO Dog+Person Dataset Ready!")
    print("=" * 60)
    print(f"  Downloaded : {total_downloaded} images")
    print(f"  Failed     : {total_failed} images")
    print(f"  Location   : {DATASET_DIR}")
    print(f"  data.yaml  : {data_yaml}")

    for split in ["train", "valid"]:
        img_dir = DATASET_DIR / split / "images"
        lbl_dir = DATASET_DIR / split / "labels"
        n_imgs = len(list(img_dir.glob("*"))) if img_dir.exists() else 0
        n_lbls = len(list(lbl_dir.glob("*"))) if lbl_dir.exists() else 0
        print(f"  {split}: {n_imgs} images, {n_lbls} labels")

    print()
    print("  NEXT STEPS:")
    print("    python data/prepare_crops.py        # create CNN behavior crops")
    print("    python train/train_yolo.py           # fine-tune YOLO")
    print("    python train/train_ssd.py            # train SSD")
    print("    python train/train_cnn_v2.py         # train BehaviorNetV2")
    print("=" * 60)


def _download_with_progress(url, dest):
    """Download file with progress indicator."""
    from tqdm import tqdm

    class TqdmUpTo(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with TqdmUpTo(unit="B", unit_scale=True, miniters=1, desc=Path(dest).name) as t:
        urlretrieve(url, dest, reporthook=t.update_to)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download COCO dog+person subset")
    parser.add_argument("--max", type=int, default=5000,
                        help="Maximum images to download (default: 5000)")
    parser.add_argument("--split", type=float, default=0.8,
                        help="Train/val split ratio (default: 0.8)")
    args = parser.parse_args()

    download_coco_subset(max_images=args.max, train_split=args.split)
