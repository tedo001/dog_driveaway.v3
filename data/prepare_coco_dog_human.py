"""
data/prepare_coco_dog_human.py — Use COCO-pretrained YOLOv8 to detect dogs+humans
in YOUR images and auto-generate YOLO labels + CNN behavior crops.

NO TRAINING NEEDED for basic detection — COCO already knows dogs and humans!

COCO class IDs:
  0  = person
  16 = dog

Usage:
  Step 1: Put your images in data/raw_images/
  Step 2: python data/prepare_coco_dog_human.py
  Step 3: Labels saved to data/dataset/ and crops to data/crops/

This script:
  1. Runs pretrained YOLOv8n on each image
  2. Filters only dog (class 16) and person (class 0) detections
  3. Saves YOLO-format labels (remapped: 0=dog, 1=person)
  4. Crops dog regions and sorts into behavior classes for CNN training
"""

import sys
import cv2
import shutil
import random
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    DATASET_DIR,
    CROPS_DIR,
    CNN_INPUT_SIZE,
    DATA_DIR,
    THREAT_CLASSES,
)


COCO_PERSON = 0
COCO_DOG = 16


def prepare_from_images(
    images_dir=None,
    conf_threshold=0.35,
    train_split=0.8,
):
    """
    Process raw images using COCO-pretrained YOLOv8.

    Args:
        images_dir: Path to folder with your images. Default: data/raw_images/
        conf_threshold: Minimum detection confidence.
        train_split: Fraction of images for training (rest goes to validation).
    """
    from ultralytics import YOLO

    if images_dir is None:
        images_dir = DATA_DIR / "raw_images"

    images_dir = Path(images_dir)

    if not images_dir.exists():
        print("=" * 60)
        print("  PUT YOUR IMAGES HERE FIRST:")
        print("=" * 60)
        print()
        print(f"  1. Create folder: {images_dir}")
        print(f"  2. Put dog/human images inside it (.jpg, .png)")
        print(f"  3. Run this script again")
        print()
        print("  You can collect images from:")
        print("  - Google Images (search 'aggressive dog', 'dog near person')")
        print("  - Your phone camera")
        print("  - YouTube frames (use: python data/collect_youtube.py URL)")
        print()
        images_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Created empty folder: {images_dir}")
        sys.exit(0)

    # Find images
    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        image_files.extend(list(images_dir.glob(ext)))

    if not image_files:
        print(f"  No images found in {images_dir}")
        print("  Put .jpg or .png files in that folder and try again.")
        sys.exit(1)

    print("=" * 60)
    print("  COCO Dog+Human Auto-Labeler")
    print("=" * 60)
    print(f"  Images found : {len(image_files)}")
    print(f"  Confidence   : {conf_threshold}")
    print(f"  Train split  : {train_split:.0%}")
    print("=" * 60)

    # Load pretrained COCO YOLOv8
    print("\n[INIT] Loading YOLOv8n (COCO pretrained)...")
    model = YOLO("yolov8n.pt")

    # Shuffle and split
    random.shuffle(image_files)
    split_idx = int(len(image_files) * train_split)
    splits = {
        "train": image_files[:split_idx],
        "valid": image_files[split_idx:],
    }

    # Setup output directories
    for split in ["train", "valid"]:
        (DATASET_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    for class_name in THREAT_CLASSES.values():
        for split in ["train", "val"]:
            (CROPS_DIR / split / class_name).mkdir(parents=True, exist_ok=True)

    # Stats
    stats = {
        "total_images": 0,
        "images_with_dogs": 0,
        "images_with_humans": 0,
        "total_dogs": 0,
        "total_humans": 0,
        "crops": {name: 0 for name in THREAT_CLASSES.values()},
    }

    # Process each split
    for split_name, files in splits.items():
        crop_split = "train" if split_name == "train" else "val"
        print(f"\n[PROCESSING] {split_name}: {len(files)} images")

        for img_path in tqdm(files, desc=f"  {split_name}"):
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            h, w = frame.shape[:2]
            frame_area = h * w
            stats["total_images"] += 1

            # Run COCO detection — only person (0) and dog (16)
            results = model.predict(
                source=frame,
                conf=conf_threshold,
                classes=[COCO_PERSON, COCO_DOG],
                verbose=False,
            )

            label_lines = []
            dog_boxes = []
            has_dog = False
            has_human = False

            for result in results:
                for box in result.boxes:
                    coco_cls = int(box.cls[0])
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])

                    # Remap: COCO dog(16) → our class 0, COCO person(0) → our class 1
                    if coco_cls == COCO_DOG:
                        our_cls = 0  # dog
                        has_dog = True
                        stats["total_dogs"] += 1
                        dog_boxes.append((x1, y1, x2, y2, conf))
                    elif coco_cls == COCO_PERSON:
                        our_cls = 1  # person
                        has_human = True
                        stats["total_humans"] += 1
                    else:
                        continue

                    # YOLO format: class cx cy bw bh (normalized)
                    cx = ((x1 + x2) / 2) / w
                    cy = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    label_lines.append(f"{our_cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            if has_dog:
                stats["images_with_dogs"] += 1
            if has_human:
                stats["images_with_humans"] += 1

            # Save image + labels
            if label_lines:
                dst_img = DATASET_DIR / split_name / "images" / img_path.name
                dst_lbl = DATASET_DIR / split_name / "labels" / f"{img_path.stem}.txt"
                shutil.copy2(img_path, dst_img)
                dst_lbl.write_text("\n".join(label_lines))

            # Create CNN crops from dog detections
            for i, (bx1, by1, bx2, by2, bconf) in enumerate(dog_boxes):
                bx1, by1 = max(0, int(bx1)), max(0, int(by1))
                bx2, by2 = min(w, int(bx2)), min(h, int(by2))

                if bx2 - bx1 < 20 or by2 - by1 < 20:
                    continue

                crop = frame[by1:by2, bx1:bx2]
                crop = cv2.resize(crop, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))

                # Auto-label behavior by box size ratio
                bbox_area = (bx2 - bx1) * (by2 - by1)
                ratio = bbox_area / frame_area

                if has_human and ratio > 0.20:
                    label = "DANGER"
                elif has_human and ratio > 0.08:
                    label = "ALERT"
                elif not has_human and len(dog_boxes) >= 2:
                    label = "DOG_FIGHT"
                else:
                    label = "IDLE"

                out_dir = CROPS_DIR / crop_split / label
                filename = f"{img_path.stem}_dog{i}.jpg"
                cv2.imwrite(str(out_dir / filename), crop)
                stats["crops"][label] += 1

    # Create data.yaml for YOLO training
    data_yaml = DATASET_DIR / "data.yaml"
    data_yaml.write_text(
        f"train: {DATASET_DIR / 'train' / 'images'}\n"
        f"val: {DATASET_DIR / 'valid' / 'images'}\n"
        f"\n"
        f"nc: 2\n"
        f"names: ['dog', 'person']\n"
    )

    # Print results
    print("\n" + "=" * 60)
    print("  DONE! Dataset Ready")
    print("=" * 60)
    print(f"  Total images processed : {stats['total_images']}")
    print(f"  Images with dogs       : {stats['images_with_dogs']}")
    print(f"  Images with humans     : {stats['images_with_humans']}")
    print(f"  Total dog detections   : {stats['total_dogs']}")
    print(f"  Total human detections : {stats['total_humans']}")
    print()
    print("  YOLO labels saved to:")
    print(f"    {DATASET_DIR}")
    print(f"    data.yaml: {data_yaml}")
    print()
    print("  CNN behavior crops:")
    for label, count in stats["crops"].items():
        print(f"    {label:>10}: {count} crops")
    print()
    print("  NEXT STEPS:")
    print("    python train/train_yolo.py     # fine-tune YOLO on your data")
    print("    python train/train_ssd.py      # train SSD on your data")
    print("    python train/train_cnn.py      # train behavior classifier")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Auto-label images with COCO YOLOv8")
    parser.add_argument("--images", type=str, default=None,
                        help="Path to folder with your images")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="Detection confidence threshold")
    args = parser.parse_args()

    prepare_from_images(args.images, args.conf)
