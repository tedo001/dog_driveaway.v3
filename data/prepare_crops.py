"""
data/prepare_crops.py — Auto-label and crop dog images for CNN training.
Uses YOLO detections to extract dog crops and assigns behavior labels
based on bounding box size relative to frame (proxy for distance/threat).
"""

import sys
import cv2
import os
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    DATASET_DIR,
    CROPS_DIR,
    CNN_INPUT_SIZE,
    YOLO_MODEL_PATH,
    YOLO_BASE_MODEL,
    THREAT_CLASSES,
)


def get_label_from_bbox_ratio(bbox_area, frame_area):
    """
    Auto-label based on dog bounding box size relative to frame.
    Larger box → closer dog → higher threat level.

    This is a proxy labeling strategy since we lack real attack labels.

    Args:
        bbox_area: area of dog bounding box in pixels.
        frame_area: total area of frame in pixels.

    Returns:
        int: threat class (0=IDLE, 1=ALERT, 2=DANGER, 3=DOG_FIGHT)
    """
    ratio = bbox_area / frame_area

    if ratio < 0.05:
        return 0  # IDLE — small/far dog
    elif ratio < 0.15:
        return 1  # ALERT — medium distance
    elif ratio < 0.30:
        return 2  # DANGER — close/large
    else:
        return 3  # DOG_FIGHT — very large (multiple dogs overlapping)


def prepare_crops():
    """Extract dog crops from dataset and auto-label them."""
    from ultralytics import YOLO

    # Load YOLO model
    model_path = YOLO_MODEL_PATH if YOLO_MODEL_PATH.exists() else YOLO_BASE_MODEL
    print(f"[CROPS] Loading YOLO model: {model_path}")
    model = YOLO(str(model_path))

    # Setup output directories
    for class_id, class_name in THREAT_CLASSES.items():
        for split in ["train", "val"]:
            out_dir = CROPS_DIR / split / class_name
            out_dir.mkdir(parents=True, exist_ok=True)

    # Process each split
    total_crops = {name: 0 for name in THREAT_CLASSES.values()}

    for split_name, out_split in [("train", "train"), ("valid", "val"), ("test", "val")]:
        img_dir = DATASET_DIR / split_name / "images"
        if not img_dir.exists():
            print(f"[CROPS] Skipping {split_name} — not found at {img_dir}")
            continue

        image_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        print(f"[CROPS] Processing {split_name}: {len(image_files)} images")

        for img_path in tqdm(image_files, desc=f"  {split_name}"):
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            h, w = frame.shape[:2]
            frame_area = h * w

            # Run YOLO detection
            results = model.predict(source=frame, conf=0.3, verbose=False)

            for result in results:
                for i, box in enumerate(result.boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    if x2 - x1 < 20 or y2 - y1 < 20:
                        continue

                    crop = frame[y1:y2, x1:x2]
                    crop = cv2.resize(crop, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))

                    bbox_area = (x2 - x1) * (y2 - y1)
                    label = get_label_from_bbox_ratio(bbox_area, frame_area)
                    label_name = THREAT_CLASSES[label]

                    out_dir = CROPS_DIR / out_split / label_name
                    filename = f"{img_path.stem}_crop{i}.jpg"
                    cv2.imwrite(str(out_dir / filename), crop)
                    total_crops[label_name] += 1

    print("\n[CROPS] Done! Crop distribution:")
    for name, count in total_crops.items():
        print(f"  {name}: {count}")


if __name__ == "__main__":
    prepare_crops()
