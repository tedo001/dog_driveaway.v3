"""
models/ssd_dataset.py — PyTorch Dataset for SSD training.
Converts YOLO-format labels (class cx cy w h) to SSD-format
(absolute x1 y1 x2 y2 boxes + integer labels).

SSD training requires:
  - images: list of float32 tensors (C, H, W) in [0, 1]
  - targets: list of dicts with:
      'boxes': FloatTensor (N, 4) in [x1, y1, x2, y2] absolute pixels
      'labels': Int64Tensor (N,) class indices (0=background, 1=dog, 2=person)
"""

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import torchvision.transforms as T

from config import SSD_IMGSZ, SSD_NUM_CLASSES


class SSDDogDataset(Dataset):
    """
    Loads YOLO-format dataset and converts to SSD training format.

    YOLO label format (per line):
        class_id  center_x  center_y  width  height   (all normalized 0-1)

    SSD requires:
        boxes: absolute pixel coordinates [x1, y1, x2, y2]
        labels: integer class IDs (shifted +1 for background class 0)
    """

    def __init__(self, images_dir, labels_dir, img_size=SSD_IMGSZ, augment=False):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.img_size = img_size
        self.augment = augment

        # Find all images with matching labels
        self.image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            self.image_files.extend(list(self.images_dir.glob(ext)))
        self.image_files.sort()

        print(f"[SSD-DATA] Found {len(self.image_files)} images in {images_dir}")

        # Augmentation transforms
        self.color_jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            # Return empty sample on read failure
            return self._empty_sample()

        orig_h, orig_w = img.shape[:2]

        # Resize to SSD input size
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load YOLO labels
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        boxes = []
        labels = []

        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    cls_id = int(parts[0])
                    cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

                    # Convert YOLO normalized (cx, cy, w, h) → absolute (x1, y1, x2, y2)
                    x1 = (cx - bw / 2) * self.img_size
                    y1 = (cy - bh / 2) * self.img_size
                    x2 = (cx + bw / 2) * self.img_size
                    y2 = (cy + bh / 2) * self.img_size

                    # Clamp to image bounds
                    x1 = max(0, min(self.img_size, x1))
                    y1 = max(0, min(self.img_size, y1))
                    x2 = max(0, min(self.img_size, x2))
                    y2 = max(0, min(self.img_size, y2))

                    # Skip degenerate boxes
                    if x2 - x1 < 2 or y2 - y1 < 2:
                        continue

                    boxes.append([x1, y1, x2, y2])
                    # SSD class 0 = background, so shift YOLO classes by +1
                    labels.append(cls_id + 1)

        # Convert to tensors
        img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)

        if self.augment:
            img_tensor = self.color_jitter(img_tensor)
            # Random horizontal flip
            if torch.rand(1).item() > 0.5:
                img_tensor = torch.flip(img_tensor, [2])
                for i, box in enumerate(boxes):
                    x1_new = self.img_size - box[2]
                    x2_new = self.img_size - box[0]
                    boxes[i] = [x1_new, box[1], x2_new, box[3]]

        if len(boxes) == 0:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
            }
        else:
            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64),
            }

        return img_tensor, target

    def _empty_sample(self):
        img = torch.zeros(3, self.img_size, self.img_size, dtype=torch.float32)
        target = {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
        }
        return img, target


def ssd_collate_fn(batch):
    """
    Custom collate function for SSD DataLoader.
    SSD needs list of images and list of targets (not stacked tensors).
    """
    images = []
    targets = []
    for img, target in batch:
        images.append(img)
        targets.append(target)
    return images, targets
