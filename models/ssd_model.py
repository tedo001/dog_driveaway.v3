"""
models/ssd_model.py — SSD (Single Shot Detector) for dog + person detection.

Architecture: SSD300 with MobileNetV2 backbone (lightweight, RTX 4060 optimized).
Uses torchvision's SSD implementation with custom head for 3 classes:
  0 = background
  1 = dog
  2 = person

SSD vs YOLO comparison:
  SSD  → single-pass detection, anchor-based, good for fixed-size input
  YOLO → grid-based, anchor-free (v8), generally faster

Both are available — switch via DETECTOR_BACKEND in config.py.
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models.detection.ssd import SSDHead
import cv2
import numpy as np
from pathlib import Path

from config import (
    SSD_MODEL_PATH,
    SSD_NUM_CLASSES,
    SSD_CONF_THRESHOLD,
    SSD_NMS_THRESHOLD,
    SSD_IMGSZ,
    SSD_CLASS_NAMES,
    YOLO_DEVICE,
)


class SSDDogDetector(nn.Module):
    """
    SSD300 with VGG16 backbone for dog + person detection.

    Why SSD:
      - Single forward pass (no region proposals like Faster R-CNN)
      - Multi-scale feature maps detect objects at different sizes
      - Anchor boxes at 6 different aspect ratios per feature map
      - Non-maximum suppression filters overlapping detections
      - 300x300 input → fast inference on RTX 4060

    Architecture:
      VGG16 backbone (pretrained ImageNet)
        → Extra feature layers (conv6, conv7, conv8, conv9, conv10, conv11)
        → SSD Head: classification + regression per anchor box
        → NMS post-processing
    """

    def __init__(self, num_classes=SSD_NUM_CLASSES, pretrained_backbone=True):
        super().__init__()
        self.num_classes = num_classes

        # Load pretrained SSD300-VGG16 from torchvision
        if pretrained_backbone:
            weights = SSD300_VGG16_Weights.COCO_V1
            self.model = ssd300_vgg16(weights=weights)
        else:
            self.model = ssd300_vgg16(weights=None)

        # Replace the classification head for our custom classes
        # SSD head needs: in_channels list, num_anchors list, num_classes
        in_channels = [512, 1024, 512, 256, 256, 256]  # VGG16 SSD feature map channels
        num_anchors = self.model.anchor_generator.num_anchors_per_location()
        self.model.head = SSDHead(in_channels, num_anchors, num_classes)

    def forward(self, images, targets=None):
        """
        Forward pass.
        Training: images + targets → losses dict
        Inference: images → list of {boxes, labels, scores}
        """
        return self.model(images, targets)


class SSDDetector:
    """
    Inference wrapper for SSD dog+person detector.
    Same interface as DualYOLODetector so it's a drop-in replacement.
    """

    def __init__(self, model_path=None, device=None):
        self.device_str = device or YOLO_DEVICE
        if self.device_str != "cpu":
            self.device_str = f"cuda:{self.device_str}" if not str(self.device_str).startswith("cuda") else self.device_str
        self.torch_device = torch.device(self.device_str if torch.cuda.is_available() else "cpu")

        model_file = Path(model_path) if model_path else SSD_MODEL_PATH

        self.model = SSDDogDetector(num_classes=SSD_NUM_CLASSES, pretrained_backbone=True)

        if model_file.exists():
            print(f"[SSD] Loading trained model: {model_file}")
            state_dict = torch.load(str(model_file), map_location=self.torch_device, weights_only=True)
            self.model.load_state_dict(state_dict)
        else:
            print(f"[SSD] No trained model at {model_file}")
            print("[SSD] Using COCO-pretrained SSD300 (will detect COCO classes)")
            self._use_coco_fallback = True

        self.model.to(self.torch_device)
        self.model.eval()

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"[SSD] SSD300-VGG16 loaded on {self.torch_device} ({total_params:,} params)")

        self._use_coco_fallback = not model_file.exists()

        # COCO class IDs for fallback mode
        self._coco_dog_id = 18     # COCO: dog
        self._coco_person_id = 1   # COCO: person

    def preprocess(self, frame):
        """
        Preprocess BGR frame for SSD input.
        SSD300 expects: float32 tensor [0-1], shape (3, 300, 300).
        """
        img = cv2.resize(frame, (SSD_IMGSZ, SSD_IMGSZ))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        tensor = torch.from_numpy(img).unsqueeze(0)
        return tensor.to(self.torch_device)

    def detect(self, frame):
        """
        Run SSD detection on a single BGR frame.
        Returns same format as DualYOLODetector.detect().

        Returns:
            list of dicts: [
                {'bbox': (x1, y1, x2, y2), 'class': 'dog'/'person', 'confidence': float},
                ...
            ]
        """
        h_orig, w_orig = frame.shape[:2]
        tensor = self.preprocess(frame)

        with torch.no_grad():
            predictions = self.model(tensor)

        detections = []
        pred = predictions[0]  # first (only) image in batch

        boxes = pred["boxes"].cpu().numpy()
        labels = pred["labels"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()

        for box, label, score in zip(boxes, labels, scores):
            if score < SSD_CONF_THRESHOLD:
                continue

            # Map label to class name
            if self._use_coco_fallback:
                # COCO pretrained — map COCO IDs to our classes
                if label == self._coco_dog_id:
                    cls_name = "dog"
                elif label == self._coco_person_id:
                    cls_name = "person"
                else:
                    continue  # skip other COCO classes
            else:
                # Custom trained model
                if label < len(SSD_CLASS_NAMES):
                    cls_name = SSD_CLASS_NAMES[label]
                    if cls_name == "__background__":
                        continue
                else:
                    continue

            # Scale boxes from 300x300 back to original frame size
            x1 = box[0] * w_orig / SSD_IMGSZ
            y1 = box[1] * h_orig / SSD_IMGSZ
            x2 = box[2] * w_orig / SSD_IMGSZ
            y2 = box[3] * h_orig / SSD_IMGSZ

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "class": cls_name,
                "confidence": float(score),
            })

        return detections

    def get_dog_crops(self, frame, detections=None):
        """
        Extract cropped dog regions. Same interface as DualYOLODetector.

        Returns:
            list of (crop, bbox) tuples.
        """
        if detections is None:
            detections = self.detect(frame)

        crops = []
        h, w = frame.shape[:2]
        for det in detections:
            if det["class"] != "dog":
                continue
            x1, y1, x2, y2 = [int(c) for c in det["bbox"]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 > 10 and y2 - y1 > 10:
                crop = frame[y1:y2, x1:x2]
                crops.append((crop, det["bbox"]))

        return crops
