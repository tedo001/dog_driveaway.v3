"""
models/cnn_model.py — BehaviorNet: Custom CNN for dog behavior classification.
4 conv blocks → FC layers → 4 threat classes.
Optimized for 128x128 RGB input on RTX 4060.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path

from config import (
    CNN_MODEL_PATH,
    CNN_INPUT_SIZE,
    CNN_NUM_CLASSES,
    THREAT_CLASSES,
    YOLO_DEVICE,
)


class BehaviorNet(nn.Module):
    """
    Custom CNN architecture for dog behavior classification.

    Architecture:
        Block 1: Conv(3→32) → BN → ReLU → MaxPool
        Block 2: Conv(32→64) → BN → ReLU → MaxPool
        Block 3: Conv(64→128) → BN → ReLU → MaxPool
        Block 4: Conv(128→256) → BN → ReLU → MaxPool
        FC: 256*8*8 → 512 → dropout → 4 classes

    Total parameters: ~2,748,708
    """

    def __init__(self, num_classes=CNN_NUM_CLASSES):
        super().__init__()

        # Block 1: 128x128 → 64x64
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Block 2: 64x64 → 32x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Block 3: 32x32 → 16x16
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Block 4: 16x16 → 8x8
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Classifier
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class BehaviorClassifier:
    """
    Wrapper for BehaviorNet inference.
    Loads trained weights and classifies dog crop images.
    """

    def __init__(self, model_path=None, device=None):
        self.device = device or YOLO_DEVICE
        if self.device != "cpu":
            self.device = f"cuda:{self.device}" if not str(self.device).startswith("cuda") else self.device
        self.torch_device = torch.device(self.device if torch.cuda.is_available() else "cpu")

        self.model = BehaviorNet(num_classes=CNN_NUM_CLASSES)
        model_file = Path(model_path) if model_path else CNN_MODEL_PATH

        if model_file.exists():
            print(f"[CNN] Loading BehaviorNet: {model_file}")
            state_dict = torch.load(str(model_file), map_location=self.torch_device, weights_only=True)
            self.model.load_state_dict(state_dict)
        else:
            print(f"[CNN] No trained model at {model_file} — using random weights")

        self.model.to(self.torch_device)
        self.model.eval()
        print(f"[CNN] BehaviorNet loaded on {self.torch_device} "
              f"({sum(p.numel() for p in self.model.parameters()):,} params)")

    def preprocess(self, crop):
        """
        Preprocess a BGR crop for CNN input.

        Args:
            crop: BGR numpy array of any size.

        Returns:
            torch.Tensor of shape (1, 3, 128, 128), normalized to [0, 1].
        """
        img = cv2.resize(crop, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        tensor = torch.from_numpy(img).unsqueeze(0)  # add batch dim
        return tensor.to(self.torch_device)

    def classify(self, crop):
        """
        Classify a single dog crop image.

        Args:
            crop: BGR numpy array (dog bounding box crop).

        Returns:
            (threat_class_int, threat_label_str, confidence_float)
        """
        tensor = self.preprocess(crop)
        with torch.no_grad():
            output = self.model(tensor)
            probs = F.softmax(output, dim=1)
            conf, pred = torch.max(probs, dim=1)

        threat_class = int(pred.item())
        threat_label = THREAT_CLASSES.get(threat_class, "UNKNOWN")
        confidence = float(conf.item())

        return threat_class, threat_label, confidence

    def classify_batch(self, crops):
        """
        Classify multiple dog crops in a single batch.

        Args:
            crops: list of BGR numpy arrays.

        Returns:
            list of (threat_class, threat_label, confidence) tuples.
        """
        if not crops:
            return []

        tensors = [self.preprocess(crop) for crop in crops]
        batch = torch.cat(tensors, dim=0)

        with torch.no_grad():
            output = self.model(batch)
            probs = F.softmax(output, dim=1)
            confs, preds = torch.max(probs, dim=1)

        results = []
        for i in range(len(crops)):
            tc = int(preds[i].item())
            results.append((tc, THREAT_CLASSES.get(tc, "UNKNOWN"), float(confs[i].item())))

        return results
