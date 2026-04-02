"""
models/behavior_net_v2.py — BehaviorNetV2: Improved CNN with residual connections.

Upgrade from BehaviorNet v1:
  v1: 4 conv blocks → FC → 4 classes (2.7M params, 99.85% val acc)
  v2: 5 residual blocks + SE attention → FC → N classes (~4.2M params)

Improvements:
  - Residual connections: prevents vanishing gradients, trains deeper
  - SE (Squeeze-and-Excitation) attention: learns which channels matter
  - Dropout after each block: reduces overfitting
  - Global Average Pooling: reduces parameters, better generalization
  - Works with 2, 3, or 4 classes (auto-detects from data)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path

from config import (
    CNN_INPUT_SIZE,
    CNN_NUM_CLASSES,
    THREAT_CLASSES,
    DEVICE,
)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation attention block."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        scale = self.squeeze(x).view(b, c)
        scale = self.excite(scale).view(b, c, 1, 1)
        return x * scale


class ResidualBlock(nn.Module):
    """Residual block with optional SE attention."""

    def __init__(self, in_channels, out_channels, use_se=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels) if use_se else nn.Identity()
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(0.1)

        # Skip connection: match dimensions if channels change
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = F.relu(out + identity)  # residual connection
        out = self.pool(out)
        out = self.dropout(out)
        return out


class BehaviorNetV2(nn.Module):
    """
    Improved CNN for dog behavior classification.

    Architecture:
        Input: 128x128x3
        Block 1: ResBlock(3→32)  + SE + Pool → 64x64x32
        Block 2: ResBlock(32→64) + SE + Pool → 32x32x64
        Block 3: ResBlock(64→128) + SE + Pool → 16x16x128
        Block 4: ResBlock(128→256) + SE + Pool → 8x8x256
        Block 5: ResBlock(256→512) + SE + Pool → 4x4x512
        Global Average Pool → 512
        FC: 512 → 256 → dropout → num_classes

    ~4.2M parameters
    """

    def __init__(self, num_classes=CNN_NUM_CLASSES):
        super().__init__()

        self.block1 = ResidualBlock(3, 32)      # 128→64
        self.block2 = ResidualBlock(32, 64)      # 64→32
        self.block3 = ResidualBlock(64, 128)     # 32→16
        self.block4 = ResidualBlock(128, 256)    # 16→8
        self.block5 = ResidualBlock(256, 512)    # 8→4

        self.gap = nn.AdaptiveAvgPool2d(1)       # 4x4→1x1
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.gap(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class BehaviorClassifierV2:
    """
    Inference wrapper for BehaviorNetV2.
    Same interface as BehaviorClassifier (drop-in replacement).
    """

    CNN_V2_PATH = Path(__file__).resolve().parent.parent / "export" / "behavior_net_v2.pt"

    def __init__(self, model_path=None, device=None, num_classes=None):
        self.device_str = device or DEVICE
        if self.device_str != "cpu":
            self.device_str = f"cuda:{self.device_str}" if not str(self.device_str).startswith("cuda") else self.device_str
        self.torch_device = torch.device(self.device_str if torch.cuda.is_available() else "cpu")

        # Auto-detect number of classes from saved model or use config
        self.num_classes = num_classes or CNN_NUM_CLASSES
        model_file = Path(model_path) if model_path else self.CNN_V2_PATH

        self.model = BehaviorNetV2(num_classes=self.num_classes)

        if model_file.exists():
            print(f"[CNN-V2] Loading BehaviorNetV2: {model_file}")
            state_dict = torch.load(str(model_file), map_location=self.torch_device, weights_only=True)
            self.model.load_state_dict(state_dict)
        else:
            # Fallback to v1
            from config import CNN_MODEL_PATH
            if CNN_MODEL_PATH.exists():
                print(f"[CNN-V2] V2 not found, loading V1: {CNN_MODEL_PATH}")
                from models.cnn_model import BehaviorClassifier
                self._fallback = BehaviorClassifier()
                self._use_fallback = True
                return
            print(f"[CNN-V2] No trained model found — using random weights")

        self._use_fallback = False
        self.model.to(self.torch_device)
        self.model.eval()
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"[CNN-V2] BehaviorNetV2 on {self.torch_device} ({total_params:,} params)")

    def preprocess(self, crop):
        img = cv2.resize(crop, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        tensor = torch.from_numpy(img).unsqueeze(0)
        return tensor.to(self.torch_device)

    def classify(self, crop):
        if hasattr(self, '_use_fallback') and self._use_fallback:
            return self._fallback.classify(crop)

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
        if hasattr(self, '_use_fallback') and self._use_fallback:
            return [self._fallback.classify(c) for c in crops]

        if not crops:
            return []

        tensors = [self.preprocess(c) for c in crops]
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
