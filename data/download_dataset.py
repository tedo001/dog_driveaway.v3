"""
data/download_dataset.py — Universal dataset downloader.
Downloads dog behaviour dataset from Roboflow.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    ROBOFLOW_API_KEY,
    ROBOFLOW_WORKSPACE,
    ROBOFLOW_PROJECT,
    ROBOFLOW_VERSION,
    DATASET_DIR,
)


def download_dataset():
    """Download dataset from Roboflow."""
    if not ROBOFLOW_API_KEY or ROBOFLOW_API_KEY == "your_roboflow_api_key_here":
        print("ERROR: Set your ROBOFLOW_API_KEY in .env file")
        print("  1. Go to https://roboflow.com → Settings → API Key")
        print("  2. Paste it in .env: ROBOFLOW_API_KEY=your_key")
        sys.exit(1)

    from roboflow import Roboflow

    print(f"[DOWNLOAD] Connecting to Roboflow...")
    print(f"  Workspace : {ROBOFLOW_WORKSPACE}")
    print(f"  Project   : {ROBOFLOW_PROJECT}")
    print(f"  Version   : {ROBOFLOW_VERSION}")

    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    version = project.version(ROBOFLOW_VERSION)

    print(f"[DOWNLOAD] Downloading YOLOv8 format to {DATASET_DIR}...")
    dataset = version.download("yolov8", location=str(DATASET_DIR))

    print(f"[DOWNLOAD] Dataset downloaded successfully!")
    print(f"  Location: {DATASET_DIR}")

    # Print dataset stats
    for split in ["train", "valid", "test"]:
        split_dir = DATASET_DIR / split / "images"
        if split_dir.exists():
            count = len(list(split_dir.glob("*")))
            print(f"  {split}: {count} images")

    return dataset


if __name__ == "__main__":
    download_dataset()
