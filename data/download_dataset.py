"""
data/download_dataset.py — Download dog detection dataset from Roboflow Universe.

Two modes:
  Mode 1 (default): Download public dog dataset from Roboflow Universe
  Mode 2 (custom):  Download your own dataset from your Roboflow workspace

Usage:
  python data/download_dataset.py                          → download public dog dataset
  python data/download_dataset.py --custom                 → download from your workspace
  python data/download_dataset.py --url "roboflow-url"     → download from any Universe URL
"""

import os
import sys
import argparse
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

# ── Public dog detection datasets on Roboflow Universe ──────────
# These are FREE and work with any API key (private or publishable)
PUBLIC_DATASETS = {
    "dog-detection": {
        "workspace": "dog-breeds-ssvtb",
        "project": "dog-detection-fao1e",
        "version": 2,
        "description": "Dog detection dataset (~5,000 images, YOLO format)",
    },
    "dog-and-cat": {
        "workspace": "brad-dwyer",
        "project": "aerial-sheep",
        "version": 1,
        "description": "Animal detection dataset",
    },
}


def check_api_key():
    """Validate API key is set and correct type."""
    if not ROBOFLOW_API_KEY or ROBOFLOW_API_KEY == "your_private_api_key_here":
        print("=" * 60)
        print("  SETUP: Set your Roboflow API key in .env")
        print("=" * 60)
        print()
        print("  1. Go to https://app.roboflow.com")
        print("  2. Click Settings (bottom left) → API Keys")
        print("  3. Copy your PRIVATE API Key")
        print("     (click the eye icon next to the key starting with Q1g2...)")
        print("  4. Open .env file and set:")
        print("     ROBOFLOW_API_KEY=your_private_key_here")
        print()
        print("  NOTE: The Publishable key (rf_...) works ONLY for")
        print("        public Universe datasets, not your own workspace.")
        print()
        sys.exit(1)
    return True


def download_public_dataset(dataset_key="dog-detection"):
    """Download a public dataset from Roboflow Universe."""
    from roboflow import Roboflow

    check_api_key()

    ds_info = PUBLIC_DATASETS.get(dataset_key)
    if not ds_info:
        print(f"Unknown dataset: {dataset_key}")
        print(f"Available: {list(PUBLIC_DATASETS.keys())}")
        sys.exit(1)

    print("=" * 60)
    print("  Downloading Public Dog Dataset from Roboflow Universe")
    print("=" * 60)
    print(f"  Dataset   : {ds_info['description']}")
    print(f"  Workspace : {ds_info['workspace']}")
    print(f"  Project   : {ds_info['project']}")
    print(f"  Version   : {ds_info['version']}")
    print("=" * 60)

    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(ds_info["workspace"]).project(ds_info["project"])
    version = project.version(ds_info["version"])

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[DOWNLOAD] Downloading YOLOv8 format to {DATASET_DIR}...")
    dataset = version.download("yolov8", location=str(DATASET_DIR))

    _print_stats()
    return dataset


def download_custom_dataset():
    """Download dataset from your own Roboflow workspace."""
    from roboflow import Roboflow

    check_api_key()

    if ROBOFLOW_API_KEY.startswith("rf_"):
        print("  WARNING: You're using the Publishable key (rf_...).")
        print("  This may not work for your own workspace datasets.")
        print("  Use the Private key for workspace access.\n")

    print("=" * 60)
    print("  Downloading from YOUR Roboflow Workspace")
    print("=" * 60)
    print(f"  Workspace : {ROBOFLOW_WORKSPACE}")
    print(f"  Project   : {ROBOFLOW_PROJECT}")
    print(f"  Version   : {ROBOFLOW_VERSION}")
    print("=" * 60)

    rf = Roboflow(api_key=ROBOFLOW_API_KEY)

    # Try to connect to workspace
    try:
        ws = rf.workspace(ROBOFLOW_WORKSPACE)
    except Exception as e:
        print(f"\n  ERROR: Workspace '{ROBOFLOW_WORKSPACE}' not found.")
        print(f"  {e}")
        print()
        print("  HOW TO FIX:")
        print("  1. Go to https://app.roboflow.com")
        print("  2. Look at URL: app.roboflow.com/YOUR-WORKSPACE-ID/...")
        print("  3. Update .env: ROBOFLOW_WORKSPACE=your-workspace-id")
        sys.exit(1)

    # Try to access project
    try:
        project = ws.project(ROBOFLOW_PROJECT)
    except Exception:
        print(f"\n  ERROR: Project '{ROBOFLOW_PROJECT}' not found.")
        print("  Available projects in your workspace:")
        try:
            for p in ws.project_list:
                name = p if isinstance(p, str) else getattr(p, 'id', str(p))
                print(f"    - {name}")
        except Exception:
            print("    (could not list projects)")
        print("\n  Update .env: ROBOFLOW_PROJECT=correct-project-name")
        sys.exit(1)

    # Download
    version = project.version(ROBOFLOW_VERSION)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[DOWNLOAD] Downloading YOLOv8 format to {DATASET_DIR}...")
    dataset = version.download("yolov8", location=str(DATASET_DIR))

    _print_stats()
    return dataset


def download_from_url(url):
    """Download dataset from a Roboflow Universe URL."""
    from roboflow import Roboflow

    check_api_key()

    # Parse URL: https://universe.roboflow.com/workspace/project/version
    # or: https://app.roboflow.com/workspace/project/version
    parts = url.rstrip("/").split("/")
    workspace = None
    project = None
    version_num = 1

    for i, part in enumerate(parts):
        if part in ("universe.roboflow.com", "app.roboflow.com"):
            if i + 1 < len(parts):
                workspace = parts[i + 1]
            if i + 2 < len(parts):
                project = parts[i + 2]
            if i + 3 < len(parts):
                try:
                    version_num = int(parts[i + 3])
                except ValueError:
                    pass
            break

    if not workspace or not project:
        print(f"  Could not parse URL: {url}")
        print("  Expected format: https://universe.roboflow.com/WORKSPACE/PROJECT/VERSION")
        sys.exit(1)

    print("=" * 60)
    print(f"  Downloading from URL")
    print("=" * 60)
    print(f"  Workspace : {workspace}")
    print(f"  Project   : {project}")
    print(f"  Version   : {version_num}")
    print("=" * 60)

    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    proj = rf.workspace(workspace).project(project)
    ver = proj.version(version_num)

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[DOWNLOAD] Downloading YOLOv8 format to {DATASET_DIR}...")
    dataset = ver.download("yolov8", location=str(DATASET_DIR))

    _print_stats()
    return dataset


def _print_stats():
    """Print dataset statistics."""
    print(f"\n[DOWNLOAD] Dataset downloaded successfully!")
    print(f"  Location: {DATASET_DIR}")
    total = 0
    for split in ["train", "valid", "test"]:
        split_dir = DATASET_DIR / split / "images"
        if split_dir.exists():
            count = len(list(split_dir.glob("*")))
            total += count
            print(f"  {split}: {count} images")
    print(f"  TOTAL: {total} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download dog detection dataset")
    parser.add_argument("--custom", action="store_true",
                        help="Download from your own Roboflow workspace")
    parser.add_argument("--url", type=str, default=None,
                        help="Download from a Roboflow Universe URL")
    parser.add_argument("--dataset", type=str, default="dog-detection",
                        choices=list(PUBLIC_DATASETS.keys()),
                        help="Which public dataset to download")
    args = parser.parse_args()

    if args.url:
        download_from_url(args.url)
    elif args.custom:
        download_custom_dataset()
    else:
        download_public_dataset(args.dataset)
