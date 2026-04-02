"""
mlops/data_loader.py — Unified data loading for all dataset formats.

Supports:
  1. Local ZIP file (Roboflow export) — auto-extracts and detects format
  2. Local folder (already extracted) — auto-detects structure
  3. COCO download — downloads dog+person subset from COCO 2017

Auto-detects:
  - YOLO format (images/ + labels/ with .txt)
  - Roboflow format (data.yaml with class names)
  - Custom nested folders (image/ + label/ pairs)

Usage from MLOps app — user just points to a path, everything else is automatic.
"""

import sys
import os
import shutil
import zipfile
import random
import cv2
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATASET_DIR, CROPS_DIR, CNN_INPUT_SIZE


# ── Format Detection ────────────────────────────────────────────────────────

def detect_format(src_dir):
    """
    Auto-detect dataset format from a directory.

    Returns dict with:
      - format: "roboflow_yolo" | "yolo" | "coco" | "custom_nested" | "unknown"
      - data_yaml: path to data.yaml if found
      - classes: list of class names if detected
      - img_dir: path to images directory
      - lbl_dir: path to labels directory
    """
    src = Path(src_dir)
    result = {"format": "unknown", "data_yaml": None, "classes": [], "img_dir": None, "lbl_dir": None}

    # Check for data.yaml (Roboflow / YOLO format)
    yaml_path = None
    for candidate in [src / "data.yaml", src / "data.yml"]:
        if candidate.exists():
            yaml_path = candidate
            break

    if yaml_path:
        classes = _parse_yaml_classes(yaml_path)
        result["data_yaml"] = yaml_path
        result["classes"] = classes

        # Check for Roboflow marker
        yaml_text = yaml_path.read_text()
        if "roboflow" in yaml_text.lower():
            result["format"] = "roboflow_yolo"
        else:
            result["format"] = "yolo"

    # Find image/label directories
    for img_name in ["images", "image", "img"]:
        for sub in [src / "train" / img_name, src / img_name]:
            if sub.exists():
                result["img_dir"] = sub
                break
        if result["img_dir"]:
            break

    for lbl_name in ["labels", "label", "lbl"]:
        for sub in [src / "train" / lbl_name, src / lbl_name]:
            if sub.exists():
                result["lbl_dir"] = sub
                break
        if result["lbl_dir"]:
            break

    # If no standard dirs found, check for nested custom structure
    if not result["img_dir"]:
        nested = _find_nested_pairs(src)
        if nested:
            result["format"] = "custom_nested"
            result["img_dir"] = src  # root contains nested pairs

    # If we have img+lbl dirs but no yaml, it's plain YOLO
    if result["img_dir"] and result["lbl_dir"] and result["format"] == "unknown":
        result["format"] = "yolo"

    return result


def _parse_yaml_classes(yaml_path):
    """Parse class names from data.yaml without requiring PyYAML."""
    classes = []
    text = yaml_path.read_text()
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("names:"):
            # Parse inline list: names: ['angry', 'happy', 'relaxed', 'sad']
            if "[" in line:
                inner = line.split("[")[1].split("]")[0]
                classes = [c.strip().strip("'\"") for c in inner.split(",")]
            break
    # Also try multi-line names block
    if not classes:
        in_names = False
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped.startswith("names:"):
                in_names = True
                continue
            if in_names:
                if stripped.startswith("- "):
                    classes.append(stripped[2:].strip().strip("'\""))
                elif stripped and not stripped.startswith("#"):
                    break
    return classes


def _find_nested_pairs(root):
    """Find nested image/label folder pairs (custom format)."""
    pairs = []
    for dirpath, dirnames, _ in os.walk(root):
        dirpath = Path(dirpath)
        subdirs = {d.lower(): d for d in dirnames}
        img_dir = lbl_dir = None
        for name in ["image", "images", "img"]:
            if name in subdirs:
                img_dir = dirpath / subdirs[name]
        for name in ["label", "labels", "lbl"]:
            if name in subdirs:
                lbl_dir = dirpath / subdirs[name]
        if img_dir and lbl_dir:
            pairs.append((img_dir, lbl_dir))
    return pairs


# ── ZIP Extraction ──────────────────────────────────────────────────────────

def extract_zip(zip_path, dest_dir=None):
    """
    Extract a ZIP file and return the extracted directory path.
    If dest_dir is None, extracts next to the ZIP file.
    """
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    if dest_dir is None:
        dest_dir = zip_path.parent / zip_path.stem

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)

    print(f"  Extracted to: {dest_dir}")
    return dest_dir


# ── Unified Data Loader ─────────────────────────────────────────────────────

def count_images(directory):
    """Count image files in a directory."""
    count = 0
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        count += len(list(Path(directory).glob(ext)))
    return count


def load_dataset_from_path(src_path, purpose="detection"):
    """
    Load a dataset from any local path (ZIP or folder).

    Args:
        src_path: Path to ZIP file or extracted folder
        purpose: "detection" (YOLO/SSD) or "behavior" (CNN crops)

    Returns:
        dict with dataset info and status
    """
    src = Path(src_path)

    # Handle ZIP files
    if src.suffix.lower() == ".zip":
        src = extract_zip(src)

    if not src.exists():
        return {"success": False, "error": f"Path not found: {src}"}

    # Detect format
    fmt = detect_format(src)

    print(f"\n  Auto-detected format: {fmt['format']}")
    if fmt["classes"]:
        print(f"  Classes found: {fmt['classes']}")
    if fmt["img_dir"]:
        n_imgs = count_images(fmt["img_dir"])
        print(f"  Images found: {n_imgs}")

    return {
        "success": True,
        "src_path": str(src),
        "format": fmt["format"],
        "classes": fmt["classes"],
        "img_dir": str(fmt["img_dir"]) if fmt["img_dir"] else None,
        "lbl_dir": str(fmt["lbl_dir"]) if fmt["lbl_dir"] else None,
        "data_yaml": str(fmt["data_yaml"]) if fmt["data_yaml"] else None,
    }


# ── COCO Dog+Human Downloader ──────────────────────────────────────────────

def download_coco(max_images=5000):
    """
    Download COCO 2017 dog+person subset.
    Wraps data/download_coco_dogs.py functionality.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    try:
        from data.download_coco_dogs import download_coco_subset
        download_coco_subset(max_images=max_images)
        return {"success": True, "images": max_images}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Copy YOLO Dataset for Detection Training ───────────────────────────────

def copy_for_detection(src_info, clear_old=True):
    """
    Copy a YOLO-format dataset to DATASET_DIR for YOLO/SSD training.

    Args:
        src_info: dict from load_dataset_from_path()
        clear_old: whether to clear existing dataset first
    """
    if clear_old and DATASET_DIR.exists():
        print(f"  Clearing old dataset at {DATASET_DIR}...")
        shutil.rmtree(DATASET_DIR)

    img_dir = Path(src_info["img_dir"])
    lbl_dir = Path(src_info["lbl_dir"])

    # Create output dirs
    train_imgs = DATASET_DIR / "train" / "images"
    train_lbls = DATASET_DIR / "train" / "labels"
    val_imgs = DATASET_DIR / "valid" / "images"
    val_lbls = DATASET_DIR / "valid" / "labels"

    for d in [train_imgs, train_lbls, val_imgs, val_lbls]:
        d.mkdir(parents=True, exist_ok=True)

    # Collect all image-label pairs
    images = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        images.extend(list(img_dir.glob(ext)))

    random.seed(42)
    random.shuffle(images)
    split = int(len(images) * 0.8)

    copied = 0
    for idx, img_path in enumerate(tqdm(images, desc="  Copying")):
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue

        if idx < split:
            dst_img_dir, dst_lbl_dir = train_imgs, train_lbls
        else:
            dst_img_dir, dst_lbl_dir = val_imgs, val_lbls

        shutil.copy2(img_path, dst_img_dir / img_path.name)
        shutil.copy2(lbl_path, dst_lbl_dir / lbl_path.name)
        copied += 1

    # Create data.yaml
    nc = len(src_info.get("classes", [])) or 2
    names = src_info.get("classes", []) or ["dog", "person"]
    yaml_content = (
        f"train: {train_imgs}\n"
        f"val: {val_imgs}\n\n"
        f"nc: {nc}\n"
        f"names: {names}\n"
    )
    (DATASET_DIR / "data.yaml").write_text(yaml_content)

    print(f"\n  Copied {copied} image-label pairs to {DATASET_DIR}")
    print(f"  Train: {split} | Val: {copied - split}")
    return {"success": True, "total": copied, "train": split, "val": copied - split}
