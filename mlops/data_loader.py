"""
mlops/data_loader.py — Unified data loading for all dataset formats.

Supports:
  1. Local ZIP file (Roboflow export) — auto-extracts and detects format
  2. Local folder (already extracted) — auto-detects structure
  3. COCO download — downloads dog+person subset from COCO 2017

Auto-detects:
  - YOLO format (images/ + labels/ with .txt)
  - Roboflow format (coco128.yaml with class names)
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
      - data_yaml: path to coco128.yaml if found
      - classes: list of class names if detected
      - img_dir: path to images directory
      - lbl_dir: path to labels directory
    """
    src = Path(src_dir)
    result = {"format": "unknown", "data_yaml": None, "classes": [], "img_dir": None, "lbl_dir": None, "all_splits": []}

    # Check for coco128.yaml (Roboflow / YOLO format)
    yaml_path = None
    for candidate in [src / "coco128.yaml", src / "data.yml"]:
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

    # Collect all split directories (train, valid, test)
    all_splits = []
    for split_name in ["train", "valid", "val", "test"]:
        split_dir = src / split_name
        if split_dir.exists():
            s_img = s_lbl = None
            for img_name in ["images", "image", "img"]:
                if (split_dir / img_name).exists():
                    s_img = split_dir / img_name
                    break
            for lbl_name in ["labels", "label", "lbl"]:
                if (split_dir / lbl_name).exists():
                    s_lbl = split_dir / lbl_name
                    break
            if s_img and s_lbl:
                all_splits.append({"split": split_name, "img_dir": s_img, "lbl_dir": s_lbl})
    result["all_splits"] = all_splits

    # If we have img+lbl dirs but no yaml, it's plain YOLO
    if result["img_dir"] and result["lbl_dir"] and result["format"] == "unknown":
        result["format"] = "yolo"

    return result


def _parse_yaml_classes(yaml_path):
    """Parse class names from coco128.yaml without requiring PyYAML."""
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


# ── Image Counting Utilities ────────────────────────────────────────────────

# FIX: use suffix.lower() so .JPG / .PNG / .JPEG are counted correctly.
# The old glob-based approach ("*.jpg") was case-sensitive and silently
# skipped uppercase-extension files, causing IDLE to show 0 in the dashboard.
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def count_images(directory):
    """
    Count image files in a directory.

    FIX: uses suffix.lower() for case-insensitive matching so files saved
    as .JPG / .PNG / .JPEG are counted correctly on all platforms.
    Previously used glob('*.jpg') which silently skipped uppercase extensions.
    """
    return sum(
        1 for f in Path(directory).iterdir()
        if f.is_file() and f.suffix.lower() in _IMG_EXTS
    )


def get_behavior_class_counts():
    """
    Return per-class image counts aggregated from BOTH train AND val splits.

    This is the function the dashboard should call to display accurate
    DANGER / IDLE counts. Scanning only 'train' was causing IDLE: 0 when
    crops existed in val but not exclusively in train.

    Returns:
        dict: { "DANGER": <int>, "IDLE": <int>, ... }
              Keys are the actual subfolder names found on disk.
    """
    counts = {}

    print("\n[DEBUG] get_behavior_class_counts()")
    print(f"[DEBUG] CROPS_DIR = {CROPS_DIR}")

    for split in ["train", "val"]:
        split_dir = CROPS_DIR / split
        if not split_dir.exists():
            print(f"[DEBUG]   split '{split}' dir missing: {split_dir}")
            continue

        cls_dirs = sorted(d for d in split_dir.iterdir() if d.is_dir())
        print(f"[DEBUG]   {split}/ → class folders: {[d.name for d in cls_dirs]}")

        for cls_dir in cls_dirs:
            cls_name = cls_dir.name
            n = sum(
                1 for f in cls_dir.iterdir()
                if f.is_file() and f.suffix.lower() in _IMG_EXTS
            )
            counts[cls_name] = counts.get(cls_name, 0) + n
            print(f"[DEBUG]     {split}/{cls_name}: {n} images")

    print(f"[DEBUG] Final class counts: {counts}\n")
    return counts


# ── Unified Data Loader ─────────────────────────────────────────────────────

def load_dataset_from_path(src_path, purpose="detection"):
    """
    Load a dataset from any local path (ZIP or folder).

    Args:
        src_path: Path to ZIP file or extracted folder
        purpose: "detection" (YOLO) or "behavior" (CNN crops)

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
        "all_splits": [{"split": s["split"], "img_dir": str(s["img_dir"]), "lbl_dir": str(s["lbl_dir"])} for s in fmt.get("all_splits", [])],
    }


# ── COCO Dog+Human Downloader ──────────────────────────────────────────────

# COCO 2017 URLs and class IDs
COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_PERSON_ID = 1
COCO_DOG_ID = 18
COCO_REMAP = {COCO_DOG_ID: 0, COCO_PERSON_ID: 1}  # 0=dog, 1=person


def download_coco(max_images=5000, train_split=0.8):
    """
    Download COCO 2017 dog+person subset directly.

    Downloads annotations, filters dog/person images, downloads those images,
    creates YOLO labels and coco128.yaml.
    """
    import json
    import zipfile
    from urllib.request import urlretrieve

    try:
        coco_dir = DATASET_DIR.parent / "coco_cache"
        coco_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Download annotations
        ann_zip = coco_dir / "annotations_trainval2017.zip"
        ann_file = coco_dir / "annotations" / "instances_train2017.json"

        if not ann_file.exists():
            print("[COCO] Downloading annotations (252MB, one time only)...")
            urlretrieve(COCO_ANNOTATIONS_URL, str(ann_zip))
            print("[COCO] Extracting annotations...")
            with zipfile.ZipFile(str(ann_zip), "r") as z:
                z.extractall(str(coco_dir))

        # Step 2: Parse and filter
        print("[COCO] Parsing annotations...")
        with open(str(ann_file), "r") as f:
            coco_data = json.load(f)

        id_to_image = {img["id"]: img for img in coco_data["images"]}
        dog_image_ids = set()
        person_image_ids = set()
        image_annotations = {}

        for ann in coco_data["annotations"]:
            cat_id = ann["category_id"]
            img_id = ann["image_id"]
            if cat_id not in COCO_REMAP:
                continue
            if cat_id == COCO_DOG_ID:
                dog_image_ids.add(img_id)
            elif cat_id == COCO_PERSON_ID:
                person_image_ids.add(img_id)
            if img_id not in image_annotations:
                image_annotations[img_id] = []
            image_annotations[img_id].append(ann)

        selected_ids = list(dog_image_ids)
        person_only = list(person_image_ids - dog_image_ids)
        random.shuffle(person_only)
        selected_ids.extend(person_only[:max_images // 4])
        random.shuffle(selected_ids)
        selected_ids = selected_ids[:max_images]

        print(f"[COCO] {len(dog_image_ids)} dog images, {len(person_image_ids)} person images")
        print(f"[COCO] Selected {len(selected_ids)} to download")

        # Step 3: Download images and create YOLO labels
        split_idx = int(len(selected_ids) * train_split)
        splits = {"train": selected_ids[:split_idx], "valid": selected_ids[split_idx:]}

        for split in ["train", "valid"]:
            (DATASET_DIR / split / "images").mkdir(parents=True, exist_ok=True)
            (DATASET_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

        total_downloaded = 0
        for split_name, img_ids in splits.items():
            print(f"[COCO] Downloading {split_name}: {len(img_ids)} images")
            for i, img_id in enumerate(tqdm(img_ids, desc=f"  {split_name}")):
                img_info = id_to_image.get(img_id)
                if not img_info:
                    continue

                filename = img_info["file_name"]
                img_w, img_h = img_info["width"], img_info["height"]
                dst_img = DATASET_DIR / split_name / "images" / filename
                dst_lbl = DATASET_DIR / split_name / "labels" / f"{Path(filename).stem}.txt"

                if not dst_img.exists():
                    try:
                        urlretrieve(img_info["coco_url"], str(dst_img))
                        total_downloaded += 1
                    except Exception:
                        continue

                anns = image_annotations.get(img_id, [])
                lines = []
                for ann in anns:
                    if ann["category_id"] not in COCO_REMAP:
                        continue
                    cls = COCO_REMAP[ann["category_id"]]
                    bx, by, bw, bh = ann["bbox"]
                    cx, cy = (bx + bw / 2) / img_w, (by + bh / 2) / img_h
                    nw, nh = bw / img_w, bh / img_h
                    cx, cy = max(0, min(1, cx)), max(0, min(1, cy))
                    nw, nh = max(0, min(1, nw)), max(0, min(1, nh))
                    if nw > 0.01 and nh > 0.01:
                        lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                if lines:
                    dst_lbl.write_text("\n".join(lines))

        # Step 4: Create coco128.yaml
        (DATASET_DIR / "coco128.yaml").write_text(
            f"train: {DATASET_DIR / 'train' / 'images'}\n"
            f"val: {DATASET_DIR / 'valid' / 'images'}\n\n"
            f"nc: 2\nnames: ['dog', 'person']\n"
        )

        print(f"[COCO] Done! {total_downloaded} images downloaded to {DATASET_DIR}")
        return {"success": True, "images": total_downloaded}

    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Copy YOLO Dataset for Detection Training ───────────────────────────────

def copy_for_detection(src_info, clear_old=True):
    """
    Copy a YOLO-format dataset to DATASET_DIR for YOLO training.

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

    # Create coco128.yaml
    nc = len(src_info.get("classes", [])) or 2
    names = src_info.get("classes", []) or ["dog", "person"]
    yaml_content = (
        f"train: {train_imgs}\n"
        f"val: {val_imgs}\n\n"
        f"nc: {nc}\n"
        f"names: {names}\n"
    )
    (DATASET_DIR / "coco128.yaml").write_text(yaml_content)

    print(f"\n  Copied {copied} image-label pairs to {DATASET_DIR}")
    print(f"  Train: {split} | Val: {copied - split}")
    return {"success": True, "total": copied, "train": split, "val": copied - split}


# ── CNN Data Loader ────────────────────────────────────────────────────────

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


def load_cnn_data(batch_size=32):
    """
    CNN data loader using BOTH train and val folders from CROPS_DIR.

    FIX: uses is_valid_file with suffix.lower() so .JPG/.PNG files are
    recognized by torchvision regardless of extension case.
    Raises RuntimeError with a clear message if DANGER or IDLE is missing.
    """
    train_dir = CROPS_DIR / "train"
    val_dir   = CROPS_DIR / "val"

    print("\n[DEBUG] load_cnn_data()")
    print(f"[DEBUG] train_dir = {train_dir}  (exists={train_dir.exists()})")
    print(f"[DEBUG] val_dir   = {val_dir}  (exists={val_dir.exists()})")

    if not train_dir.exists() or not val_dir.exists():
        raise RuntimeError(
            f"Train or Val folder missing under CROPS_DIR={CROPS_DIR}. "
            f"Expected: {train_dir} and {val_dir}"
        )

    transform = transforms.Compose([
        transforms.Resize((CNN_INPUT_SIZE, CNN_INPUT_SIZE)),
        transforms.ToTensor(),
    ])

    # Case-insensitive extension check — the core fix
    _valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    def _is_valid_file(path: str) -> bool:
        return Path(path).suffix.lower() in _valid_exts

    train_dataset = ImageFolder(root=str(train_dir), transform=transform,
                                is_valid_file=_is_valid_file)
    val_dataset   = ImageFolder(root=str(val_dir),   transform=transform,
                                is_valid_file=_is_valid_file)

    print(f"\n[DEBUG] Train classes : {train_dataset.classes}")
    print(f"[DEBUG] Val   classes : {val_dataset.classes}")
    print(f"[DEBUG] Train samples : {len(train_dataset)}")
    print(f"[DEBUG] Val   samples : {len(val_dataset)}")

    for cls_name, cls_idx in train_dataset.class_to_idx.items():
        n_train = sum(1 for _, lbl in train_dataset.samples if lbl == cls_idx)
        n_val   = sum(1 for _, lbl in val_dataset.samples   if lbl == cls_idx)
        print(f"[DEBUG]   {cls_name:10s} → train={n_train:5d}, val={n_val:5d}")

    if "IDLE" not in train_dataset.classes:
        raise RuntimeError(
            f"IDLE class missing from train dataset. "
            f"Found classes: {train_dataset.classes}. "
            f"Check folder: {train_dir / 'IDLE'}"
        )
    if "DANGER" not in train_dataset.classes:
        raise RuntimeError(
            f"DANGER class missing from train dataset. "
            f"Found classes: {train_dataset.classes}. "
            f"Check folder: {train_dir / 'DANGER'}"
        )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    print("\n[DEBUG] ✅ CNN dataset loaded successfully\n")
    return train_loader, val_loader