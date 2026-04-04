"""
Run this from D:\Pycharm\dog_driveaway.v3\
python check_data.py
"""
from pathlib import Path

CROPS_DIR = Path("data/crops")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

print("=" * 50)
print("DATASET DIAGNOSTIC")
print("=" * 50)

for split in ["train", "val"]:
    split_dir = CROPS_DIR / split
    print(f"\n[{split.upper()}]")
    if not split_dir.exists():
        print(f"  ❌ MISSING: {split_dir}")
        continue

    for cls_dir in sorted(split_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        imgs = [f for f in cls_dir.iterdir()
                if f.is_file() and f.suffix.lower() in IMG_EXTS]
        status = "✅" if len(imgs) > 0 else "❌ EMPTY"
        print(f"  {status}  {cls_dir.name}: {len(imgs)} images")

print("\n" + "=" * 50)
print("CONFIG CHECK")
print("=" * 50)

try:
    import sys

    sys.path.insert(0, ".")
    from config import CROPS_DIR as CFG_CROPS, CNN_NUM_CLASSES, THREAT_CLASSES

    print(f"  CROPS_DIR      = {CFG_CROPS}")
    print(f"  CNN_NUM_CLASSES= {CNN_NUM_CLASSES}")
    print(f"  THREAT_CLASSES = {THREAT_CLASSES}")
except Exception as e:
    print(f"  Could not import config: {e}")