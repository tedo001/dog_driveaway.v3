"""
fix_idle_split.py
Run once from the project root to redistribute IDLE images
from val/IDLE into train/IDLE (80/20 split).
"""

import shutil
import random
from pathlib import Path

CROPS_DIR = Path("data/crops")   # adjust if your path differs

train_idle = CROPS_DIR / "train" / "IDLE"
val_idle   = CROPS_DIR / "val"   / "IDLE"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ── Diagnose first ────────────────────────────────────────────────────────
train_imgs = [f for f in train_idle.iterdir() if f.suffix.lower() in IMG_EXTS]
val_imgs   = [f for f in val_idle.iterdir()   if f.suffix.lower() in IMG_EXTS]

print(f"train/IDLE images : {len(train_imgs)}")
print(f"val/IDLE   images : {len(val_imgs)}")

# ── If train/IDLE is empty, pull 80% from val/IDLE ───────────────────────
if len(train_imgs) == 0 and len(val_imgs) > 0:
    print("\ntrain/IDLE is empty — redistributing from val/IDLE (80/20 split)...")

    random.seed(42)
    random.shuffle(val_imgs)

    split      = int(len(val_imgs) * 0.8)
    to_train   = val_imgs[:split]
    keep_val   = val_imgs[split:]

    train_idle.mkdir(parents=True, exist_ok=True)

    for f in to_train:
        shutil.move(str(f), str(train_idle / f.name))

    print(f"\n✅ Done!")
    print(f"   train/IDLE : {len(to_train)} images")
    print(f"   val/IDLE   : {len(keep_val)} images remaining")

elif len(train_imgs) > 0:
    print("\n✅ train/IDLE already has images — no action needed.")
    print("   If training still fails, check the file extensions:")
    for f in train_imgs[:5]:
        print(f"   {f.name}")

else:
    print("\n❌ Both train/IDLE and val/IDLE are empty!")
    print("   You need to add IDLE images to data/crops/val/IDLE first.")