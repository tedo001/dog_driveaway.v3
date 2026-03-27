"""
data/auto_label.py — YOLO-based auto-labeling for unlabeled dog images.
Runs YOLO on raw images to generate bounding box annotations in YOLO format.
"""

import sys
import cv2
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import YOLO_MODEL_PATH, YOLO_BASE_MODEL


def auto_label(
    image_dir,
    output_dir=None,
    conf_threshold=0.4,
    model_path=None,
):
    """
    Auto-label images using YOLO model.

    Args:
        image_dir: Path to directory containing images.
        output_dir: Path for YOLO-format label files. Defaults to image_dir/../labels.
        conf_threshold: Minimum confidence for detections.
        model_path: Path to YOLO model. Uses trained model if available.
    """
    from ultralytics import YOLO

    image_dir = Path(image_dir)
    if output_dir is None:
        output_dir = image_dir.parent / "labels"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    if model_path:
        model = YOLO(str(model_path))
    elif YOLO_MODEL_PATH.exists():
        model = YOLO(str(YOLO_MODEL_PATH))
    else:
        model = YOLO(YOLO_BASE_MODEL)

    print(f"[AUTO-LABEL] Model loaded")
    print(f"[AUTO-LABEL] Input:  {image_dir}")
    print(f"[AUTO-LABEL] Output: {output_dir}")

    image_files = (
        list(image_dir.glob("*.jpg"))
        + list(image_dir.glob("*.jpeg"))
        + list(image_dir.glob("*.png"))
    )
    print(f"[AUTO-LABEL] Found {len(image_files)} images")

    labeled_count = 0

    for img_path in tqdm(image_files, desc="Auto-labeling"):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        h, w = frame.shape[:2]
        results = model.predict(source=frame, conf=conf_threshold, verbose=False)

        lines = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls = int(box.cls[0])

                # Convert to YOLO format (normalized center x, center y, w, h)
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h

                lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if lines:
            label_path = output_dir / f"{img_path.stem}.txt"
            label_path.write_text("\n".join(lines))
            labeled_count += 1

    print(f"[AUTO-LABEL] Labeled {labeled_count}/{len(image_files)} images")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Auto-label images with YOLO")
    parser.add_argument("image_dir", type=str, help="Directory containing images")
    parser.add_argument("--output", type=str, default=None, help="Output label directory")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    args = parser.parse_args()

    auto_label(args.image_dir, args.output, args.conf)
