"""
data/collect_youtube.py — YouTube frame extractor for building training data.
Downloads videos and extracts frames at configurable intervals.
Requires yt-dlp.
"""

import sys
import os
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATA_DIR


def extract_frames(
    video_url,
    output_dir=None,
    frame_interval=30,
    max_frames=500,
    resolution="480p",
):
    """
    Download a YouTube video and extract frames.

    Args:
        video_url: YouTube video URL.
        output_dir: Directory to save extracted frames.
        frame_interval: Extract every Nth frame (default: every 30th = ~1fps at 30fps).
        max_frames: Maximum frames to extract.
        resolution: Video resolution to download.
    """
    import yt_dlp

    if output_dir is None:
        output_dir = DATA_DIR / "youtube_frames"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download video
    temp_video = output_dir / "temp_video.mp4"
    ydl_opts = {
        "format": f"best[height<={resolution.replace('p', '')}]",
        "outtmpl": str(temp_video),
        "quiet": True,
        "no_warnings": True,
    }

    print(f"[YOUTUBE] Downloading: {video_url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        video_title = info.get("title", "unknown")

    print(f"[YOUTUBE] Video: {video_title}")

    # Extract frames
    cap = cv2.VideoCapture(str(temp_video))
    if not cap.isOpened():
        print("[YOUTUBE] ERROR: Could not open downloaded video")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[YOUTUBE] Total frames: {total_frames}, FPS: {fps:.1f}")

    frame_count = 0
    saved_count = 0
    safe_title = "".join(c if c.isalnum() or c in "-_" else "_" for c in video_title[:30])

    while cap.isOpened() and saved_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            filename = f"{safe_title}_{saved_count:05d}.jpg"
            cv2.imwrite(str(output_dir / filename), frame)
            saved_count += 1

        frame_count += 1

    cap.release()

    # Clean up temp video
    if temp_video.exists():
        os.remove(temp_video)

    print(f"[YOUTUBE] Extracted {saved_count} frames to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract frames from YouTube videos")
    parser.add_argument("url", type=str, help="YouTube video URL")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--interval", type=int, default=30, help="Frame interval")
    parser.add_argument("--max", type=int, default=500, help="Max frames")
    args = parser.parse_args()

    extract_frames(args.url, args.output, args.interval, args.max)
