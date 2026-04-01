"""
data/download_dataset.py — Universal dataset downloader.
Downloads dog behaviour dataset from Roboflow.
Auto-detects workspace and lists available projects if names are wrong.
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
        print("=" * 60)
        print("  ERROR: Set your ROBOFLOW_API_KEY in .env file")
        print("=" * 60)
        print()
        print("  Steps:")
        print("  1. Go to https://app.roboflow.com → Settings → API Keys")
        print("  2. Copy the PRIVATE API Key (NOT the Publishable one)")
        print("     The Private key looks like: Q1g2xxxxxxxxxxxxx")
        print("     The Publishable key (rf_xxx) will NOT work for downloads")
        print("  3. Open .env and set:")
        print("     ROBOFLOW_API_KEY=your_private_key_here")
        print()
        sys.exit(1)

    if ROBOFLOW_API_KEY.startswith("rf_"):
        print("=" * 60)
        print("  ERROR: You are using the PUBLISHABLE API key (rf_...)")
        print("=" * 60)
        print()
        print("  The publishable key does NOT work for dataset downloads.")
        print("  You need the PRIVATE API key instead.")
        print()
        print("  Steps:")
        print("  1. Go to https://app.roboflow.com → Settings → API Keys")
        print("  2. Click the EYE icon next to your Private key to reveal it")
        print("  3. Copy it and paste in .env:")
        print("     ROBOFLOW_API_KEY=your_private_key_here")
        print()
        sys.exit(1)

    from roboflow import Roboflow

    print("[DOWNLOAD] Connecting to Roboflow...")

    try:
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    except Exception as e:
        print(f"\n  ERROR: API key rejected — {e}")
        print("  Make sure you're using the PRIVATE API key (not publishable)")
        sys.exit(1)

    # ── Auto-detect workspace ────────────────────────────────────
    workspace_id = ROBOFLOW_WORKSPACE
    try:
        ws = rf.workspace(workspace_id)
    except Exception:
        # Workspace name is wrong — try to auto-detect
        print(f"\n  Workspace '{workspace_id}' not found.")
        print("  Auto-detecting your workspace...\n")
        try:
            ws = rf.workspace()
            workspace_id = ws.name if hasattr(ws, 'name') else str(ws)
            print(f"  Found workspace: {workspace_id}")
        except Exception:
            print("  Could not auto-detect workspace.")
            print()
            print("  HOW TO FIND YOUR WORKSPACE NAME:")
            print("  1. Go to https://app.roboflow.com")
            print("  2. Look at the URL: app.roboflow.com/YOUR-WORKSPACE-NAME/...")
            print("  3. Copy that name and put it in .env:")
            print("     ROBOFLOW_WORKSPACE=your-workspace-name")
            print()
            print("  From your screenshot, your workspace is:")
            print("     ROBOFLOW_WORKSPACE=durgamanis-workspace-on48g")
            sys.exit(1)

    # ── List available projects ──────────────────────────────────
    project_id = ROBOFLOW_PROJECT
    try:
        project = ws.project(project_id)
    except Exception:
        print(f"\n  Project '{project_id}' not found in workspace '{workspace_id}'.")
        print("  Listing available projects in your workspace...\n")
        try:
            projects = ws.project_list
            if projects:
                print("  Available projects:")
                print("  " + "-" * 50)
                for i, p in enumerate(projects):
                    name = p if isinstance(p, str) else getattr(p, 'id', str(p))
                    print(f"    {i + 1}. {name}")
                print()
                print("  Update .env with the correct project name:")
                print(f"    ROBOFLOW_PROJECT=<project-name-from-above>")
            else:
                print("  No projects found. Create one at https://app.roboflow.com")
        except Exception as e:
            print(f"  Could not list projects: {e}")
            print()
            print("  HOW TO FIND YOUR PROJECT NAME:")
            print("  1. Go to https://app.roboflow.com")
            print("  2. Click on your project")
            print("  3. The URL shows: app.roboflow.com/workspace/PROJECT-NAME")
            print("  4. Put it in .env:")
            print("     ROBOFLOW_PROJECT=your-project-name")
        sys.exit(1)

    # ── Download dataset ─────────────────────────────────────────
    print(f"\n[DOWNLOAD] Found project: {project_id}")
    try:
        version = project.version(ROBOFLOW_VERSION)
    except Exception:
        print(f"\n  Version {ROBOFLOW_VERSION} not found.")
        print("  Available versions:")
        try:
            versions = project.versions()
            for v in versions:
                vid = v if isinstance(v, (int, str)) else getattr(v, 'version', str(v))
                print(f"    Version {vid}")
        except Exception:
            pass
        print(f"\n  Update .env: ROBOFLOW_VERSION=<correct-version>")
        sys.exit(1)

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[DOWNLOAD] Downloading YOLOv8 format to {DATASET_DIR}...")
    dataset = version.download("yolov8", location=str(DATASET_DIR))

    print(f"\n[DOWNLOAD] Dataset downloaded successfully!")
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
