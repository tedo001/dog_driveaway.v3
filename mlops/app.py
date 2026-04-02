"""
mlops/app.py — MLOps Pipeline Dashboard.

One application to manage everything:
  - Load datasets (ZIP, folder, COCO download)
  - Preprocess (class mapping, crop extraction, train/val split)
  - Train models (YOLO, SSD, CNN)
  - View status and history
  - Clear old models

Usage:
  python mlops/app.py                  # Interactive menu
  python mlops/app.py --load "D:\\dog_cnn"  # Direct: load a dataset
  python mlops/app.py --train all      # Direct: train all models
  python mlops/app.py --status         # Direct: view pipeline status
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlops.state import load_state, save_state, log_event, update_dataset, update_model, get_status_summary
from mlops.data_loader import load_dataset_from_path, download_coco, count_images, detect_format
from mlops.preprocessor import auto_select_mapping, create_custom_mapping, extract_behavior_crops, MAPPING_PRESETS
from mlops.trainer import get_device_info, clear_old_models, train_yolo, train_ssd, train_cnn, train_all


# ── UI Helpers ──────────────────────────────────────────────────────────────

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_banner():
    print()
    print("=" * 60)
    print("     DOG THREAT DETECTION — MLOps Pipeline")
    print("=" * 60)
    device = get_device_info()
    print(f"  Device: {device['display']}")
    print("=" * 60)


def prompt_choice(options, prompt_text="  Select"):
    """Show numbered options and get user choice."""
    print()
    for i, (label, _) in enumerate(options):
        print(f"    [{i + 1}] {label}")
    print(f"    [0] Back / Cancel")
    print()
    while True:
        try:
            choice = input(f"  {prompt_text} (0-{len(options)}): ").strip()
            if choice == "0":
                return None
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx]
        except (ValueError, IndexError):
            pass
        print(f"    Invalid. Enter 0-{len(options)}")


def prompt_path(prompt_text="  Path"):
    """Get a file/folder path from user."""
    while True:
        path = input(f"  {prompt_text}: ").strip().strip('"').strip("'")
        if not path:
            return None
        p = Path(path)
        if p.exists():
            return p
        print(f"    Not found: {path}")
        print(f"    Try again (or press Enter to cancel)")


def prompt_int(prompt_text, default=None):
    """Get an integer from user."""
    suffix = f" [{default}]" if default else ""
    while True:
        val = input(f"  {prompt_text}{suffix}: ").strip()
        if not val and default is not None:
            return default
        try:
            return int(val)
        except ValueError:
            print(f"    Invalid number. Try again.")


# ── Menu Handlers ───────────────────────────────────────────────────────────

def menu_load_data(state):
    """Data loading submenu."""
    options = [
        ("Load from local folder / ZIP  (Roboflow export, custom dataset)", "local"),
        ("Download COCO dog+human subset (for YOLO/SSD detection)", "coco"),
    ]

    choice = prompt_choice(options, "Data source")
    if not choice:
        return

    if choice[1] == "coco":
        _handle_coco_download(state)
    elif choice[1] == "local":
        _handle_local_load(state)


def _handle_coco_download(state):
    """Download COCO dog+human dataset."""
    max_imgs = prompt_int("Max images to download", default=5000)

    print(f"\n  Downloading COCO dog+human subset ({max_imgs} images)...")
    print("  This may take a while on first run...\n")

    result = download_coco(max_images=max_imgs)

    if result["success"]:
        update_dataset(state, "coco", loaded=True, images=max_imgs)
        log_event(state, "COCO Download", f"{max_imgs} images downloaded")
        print(f"\n  COCO dataset ready! {max_imgs} images loaded.")
    else:
        print(f"\n  ERROR: {result.get('error')}")


def _handle_local_load(state):
    """Load dataset from local path."""
    print("\n  Point to your dataset folder or ZIP file.")
    print("  Supports: Roboflow ZIP, YOLO folder, custom nested folders")
    print()

    src = prompt_path("Dataset path (folder or .zip)")
    if not src:
        return

    # Load and detect format
    info = load_dataset_from_path(str(src))

    if not info["success"]:
        print(f"\n  ERROR: {info.get('error')}")
        return

    print(f"\n  Format detected: {info['format']}")
    if info["classes"]:
        print(f"  Classes: {info['classes']}")

    # Ask what to do with this data
    options = [
        ("Use for CNN BEHAVIOR training (crop dogs → classify DANGER/IDLE)", "behavior"),
        ("Use for DETECTION training (YOLO/SSD dog+person)", "detection"),
    ]

    purpose = prompt_choice(options, "Purpose")
    if not purpose:
        return

    if purpose[1] == "behavior":
        _handle_behavior_preprocessing(state, info)
    elif purpose[1] == "detection":
        _handle_detection_copy(state, info)


def _handle_behavior_preprocessing(state, info):
    """Process dataset for CNN behavior training."""
    classes = info.get("classes", [])

    if not classes:
        print("\n  WARNING: No class names detected from data.yaml")
        print("  Enter class names manually (comma-separated):")
        raw = input("  Classes: ").strip()
        classes = [c.strip() for c in raw.split(",") if c.strip()]
        if not classes:
            print("  Cancelled.")
            return

    # Try auto-mapping
    preset_name, mapping = auto_select_mapping(classes)

    if mapping:
        print(f"\n  Auto-detected mapping: {preset_name}")
        print(f"  " + "-" * 40)
        for src_cls, dst_cls in mapping.items():
            print(f"    {src_cls:12s} → {dst_cls}")
        print()

        use_auto = input("  Use this mapping? [Y/n]: ").strip().lower()
        if use_auto in ("n", "no"):
            mapping = None

    if not mapping:
        # Show preset options or custom
        print("\n  Choose a class mapping:")
        opts = []
        for name, preset in MAPPING_PRESETS.items():
            opts.append((f"{name}: {preset['description']}", name))
        opts.append(("Custom mapping (you define each class)", "custom"))

        choice = prompt_choice(opts, "Mapping")
        if not choice:
            return

        if choice[1] == "custom":
            mapping = create_custom_mapping(classes)
        else:
            mapping = MAPPING_PRESETS[choice[1]]["mapping"]
            # Filter to only classes that exist in data
            mapping = {k: v for k, v in mapping.items() if k.lower() in [c.lower() for c in classes]}

    if not mapping:
        print("  No valid mapping. Cancelled.")
        return

    # Clear old crops?
    clear = input("\n  Clear old behavior crops first? [Y/n]: ").strip().lower()
    clear_old = clear not in ("n", "no")

    # Extract crops
    img_dir = info.get("img_dir")
    lbl_dir = info.get("lbl_dir")

    if not img_dir or not lbl_dir:
        print("  ERROR: Could not find images/labels directories")
        return

    result = extract_behavior_crops(
        img_dir=img_dir,
        lbl_dir=lbl_dir,
        class_names=classes,
        class_mapping=mapping,
        train_split=0.8,
        clear_old=clear_old,
    )

    if result["success"]:
        update_dataset(state, "behavior", loaded=True, crops=result["stats"].get("train", {}), path=str(img_dir))
        log_event(state, "Behavior Data Loaded", f"{result['total']} crops extracted")
        print(f"\n  Behavior dataset ready! Run training next.")
    else:
        print(f"\n  ERROR: {result.get('error')}")


def _handle_detection_copy(state, info):
    """Copy dataset for YOLO/SSD detection training."""
    from mlops.data_loader import copy_for_detection

    if not info.get("img_dir") or not info.get("lbl_dir"):
        print("  ERROR: Need images/ and labels/ directories for detection training")
        return

    clear = input("\n  Clear old detection dataset? [Y/n]: ").strip().lower()
    clear_old = clear not in ("n", "no")

    result = copy_for_detection(info, clear_old=clear_old)

    if result["success"]:
        update_dataset(state, "coco", loaded=True, images=result["total"], path=info["img_dir"])
        log_event(state, "Detection Data Loaded", f"{result['total']} images copied")
    else:
        print(f"\n  ERROR: {result.get('error')}")


# ── Training Menu ───────────────────────────────────────────────────────────

def menu_train(state):
    """Training submenu."""
    options = [
        ("Train YOLO  (dog+person detector)", "yolo"),
        ("Train SSD   (dog+person detector)", "ssd"),
        ("Train CNN   (BehaviorNetV2 — DANGER/IDLE classifier)", "cnn"),
        ("Train ALL   (YOLO → SSD → CNN — full pipeline)", "all"),
    ]

    choice = prompt_choice(options, "Train")
    if not choice:
        return

    # Ask for epochs
    from config import YOLO_EPOCHS, SSD_EPOCHS, CNN_EPOCHS
    defaults = {"yolo": YOLO_EPOCHS, "ssd": SSD_EPOCHS, "cnn": CNN_EPOCHS, "all": 60}
    epochs = prompt_int("Epochs", default=defaults.get(choice[1], 60))

    model_name = choice[1]

    if model_name == "all":
        results = train_all(epochs=epochs)
        for name, res in results.items():
            if res.get("success"):
                update_model(state, name, trained=True, epochs=res.get("epochs", epochs),
                             best_metric=res.get("best_val_acc") or res.get("best_val_loss"),
                             path=res.get("model_path", ""))
                log_event(state, f"{name.upper()} Trained", f"epochs={res.get('epochs')}")
    elif model_name == "yolo":
        res = train_yolo(epochs=epochs)
        if res.get("success"):
            update_model(state, "yolo", trained=True, epochs=epochs,
                         best_metric="see runs/", path=res.get("model_path", ""))
            log_event(state, "YOLO Trained", f"epochs={epochs}")
    elif model_name == "ssd":
        res = train_ssd(epochs=epochs)
        if res.get("success"):
            update_model(state, "ssd", trained=True, epochs=res.get("epochs", epochs),
                         best_metric=res.get("best_val_loss"),
                         path=res.get("model_path", ""))
            log_event(state, "SSD Trained", f"epochs={res.get('epochs')}, loss={res.get('best_val_loss')}")
    elif model_name == "cnn":
        res = train_cnn(epochs=epochs)
        if res.get("success"):
            update_model(state, "cnn", trained=True, epochs=res.get("epochs", epochs),
                         best_metric=res.get("best_val_acc"),
                         path=res.get("model_path", ""))
            log_event(state, "CNN Trained", f"epochs={res.get('epochs')}, acc={res.get('best_val_acc')}")


# ── Other Menus ─────────────────────────────────────────────────────────────

def menu_clear_models(state):
    """Clear old model weights."""
    options = [
        ("Clear ALL models", ["yolo", "ssd", "cnn", "cnn_v1", "cnn_v2", "onnx", "scripted"]),
        ("Clear YOLO only", ["yolo"]),
        ("Clear SSD only", ["ssd"]),
        ("Clear CNN only", ["cnn", "cnn_v1", "cnn_v2"]),
    ]

    choice = prompt_choice(options, "Clear")
    if not choice:
        return

    confirm = input(f"\n  Confirm delete {choice[0]}? [y/N]: ").strip().lower()
    if confirm not in ("y", "yes"):
        print("  Cancelled.")
        return

    cleared = clear_old_models(choice[1])
    if cleared:
        log_event(state, "Models Cleared", f"Deleted {len(cleared)} files")
        # Reset model state
        for name in choice[1]:
            if name in state["models"]:
                state["models"][name] = {"trained": False, "epochs": 0, "best_metric": None, "path": ""}
        save_state(state)


def menu_status(state):
    """Show pipeline status."""
    print(get_status_summary(state))
    input("\n  Press Enter to continue...")


def menu_quick_pipeline(state):
    """Run the complete pipeline: data → preprocess → train."""
    print("\n" + "=" * 60)
    print("  QUICK PIPELINE — Complete End-to-End")
    print("=" * 60)
    print()
    print("  This will:")
    print("    1. Ask for your dataset path (ZIP or folder)")
    print("    2. Auto-detect format and map classes")
    print("    3. Extract behavior crops for CNN")
    print("    4. Train ALL models (YOLO → SSD → CNN)")
    print()

    proceed = input("  Continue? [Y/n]: ").strip().lower()
    if proceed in ("n", "no"):
        return

    # Step 1: Load data
    print("\n  STEP 1: Load Dataset")
    print("  " + "-" * 40)
    src = prompt_path("Dataset path (folder or .zip)")
    if not src:
        return

    info = load_dataset_from_path(str(src))
    if not info["success"]:
        print(f"\n  ERROR: {info.get('error')}")
        return

    # Step 2: Preprocess for behavior
    if info.get("classes"):
        print("\n  STEP 2: Preprocess for CNN Behavior")
        print("  " + "-" * 40)
        _handle_behavior_preprocessing(state, info)

    # Step 3: Check if COCO data exists for detection
    from config import DATASET_DIR
    coco_yaml = DATASET_DIR / "data.yaml"
    if not coco_yaml.exists():
        print("\n  STEP 2b: No COCO detection data found!")
        dl = input("  Download COCO dog+human now? [Y/n]: ").strip().lower()
        if dl not in ("n", "no"):
            _handle_coco_download(state)

    # Step 4: Train all
    print("\n  STEP 3: Train All Models")
    print("  " + "-" * 40)
    epochs = prompt_int("Epochs for all models", default=60)

    results = train_all(epochs=epochs)
    for name, res in results.items():
        if res.get("success"):
            update_model(state, name, trained=True, epochs=res.get("epochs", epochs),
                         best_metric=res.get("best_val_acc") or res.get("best_val_loss"),
                         path=res.get("model_path", ""))
            log_event(state, f"{name.upper()} Trained (Pipeline)", f"epochs={res.get('epochs')}")

    print("\n  Pipeline complete! Run 'python main.py' to use the system.")


# ── Main Menu Loop ──────────────────────────────────────────────────────────

def main_menu():
    """Main interactive menu loop."""
    state = load_state()

    while True:
        clear_screen()
        print_banner()

        # Quick status
        coco = state["datasets"]["coco"]
        beh = state["datasets"]["behavior"]
        print(f"\n  Data:   COCO={'LOADED' if coco['loaded'] else 'NO'}  |  "
              f"Behavior={'LOADED' if beh['loaded'] else 'NO'}")
        models_status = []
        for name, info in state["models"].items():
            models_status.append(f"{name.upper()}={'OK' if info['trained'] else 'NO'}")
        print(f"  Models: {' | '.join(models_status)}")

        options = [
            ("Load Data       (ZIP, folder, COCO download)", "load"),
            ("Train Models    (YOLO, SSD, CNN, or ALL)", "train"),
            ("Quick Pipeline  (data → preprocess → train ALL)", "quick"),
            ("Clear Models    (remove old weights)", "clear"),
            ("View Status     (datasets, models, history)", "status"),
            ("Exit", "exit"),
        ]

        choice = prompt_choice(options, "Choose")

        if choice is None or choice[1] == "exit":
            print("\n  Goodbye!\n")
            break
        elif choice[1] == "load":
            menu_load_data(state)
        elif choice[1] == "train":
            menu_train(state)
        elif choice[1] == "quick":
            menu_quick_pipeline(state)
        elif choice[1] == "clear":
            menu_clear_models(state)
        elif choice[1] == "status":
            menu_status(state)

        # Reload state in case it was modified
        state = load_state()


# ── CLI Entry Point ─────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Dog Threat Detection — MLOps Pipeline Dashboard",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--load", type=str, metavar="PATH",
                        help="Load dataset from path (ZIP or folder)")
    parser.add_argument("--train", type=str, choices=["yolo", "ssd", "cnn", "all"],
                        help="Train a specific model or all")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training epochs")
    parser.add_argument("--status", action="store_true",
                        help="Show pipeline status")
    parser.add_argument("--clear", action="store_true",
                        help="Clear all old models")
    parser.add_argument("--coco", type=int, metavar="N",
                        help="Download COCO dog+human dataset (N images)")

    args = parser.parse_args()

    # If no arguments, run interactive menu
    if not any([args.load, args.train, args.status, args.clear, args.coco]):
        main_menu()
        return

    state = load_state()

    # Handle CLI commands
    if args.status:
        print(get_status_summary(state))
        return

    if args.clear:
        clear_old_models()
        log_event(state, "Models Cleared (CLI)", "All models deleted")
        return

    if args.coco:
        result = download_coco(max_images=args.coco)
        if result["success"]:
            update_dataset(state, "coco", loaded=True, images=args.coco)
            log_event(state, "COCO Download (CLI)", f"{args.coco} images")
        return

    if args.load:
        info = load_dataset_from_path(args.load)
        if info["success"]:
            print(f"\n  Dataset loaded: {info['format']}")
            print(f"  Classes: {info.get('classes', [])}")

            # Auto-preprocess for behavior if classes detected
            classes = info.get("classes", [])
            if classes:
                preset_name, mapping = auto_select_mapping(classes)
                if mapping:
                    print(f"  Auto-mapping: {preset_name}")
                    result = extract_behavior_crops(
                        img_dir=info["img_dir"],
                        lbl_dir=info["lbl_dir"],
                        class_names=classes,
                        class_mapping=mapping,
                        clear_old=True,
                    )
                    if result["success"]:
                        update_dataset(state, "behavior", loaded=True,
                                       crops=result["stats"].get("train", {}))
                        log_event(state, "Data Loaded (CLI)", f"{result['total']} crops")
        else:
            print(f"  ERROR: {info.get('error')}")
        return

    if args.train:
        epochs = args.epochs

        if args.train == "all":
            results = train_all(epochs=epochs)
            for name, res in results.items():
                if res.get("success"):
                    update_model(state, name, trained=True, epochs=res.get("epochs", epochs or 60),
                                 best_metric=res.get("best_val_acc") or res.get("best_val_loss"),
                                 path=res.get("model_path", ""))
        elif args.train == "yolo":
            res = train_yolo(epochs=epochs)
            if res.get("success"):
                update_model(state, "yolo", trained=True, epochs=epochs or 60, path=res.get("model_path", ""))
        elif args.train == "ssd":
            res = train_ssd(epochs=epochs)
            if res.get("success"):
                update_model(state, "ssd", trained=True, epochs=res.get("epochs"),
                             best_metric=res.get("best_val_loss"), path=res.get("model_path", ""))
        elif args.train == "cnn":
            res = train_cnn(epochs=epochs)
            if res.get("success"):
                update_model(state, "cnn", trained=True, epochs=res.get("epochs"),
                             best_metric=res.get("best_val_acc"), path=res.get("model_path", ""))


if __name__ == "__main__":
    main()
