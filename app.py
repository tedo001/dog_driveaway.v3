"""
app.py - Smart Dog Threat Detection System - GUI Application.

Run: python app.py

Tkinter GUI with 6 pages:
  1. Dashboard    - system status, GPU/CPU toggle, model graph
  2. Data Pipeline - load dataset, Roboflow API, COCO download
  3. Training     - train YOLO / CNN with progress monitoring
  4. Run Detection - simulation / live / hardware with logging
  5. Retrain      - retrain, fine-tune, add new data
  6. Admin        - API key management, admin password
"""

import sys
import os
import json
import threading
import logging
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Logging Setup ─────────────────────────────────────────────────────────
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"system_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(str(LOG_FILE)),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("DogThreat")

# ── Admin Config ──────────────────────────────────────────────────────────
ADMIN_CONFIG_FILE = PROJECT_ROOT / "admin_config.json"
DEFAULT_ADMIN = {"password": "admin123", "roboflow_api_key": "", "roboflow_workspace": ""}


def load_admin_config():
    if ADMIN_CONFIG_FILE.exists():
        try:
            return json.loads(ADMIN_CONFIG_FILE.read_text())
        except Exception:
            pass
    return dict(DEFAULT_ADMIN)


def save_admin_config(cfg):
    ADMIN_CONFIG_FILE.write_text(json.dumps(cfg, indent=2))


# ── RED Color Theme ───────────────────────────────────────────────────────
BG = "#1a1a1a"
BG_CARD = "#2d1a1a"
FG = "#e0d0d0"
FG_DIM = "#8a7070"
ACCENT = "#cc3333"
ACCENT_LIGHT = "#ff4444"
GREEN = "#4caf50"
RED = "#ff3333"
YELLOW = "#ffaa00"
ORANGE = "#ff6633"
DARK_RED = "#441111"


# ── Main Application ──────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Dog Threat Detection System")
        self.geometry("1050x720")
        self.configure(bg=BG)
        self.resizable(True, True)

        # Sidebar
        sidebar = tk.Frame(self, bg=BG_CARD, width=200)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        tk.Label(sidebar, text="DOG THREAT\nDETECTION", font=("Segoe UI", 14, "bold"),
                 bg=BG_CARD, fg=ACCENT, justify="center").pack(pady=(20, 30))

        self.pages = {}
        self.nav_buttons = {}
        self.current_page = None

        nav_items = [
            ("Dashboard", DashboardPage),
            ("Data Pipeline", DataPage),
            ("Training", TrainingPage),
            ("Run Detection", DetectionPage),
            ("Retrain", RetrainPage),
            ("Admin", AdminPage),
        ]

        for name, page_class in nav_items:
            btn = tk.Button(sidebar, text=name, font=("Segoe UI", 11),
                            bg=BG_CARD, fg=FG, bd=0, anchor="w", padx=20, pady=10,
                            activebackground=ACCENT, activeforeground=BG,
                            cursor="hand2",
                            command=lambda n=name: self.show_page(n))
            btn.pack(fill="x")
            self.nav_buttons[name] = btn

        # Version label at bottom
        tk.Label(sidebar, text="v3.0 CUDA", font=("Segoe UI", 8),
                 bg=BG_CARD, fg=FG_DIM).pack(side="bottom", pady=10)

        # Content area
        self.content = tk.Frame(self, bg=BG)
        self.content.pack(side="right", fill="both", expand=True)

        for name, page_class in nav_items:
            page = page_class(self.content, self)
            page.place(relx=0, rely=0, relwidth=1, relheight=1)
            self.pages[name] = page

        self.show_page("Dashboard")
        logger.info("Application started")

    def show_page(self, name):
        if self.current_page:
            self.nav_buttons[self.current_page].configure(bg=BG_CARD, fg=FG)
        self.current_page = name
        self.nav_buttons[name].configure(bg=ACCENT, fg="#ffffff")
        self.pages[name].tkraise()
        if hasattr(self.pages[name], "on_show"):
            self.pages[name].on_show()


# ── Helper: scrollable log widget ─────────────────────────────────────────

def make_log(parent):
    log = tk.Text(parent, font=("Consolas", 9), bg=BG_CARD, fg=FG,
                  insertbackground=FG, height=14, state="disabled",
                  relief="flat", padx=10, pady=10)
    return log


def log_msg(widget, msg):
    widget.configure(state="normal")
    ts = datetime.now().strftime("%H:%M:%S")
    widget.insert("end", f"[{ts}] {msg}\n")
    widget.see("end")
    widget.configure(state="disabled")


# ── Dashboard Page ────────────────────────────────────────────────────────

class DashboardPage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG)
        self.app = app
        self.gpu_on = tk.BooleanVar(value=True)
        self.cpu_on = tk.BooleanVar(value=False)

        tk.Label(self, text="Dashboard", font=("Segoe UI", 20, "bold"),
                 bg=BG, fg=FG).pack(anchor="w", padx=30, pady=(20, 10))

        # GPU/CPU toggle row
        toggle_frame = tk.Frame(self, bg=BG)
        toggle_frame.pack(fill="x", padx=30, pady=(0, 10))

        tk.Label(toggle_frame, text="Device:", font=("Segoe UI", 11, "bold"),
                 bg=BG, fg=FG).pack(side="left")

        self.gpu_btn = tk.Button(toggle_frame, text="GPU ON", font=("Segoe UI", 10, "bold"),
                                  bg=GREEN, fg="#ffffff", bd=0, padx=15, pady=5, cursor="hand2",
                                  command=self._toggle_gpu)
        self.gpu_btn.pack(side="left", padx=(15, 5))

        self.cpu_btn = tk.Button(toggle_frame, text="CPU OFF", font=("Segoe UI", 10, "bold"),
                                  bg=FG_DIM, fg="#ffffff", bd=0, padx=15, pady=5, cursor="hand2",
                                  command=self._toggle_cpu)
        self.cpu_btn.pack(side="left", padx=5)

        self.device_label = tk.Label(toggle_frame, text="", font=("Segoe UI", 10),
                                      bg=BG, fg=YELLOW)
        self.device_label.pack(side="left", padx=15)

        self.info_frame = tk.Frame(self, bg=BG)
        self.info_frame.pack(fill="both", expand=True, padx=30, pady=10)

        self._init_device()

    def _init_device(self):
        try:
            from config import DEVICE
            if DEVICE != "cpu":
                self.gpu_on.set(True)
                self.cpu_on.set(False)
                self.gpu_btn.configure(text="GPU ON", bg=GREEN)
                self.cpu_btn.configure(text="CPU OFF", bg=FG_DIM)
            else:
                self.gpu_on.set(False)
                self.cpu_on.set(True)
                self.gpu_btn.configure(text="GPU OFF", bg=FG_DIM)
                self.cpu_btn.configure(text="CPU ON", bg=GREEN)
        except Exception:
            pass

    def _toggle_gpu(self):
        import torch
        if torch.cuda.is_available():
            self.gpu_on.set(True)
            self.cpu_on.set(False)
            self.gpu_btn.configure(text="GPU ON", bg=GREEN)
            self.cpu_btn.configure(text="CPU OFF", bg=FG_DIM)
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            self.device_label.configure(text="Switched to GPU", fg=GREEN)
            logger.info("Device switched to GPU")
        else:
            messagebox.showwarning("No GPU", "CUDA GPU not available on this system.")

    def _toggle_cpu(self):
        self.gpu_on.set(False)
        self.cpu_on.set(True)
        self.gpu_btn.configure(text="GPU OFF", bg=FG_DIM)
        self.cpu_btn.configure(text="CPU ON", bg=GREEN)
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        self.device_label.configure(text="Switched to CPU", fg=YELLOW)
        logger.info("Device switched to CPU")

    def on_show(self):
        for w in self.info_frame.winfo_children():
            w.destroy()

        # GPU Card
        gpu_card = self._card("GPU / Device")
        try:
            from config import DEVICE, DEVICE_NAME, DEVICE_VRAM_GB
            if DEVICE != "cpu":
                self._row(gpu_card, "GPU", DEVICE_NAME, GREEN)
                self._row(gpu_card, "VRAM", f"{DEVICE_VRAM_GB} GB", GREEN)
                import torch
                self._row(gpu_card, "CUDA", torch.version.cuda, GREEN)
            else:
                self._row(gpu_card, "Device", "CPU (no GPU)", YELLOW)
        except Exception as e:
            self._row(gpu_card, "Error", str(e), RED)

        # Models Card
        model_card = self._card("Models")
        try:
            from config import YOLO_MODEL_PATH, CNN_MODEL_PATH
            yolo_ok = YOLO_MODEL_PATH.exists()
            cnn_ok = CNN_MODEL_PATH.exists()
            self._row(model_card, "YOLO Detector",
                      "Ready" if yolo_ok else "Not trained",
                      GREEN if yolo_ok else RED)
            self._row(model_card, "CNN Behavior",
                      "Ready" if cnn_ok else "Not trained",
                      GREEN if cnn_ok else RED)

            # Model file sizes
            if yolo_ok:
                sz = YOLO_MODEL_PATH.stat().st_size / (1024 * 1024)
                self._row(model_card, "YOLO Size", f"{sz:.1f} MB", FG)
            if cnn_ok:
                sz = CNN_MODEL_PATH.stat().st_size / (1024 * 1024)
                self._row(model_card, "CNN Size", f"{sz:.1f} MB", FG)
        except Exception as e:
            self._row(model_card, "Error", str(e), RED)

        # Data Card
        data_card = self._card("Data")
        try:
            from config import DATASET_DIR, CROPS_DIR
            det_ok = (DATASET_DIR / "data.yaml").exists()
            crops_ok = (CROPS_DIR / "train").exists()
            self._row(data_card, "Detection Data",
                      "Loaded" if det_ok else "Not loaded",
                      GREEN if det_ok else FG_DIM)
            self._row(data_card, "Behavior Crops",
                      "Loaded" if crops_ok else "Not loaded",
                      GREEN if crops_ok else FG_DIM)

            # Count images if available
            if det_ok:
                train_imgs = DATASET_DIR / "train" / "images"
                if train_imgs.exists():
                    n = len(list(train_imgs.glob("*.*")))
                    self._row(data_card, "Train Images", str(n), FG)
            if crops_ok:
                for cls_dir in sorted((CROPS_DIR / "train").iterdir()):
                    if cls_dir.is_dir():
                        n = len(list(cls_dir.glob("*.*")))
                        self._row(data_card, f"  {cls_dir.name}", str(n), FG)
        except Exception as e:
            self._row(data_card, "Error", str(e), RED)

        # Pipeline Card
        pipe_card = self._card("Pipeline Status")
        try:
            from mlops.state import load_state
            state = load_state()
            for name, info in state["models"].items():
                if info["trained"]:
                    self._row(pipe_card, f"{name.upper()} Training",
                              f"Done (epoch {info['epochs']})", GREEN)
                else:
                    self._row(pipe_card, f"{name.upper()} Training", "Pending", FG_DIM)
        except Exception:
            self._row(pipe_card, "State", "No history yet", FG_DIM)

    def _card(self, title):
        frame = tk.LabelFrame(self.info_frame, text=f"  {title}  ",
                               font=("Segoe UI", 11, "bold"),
                               bg=BG_CARD, fg=ACCENT, bd=1, relief="groove",
                               padx=15, pady=10)
        frame.pack(fill="x", pady=5)
        return frame

    def _row(self, parent, label, value, color=FG):
        row = tk.Frame(parent, bg=BG_CARD)
        row.pack(fill="x", pady=2)
        tk.Label(row, text=label, font=("Segoe UI", 10),
                 bg=BG_CARD, fg=FG_DIM, width=18, anchor="w").pack(side="left")
        tk.Label(row, text=value, font=("Segoe UI", 10, "bold"),
                 bg=BG_CARD, fg=color, anchor="w").pack(side="left")


# ── Data Pipeline Page ────────────────────────────────────────────────────

class DataPage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG)
        self.app = app

        tk.Label(self, text="Data Pipeline", font=("Segoe UI", 20, "bold"),
                 bg=BG, fg=FG).pack(anchor="w", padx=30, pady=(20, 5))

        # Create a scrollable canvas for all the controls
        container = tk.Frame(self, bg=BG)
        container.pack(fill="both", expand=True, padx=30, pady=5)

        # ── Section 1: Load Raw Data ──
        sec1 = tk.LabelFrame(container, text="  Load Raw Data (auto-detect format)  ",
                              font=("Segoe UI", 10, "bold"), bg=BG_CARD, fg=ACCENT,
                              bd=1, relief="groove", padx=10, pady=8)
        sec1.pack(fill="x", pady=5)

        path_row = tk.Frame(sec1, bg=BG_CARD)
        path_row.pack(fill="x", pady=3)
        tk.Label(path_row, text="Path:", font=("Segoe UI", 10), bg=BG_CARD, fg=FG).pack(side="left")
        self.raw_path_var = tk.StringVar()
        tk.Entry(path_row, textvariable=self.raw_path_var, font=("Segoe UI", 10),
                 bg=DARK_RED, fg=FG, insertbackground=FG, width=45).pack(side="left", padx=8, fill="x", expand=True)
        tk.Button(path_row, text="Browse", font=("Segoe UI", 9), bg=ACCENT, fg="#fff",
                  bd=0, padx=10, cursor="hand2", command=self._browse_raw).pack(side="left")
        tk.Button(path_row, text="Load & Process", font=("Segoe UI", 9, "bold"), bg=GREEN, fg="#fff",
                  bd=0, padx=12, cursor="hand2", command=self._load_raw).pack(side="left", padx=5)

        # Custom classes input
        cls_row = tk.Frame(sec1, bg=BG_CARD)
        cls_row.pack(fill="x", pady=3)
        tk.Label(cls_row, text="Custom Classes:", font=("Segoe UI", 10), bg=BG_CARD, fg=FG).pack(side="left")
        self.custom_classes_var = tk.StringVar(value="")
        tk.Entry(cls_row, textvariable=self.custom_classes_var, font=("Segoe UI", 10),
                 bg=DARK_RED, fg=FG, insertbackground=FG, width=40).pack(side="left", padx=8, fill="x", expand=True)
        tk.Label(cls_row, text="(comma separated, leave empty for auto)", font=("Segoe UI", 8),
                 bg=BG_CARD, fg=FG_DIM).pack(side="left")

        # ── Section 2: Pre-Labeled Roboflow YOLOv8 Data ──
        sec2 = tk.LabelFrame(container, text="  Pre-Labeled Data (any format → auto-detect → loads for YOLO + CNN)  ",
                              font=("Segoe UI", 10, "bold"), bg=BG_CARD, fg=ACCENT,
                              bd=1, relief="groove", padx=10, pady=8)
        sec2.pack(fill="x", pady=5)

        labeled_row = tk.Frame(sec2, bg=BG_CARD)
        labeled_row.pack(fill="x", pady=3)
        tk.Label(labeled_row, text="Path:", font=("Segoe UI", 10), bg=BG_CARD, fg=FG).pack(side="left")
        self.labeled_path_var = tk.StringVar()
        tk.Entry(labeled_row, textvariable=self.labeled_path_var, font=("Segoe UI", 10),
                 bg=DARK_RED, fg=FG, insertbackground=FG, width=45).pack(side="left", padx=8, fill="x", expand=True)
        tk.Button(labeled_row, text="Browse", font=("Segoe UI", 9), bg=ACCENT, fg="#fff",
                  bd=0, padx=10, cursor="hand2", command=self._browse_labeled).pack(side="left")
        tk.Button(labeled_row, text="Load for Training", font=("Segoe UI", 9, "bold"), bg=GREEN, fg="#fff",
                  bd=0, padx=12, cursor="hand2", command=self._load_labeled).pack(side="left", padx=5)

        # Target model selector
        target_row = tk.Frame(sec2, bg=BG_CARD)
        target_row.pack(fill="x", pady=3)
        tk.Label(target_row, text="Load for:", font=("Segoe UI", 10), bg=BG_CARD, fg=FG).pack(side="left")
        self.load_target_var = tk.StringVar(value="both")
        for val, txt in [("both", "YOLO + CNN"), ("yolo", "YOLO only"), ("cnn", "CNN only")]:
            tk.Radiobutton(target_row, text=txt, variable=self.load_target_var, value=val,
                           font=("Segoe UI", 9), bg=BG_CARD, fg=FG, selectcolor=DARK_RED,
                           activebackground=BG_CARD, activeforeground=FG).pack(side="left", padx=8)

        # ── Section 3: COCO Download ──
        sec3 = tk.LabelFrame(container, text="  COCO Download  ",
                              font=("Segoe UI", 10, "bold"), bg=BG_CARD, fg=ACCENT,
                              bd=1, relief="groove", padx=10, pady=8)
        sec3.pack(fill="x", pady=5)

        coco_row = tk.Frame(sec3, bg=BG_CARD)
        coco_row.pack(fill="x", pady=3)
        tk.Label(coco_row, text="Max Images:", font=("Segoe UI", 10), bg=BG_CARD, fg=FG).pack(side="left")
        self.coco_count_var = tk.StringVar(value="5000")
        tk.Entry(coco_row, textvariable=self.coco_count_var, font=("Segoe UI", 10),
                 bg=DARK_RED, fg=FG, insertbackground=FG, width=8).pack(side="left", padx=8)
        tk.Button(coco_row, text="Download COCO", font=("Segoe UI", 9, "bold"), bg=ORANGE, fg="#fff",
                  bd=0, padx=12, cursor="hand2", command=self._download_coco).pack(side="left", padx=5)

        # ── Section 4: Roboflow API Download ──
        sec4 = tk.LabelFrame(container, text="  Roboflow API Download  ",
                              font=("Segoe UI", 10, "bold"), bg=BG_CARD, fg=ACCENT,
                              bd=1, relief="groove", padx=10, pady=8)
        sec4.pack(fill="x", pady=5)

        rf_row1 = tk.Frame(sec4, bg=BG_CARD)
        rf_row1.pack(fill="x", pady=2)
        tk.Label(rf_row1, text="Dataset URL:", font=("Segoe UI", 10), bg=BG_CARD, fg=FG).pack(side="left")
        self.rf_url_var = tk.StringVar()
        tk.Entry(rf_row1, textvariable=self.rf_url_var, font=("Segoe UI", 10),
                 bg=DARK_RED, fg=FG, insertbackground=FG, width=50).pack(side="left", padx=8, fill="x", expand=True)

        rf_row2 = tk.Frame(sec4, bg=BG_CARD)
        rf_row2.pack(fill="x", pady=2)
        tk.Label(rf_row2, text="Format:", font=("Segoe UI", 10), bg=BG_CARD, fg=FG).pack(side="left")
        self.rf_format_var = tk.StringVar(value="yolov8")
        for fmt in ["yolov8", "yolov5", "coco", "voc"]:
            tk.Radiobutton(rf_row2, text=fmt, variable=self.rf_format_var, value=fmt,
                           font=("Segoe UI", 9), bg=BG_CARD, fg=FG, selectcolor=DARK_RED,
                           activebackground=BG_CARD, activeforeground=FG).pack(side="left", padx=5)

        tk.Button(sec4, text="Download from Roboflow", font=("Segoe UI", 9, "bold"),
                  bg=ACCENT, fg="#fff", bd=0, padx=12, pady=5, cursor="hand2",
                  command=self._download_roboflow).pack(anchor="w", pady=5)

        # ── Log ──
        self.log = make_log(container)
        self.log.pack(fill="both", expand=True, pady=(5, 5))

    def _browse_raw(self):
        path = filedialog.askdirectory(title="Select raw dataset folder")
        if not path:
            path = filedialog.askopenfilename(title="Select dataset ZIP",
                                               filetypes=[("ZIP files", "*.zip")])
        if path:
            self.raw_path_var.set(path)

    def _browse_labeled(self):
        path = filedialog.askdirectory(title="Select pre-labeled Roboflow YOLOv8 folder")
        if path:
            self.labeled_path_var.set(path)

    def _validate_labeled(self, path):
        """Check if data is labeled: needs images + labels dirs with matching files."""
        p = Path(path)
        has_labels = False
        for lbl_name in ["labels", "label"]:
            for sub in [p / "train" / lbl_name, p / lbl_name]:
                if sub.exists() and len(list(sub.glob("*.txt"))) > 0:
                    has_labels = True
                    break
        return has_labels

    def _load_raw(self):
        path = self.raw_path_var.get().strip()
        if not path:
            messagebox.showwarning("No path", "Select a dataset folder or ZIP file.")
            return

        def run():
            try:
                log_msg(self.log, f"Loading: {path}")
                logger.info(f"Data pipeline: loading raw data from {path}")
                from mlops.data_loader import load_dataset_from_path
                from mlops.preprocessor import auto_select_mapping, extract_behavior_crops
                from mlops.state import load_state, update_dataset, log_event

                info = load_dataset_from_path(path)
                if not info["success"]:
                    log_msg(self.log, f"ERROR: {info.get('error')}")
                    return

                log_msg(self.log, f"Format: {info['format']}")
                log_msg(self.log, f"Classes: {info.get('classes', [])}")

                # Use custom classes if provided
                custom = self.custom_classes_var.get().strip()
                if custom:
                    classes = [c.strip() for c in custom.split(",") if c.strip()]
                    log_msg(self.log, f"Using custom classes: {classes}")
                else:
                    classes = info.get("classes", [])

                if classes and info.get("img_dir") and info.get("lbl_dir"):
                    preset_name, mapping = auto_select_mapping(classes)
                    if mapping:
                        log_msg(self.log, f"Auto-mapping: {preset_name}")
                        for src, dst in mapping.items():
                            log_msg(self.log, f"  {src} -> {dst}")

                        log_msg(self.log, "Extracting behavior crops...")
                        result = extract_behavior_crops(
                            img_dir=info["img_dir"], lbl_dir=info["lbl_dir"],
                            class_names=classes, class_mapping=mapping, clear_old=True,
                            all_splits=info.get("all_splits"),
                        )
                        if result["success"]:
                            state = load_state()
                            update_dataset(state, "behavior", loaded=True,
                                           crops=result["stats"].get("train", {}))
                            log_event(state, "Data Loaded", f"{result['total']} crops")
                            log_msg(self.log, f"Done! {result['total']} crops extracted.")
                            logger.info(f"Behavior crops extracted: {result['total']}")
                        else:
                            log_msg(self.log, f"Crop ERROR: {result.get('error')}")
                    else:
                        log_msg(self.log, "No auto-mapping found. Using as detection data.")
                        self._copy_detection(info)
                else:
                    self._copy_detection(info)

                log_msg(self.log, "Pipeline complete.")
            except Exception as e:
                log_msg(self.log, f"ERROR: {e}")
                logger.error(f"Data pipeline error: {e}")

        threading.Thread(target=run, daemon=True).start()

    def _copy_detection(self, info):
        from mlops.data_loader import copy_for_detection
        from mlops.state import load_state, update_dataset, log_event
        if info.get("img_dir") and info.get("lbl_dir"):
            log_msg(self.log, "Copying for YOLO training...")
            result = copy_for_detection(info, clear_old=True)
            if result["success"]:
                state = load_state()
                update_dataset(state, "detection", loaded=True, images=result["total"])
                log_event(state, "Detection Data", f"{result['total']} images")
                log_msg(self.log, f"Done! {result['total']} images ready.")
                logger.info(f"Detection data loaded: {result['total']} images")

    def _load_labeled(self):
        path = self.labeled_path_var.get().strip()
        if not path:
            messagebox.showwarning("No path", "Select a pre-labeled data folder.")
            return

        if not self._validate_labeled(path):
            messagebox.showerror("Not Labeled",
                "This data does not appear to be labeled.\n"
                "Expected: images + labels folders with .txt files.\n\n"
                "Unlabeled data is rejected. Please use labeled data.")
            log_msg(self.log, "REJECTED: Data is not labeled. Need labels/ with .txt files.")
            return

        target = self.load_target_var.get()  # "both", "yolo", or "cnn"

        def run():
            try:
                log_msg(self.log, f"Loading pre-labeled data: {path}")
                log_msg(self.log, f"Target: {target.upper()}")
                logger.info(f"Loading pre-labeled data from {path} for {target}")
                from mlops.data_loader import load_dataset_from_path, copy_for_detection
                from mlops.preprocessor import auto_select_mapping, extract_behavior_crops
                from mlops.state import load_state, update_dataset, log_event

                info = load_dataset_from_path(path)
                if not info["success"]:
                    log_msg(self.log, f"ERROR: {info.get('error')}")
                    return

                log_msg(self.log, f"Format: {info['format']} | Classes: {info.get('classes', [])}")
                state = load_state()

                # Load for YOLO training
                if target in ("both", "yolo"):
                    if info.get("img_dir") and info.get("lbl_dir"):
                        log_msg(self.log, "Copying to YOLO training directory...")
                        result = copy_for_detection(info, clear_old=True)
                        if result["success"]:
                            update_dataset(state, "detection", loaded=True, images=result["total"])
                            log_event(state, "Labeled Data Loaded", f"{result['total']} images")
                            log_msg(self.log, f"YOLO data ready: {result['total']} images")
                        else:
                            log_msg(self.log, f"YOLO copy error: {result.get('error')}")
                    else:
                        log_msg(self.log, "WARNING: No image/label dirs found for YOLO")

                # Load for CNN training (extract behavior crops)
                if target in ("both", "cnn"):
                    classes = info.get("classes", [])
                    if classes and info.get("img_dir") and info.get("lbl_dir"):
                        preset_name, mapping = auto_select_mapping(classes)
                        if mapping:
                            log_msg(self.log, f"Extracting CNN crops ({preset_name})...")
                            crop_result = extract_behavior_crops(
                                img_dir=info["img_dir"], lbl_dir=info["lbl_dir"],
                                class_names=classes, class_mapping=mapping, clear_old=True,
                                all_splits=info.get("all_splits"),
                            )
                            if crop_result["success"]:
                                update_dataset(state, "behavior", loaded=True,
                                               crops=crop_result["stats"].get("train", {}))
                                log_msg(self.log, f"CNN crops: {crop_result['total']} extracted")
                            else:
                                log_msg(self.log, f"CNN crop error: {crop_result.get('error')}")
                        else:
                            log_msg(self.log, "No class mapping found for CNN. Data loaded for YOLO only.")
                    else:
                        log_msg(self.log, "No classes detected for CNN crop extraction.")

                log_msg(self.log, "Pre-labeled data loaded successfully.")
                logger.info("Pre-labeled data pipeline complete")
            except Exception as e:
                log_msg(self.log, f"ERROR: {e}")
                logger.error(f"Labeled data error: {e}")

        threading.Thread(target=run, daemon=True).start()

    def _download_coco(self):
        try:
            count = int(self.coco_count_var.get().strip())
        except ValueError:
            messagebox.showwarning("Invalid", "Enter a valid number for max images.")
            return

        def run():
            try:
                log_msg(self.log, f"Downloading COCO ({count} images)... This may take a while.")
                logger.info(f"Starting COCO download: {count} images")
                from mlops.data_loader import download_coco
                from mlops.state import load_state, update_dataset, log_event

                result = download_coco(max_images=count)
                if result["success"]:
                    state = load_state()
                    update_dataset(state, "detection", loaded=True, images=count)
                    log_event(state, "COCO Download", f"{count} images")
                    log_msg(self.log, f"COCO dataset ready! {count} images.")
                    logger.info(f"COCO download complete: {count} images")
                else:
                    log_msg(self.log, f"ERROR: {result.get('error')}")
            except Exception as e:
                log_msg(self.log, f"ERROR: {e}")

        threading.Thread(target=run, daemon=True).start()

    def _download_roboflow(self):
        url = self.rf_url_var.get().strip()
        fmt = self.rf_format_var.get()
        cfg = load_admin_config()
        api_key = cfg.get("roboflow_api_key", "")

        if not api_key:
            messagebox.showwarning("No API Key",
                "Roboflow API key not set.\nGo to Admin tab to set it.")
            return
        if not url:
            messagebox.showwarning("No URL", "Enter a Roboflow dataset URL or ID.")
            return

        def run():
            try:
                log_msg(self.log, f"Downloading from Roboflow ({fmt})...")
                logger.info(f"Roboflow download: {url} format={fmt}")

                from roboflow import Roboflow
                rf = Roboflow(api_key=api_key)

                # Parse URL: workspace/project/version
                parts = url.strip("/").split("/")
                if len(parts) >= 3:
                    workspace = parts[-3] if len(parts) >= 3 else parts[0]
                    project_name = parts[-2]
                    version = int(parts[-1])
                elif len(parts) == 2:
                    workspace = cfg.get("roboflow_workspace", parts[0])
                    project_name = parts[0]
                    version = int(parts[1])
                else:
                    log_msg(self.log, "ERROR: URL format should be workspace/project/version")
                    return

                project = rf.workspace(workspace).project(project_name)
                dataset = project.version(version).download(fmt,
                    location=str(PROJECT_ROOT / "data" / "roboflow_download"))

                log_msg(self.log, f"Downloaded to: {dataset.location}")
                log_msg(self.log, "Now load it using the 'Pre-Labeled Data' section above.")
                logger.info(f"Roboflow download complete: {dataset.location}")

            except ImportError:
                log_msg(self.log, "ERROR: pip install roboflow  (package not installed)")
            except Exception as e:
                log_msg(self.log, f"ERROR: {e}")
                logger.error(f"Roboflow download error: {e}")

        threading.Thread(target=run, daemon=True).start()


# ── Training Page ─────────────────────────────────────────────────────────

class TrainingPage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG)
        self.app = app
        self.training = False

        tk.Label(self, text="Training", font=("Segoe UI", 20, "bold"),
                 bg=BG, fg=FG).pack(anchor="w", padx=30, pady=(20, 5))
        tk.Label(self, text="Train YOLO detector and CNN behavior classifier",
                 font=("Segoe UI", 10), bg=BG, fg=FG_DIM).pack(anchor="w", padx=30, pady=(0, 10))

        # Settings
        settings_frame = tk.Frame(self, bg=BG)
        settings_frame.pack(fill="x", padx=30)

        tk.Label(settings_frame, text="Epochs:", font=("Segoe UI", 10),
                 bg=BG, fg=FG).pack(side="left")
        self.epochs_var = tk.StringVar(value="60")
        tk.Entry(settings_frame, textvariable=self.epochs_var, width=6,
                 font=("Segoe UI", 10), bg=DARK_RED, fg=FG,
                 insertbackground=FG).pack(side="left", padx=10)

        tk.Label(settings_frame, text="Batch:", font=("Segoe UI", 10),
                 bg=BG, fg=FG).pack(side="left", padx=(15, 0))
        self.batch_var = tk.StringVar(value="16")
        tk.Entry(settings_frame, textvariable=self.batch_var, width=6,
                 font=("Segoe UI", 10), bg=DARK_RED, fg=FG,
                 insertbackground=FG).pack(side="left", padx=10)

        # Buttons
        btn_frame = tk.Frame(self, bg=BG)
        btn_frame.pack(fill="x", padx=30, pady=10)

        self.btn_yolo = tk.Button(btn_frame, text="Train YOLO", font=("Segoe UI", 11, "bold"),
                                   bg=ACCENT, fg="#fff", bd=0, padx=20, pady=10, cursor="hand2",
                                   command=lambda: self._train("yolo"))
        self.btn_yolo.pack(side="left", padx=5)

        self.btn_cnn = tk.Button(btn_frame, text="Train CNN", font=("Segoe UI", 11, "bold"),
                                  bg=ACCENT, fg="#fff", bd=0, padx=20, pady=10, cursor="hand2",
                                  command=lambda: self._train("cnn"))
        self.btn_cnn.pack(side="left", padx=5)

        self.btn_all = tk.Button(btn_frame, text="Train ALL", font=("Segoe UI", 11, "bold"),
                                  bg=GREEN, fg="#fff", bd=0, padx=20, pady=10, cursor="hand2",
                                  command=lambda: self._train("all"))
        self.btn_all.pack(side="left", padx=5)

        self.btn_clear = tk.Button(btn_frame, text="Clear Models", font=("Segoe UI", 10),
                  bg=RED, fg="#fff", bd=0, padx=15, pady=10, cursor="hand2",
                  command=self._clear_models)
        self.btn_clear.pack(side="right", padx=5)

        # Progress
        self.progress_var = tk.DoubleVar(value=0)
        style = ttk.Style()
        style.theme_use("default")
        style.configure("red.Horizontal.TProgressbar", troughcolor=DARK_RED,
                         background=ACCENT, thickness=20)
        self.progress_bar = ttk.Progressbar(self, variable=self.progress_var,
                                             maximum=100, style="red.Horizontal.TProgressbar")
        self.progress_bar.pack(fill="x", padx=30, pady=5)

        self.progress_label = tk.Label(self, text="", font=("Segoe UI", 10),
                                        bg=BG, fg=YELLOW)
        self.progress_label.pack(anchor="w", padx=30)

        # Log
        self.log = make_log(self)
        self.log.pack(fill="both", expand=True, padx=30, pady=(5, 20))

    def _set_buttons(self, enabled):
        state = "normal" if enabled else "disabled"
        self.btn_yolo.configure(state=state)
        self.btn_cnn.configure(state=state)
        self.btn_all.configure(state=state)

    def _train(self, model_name):
        if self.training:
            messagebox.showinfo("Busy", "Training already in progress.")
            return

        try:
            epochs = int(self.epochs_var.get())
        except ValueError:
            messagebox.showwarning("Invalid", "Epochs must be a number.")
            return

        self.training = True
        self._set_buttons(False)
        self.progress_var.set(0)

        def progress_cb(epoch, total, train_metric, val_metric):
            pct = (epoch / total) * 100
            self.progress_var.set(pct)
            self.progress_label.configure(
                text=f"Epoch {epoch}/{total} | Train: {train_metric:.4f} | Val: {val_metric:.4f}")
            log_msg(self.log, f"Epoch {epoch}/{total} | Train: {train_metric:.4f} | Val: {val_metric:.4f}")

        def run():
            try:
                from mlops.trainer import train_yolo, train_cnn, train_all
                from mlops.state import load_state, update_model, log_event

                state = load_state()

                if model_name == "yolo":
                    log_msg(self.log, f"Training YOLO ({epochs} epochs)...")
                    logger.info(f"Started YOLO training: {epochs} epochs")
                    res = train_yolo(epochs=epochs, progress_callback=progress_cb)
                    if res["success"]:
                        update_model(state, "yolo", trained=True, epochs=epochs,
                                     path=res.get("model_path", ""))
                        log_event(state, "YOLO Trained", f"epochs={epochs}")
                        log_msg(self.log, f"YOLO done! Model: {res['model_path']}")
                        logger.info(f"YOLO training complete: {res['model_path']}")
                    else:
                        log_msg(self.log, f"YOLO FAILED: {res.get('error')}")
                        logger.error(f"YOLO training failed: {res.get('error')}")

                elif model_name == "cnn":
                    log_msg(self.log, f"Training CNN BehaviorNetV2 ({epochs} epochs)...")
                    logger.info(f"Started CNN training: {epochs} epochs")

                    # Check if crops exist
                    from config import CROPS_DIR
                    train_dir = CROPS_DIR / "train"
                    if not train_dir.exists() or not any(train_dir.iterdir()):
                        log_msg(self.log, "ERROR: No behavior crops found!")
                        log_msg(self.log, "Go to Data Pipeline → load labeled data first.")
                        log_msg(self.log, "Need: data/crops/train/DANGER/ and data/crops/train/IDLE/")
                        self.training = False
                        self._set_buttons(True)
                        return

                    res = train_cnn(epochs=epochs, progress_callback=progress_cb)
                    if res["success"]:
                        update_model(state, "cnn", trained=True, epochs=res["epochs"],
                                     best_metric=res.get("best_val_acc"),
                                     path=res.get("model_path", ""))
                        log_event(state, "CNN Trained", f"acc={res.get('best_val_acc')}")
                        log_msg(self.log, f"CNN done! Best acc: {res.get('best_val_acc')}")
                        logger.info(f"CNN training complete: acc={res.get('best_val_acc')}")
                    else:
                        log_msg(self.log, f"CNN FAILED: {res.get('error')}")
                        logger.error(f"CNN training failed: {res.get('error')}")

                elif model_name == "all":
                    log_msg(self.log, f"Training ALL ({epochs} epochs each)...")
                    logger.info(f"Started full training: {epochs} epochs")
                    results = train_all(epochs=epochs, progress_callback=progress_cb)
                    for name, res in results.items():
                        if res.get("success"):
                            update_model(state, name, trained=True,
                                         epochs=res.get("epochs", epochs),
                                         best_metric=res.get("best_val_acc"),
                                         path=res.get("model_path", ""))
                            log_event(state, f"{name.upper()} Trained", "via Train ALL")
                            log_msg(self.log, f"  {name.upper()}: Done!")
                        else:
                            log_msg(self.log, f"  {name.upper()}: FAILED - {res.get('error')}")

                self.progress_var.set(100)
                self.progress_label.configure(text="Training complete!")
            except Exception as e:
                log_msg(self.log, f"ERROR: {e}")
                logger.error(f"Training error: {e}")
            finally:
                self.training = False
                self._set_buttons(True)

        threading.Thread(target=run, daemon=True).start()

    def _clear_models(self):
        # Show available models first
        from config import YOLO_MODEL_PATH, CNN_MODEL_PATH
        models = []
        if YOLO_MODEL_PATH.exists():
            sz = YOLO_MODEL_PATH.stat().st_size / (1024 * 1024)
            models.append(f"YOLO: {YOLO_MODEL_PATH.name} ({sz:.1f} MB)")
        if CNN_MODEL_PATH.exists():
            sz = CNN_MODEL_PATH.stat().st_size / (1024 * 1024)
            models.append(f"CNN: {CNN_MODEL_PATH.name} ({sz:.1f} MB)")

        if not models:
            messagebox.showinfo("Clear Models", "No trained models found.")
            return

        model_list = "\n".join(models)

        # Ask for admin password
        pwd = simpledialog.askstring("Admin Password",
            f"Models found:\n{model_list}\n\nEnter admin password to delete:",
            show="*", parent=self)

        if pwd is None:
            return

        cfg = load_admin_config()
        if pwd != cfg.get("password", "admin123"):
            messagebox.showerror("Wrong Password", "Incorrect admin password.")
            return

        if messagebox.askyesno("Confirm", f"Delete ALL trained models?\n\n{model_list}"):
            from mlops.trainer import clear_old_models
            cleared = clear_old_models()
            log_msg(self.log, f"Cleared {len(cleared)} model files.")
            logger.info(f"Models cleared: {cleared}")


# ── Detection Page ────────────────────────────────────────────────────────

class DetectionPage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG)
        self.app = app
        self.running = False

        tk.Label(self, text="Run Detection", font=("Segoe UI", 20, "bold"),
                 bg=BG, fg=FG).pack(anchor="w", padx=30, pady=(20, 5))
        tk.Label(self, text="Choose a mode to start the detection system",
                 font=("Segoe UI", 10), bg=BG, fg=FG_DIM).pack(anchor="w", padx=30, pady=(0, 10))

        # Mode cards
        modes = [
            ("Simulation", "No camera needed - animated test scenarios.\n"
             "Controls: 1-6 switch scenario, SPACE pause, Q quit.", GREEN, self._run_simulation),
            ("Live Camera", "Webcam + YOLO detection + CNN behavior + ultrasonic.\n"
             "Requires: trained models + webcam.", ACCENT, self._run_live),
            ("Hardware", "Webcam + Arduino + external ultrasonic sensor.\n"
             "Requires: trained models + webcam + Arduino.", ORANGE, self._run_hardware),
        ]

        for title, desc, color, cmd in modes:
            card = tk.Frame(self, bg=BG_CARD, padx=20, pady=12)
            card.pack(fill="x", padx=30, pady=4)

            header = tk.Frame(card, bg=BG_CARD)
            header.pack(fill="x")

            tk.Label(header, text=title, font=("Segoe UI", 13, "bold"),
                     bg=BG_CARD, fg=color).pack(side="left")

            tk.Button(header, text="START", font=("Segoe UI", 10, "bold"),
                      bg=color, fg="#fff", bd=0, padx=20, pady=5, cursor="hand2",
                      command=cmd).pack(side="right")

            tk.Label(card, text=desc, font=("Segoe UI", 9),
                     bg=BG_CARD, fg=FG_DIM, justify="left").pack(anchor="w", pady=(5, 0))

        # Runtime log
        tk.Label(self, text="Runtime Log", font=("Segoe UI", 10, "bold"),
                 bg=BG, fg=ACCENT).pack(anchor="w", padx=30, pady=(10, 0))
        self.log = make_log(self)
        self.log.configure(height=8)
        self.log.pack(fill="both", expand=True, padx=30, pady=(2, 15))

        self.status_label = tk.Label(self, text="", font=("Segoe UI", 10),
                                      bg=BG, fg=YELLOW)
        self.status_label.pack(anchor="w", padx=30, pady=(0, 10))

    def _run_simulation(self):
        if self.running:
            return
        self.running = True
        self.status_label.configure(text="Starting simulation...")
        log_msg(self.log, "Starting simulation mode...")
        logger.info("Detection: simulation started")
        threading.Thread(target=self._run_detection_mode, args=("simulation",), daemon=True).start()

    def _run_live(self):
        if self.running:
            return
        from config import YOLO_MODEL_PATH
        if not YOLO_MODEL_PATH.exists():
            messagebox.showwarning("No Model", "YOLO model not trained yet.\nGo to Training tab first.")
            return
        self.running = True
        self.status_label.configure(text="Starting live detection...")
        log_msg(self.log, "Starting live camera mode...")
        logger.info("Detection: live mode started")
        threading.Thread(target=self._run_detection_mode, args=("live",), daemon=True).start()

    def _run_hardware(self):
        if self.running:
            return
        from config import YOLO_MODEL_PATH
        if not YOLO_MODEL_PATH.exists():
            messagebox.showwarning("No Model", "YOLO model not trained yet.\nGo to Training tab first.")
            return
        self.running = True
        self.status_label.configure(text="Starting hardware mode...")
        log_msg(self.log, "Starting hardware mode...")
        logger.info("Detection: hardware mode started")
        threading.Thread(target=self._run_detection_mode, args=("hardware",), daemon=True).start()

    def _run_detection_mode(self, mode):
        try:
            import cv2
            import time as _time

            if mode == "simulation":
                from config import SIM_FPS, WINDOW_NAME
                from simulation.simulator import Simulator
                from simulation.scenarios import ScenarioManager, update_scenario_behavior

                sim = Simulator()
                current_scenario = 1
                scenario_name = ScenarioManager.load(sim, current_scenario)
                frame_delay = 1.0 / SIM_FPS
                frame_n = 0

                while True:
                    start = _time.time()
                    if not sim.paused:
                        update_scenario_behavior(sim, current_scenario)
                        sim.frame_count += 1
                        frame_n += 1

                    result, cnn_results, audio_state = sim.classify_threats()
                    dt = frame_delay if not sim.paused else 0
                    sim.update_ultrasonic(result, dt)

                    # Log every 60 frames
                    if frame_n % 60 == 0 and frame_n > 0:
                        threat = result.get("threat_label", "NONE")
                        conf = result.get("confidence", 0)
                        log_msg(self.log, f"Frame {frame_n} | Threat: {threat} ({conf:.2f})")
                        logger.info(f"Sim frame {frame_n}: {threat} ({conf:.2f})")

                    canvas = sim.render()
                    canvas = sim.draw_hud(canvas, result, audio_state, scenario_name)
                    cv2.imshow(WINDOW_NAME, canvas)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord(" "):
                        sim.paused = not sim.paused
                    elif key == ord("r"):
                        scenario_name = ScenarioManager.load(sim, current_scenario)
                    elif ord("1") <= key <= ord("6"):
                        current_scenario = key - ord("0")
                        scenario_name = ScenarioManager.load(sim, current_scenario)
                        log_msg(self.log, f"Scenario switched to {current_scenario}: {scenario_name}")

                    elapsed = _time.time() - start
                    if frame_delay - elapsed > 0:
                        _time.sleep(frame_delay - elapsed)

                cv2.destroyAllWindows()

            elif mode in ("live", "hardware"):
                from models.behavior_net_v2 import BehaviorClassifierV2
                from models.yolo_model import DualYOLODetector
                from models.threat_engine import ThreatEngine
                from audio.audio_detector import AudioDetector
                from audio.audio_combiner import AudioCombiner
                from utils.ui import UIRenderer
                from config import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT

                detector = DualYOLODetector()
                cnn = BehaviorClassifierV2()
                engine = ThreatEngine()
                audio_detector = AudioDetector()
                audio_combiner = AudioCombiner()
                audio_detector.start()
                ui = UIRenderer()

                if mode == "hardware":
                    from hardware.ultrasonic_hw import UltrasonicHardware
                    ultrasonic = UltrasonicHardware()
                else:
                    from audio.ultrasonic_trigger import UltrasonicTrigger
                    ultrasonic = UltrasonicTrigger()

                cap = cv2.VideoCapture(CAMERA_INDEX)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

                if not cap.isOpened():
                    self.status_label.configure(text="ERROR: Could not open camera!")
                    log_msg(self.log, "ERROR: Camera not available")
                    self.running = False
                    return

                frame_n = 0
                try:
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            _time.sleep(0.1)
                            continue

                        frame_n += 1
                        detections = detector.detect(frame)
                        num_dogs = sum(1 for d in detections if d["class"] == "dog")
                        num_humans = sum(1 for d in detections if d["class"] == "person")

                        cnn_results = []
                        if num_dogs > 0:
                            dog_crops = detector.get_dog_crops(frame, detections)
                            for crop, bbox in dog_crops:
                                cnn_results.append(cnn.classify(crop))

                        raw_audio = audio_detector.get_state()
                        audio_combiner.update(raw_audio)
                        audio_state = audio_combiner.get_combined_state(num_dogs, num_humans)

                        result = engine.evaluate(
                            cnn_results=cnn_results, num_dogs=num_dogs,
                            num_humans=num_humans, audio_state=audio_state)

                        if result["trigger_ultrasonic"]:
                            ultrasonic.trigger()

                        # Log periodically
                        if frame_n % 30 == 0:
                            threat = result.get("threat_label", "NONE")
                            conf = result.get("confidence", 0)
                            log_msg(self.log, f"Frame {frame_n} | Dogs:{num_dogs} Humans:{num_humans} | {threat} ({conf:.2f})")
                            logger.info(f"Live {frame_n}: dogs={num_dogs} humans={num_humans} threat={threat}")

                            if result["trigger_ultrasonic"]:
                                log_msg(self.log, ">>> ULTRASONIC TRIGGERED <<<")
                                logger.warning(f"Ultrasonic triggered: {threat}")

                        frame = ui.render(frame, detections, result["threat_label"],
                                          result["confidence"], audio_state)
                        key = ui.show(frame)
                        if key == ord("q"):
                            break
                finally:
                    audio_detector.stop()
                    cap.release()
                    ui.cleanup()
                    if mode == "hardware":
                        ultrasonic.cleanup()

            self.status_label.configure(text="Detection stopped.")
            log_msg(self.log, "Detection stopped.")
            logger.info("Detection stopped")
        except Exception as e:
            self.status_label.configure(text=f"Error: {e}")
            log_msg(self.log, f"ERROR: {e}")
            logger.error(f"Detection error: {e}")
        finally:
            self.running = False


# ── Retrain Page ──────────────────────────────────────────────────────────

class RetrainPage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG)
        self.app = app
        self.training = False

        tk.Label(self, text="Retrain", font=("Segoe UI", 20, "bold"),
                 bg=BG, fg=FG).pack(anchor="w", padx=30, pady=(20, 5))
        tk.Label(self, text="Retrain existing models, fine-tune, or add new data",
                 font=("Segoe UI", 10), bg=BG, fg=FG_DIM).pack(anchor="w", padx=30, pady=(0, 10))

        # ── Section 1: Retrain from Scratch ──
        sec1 = tk.LabelFrame(self, text="  Retrain from Scratch  ",
                              font=("Segoe UI", 10, "bold"), bg=BG_CARD, fg=ACCENT,
                              bd=1, relief="groove", padx=10, pady=8)
        sec1.pack(fill="x", padx=30, pady=5)

        row1 = tk.Frame(sec1, bg=BG_CARD)
        row1.pack(fill="x", pady=3)
        tk.Label(row1, text="Epochs:", font=("Segoe UI", 10), bg=BG_CARD, fg=FG).pack(side="left")
        self.retrain_epochs_var = tk.StringVar(value="60")
        tk.Entry(row1, textvariable=self.retrain_epochs_var, width=6,
                 font=("Segoe UI", 10), bg=DARK_RED, fg=FG, insertbackground=FG).pack(side="left", padx=8)

        tk.Button(row1, text="Retrain YOLO", font=("Segoe UI", 9, "bold"),
                  bg=ACCENT, fg="#fff", bd=0, padx=12, cursor="hand2",
                  command=lambda: self._retrain("yolo")).pack(side="left", padx=5)
        tk.Button(row1, text="Retrain CNN", font=("Segoe UI", 9, "bold"),
                  bg=ACCENT, fg="#fff", bd=0, padx=12, cursor="hand2",
                  command=lambda: self._retrain("cnn")).pack(side="left", padx=5)
        tk.Button(row1, text="Retrain ALL", font=("Segoe UI", 9, "bold"),
                  bg=GREEN, fg="#fff", bd=0, padx=12, cursor="hand2",
                  command=lambda: self._retrain("all")).pack(side="left", padx=5)

        # ── Section 2: Fine-tune Existing ──
        sec2 = tk.LabelFrame(self, text="  Fine-tune Existing Model  ",
                              font=("Segoe UI", 10, "bold"), bg=BG_CARD, fg=ACCENT,
                              bd=1, relief="groove", padx=10, pady=8)
        sec2.pack(fill="x", padx=30, pady=5)

        row2 = tk.Frame(sec2, bg=BG_CARD)
        row2.pack(fill="x", pady=3)
        tk.Label(row2, text="Extra Epochs:", font=("Segoe UI", 10), bg=BG_CARD, fg=FG).pack(side="left")
        self.finetune_epochs_var = tk.StringVar(value="20")
        tk.Entry(row2, textvariable=self.finetune_epochs_var, width=6,
                 font=("Segoe UI", 10), bg=DARK_RED, fg=FG, insertbackground=FG).pack(side="left", padx=8)
        tk.Label(row2, text="LR:", font=("Segoe UI", 10), bg=BG_CARD, fg=FG).pack(side="left", padx=(10, 0))
        self.finetune_lr_var = tk.StringVar(value="0.0001")
        tk.Entry(row2, textvariable=self.finetune_lr_var, width=10,
                 font=("Segoe UI", 10), bg=DARK_RED, fg=FG, insertbackground=FG).pack(side="left", padx=8)

        tk.Button(row2, text="Fine-tune YOLO", font=("Segoe UI", 9, "bold"),
                  bg=ORANGE, fg="#fff", bd=0, padx=12, cursor="hand2",
                  command=lambda: self._finetune("yolo")).pack(side="left", padx=5)
        tk.Button(row2, text="Fine-tune CNN", font=("Segoe UI", 9, "bold"),
                  bg=ORANGE, fg="#fff", bd=0, padx=12, cursor="hand2",
                  command=lambda: self._finetune("cnn")).pack(side="left", padx=5)

        # ── Section 3: Add New Data ──
        sec3 = tk.LabelFrame(self, text="  Add New Data to Model  ",
                              font=("Segoe UI", 10, "bold"), bg=BG_CARD, fg=ACCENT,
                              bd=1, relief="groove", padx=10, pady=8)
        sec3.pack(fill="x", padx=30, pady=5)

        row3 = tk.Frame(sec3, bg=BG_CARD)
        row3.pack(fill="x", pady=3)
        tk.Label(row3, text="New Data Path:", font=("Segoe UI", 10), bg=BG_CARD, fg=FG).pack(side="left")
        self.new_data_var = tk.StringVar()
        tk.Entry(row3, textvariable=self.new_data_var, font=("Segoe UI", 10),
                 bg=DARK_RED, fg=FG, insertbackground=FG, width=35).pack(side="left", padx=8, fill="x", expand=True)
        tk.Button(row3, text="Browse", font=("Segoe UI", 9), bg=ACCENT, fg="#fff",
                  bd=0, padx=10, cursor="hand2",
                  command=lambda: self.new_data_var.set(
                      filedialog.askdirectory(title="Select new data folder") or "")).pack(side="left")
        tk.Button(row3, text="Add & Retrain", font=("Segoe UI", 9, "bold"),
                  bg=GREEN, fg="#fff", bd=0, padx=12, cursor="hand2",
                  command=self._add_data_retrain).pack(side="left", padx=5)

        # Progress
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(self, variable=self.progress_var,
                                             maximum=100, style="red.Horizontal.TProgressbar")
        self.progress_bar.pack(fill="x", padx=30, pady=5)
        self.progress_label = tk.Label(self, text="", font=("Segoe UI", 10), bg=BG, fg=YELLOW)
        self.progress_label.pack(anchor="w", padx=30)

        # Log
        self.log = make_log(self)
        self.log.pack(fill="both", expand=True, padx=30, pady=(5, 15))

    def _progress_cb(self, epoch, total, train_metric, val_metric):
        pct = (epoch / total) * 100
        self.progress_var.set(pct)
        self.progress_label.configure(
            text=f"Epoch {epoch}/{total} | Train: {train_metric:.4f} | Val: {val_metric:.4f}")
        log_msg(self.log, f"Epoch {epoch}/{total} | Train: {train_metric:.4f} | Val: {val_metric:.4f}")

    def _retrain(self, model_name):
        if self.training:
            messagebox.showinfo("Busy", "Training in progress.")
            return
        try:
            epochs = int(self.retrain_epochs_var.get())
        except ValueError:
            messagebox.showwarning("Invalid", "Epochs must be a number.")
            return

        self.training = True
        self.progress_var.set(0)

        def run():
            try:
                from mlops.trainer import train_yolo, train_cnn, train_all, clear_old_models
                from mlops.state import load_state, update_model, log_event

                # Clear old models first
                if model_name in ("yolo", "all"):
                    clear_old_models(["yolo"])
                    log_msg(self.log, "Cleared old YOLO model")
                if model_name in ("cnn", "all"):
                    clear_old_models(["cnn"])
                    log_msg(self.log, "Cleared old CNN model")

                state = load_state()
                log_msg(self.log, f"Retraining {model_name.upper()} from scratch ({epochs} epochs)...")
                logger.info(f"Retrain {model_name}: {epochs} epochs")

                if model_name == "yolo":
                    res = train_yolo(epochs=epochs, progress_callback=self._progress_cb)
                    if res["success"]:
                        update_model(state, "yolo", trained=True, epochs=epochs, path=res["model_path"])
                        log_event(state, "YOLO Retrained", f"epochs={epochs}")
                        log_msg(self.log, f"YOLO retrained: {res['model_path']}")
                    else:
                        log_msg(self.log, f"FAILED: {res.get('error')}")
                elif model_name == "cnn":
                    res = train_cnn(epochs=epochs, progress_callback=self._progress_cb)
                    if res["success"]:
                        update_model(state, "cnn", trained=True, epochs=res["epochs"], path=res["model_path"])
                        log_event(state, "CNN Retrained", f"acc={res.get('best_val_acc')}")
                        log_msg(self.log, f"CNN retrained: acc={res.get('best_val_acc')}")
                    else:
                        log_msg(self.log, f"FAILED: {res.get('error')}")
                elif model_name == "all":
                    results = train_all(epochs=epochs, progress_callback=self._progress_cb)
                    for n, r in results.items():
                        if r.get("success"):
                            update_model(state, n, trained=True, epochs=r.get("epochs", epochs), path=r.get("model_path", ""))
                            log_msg(self.log, f"  {n.upper()}: Done!")
                        else:
                            log_msg(self.log, f"  {n.upper()}: FAILED - {r.get('error')}")

                self.progress_var.set(100)
                self.progress_label.configure(text="Retrain complete!")
            except Exception as e:
                log_msg(self.log, f"ERROR: {e}")
            finally:
                self.training = False

        threading.Thread(target=run, daemon=True).start()

    def _finetune(self, model_name):
        if self.training:
            messagebox.showinfo("Busy", "Training in progress.")
            return

        try:
            epochs = int(self.finetune_epochs_var.get())
            lr = float(self.finetune_lr_var.get())
        except ValueError:
            messagebox.showwarning("Invalid", "Check epochs and learning rate values.")
            return

        from config import YOLO_MODEL_PATH, CNN_MODEL_PATH
        if model_name == "yolo" and not YOLO_MODEL_PATH.exists():
            messagebox.showwarning("No Model", "No YOLO model to fine-tune. Train first.")
            return
        if model_name == "cnn" and not CNN_MODEL_PATH.exists():
            messagebox.showwarning("No Model", "No CNN model to fine-tune. Train first.")
            return

        self.training = True
        self.progress_var.set(0)

        def run():
            try:
                from mlops.state import load_state, update_model, log_event
                state = load_state()
                log_msg(self.log, f"Fine-tuning {model_name.upper()} ({epochs} epochs, lr={lr})...")
                logger.info(f"Fine-tune {model_name}: {epochs} epochs, lr={lr}")

                if model_name == "yolo":
                    from mlops.trainer import train_yolo
                    res = train_yolo(epochs=epochs, progress_callback=self._progress_cb)
                    if res["success"]:
                        update_model(state, "yolo", trained=True, epochs=epochs, path=res["model_path"])
                        log_event(state, "YOLO Fine-tuned", f"epochs={epochs}")
                        log_msg(self.log, f"Fine-tune done: {res['model_path']}")
                    else:
                        log_msg(self.log, f"FAILED: {res.get('error')}")
                elif model_name == "cnn":
                    from mlops.trainer import train_cnn
                    res = train_cnn(epochs=epochs, lr=lr, progress_callback=self._progress_cb)
                    if res["success"]:
                        update_model(state, "cnn", trained=True, epochs=res["epochs"], path=res["model_path"])
                        log_event(state, "CNN Fine-tuned", f"acc={res.get('best_val_acc')}")
                        log_msg(self.log, f"Fine-tune done: acc={res.get('best_val_acc')}")
                    else:
                        log_msg(self.log, f"FAILED: {res.get('error')}")

                self.progress_var.set(100)
                self.progress_label.configure(text="Fine-tune complete!")
            except Exception as e:
                log_msg(self.log, f"ERROR: {e}")
            finally:
                self.training = False

        threading.Thread(target=run, daemon=True).start()

    def _add_data_retrain(self):
        path = self.new_data_var.get().strip()
        if not path:
            messagebox.showwarning("No path", "Select a new data folder.")
            return
        if self.training:
            messagebox.showinfo("Busy", "Training in progress.")
            return

        self.training = True
        self.progress_var.set(0)

        def run():
            try:
                log_msg(self.log, f"Adding new data: {path}")
                logger.info(f"Adding new data from {path}")
                from mlops.data_loader import load_dataset_from_path, copy_for_detection
                from mlops.preprocessor import auto_select_mapping, extract_behavior_crops
                from mlops.state import load_state, update_dataset, log_event

                info = load_dataset_from_path(path)
                if not info["success"]:
                    log_msg(self.log, f"ERROR: {info.get('error')}")
                    self.training = False
                    return

                log_msg(self.log, f"Format: {info['format']} | Classes: {info.get('classes', [])}")

                # Add to existing data (don't clear old)
                if info.get("img_dir") and info.get("lbl_dir"):
                    result = copy_for_detection(info, clear_old=False)
                    if result["success"]:
                        log_msg(self.log, f"Added {result['total']} images to training data")

                # Also extract crops if possible
                classes = info.get("classes", [])
                if classes and info.get("img_dir") and info.get("lbl_dir"):
                    _, mapping = auto_select_mapping(classes)
                    if mapping:
                        crop_result = extract_behavior_crops(
                            img_dir=info["img_dir"], lbl_dir=info["lbl_dir"],
                            class_names=classes, class_mapping=mapping, clear_old=False,
                            all_splits=info.get("all_splits"),
                        )
                        if crop_result["success"]:
                            log_msg(self.log, f"Added {crop_result['total']} behavior crops")

                # Now retrain
                log_msg(self.log, "Retraining with expanded dataset...")
                try:
                    epochs = int(self.retrain_epochs_var.get())
                except ValueError:
                    epochs = 60

                from mlops.trainer import train_all
                results = train_all(epochs=epochs, progress_callback=self._progress_cb)
                state = load_state()
                from mlops.state import update_model
                for n, r in results.items():
                    if r.get("success"):
                        update_model(state, n, trained=True, epochs=r.get("epochs", epochs), path=r.get("model_path", ""))
                        log_msg(self.log, f"  {n.upper()}: Retrained!")
                    else:
                        log_msg(self.log, f"  {n.upper()}: FAILED - {r.get('error')}")

                self.progress_var.set(100)
                self.progress_label.configure(text="Add data + retrain complete!")
            except Exception as e:
                log_msg(self.log, f"ERROR: {e}")
            finally:
                self.training = False

        threading.Thread(target=run, daemon=True).start()


# ── Admin Page ────────────────────────────────────────────────────────────

class AdminPage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG)
        self.app = app

        tk.Label(self, text="Admin", font=("Segoe UI", 20, "bold"),
                 bg=BG, fg=FG).pack(anchor="w", padx=30, pady=(20, 5))
        tk.Label(self, text="API keys and admin settings",
                 font=("Segoe UI", 10), bg=BG, fg=FG_DIM).pack(anchor="w", padx=30, pady=(0, 15))

        # ── Roboflow API Key ──
        sec1 = tk.LabelFrame(self, text="  Roboflow API  ",
                              font=("Segoe UI", 10, "bold"), bg=BG_CARD, fg=ACCENT,
                              bd=1, relief="groove", padx=15, pady=10)
        sec1.pack(fill="x", padx=30, pady=5)

        row1 = tk.Frame(sec1, bg=BG_CARD)
        row1.pack(fill="x", pady=3)
        tk.Label(row1, text="API Key:", font=("Segoe UI", 10), bg=BG_CARD, fg=FG).pack(side="left")
        self.api_key_var = tk.StringVar()
        self.api_entry = tk.Entry(row1, textvariable=self.api_key_var, font=("Segoe UI", 10),
                                   bg=DARK_RED, fg=FG, insertbackground=FG, width=40, show="*")
        self.api_entry.pack(side="left", padx=10, fill="x", expand=True)
        self.show_key = tk.BooleanVar(value=False)
        tk.Checkbutton(row1, text="Show", variable=self.show_key, bg=BG_CARD, fg=FG,
                        selectcolor=DARK_RED, activebackground=BG_CARD,
                        command=self._toggle_key_visibility).pack(side="left")

        row2 = tk.Frame(sec1, bg=BG_CARD)
        row2.pack(fill="x", pady=3)
        tk.Label(row2, text="Workspace:", font=("Segoe UI", 10), bg=BG_CARD, fg=FG).pack(side="left")
        self.workspace_var = tk.StringVar()
        tk.Entry(row2, textvariable=self.workspace_var, font=("Segoe UI", 10),
                 bg=DARK_RED, fg=FG, insertbackground=FG, width=30).pack(side="left", padx=10)

        tk.Button(sec1, text="Save API Settings", font=("Segoe UI", 10, "bold"),
                  bg=GREEN, fg="#fff", bd=0, padx=20, pady=8, cursor="hand2",
                  command=self._save_api).pack(anchor="w", pady=8)

        # ── Admin Password ──
        sec2 = tk.LabelFrame(self, text="  Admin Password  ",
                              font=("Segoe UI", 10, "bold"), bg=BG_CARD, fg=ACCENT,
                              bd=1, relief="groove", padx=15, pady=10)
        sec2.pack(fill="x", padx=30, pady=5)

        row3 = tk.Frame(sec2, bg=BG_CARD)
        row3.pack(fill="x", pady=3)
        tk.Label(row3, text="Current Password:", font=("Segoe UI", 10), bg=BG_CARD, fg=FG).pack(side="left")
        self.old_pwd_var = tk.StringVar()
        tk.Entry(row3, textvariable=self.old_pwd_var, font=("Segoe UI", 10),
                 bg=DARK_RED, fg=FG, insertbackground=FG, width=20, show="*").pack(side="left", padx=10)

        row4 = tk.Frame(sec2, bg=BG_CARD)
        row4.pack(fill="x", pady=3)
        tk.Label(row4, text="New Password:", font=("Segoe UI", 10), bg=BG_CARD, fg=FG).pack(side="left")
        self.new_pwd_var = tk.StringVar()
        tk.Entry(row4, textvariable=self.new_pwd_var, font=("Segoe UI", 10),
                 bg=DARK_RED, fg=FG, insertbackground=FG, width=20, show="*").pack(side="left", padx=10)

        tk.Button(sec2, text="Change Password", font=("Segoe UI", 10, "bold"),
                  bg=ORANGE, fg="#fff", bd=0, padx=20, pady=8, cursor="hand2",
                  command=self._change_password).pack(anchor="w", pady=8)

        # ── System Logs ──
        sec3 = tk.LabelFrame(self, text="  System Logs  ",
                              font=("Segoe UI", 10, "bold"), bg=BG_CARD, fg=ACCENT,
                              bd=1, relief="groove", padx=15, pady=10)
        sec3.pack(fill="both", expand=True, padx=30, pady=5)

        btn_row = tk.Frame(sec3, bg=BG_CARD)
        btn_row.pack(fill="x", pady=3)
        tk.Button(btn_row, text="View Today's Log", font=("Segoe UI", 9),
                  bg=ACCENT, fg="#fff", bd=0, padx=12, cursor="hand2",
                  command=self._view_log).pack(side="left", padx=5)
        tk.Button(btn_row, text="Clear Log Display", font=("Segoe UI", 9),
                  bg=FG_DIM, fg="#fff", bd=0, padx=12, cursor="hand2",
                  command=self._clear_log_display).pack(side="left", padx=5)

        self.log_display = tk.Text(sec3, font=("Consolas", 8), bg="#0d0d0d", fg="#aaa",
                                    insertbackground=FG, height=10, state="disabled",
                                    relief="flat", padx=8, pady=8)
        self.log_display.pack(fill="both", expand=True, pady=5)

        # Status
        self.status_label = tk.Label(self, text="", font=("Segoe UI", 10), bg=BG, fg=GREEN)
        self.status_label.pack(anchor="w", padx=30, pady=(5, 15))

    def on_show(self):
        cfg = load_admin_config()
        self.api_key_var.set(cfg.get("roboflow_api_key", ""))
        self.workspace_var.set(cfg.get("roboflow_workspace", ""))

    def _toggle_key_visibility(self):
        self.api_entry.configure(show="" if self.show_key.get() else "*")

    def _save_api(self):
        cfg = load_admin_config()
        cfg["roboflow_api_key"] = self.api_key_var.get().strip()
        cfg["roboflow_workspace"] = self.workspace_var.get().strip()
        save_admin_config(cfg)
        self.status_label.configure(text="API settings saved!", fg=GREEN)
        logger.info("Admin: API settings updated")

    def _change_password(self):
        old = self.old_pwd_var.get()
        new = self.new_pwd_var.get()
        cfg = load_admin_config()

        if old != cfg.get("password", "admin123"):
            messagebox.showerror("Wrong Password", "Current password is incorrect.")
            return
        if len(new) < 4:
            messagebox.showwarning("Weak Password", "Password must be at least 4 characters.")
            return

        cfg["password"] = new
        save_admin_config(cfg)
        self.old_pwd_var.set("")
        self.new_pwd_var.set("")
        self.status_label.configure(text="Password changed!", fg=GREEN)
        logger.info("Admin: password changed")

    def _view_log(self):
        self.log_display.configure(state="normal")
        self.log_display.delete("1.0", "end")
        if LOG_FILE.exists():
            content = LOG_FILE.read_text()
            # Show last 200 lines
            lines = content.strip().split("\n")
            self.log_display.insert("end", "\n".join(lines[-200:]))
        else:
            self.log_display.insert("end", "No log file found for today.")
        self.log_display.see("end")
        self.log_display.configure(state="disabled")

    def _clear_log_display(self):
        self.log_display.configure(state="normal")
        self.log_display.delete("1.0", "end")
        self.log_display.configure(state="disabled")


# ── Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = App()
    app.mainloop()
