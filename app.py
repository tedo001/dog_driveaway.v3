"""
app.py — Smart Dog Threat Detection System — GUI Application.

Run: python app.py

Tkinter GUI with 4 pages:
  1. Dashboard   — system status, GPU info, model status
  2. Data        — load dataset, preprocess, view status
  3. Training    — train YOLO / CNN with progress
  4. Detection   — run simulation / live / hardware mode
"""

import sys
import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Color Theme ────────────────────────────────────────────────────────────

BG = "#1e1e2e"
BG_CARD = "#2a2a3d"
FG = "#cdd6f4"
FG_DIM = "#6c7086"
ACCENT = "#89b4fa"
GREEN = "#a6e3a1"
RED = "#f38ba8"
YELLOW = "#f9e2af"
ORANGE = "#fab387"


# ── Main Application ───────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Dog Threat Detection System")
        self.geometry("900x650")
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
        ]

        for name, page_class in nav_items:
            btn = tk.Button(sidebar, text=name, font=("Segoe UI", 11),
                            bg=BG_CARD, fg=FG, bd=0, anchor="w", padx=20, pady=10,
                            activebackground=ACCENT, activeforeground=BG,
                            cursor="hand2",
                            command=lambda n=name: self.show_page(n))
            btn.pack(fill="x")
            self.nav_buttons[name] = btn

        # Content area
        self.content = tk.Frame(self, bg=BG)
        self.content.pack(side="right", fill="both", expand=True)

        for name, page_class in nav_items:
            page = page_class(self.content, self)
            page.place(relx=0, rely=0, relwidth=1, relheight=1)
            self.pages[name] = page

        self.show_page("Dashboard")

    def show_page(self, name):
        if self.current_page:
            self.nav_buttons[self.current_page].configure(bg=BG_CARD, fg=FG)
        self.current_page = name
        self.nav_buttons[name].configure(bg=ACCENT, fg=BG)
        self.pages[name].tkraise()
        if hasattr(self.pages[name], "on_show"):
            self.pages[name].on_show()


# ── Dashboard Page ─────────────────────────────────────────────────────────

class DashboardPage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG)
        self.app = app

        tk.Label(self, text="Dashboard", font=("Segoe UI", 20, "bold"),
                 bg=BG, fg=FG).pack(anchor="w", padx=30, pady=(20, 10))

        self.info_frame = tk.Frame(self, bg=BG)
        self.info_frame.pack(fill="both", expand=True, padx=30, pady=10)

        self.status_labels = {}

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


# ── Data Pipeline Page ─────────────────────────────────────────────────────

class DataPage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG)
        self.app = app

        tk.Label(self, text="Data Pipeline", font=("Segoe UI", 20, "bold"),
                 bg=BG, fg=FG).pack(anchor="w", padx=30, pady=(20, 5))
        tk.Label(self, text="Load dataset → Auto-detect format → Preprocess → Ready for training",
                 font=("Segoe UI", 10), bg=BG, fg=FG_DIM).pack(anchor="w", padx=30, pady=(0, 15))

        # Path selector
        path_frame = tk.Frame(self, bg=BG)
        path_frame.pack(fill="x", padx=30, pady=5)

        tk.Label(path_frame, text="Dataset Path:", font=("Segoe UI", 10),
                 bg=BG, fg=FG).pack(side="left")
        self.path_var = tk.StringVar()
        self.path_entry = tk.Entry(path_frame, textvariable=self.path_var,
                                    font=("Segoe UI", 10), bg=BG_CARD, fg=FG,
                                    insertbackground=FG, width=50)
        self.path_entry.pack(side="left", padx=10, fill="x", expand=True)

        tk.Button(path_frame, text="Browse", font=("Segoe UI", 10),
                  bg=ACCENT, fg=BG, bd=0, padx=15, pady=5, cursor="hand2",
                  command=self._browse).pack(side="left")

        # Action buttons
        btn_frame = tk.Frame(self, bg=BG)
        btn_frame.pack(fill="x", padx=30, pady=15)

        tk.Button(btn_frame, text="Load & Process Dataset", font=("Segoe UI", 11, "bold"),
                  bg=GREEN, fg=BG, bd=0, padx=20, pady=10, cursor="hand2",
                  command=self._load_dataset).pack(side="left", padx=5)

        tk.Button(btn_frame, text="Download COCO (5000 imgs)", font=("Segoe UI", 11),
                  bg=ORANGE, fg=BG, bd=0, padx=20, pady=10, cursor="hand2",
                  command=self._download_coco).pack(side="left", padx=5)

        # Log
        self.log = tk.Text(self, font=("Consolas", 9), bg=BG_CARD, fg=FG,
                           insertbackground=FG, height=20, state="disabled",
                           relief="flat", padx=10, pady=10)
        self.log.pack(fill="both", expand=True, padx=30, pady=(0, 20))

    def _browse(self):
        path = filedialog.askdirectory(title="Select dataset folder")
        if not path:
            path = filedialog.askopenfilename(title="Select dataset ZIP",
                                               filetypes=[("ZIP files", "*.zip")])
        if path:
            self.path_var.set(path)

    def _log(self, msg):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")
        self.update_idletasks()

    def _load_dataset(self):
        path = self.path_var.get().strip()
        if not path:
            messagebox.showwarning("No path", "Please select a dataset folder or ZIP file.")
            return

        def run():
            try:
                self._log(f"Loading: {path}")
                from mlops.data_loader import load_dataset_from_path
                from mlops.preprocessor import auto_select_mapping, extract_behavior_crops
                from mlops.state import load_state, update_dataset, log_event

                info = load_dataset_from_path(path)
                if not info["success"]:
                    self._log(f"ERROR: {info.get('error')}")
                    return

                self._log(f"Format: {info['format']}")
                self._log(f"Classes: {info.get('classes', [])}")

                classes = info.get("classes", [])
                if classes and info.get("img_dir") and info.get("lbl_dir"):
                    preset_name, mapping = auto_select_mapping(classes)
                    if mapping:
                        self._log(f"Auto-mapping: {preset_name}")
                        for src, dst in mapping.items():
                            self._log(f"  {src} -> {dst}")

                        self._log("Extracting behavior crops...")
                        result = extract_behavior_crops(
                            img_dir=info["img_dir"],
                            lbl_dir=info["lbl_dir"],
                            class_names=classes,
                            class_mapping=mapping,
                            clear_old=True,
                        )
                        if result["success"]:
                            state = load_state()
                            update_dataset(state, "behavior", loaded=True,
                                           crops=result["stats"].get("train", {}))
                            log_event(state, "Data Loaded", f"{result['total']} crops")
                            self._log(f"Done! {result['total']} crops extracted.")
                        else:
                            self._log(f"Crop ERROR: {result.get('error')}")
                    else:
                        self._log("No auto-mapping found for these classes.")
                else:
                    # Detection dataset (copy for YOLO)
                    from mlops.data_loader import copy_for_detection
                    if info.get("img_dir") and info.get("lbl_dir"):
                        self._log("Copying for YOLO training...")
                        result = copy_for_detection(info, clear_old=True)
                        if result["success"]:
                            state = load_state()
                            update_dataset(state, "detection", loaded=True, images=result["total"])
                            log_event(state, "Detection Data Loaded", f"{result['total']} images")
                            self._log(f"Done! {result['total']} images ready.")

                self._log("Pipeline complete. Go to Training tab.")
            except Exception as e:
                self._log(f"ERROR: {e}")

        threading.Thread(target=run, daemon=True).start()

    def _download_coco(self):
        def run():
            try:
                self._log("Downloading COCO dog+human dataset (5000 images)...")
                self._log("This may take a while...")
                from mlops.data_loader import download_coco
                from mlops.state import load_state, update_dataset, log_event

                result = download_coco(max_images=5000)
                if result["success"]:
                    state = load_state()
                    update_dataset(state, "detection", loaded=True, images=5000)
                    log_event(state, "COCO Download", "5000 images")
                    self._log("COCO dataset ready!")
                else:
                    self._log(f"ERROR: {result.get('error')}")
            except Exception as e:
                self._log(f"ERROR: {e}")

        threading.Thread(target=run, daemon=True).start()


# ── Training Page ──────────────────────────────────────────────────────────

class TrainingPage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG)
        self.app = app
        self.training = False

        tk.Label(self, text="Training", font=("Segoe UI", 20, "bold"),
                 bg=BG, fg=FG).pack(anchor="w", padx=30, pady=(20, 5))
        tk.Label(self, text="Train YOLO detector and CNN behavior classifier",
                 font=("Segoe UI", 10), bg=BG, fg=FG_DIM).pack(anchor="w", padx=30, pady=(0, 15))

        # Settings
        settings_frame = tk.Frame(self, bg=BG)
        settings_frame.pack(fill="x", padx=30)

        tk.Label(settings_frame, text="Epochs:", font=("Segoe UI", 10),
                 bg=BG, fg=FG).pack(side="left")
        self.epochs_var = tk.StringVar(value="60")
        tk.Entry(settings_frame, textvariable=self.epochs_var, width=6,
                 font=("Segoe UI", 10), bg=BG_CARD, fg=FG,
                 insertbackground=FG).pack(side="left", padx=10)

        # Buttons
        btn_frame = tk.Frame(self, bg=BG)
        btn_frame.pack(fill="x", padx=30, pady=15)

        self.btn_yolo = tk.Button(btn_frame, text="Train YOLO", font=("Segoe UI", 11, "bold"),
                                   bg=ACCENT, fg=BG, bd=0, padx=20, pady=10, cursor="hand2",
                                   command=lambda: self._train("yolo"))
        self.btn_yolo.pack(side="left", padx=5)

        self.btn_cnn = tk.Button(btn_frame, text="Train CNN", font=("Segoe UI", 11, "bold"),
                                  bg=ACCENT, fg=BG, bd=0, padx=20, pady=10, cursor="hand2",
                                  command=lambda: self._train("cnn"))
        self.btn_cnn.pack(side="left", padx=5)

        self.btn_all = tk.Button(btn_frame, text="Train ALL", font=("Segoe UI", 11, "bold"),
                                  bg=GREEN, fg=BG, bd=0, padx=20, pady=10, cursor="hand2",
                                  command=lambda: self._train("all"))
        self.btn_all.pack(side="left", padx=5)

        tk.Button(btn_frame, text="Clear Models", font=("Segoe UI", 10),
                  bg=RED, fg=BG, bd=0, padx=15, pady=10, cursor="hand2",
                  command=self._clear_models).pack(side="right", padx=5)

        # Progress
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(self, variable=self.progress_var,
                                             maximum=100, length=400)
        self.progress_bar.pack(fill="x", padx=30, pady=5)

        self.progress_label = tk.Label(self, text="", font=("Segoe UI", 10),
                                        bg=BG, fg=YELLOW)
        self.progress_label.pack(anchor="w", padx=30)

        # Log
        self.log = tk.Text(self, font=("Consolas", 9), bg=BG_CARD, fg=FG,
                           insertbackground=FG, height=18, state="disabled",
                           relief="flat", padx=10, pady=10)
        self.log.pack(fill="both", expand=True, padx=30, pady=(5, 20))

    def _log(self, msg):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")
        self.update_idletasks()

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
                text=f"Epoch {epoch}/{total} — Train: {train_metric:.4f} — Val: {val_metric:.4f}")
            self._log(f"  Epoch {epoch}/{total} | Train: {train_metric:.4f} | Val: {val_metric:.4f}")

        def run():
            try:
                from mlops.trainer import train_yolo, train_cnn, train_all
                from mlops.state import load_state, update_model, log_event

                state = load_state()

                if model_name == "yolo":
                    self._log(f"Training YOLO ({epochs} epochs)...")
                    res = train_yolo(epochs=epochs, progress_callback=progress_cb)
                    if res["success"]:
                        update_model(state, "yolo", trained=True, epochs=epochs,
                                     path=res.get("model_path", ""))
                        log_event(state, "YOLO Trained", f"epochs={epochs}")
                        self._log(f"YOLO done! Model: {res['model_path']}")
                    else:
                        self._log(f"YOLO FAILED: {res.get('error')}")

                elif model_name == "cnn":
                    self._log(f"Training CNN BehaviorNetV2 ({epochs} epochs)...")
                    res = train_cnn(epochs=epochs, progress_callback=progress_cb)
                    if res["success"]:
                        update_model(state, "cnn", trained=True, epochs=res["epochs"],
                                     best_metric=res.get("best_val_acc"),
                                     path=res.get("model_path", ""))
                        log_event(state, "CNN Trained", f"acc={res.get('best_val_acc')}")
                        self._log(f"CNN done! Acc: {res.get('best_val_acc')}")
                    else:
                        self._log(f"CNN FAILED: {res.get('error')}")

                elif model_name == "all":
                    self._log(f"Training ALL models ({epochs} epochs each)...")
                    results = train_all(epochs=epochs, progress_callback=progress_cb)
                    for name, res in results.items():
                        if res.get("success"):
                            update_model(state, name, trained=True,
                                         epochs=res.get("epochs", epochs),
                                         best_metric=res.get("best_val_acc") or None,
                                         path=res.get("model_path", ""))
                            log_event(state, f"{name.upper()} Trained", "via Train ALL")
                            self._log(f"  {name.upper()}: Done!")
                        else:
                            self._log(f"  {name.upper()}: FAILED — {res.get('error')}")

                self.progress_var.set(100)
                self.progress_label.configure(text="Training complete!")
            except Exception as e:
                self._log(f"ERROR: {e}")
            finally:
                self.training = False
                self._set_buttons(True)

        threading.Thread(target=run, daemon=True).start()

    def _clear_models(self):
        if messagebox.askyesno("Clear Models", "Delete all trained model weights?"):
            from mlops.trainer import clear_old_models
            cleared = clear_old_models()
            self._log(f"Cleared {len(cleared)} model files.")


# ── Detection Page ─────────────────────────────────────────────────────────

class DetectionPage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG)
        self.app = app

        tk.Label(self, text="Run Detection", font=("Segoe UI", 20, "bold"),
                 bg=BG, fg=FG).pack(anchor="w", padx=30, pady=(20, 5))
        tk.Label(self, text="Choose a mode to start the detection system",
                 font=("Segoe UI", 10), bg=BG, fg=FG_DIM).pack(anchor="w", padx=30, pady=(0, 20))

        # Mode cards
        modes = [
            ("Simulation", "No camera needed — animated test scenarios.\n"
             "Controls: 1-6 switch scenario, SPACE pause, Q quit.",
             GREEN, self._run_simulation),
            ("Live Camera", "Webcam + YOLO detection + CNN behavior + ultrasonic.\n"
             "Requires: trained models + webcam.",
             ACCENT, self._run_live),
            ("Hardware", "Webcam + Arduino + external ultrasonic sensor.\n"
             "Requires: trained models + webcam + Arduino.",
             ORANGE, self._run_hardware),
        ]

        for title, desc, color, cmd in modes:
            card = tk.Frame(self, bg=BG_CARD, padx=20, pady=15)
            card.pack(fill="x", padx=30, pady=5)

            header = tk.Frame(card, bg=BG_CARD)
            header.pack(fill="x")

            tk.Label(header, text=title, font=("Segoe UI", 14, "bold"),
                     bg=BG_CARD, fg=color).pack(side="left")

            tk.Button(header, text="START", font=("Segoe UI", 10, "bold"),
                      bg=color, fg=BG, bd=0, padx=20, pady=5, cursor="hand2",
                      command=cmd).pack(side="right")

            tk.Label(card, text=desc, font=("Segoe UI", 9),
                     bg=BG_CARD, fg=FG_DIM, justify="left").pack(anchor="w", pady=(5, 0))

        # Status
        self.status_label = tk.Label(self, text="", font=("Segoe UI", 10),
                                      bg=BG, fg=YELLOW)
        self.status_label.pack(anchor="w", padx=30, pady=20)

    def _run_simulation(self):
        self.status_label.configure(text="Starting simulation...")
        threading.Thread(target=self._run_detection_mode, args=("simulation",), daemon=True).start()

    def _run_live(self):
        from config import YOLO_MODEL_PATH, CNN_MODEL_PATH
        if not YOLO_MODEL_PATH.exists():
            messagebox.showwarning("No Model", "YOLO model not trained yet.\nGo to Training tab first.")
            return
        self.status_label.configure(text="Starting live detection...")
        threading.Thread(target=self._run_detection_mode, args=("live",), daemon=True).start()

    def _run_hardware(self):
        from config import YOLO_MODEL_PATH
        if not YOLO_MODEL_PATH.exists():
            messagebox.showwarning("No Model", "YOLO model not trained yet.\nGo to Training tab first.")
            return
        self.status_label.configure(text="Starting hardware mode...")
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

                while True:
                    start = _time.time()
                    if not sim.paused:
                        update_scenario_behavior(sim, current_scenario)
                        sim.frame_count += 1

                    result, cnn_results, audio_state = sim.classify_threats()
                    dt = frame_delay if not sim.paused else 0
                    sim.update_ultrasonic(result, dt)

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
                    return

                try:
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            _time.sleep(0.1)
                            continue

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
        except Exception as e:
            self.status_label.configure(text=f"Error: {e}")


# ── Entry Point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = App()
    app.mainloop()
