"""
models/pipeline.py — Unified high-accuracy detection pipeline.
Connects all models in the correct order with proper data flow.

Pipeline Architecture:
  ┌──────────────────────────────────────────────────────────┐
  │                    INPUT FRAME                           │
  │                   (640x480 BGR)                          │
  └──────────────┬───────────────────────────────────────────┘
                 │
  ┌──────────────▼───────────────────────────────────────────┐
  │  STAGE 1: DETECTION (Where are dogs and humans?)         │
  │                                                          │
  │  Option A: YOLO only         (fast, good accuracy)       │
  │  Option B: SSD only          (good for small objects)    │
  │  Option C: YOLO + SSD fusion (BEST accuracy)             │
  │                                                          │
  │  Output: list of {bbox, class: dog/person, confidence}   │
  └──────────────┬───────────────────────────────────────────┘
                 │
  ┌──────────────▼───────────────────────────────────────────┐
  │  STAGE 2: SPATIAL ANALYSIS (How close is dog to human?)  │
  │                                                          │
  │  - Measures distance between each dog and each human     │
  │  - Tracks approach speed over time                       │
  │  - Computes threat proximity score (0.0 to 1.0)          │
  │                                                          │
  │  Output: distance, speed, threat_score per dog-human pair│
  └──────────────┬───────────────────────────────────────────┘
                 │
  ┌──────────────▼───────────────────────────────────────────┐
  │  STAGE 3: BEHAVIOR CLASSIFICATION (Is the dog aggressive?)│
  │                                                          │
  │  - Crops each dog from frame                             │
  │  - BehaviorNetV2 classifies: IDLE / ALERT / DANGER       │
  │  - Uses residual CNN with SE attention                   │
  │                                                          │
  │  Output: behavior_class + confidence per dog             │
  └──────────────┬───────────────────────────────────────────┘
                 │
  ┌──────────────▼───────────────────────────────────────────┐
  │  STAGE 4: AUDIO ANALYSIS (What sounds are happening?)    │
  │                                                          │
  │  - Bark detection (300-1000Hz)                           │
  │  - Growl detection (80-300Hz)                            │
  │  - Scream detection (1000-4000Hz)                        │
  │                                                          │
  │  Output: bark/growl/scream booleans                      │
  └──────────────┬───────────────────────────────────────────┘
                 │
  ┌──────────────▼───────────────────────────────────────────┐
  │  STAGE 5: THREAT ENGINE (Final decision)                 │
  │                                                          │
  │  Fuses: detection + spatial + behavior + audio           │
  │  Applies 5 safety rules                                  │
  │  Outputs: IDLE / ALERT / DANGER / DOG_FIGHT              │
  │                                                          │
  │  DANGER + humans → TRIGGER ULTRASONIC                    │
  └──────────────────────────────────────────────────────────┘
"""

from config import DETECTOR_BACKEND, DEVICE


class DetectionPipeline:
    """
    Unified pipeline that connects all models in the correct order.
    Single call to process() runs the entire pipeline on one frame.
    """

    def __init__(self, detector_mode=None, use_v2_cnn=True):
        """
        Args:
            detector_mode: "yolo", "ssd", or "ensemble"
            use_v2_cnn: Use BehaviorNetV2 (True) or V1 (False)
        """
        mode = detector_mode or DETECTOR_BACKEND

        # Stage 1: Load detector
        if mode == "ensemble":
            from models.ensemble_detector import EnsembleDetector
            self.detector = EnsembleDetector(use_yolo=True, use_ssd=True)
        elif mode == "ssd":
            from models.ssd_model import SSDDetector
            self.detector = SSDDetector()
        else:
            from models.yolo_model import DualYOLODetector
            self.detector = DualYOLODetector()

        # Stage 2: Spatial analyzer
        from models.spatial_analyzer import SpatialAnalyzer
        self.spatial = SpatialAnalyzer()

        # Stage 3: Behavior classifier
        if use_v2_cnn:
            from models.behavior_net_v2 import BehaviorClassifierV2
            self.cnn = BehaviorClassifierV2()
        else:
            from models.cnn_model import BehaviorClassifier
            self.cnn = BehaviorClassifier()

        # Stage 5: Threat engine
        from models.threat_engine import ThreatEngine
        self.engine = ThreatEngine()

        self.detector_mode = mode
        print(f"[PIPELINE] Ready — detector={mode}, cnn={'v2' if use_v2_cnn else 'v1'}")

    def process(self, frame, audio_state):
        """
        Run the complete pipeline on one frame.

        Args:
            frame: BGR numpy array from camera.
            audio_state: dict from AudioCombiner.get_combined_state()

        Returns:
            dict: {
                'detections': list of detection dicts,
                'spatial': spatial analysis results,
                'cnn_results': list of (class, label, confidence),
                'threat': threat engine result dict,
                'num_dogs': int,
                'num_humans': int,
            }
        """
        # Stage 1: Detection
        detections = self.detector.detect(frame)
        num_dogs = sum(1 for d in detections if d["class"] == "dog")
        num_humans = sum(1 for d in detections if d["class"] == "person")

        # Stage 2: Spatial analysis
        spatial = self.spatial.analyze(detections)

        # Stage 3: Behavior classification
        cnn_results = []
        if num_dogs > 0:
            dog_crops = self.detector.get_dog_crops(frame, detections)
            if dog_crops:
                crops_only = [crop for crop, bbox in dog_crops]
                cnn_results = self.cnn.classify_batch(crops_only)

        # Boost CNN prediction using spatial data
        cnn_results = self._fuse_spatial_with_cnn(cnn_results, spatial)

        # Stage 4: Audio is passed in from outside (runs in separate thread)

        # Stage 5: Threat engine
        threat = self.engine.evaluate(
            cnn_results=cnn_results,
            num_dogs=num_dogs,
            num_humans=num_humans,
            audio_state=audio_state,
        )

        return {
            "detections": detections,
            "spatial": spatial,
            "cnn_results": cnn_results,
            "threat": threat,
            "num_dogs": num_dogs,
            "num_humans": num_humans,
        }

    def _fuse_spatial_with_cnn(self, cnn_results, spatial):
        """
        Combine CNN behavior prediction with spatial distance data.
        Spatial data overrides CNN when distance clearly indicates threat level.
        """
        if not cnn_results or not spatial["dog_human_pairs"]:
            return cnn_results

        fused = []
        pairs = spatial["dog_human_pairs"]

        for i, (tc, tl, conf) in enumerate(cnn_results):
            # Get spatial info for this dog (use closest pair)
            pair = pairs[i] if i < len(pairs) else None

            if pair:
                threat_score = pair["threat_score"]
                distance = pair["distance"]
                approaching = pair["approach_speed"] < -2.0

                # Spatial override: if dog is very close + approaching → DANGER
                if distance < self.spatial.danger_distance and approaching:
                    if tc < 2:  # upgrade to DANGER
                        tc = 2
                        tl = "DANGER"
                        conf = max(conf, 0.80 + threat_score * 0.15)

                # Spatial override: if dog is in alert zone + moving toward
                elif distance < self.spatial.alert_distance and approaching:
                    if tc < 1:  # upgrade to ALERT
                        tc = 1
                        tl = "ALERT"
                        conf = max(conf, 0.65 + threat_score * 0.2)

                # Spatial validation: if CNN says DANGER but dog is far away → downgrade
                elif tc == 2 and distance > self.spatial.alert_distance * 1.5:
                    tc = 1
                    tl = "ALERT"
                    conf *= 0.7

            fused.append((tc, tl, min(conf, 1.0)))

        return fused
