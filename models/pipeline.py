"""
models/pipeline.py — Detection pipeline.

Pipeline: YOLO Detection → Spatial Analysis → CNN Behavior → Threat Engine
"""

from config import DEVICE


class DetectionPipeline:
    """
    Connects all models: YOLO → Spatial → CNN → ThreatEngine.
    Single call to process() runs the entire pipeline on one frame.
    """

    def __init__(self):
        from models.yolo_model import DualYOLODetector
        from models.spatial_analyzer import SpatialAnalyzer
        from models.behavior_net_v2 import BehaviorClassifierV2
        from models.threat_engine import ThreatEngine

        self.detector = DualYOLODetector()
        self.spatial = SpatialAnalyzer()
        self.cnn = BehaviorClassifierV2()
        self.engine = ThreatEngine()
        print("[PIPELINE] Ready — YOLO + BehaviorNetV2")

    def process(self, frame, audio_state):
        detections = self.detector.detect(frame)
        num_dogs = sum(1 for d in detections if d["class"] == "dog")
        num_humans = sum(1 for d in detections if d["class"] == "person")

        spatial = self.spatial.analyze(detections)

        cnn_results = []
        if num_dogs > 0:
            dog_crops = self.detector.get_dog_crops(frame, detections)
            if dog_crops:
                crops_only = [crop for crop, bbox in dog_crops]
                cnn_results = self.cnn.classify_batch(crops_only)

        cnn_results = self._fuse_spatial_with_cnn(cnn_results, spatial)

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
        if not cnn_results or not spatial["dog_human_pairs"]:
            return cnn_results

        fused = []
        pairs = spatial["dog_human_pairs"]

        for i, (tc, tl, conf) in enumerate(cnn_results):
            pair = pairs[i] if i < len(pairs) else None

            if pair:
                threat_score = pair["threat_score"]
                distance = pair["distance"]
                approaching = pair["approach_speed"] < -2.0

                if distance < self.spatial.danger_distance and approaching:
                    if tc < 2:
                        tc = 2
                        tl = "DANGER"
                        conf = max(conf, 0.80 + threat_score * 0.15)
                elif distance < self.spatial.alert_distance and approaching:
                    if tc < 1:
                        tc = 1
                        tl = "ALERT"
                        conf = max(conf, 0.65 + threat_score * 0.2)
                elif tc == 2 and distance > self.spatial.alert_distance * 1.5:
                    tc = 1
                    tl = "ALERT"
                    conf *= 0.7

            fused.append((tc, tl, min(conf, 1.0)))

        return fused
