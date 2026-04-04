"""
models/threat_engine.py — Decision engine that fuses visual + audio signals.
Enforces all 5 safety rules. Determines final threat level and ultrasonic trigger.

FIX: Rules now use threat LABEL strings instead of hardcoded class indices.
     Works correctly with 2-class CNN (DANGER/IDLE) and future 4-class CNN.
"""

from config import THREAT_CLASSES


# Priority order — higher number = more severe
THREAT_PRIORITY = {
    "IDLE":      0,
    "ALERT":     1,
    "DANGER":    2,
    "DOG_FIGHT": 3,
}


def _label(threat_class: int) -> str:
    return THREAT_CLASSES.get(threat_class, "IDLE")


def _priority(label: str) -> int:
    return THREAT_PRIORITY.get(label, 0)


class ThreatEngine:
    """
    Fuses CNN behavior classification with audio signals to produce
    a final threat assessment.

    SAFETY RULES (hardcoded, never bypass):
        Rule 1: DOG_FIGHT       → NEVER trigger ultrasonic
        Rule 2: No humans       → NEVER classify as DANGER
        Rule 3: DANGER + humans → trigger ultrasonic
        Rule 4: Scream + humans → always escalate to DANGER
        Rule 5: Growl + ALERT   → upgrade to DANGER
    """

    def __init__(self):
        self.last_threat_class  = 0
        self.last_threat_label  = "IDLE"
        self.last_confidence    = 0.0
        self.trigger_ultrasonic = False

    def evaluate(self, cnn_results, num_dogs, num_humans, audio_state):
        """
        Args:
            cnn_results : list of (threat_class:int, threat_label:str, confidence:float)
            num_dogs    : int
            num_humans  : int
            audio_state : dict  bark/growl/scream (bool), level (float)

        Returns:
            dict: threat_class, threat_label, confidence, trigger_ultrasonic, reason
        """
        audio_growl  = audio_state.get("growl",  False)
        audio_scream = audio_state.get("scream", False)

        # No dogs → IDLE, no trigger
        if not cnn_results:
            return self._make_result(
                self._idle_class(), "IDLE", 0.0, False, "No dogs detected"
            )

        # Pick highest-priority CNN prediction across all detected dogs
        best_label = "IDLE"
        best_conf  = 0.0
        for tc, tl, conf in cnn_results:
            label = tl if tl in THREAT_PRIORITY else _label(tc)
            if _priority(label) > _priority(best_label):
                best_label = label
                best_conf  = conf
            elif _priority(label) == _priority(best_label) and conf > best_conf:
                best_conf = conf

        threat_label = best_label
        threat_class = self._class_for_label(best_label)
        confidence   = best_conf
        reason       = f"CNN: {threat_label} ({confidence:.0%})"

        # Rule 5: Growl + ALERT → DANGER
        if threat_label == "ALERT" and audio_growl:
            threat_label = "DANGER"
            threat_class = self._class_for_label("DANGER")
            confidence   = max(confidence, 0.75)
            reason       = "Rule 5: Growl + ALERT → DANGER"

        # Rule 4: Scream + humans → DANGER
        if audio_scream and num_humans > 0:
            threat_label = "DANGER"
            threat_class = self._class_for_label("DANGER")
            confidence   = max(confidence, 0.90)
            reason       = "Rule 4: Human scream → DANGER"

        # Rule 1: DOG_FIGHT → never trigger
        if threat_label == "DOG_FIGHT":
            return self._make_result(
                threat_class, "DOG_FIGHT", confidence, False,
                "Rule 1: DOG_FIGHT → no ultrasonic"
            )

        # Rule 2: No humans → downgrade DANGER
        if num_humans == 0 and threat_label == "DANGER":
            fallback = "ALERT" if "ALERT" in self._available_labels() else "IDLE"
            return self._make_result(
                self._class_for_label(fallback), fallback, confidence, False,
                f"Rule 2: No humans → downgrade to {fallback}"
            )

        # Rule 3: DANGER + humans → trigger ultrasonic
        if threat_label == "DANGER" and num_humans > 0:
            return self._make_result(
                threat_class, "DANGER", confidence, True,
                "Rule 3: DANGER + humans → TRIGGER ultrasonic"
            )

        # Default — no trigger
        reasons = {"IDLE": "IDLE — no threat", "ALERT": "ALERT — monitoring"}
        reason = reasons.get(threat_label, reason)
        return self._make_result(threat_class, threat_label, confidence, False, reason)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _available_labels(self):
        return set(THREAT_CLASSES.values())

    def _idle_class(self):
        for idx, lbl in THREAT_CLASSES.items():
            if lbl == "IDLE":
                return idx
        return 0

    def _class_for_label(self, label: str) -> int:
        """Label string → class index. Falls back gracefully if label not in config."""
        for idx, lbl in THREAT_CLASSES.items():
            if lbl == label:
                return idx
        return max(THREAT_CLASSES.keys())

    def _make_result(self, threat_class, threat_label, confidence, trigger, reason):
        self.last_threat_class  = threat_class
        self.last_threat_label  = threat_label
        self.last_confidence    = confidence
        self.trigger_ultrasonic = trigger
        return {
            "threat_class":        threat_class,
            "threat_label":        threat_label,
            "confidence":          confidence,
            "trigger_ultrasonic":  trigger,
            "reason":              reason,
        }