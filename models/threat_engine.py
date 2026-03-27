"""
models/threat_engine.py — Decision engine that fuses visual + audio signals.
Enforces all 5 safety rules. Determines final threat level and ultrasonic trigger.
"""

from config import THREAT_CLASSES


class ThreatEngine:
    """
    Fuses CNN behavior classification with audio signals to produce
    a final threat assessment.

    SAFETY RULES (hardcoded — NEVER bypass):
        Rule 1: DOG_FIGHT → NEVER trigger ultrasonic
        Rule 2: No humans in frame → NEVER classify as DANGER
        Rule 3: Only DANGER + humans present → trigger ultrasonic
        Rule 4: Audio scream detected → always escalate to DANGER
        Rule 5: Audio growl + visual ALERT → upgrade to DANGER
    """

    def __init__(self):
        self.last_threat_class = 0
        self.last_threat_label = "IDLE"
        self.last_confidence = 0.0
        self.trigger_ultrasonic = False

    def evaluate(
        self,
        cnn_results,
        num_dogs,
        num_humans,
        audio_state,
    ):
        """
        Determine final threat level from all signals.

        Args:
            cnn_results: list of (threat_class, threat_label, confidence)
                         from BehaviorClassifier for each detected dog.
            num_dogs: int, number of dogs detected in frame.
            num_humans: int, number of humans detected in frame.
            audio_state: dict with keys:
                'bark' (bool), 'growl' (bool), 'scream' (bool), 'level' (float).

        Returns:
            dict: {
                'threat_class': int,
                'threat_label': str,
                'confidence': float,
                'trigger_ultrasonic': bool,
                'reason': str,
            }
        """
        audio_bark = audio_state.get("bark", False)
        audio_growl = audio_state.get("growl", False)
        audio_scream = audio_state.get("scream", False)

        # Default: IDLE
        threat_class = 0
        threat_label = "IDLE"
        confidence = 0.0
        trigger = False
        reason = "No dogs detected"

        if not cnn_results:
            # No dogs → IDLE, no trigger
            return self._make_result(0, "IDLE", 0.0, False, reason)

        # Get highest-priority CNN prediction
        # Priority: DANGER > DOG_FIGHT > ALERT > IDLE
        best_class = 0
        best_conf = 0.0
        for tc, tl, conf in cnn_results:
            if tc > best_class or (tc == best_class and conf > best_conf):
                best_class = tc
                best_conf = conf

        threat_class = best_class
        threat_label = THREAT_CLASSES.get(threat_class, "IDLE")
        confidence = best_conf

        # ── RULE 5: Audio growl + visual ALERT → upgrade to DANGER ──
        if threat_class == 1 and audio_growl:
            threat_class = 2
            threat_label = "DANGER"
            confidence = max(confidence, 0.75)
            reason = "Rule 5: Growl + ALERT → DANGER"

        # ── RULE 4: Audio scream → always DANGER ──
        if audio_scream and num_humans > 0:
            threat_class = 2
            threat_label = "DANGER"
            confidence = max(confidence, 0.90)
            reason = "Rule 4: Human scream → DANGER"

        # ── RULE 1: DOG_FIGHT → NEVER trigger ultrasonic ──
        if threat_class == 3:
            trigger = False
            reason = "Rule 1: DOG_FIGHT → no ultrasonic"
            return self._make_result(threat_class, "DOG_FIGHT", confidence, False, reason)

        # ── RULE 2: No humans → NEVER classify as DANGER ──
        if num_humans == 0 and threat_class == 2:
            threat_class = 1  # downgrade to ALERT
            threat_label = "ALERT"
            trigger = False
            reason = "Rule 2: No humans → downgrade DANGER to ALERT"
            return self._make_result(threat_class, threat_label, confidence, False, reason)

        # ── RULE 3: Only DANGER + humans → trigger ultrasonic ──
        if threat_class == 2 and num_humans > 0:
            trigger = True
            reason = "Rule 3: DANGER + humans → TRIGGER ultrasonic"
        else:
            trigger = False
            if threat_class == 0:
                reason = "IDLE — no threat"
            elif threat_class == 1:
                reason = "ALERT — monitoring"

        return self._make_result(threat_class, threat_label, confidence, trigger, reason)

    def _make_result(self, threat_class, threat_label, confidence, trigger, reason):
        self.last_threat_class = threat_class
        self.last_threat_label = threat_label
        self.last_confidence = confidence
        self.trigger_ultrasonic = trigger

        return {
            "threat_class": threat_class,
            "threat_label": threat_label,
            "confidence": confidence,
            "trigger_ultrasonic": trigger,
            "reason": reason,
        }
