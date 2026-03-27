"""
audio/audio_combiner.py — Combines audio detection state with visual detection
to produce a unified signal for the threat engine.
"""


class AudioCombiner:
    """
    Merges audio signals with visual detection results.
    Provides a clean interface between AudioDetector and ThreatEngine.
    """

    def __init__(self):
        self._last_audio_state = {
            "bark": False,
            "growl": False,
            "scream": False,
            "level": 0.0,
        }

    def update(self, audio_state):
        """
        Update with latest audio state from AudioDetector.

        Args:
            audio_state: dict from AudioDetector.get_state()
        """
        self._last_audio_state = dict(audio_state)

    def get_combined_state(self, num_dogs, num_humans):
        """
        Get combined audio state with context awareness.

        Returns audio state filtered by what makes sense given the visual scene:
        - Bark/growl only relevant if dogs are in frame
        - Scream only relevant if humans are in frame

        Args:
            num_dogs: number of dogs detected visually.
            num_humans: number of humans detected visually.

        Returns:
            dict with filtered audio state.
        """
        state = dict(self._last_audio_state)

        # Only report bark/growl if dogs are visible
        if num_dogs == 0:
            state["bark"] = False
            state["growl"] = False

        # Only report scream if humans are visible
        if num_humans == 0:
            state["scream"] = False

        return state

    @property
    def raw_state(self):
        """Get unfiltered audio state."""
        return dict(self._last_audio_state)
