"""
audio/audio_detector.py — Real-time microphone audio analysis.
Detects dog barking, growling, and human screaming using signal processing.
No ML model needed — uses frequency band energy analysis.
"""

import threading
import numpy as np
import sounddevice as sd
from scipy import signal as scipy_signal

from config import (
    AUDIO_SAMPLE_RATE,
    AUDIO_CHANNELS,
    AUDIO_BLOCK_SIZE,
    AUDIO_BARK_THRESHOLD,
    AUDIO_GROWL_THRESHOLD,
    AUDIO_SCREAM_THRESHOLD,
)


class AudioDetector:
    """
    Real-time audio detector using laptop microphone.
    Analyzes frequency bands to detect:
      - Dog barking (300–1000 Hz, sharp transients)
      - Dog growling (80–300 Hz, sustained low frequency)
      - Human screaming (1000–4000 Hz, high energy sustained)
    """

    # Frequency band definitions (Hz)
    GROWL_BAND = (80, 300)
    BARK_BAND = (300, 1000)
    SCREAM_BAND = (1000, 4000)

    def __init__(self):
        self._running = False
        self._thread = None
        self._lock = threading.Lock()

        # Current audio state
        self._state = {
            "bark": False,
            "growl": False,
            "scream": False,
            "level": 0.0,
        }

        # Rolling buffers for temporal smoothing
        self._bark_history = []
        self._growl_history = []
        self._scream_history = []
        self._history_len = 10  # number of frames to average

    def start(self):
        """Start audio capture in background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print("[AUDIO] Microphone capture started")

    def stop(self):
        """Stop audio capture."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        print("[AUDIO] Microphone capture stopped")

    def get_state(self):
        """Get current audio detection state (thread-safe)."""
        with self._lock:
            return dict(self._state)

    def _capture_loop(self):
        """Main capture loop running in background thread."""
        try:
            with sd.InputStream(
                samplerate=AUDIO_SAMPLE_RATE,
                channels=AUDIO_CHANNELS,
                blocksize=AUDIO_BLOCK_SIZE,
                callback=self._audio_callback,
            ):
                while self._running:
                    sd.sleep(100)
        except Exception as e:
            print(f"[AUDIO] Error: {e}")
            self._running = False

    def _audio_callback(self, indata, frames, time_info, status):
        """Called by sounddevice for each audio block."""
        if status:
            pass  # Ignore overflow warnings

        audio = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        audio = audio.astype(np.float64)

        # Compute overall RMS level
        rms = float(np.sqrt(np.mean(audio ** 2)))

        # Compute frequency spectrum
        freqs, psd = scipy_signal.welch(
            audio,
            fs=AUDIO_SAMPLE_RATE,
            nperseg=min(len(audio), 1024),
        )

        # Measure energy in each frequency band
        growl_energy = self._band_energy(freqs, psd, *self.GROWL_BAND)
        bark_energy = self._band_energy(freqs, psd, *self.BARK_BAND)
        scream_energy = self._band_energy(freqs, psd, *self.SCREAM_BAND)

        # Update rolling histories
        self._bark_history.append(bark_energy)
        self._growl_history.append(growl_energy)
        self._scream_history.append(scream_energy)

        # Trim histories
        for hist in (self._bark_history, self._growl_history, self._scream_history):
            while len(hist) > self._history_len:
                hist.pop(0)

        # Smoothed detection with thresholds
        avg_bark = np.mean(self._bark_history)
        avg_growl = np.mean(self._growl_history)
        avg_scream = np.mean(self._scream_history)

        with self._lock:
            self._state["bark"] = avg_bark > AUDIO_BARK_THRESHOLD
            self._state["growl"] = avg_growl > AUDIO_GROWL_THRESHOLD
            self._state["scream"] = avg_scream > AUDIO_SCREAM_THRESHOLD
            self._state["level"] = min(rms * 5.0, 1.0)  # scale for display

    @staticmethod
    def _band_energy(freqs, psd, low, high):
        """Compute energy in a frequency band."""
        mask = (freqs >= low) & (freqs <= high)
        if not np.any(mask):
            return 0.0
        return float(np.mean(psd[mask]))
