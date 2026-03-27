"""
audio/ultrasonic_trigger.py — Generates and plays ultrasonic sound (20-25kHz)
through laptop speaker when dog-to-human DANGER is confirmed.
Includes cooldown to prevent rapid re-triggering.
"""

import time
import threading
import numpy as np
import sounddevice as sd

from config import (
    AUDIO_SAMPLE_RATE,
    ULTRASONIC_FREQ_MIN,
    ULTRASONIC_FREQ_MAX,
    ULTRASONIC_DURATION,
    ULTRASONIC_COOLDOWN,
    ULTRASONIC_VOLUME,
)


class UltrasonicTrigger:
    """
    Plays ultrasonic frequency sweep (20-25kHz) through laptop speaker.
    Only triggers on DANGER threat level with humans present.
    Has built-in cooldown to prevent audio spam.
    """

    def __init__(self):
        self._last_trigger_time = 0.0
        self._is_playing = False
        self._lock = threading.Lock()
        self._trigger_count = 0

        # Pre-generate the ultrasonic waveform
        self._waveform = self._generate_waveform()
        print(f"[ULTRASONIC] Ready — {ULTRASONIC_FREQ_MIN}-{ULTRASONIC_FREQ_MAX}Hz, "
              f"duration={ULTRASONIC_DURATION}s, cooldown={ULTRASONIC_COOLDOWN}s")

    def _generate_waveform(self):
        """
        Generate a frequency sweep from FREQ_MIN to FREQ_MAX.
        Uses linear chirp for maximum coverage of ultrasonic range.
        """
        t = np.linspace(0, ULTRASONIC_DURATION, int(AUDIO_SAMPLE_RATE * ULTRASONIC_DURATION))

        # Linear frequency sweep (chirp)
        freq_sweep = np.linspace(ULTRASONIC_FREQ_MIN, ULTRASONIC_FREQ_MAX, len(t))
        phase = 2 * np.pi * np.cumsum(freq_sweep) / AUDIO_SAMPLE_RATE
        waveform = np.sin(phase)

        # Apply fade-in/fade-out envelope to prevent clicks
        fade_samples = int(0.05 * AUDIO_SAMPLE_RATE)  # 50ms fade
        envelope = np.ones_like(waveform)
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)

        waveform *= envelope * ULTRASONIC_VOLUME
        return waveform.astype(np.float32)

    def trigger(self):
        """
        Attempt to play ultrasonic sound.
        Respects cooldown period. Non-blocking (plays in background thread).

        Returns:
            bool: True if sound was triggered, False if on cooldown or already playing.
        """
        with self._lock:
            now = time.time()

            # Check cooldown
            if now - self._last_trigger_time < ULTRASONIC_COOLDOWN:
                return False

            # Check if already playing
            if self._is_playing:
                return False

            self._last_trigger_time = now
            self._is_playing = True

        # Play in background thread
        thread = threading.Thread(target=self._play, daemon=True)
        thread.start()
        return True

    def _play(self):
        """Play the ultrasonic waveform through speakers."""
        try:
            self._trigger_count += 1
            print(f"[ULTRASONIC] TRIGGERED (#{self._trigger_count}) — "
                  f"Playing {ULTRASONIC_FREQ_MIN}-{ULTRASONIC_FREQ_MAX}Hz "
                  f"for {ULTRASONIC_DURATION}s")
            sd.play(self._waveform, samplerate=AUDIO_SAMPLE_RATE)
            sd.wait()
        except Exception as e:
            print(f"[ULTRASONIC] Playback error: {e}")
        finally:
            with self._lock:
                self._is_playing = False

    @property
    def is_on_cooldown(self):
        """Check if trigger is on cooldown."""
        return time.time() - self._last_trigger_time < ULTRASONIC_COOLDOWN

    @property
    def is_playing(self):
        return self._is_playing

    @property
    def total_triggers(self):
        return self._trigger_count
