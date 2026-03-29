"""
hardware/ultrasonic_hw.py — Hardware ultrasonic trigger via Arduino.
Manages trigger/stop commands with cooldown, integrating with ArduinoBridge.
Drop-in replacement for audio/ultrasonic_trigger.py in hardware mode.
"""

import time
import threading

from config import ULTRASONIC_DURATION, ULTRASONIC_COOLDOWN
from hardware.arduino_bridge import ArduinoBridge


class UltrasonicHardware:
    """
    Controls external ultrasonic emitter via Arduino serial.
    Same interface as UltrasonicTrigger (software version) so it's
    a drop-in replacement in the main pipeline.
    """

    def __init__(self, arduino=None):
        self.arduino = arduino or ArduinoBridge()
        self._last_trigger_time = 0.0
        self._is_playing = False
        self._lock = threading.Lock()
        self._trigger_count = 0

        # Try to connect to Arduino
        if not self.arduino.is_connected:
            self.arduino.connect()

        mode = "HARDWARE" if self.arduino.is_connected else "SIMULATED"
        print(f"[ULTRASONIC-HW] Mode: {mode}, "
              f"duration={ULTRASONIC_DURATION}s, cooldown={ULTRASONIC_COOLDOWN}s")

    def trigger(self):
        """
        Trigger ultrasonic emitter via Arduino.
        Respects cooldown. Auto-stops after ULTRASONIC_DURATION seconds.

        Returns:
            bool: True if triggered, False if on cooldown or already active.
        """
        with self._lock:
            now = time.time()
            if now - self._last_trigger_time < ULTRASONIC_COOLDOWN:
                return False
            if self._is_playing:
                return False

            self._last_trigger_time = now
            self._is_playing = True

        # Trigger in background thread (auto-stops after duration)
        thread = threading.Thread(target=self._trigger_cycle, daemon=True)
        thread.start()
        return True

    def _trigger_cycle(self):
        """Send ON command, wait duration, send OFF command."""
        try:
            self._trigger_count += 1
            mode = "HW" if self.arduino.is_connected else "SIM"
            print(f"[ULTRASONIC-{mode}] TRIGGERED (#{self._trigger_count})")

            self.arduino.trigger_ultrasonic()
            time.sleep(ULTRASONIC_DURATION)
            self.arduino.stop_ultrasonic()
        except Exception as e:
            print(f"[ULTRASONIC-HW] Error: {e}")
        finally:
            with self._lock:
                self._is_playing = False

    def stop(self):
        """Force stop the ultrasonic emitter."""
        self.arduino.stop_ultrasonic()
        with self._lock:
            self._is_playing = False

    @property
    def is_on_cooldown(self):
        return time.time() - self._last_trigger_time < ULTRASONIC_COOLDOWN

    @property
    def is_playing(self):
        return self._is_playing

    @property
    def total_triggers(self):
        return self._trigger_count

    def cleanup(self):
        self.stop()
        self.arduino.disconnect()
