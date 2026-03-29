"""
hardware/arduino_bridge.py — Serial communication bridge to Arduino.
Sends ultrasonic trigger commands over USB serial.
Designed for future integration with Arduino + ultrasonic sensor hardware.

Arduino Expected Behavior:
  - Receives "ULTRASONIC_ON\n"  → activates ultrasonic emitter
  - Receives "ULTRASONIC_OFF\n" → deactivates ultrasonic emitter
  - Receives "STATUS\n"         → replies with "OK" or "EMITTING"

Wiring (future):
  Arduino Uno/Nano → USB → Laptop COM port
  Arduino Pin 9    → Ultrasonic transducer (+)
  Arduino GND      → Ultrasonic transducer (-)
"""

import time
import threading

from config import (
    ARDUINO_PORT,
    ARDUINO_BAUD,
    ARDUINO_TIMEOUT,
    ARDUINO_CMD_TRIGGER,
    ARDUINO_CMD_STOP,
    ARDUINO_CMD_STATUS,
)


class ArduinoBridge:
    """
    Serial communication with Arduino for hardware ultrasonic control.
    Falls back gracefully to simulation mode if Arduino is not connected.
    """

    def __init__(self, port=None, baud=None):
        self.port = port or ARDUINO_PORT
        self.baud = baud or ARDUINO_BAUD
        self._serial = None
        self._connected = False
        self._lock = threading.Lock()
        self._emitting = False

    def connect(self):
        """Attempt to connect to Arduino. Returns True if successful."""
        try:
            import serial
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baud,
                timeout=ARDUINO_TIMEOUT,
            )
            time.sleep(2)  # Arduino resets on serial connect
            self._connected = True
            print(f"[ARDUINO] Connected on {self.port} @ {self.baud} baud")
            return True
        except ImportError:
            print("[ARDUINO] pyserial not installed — run: pip install pyserial")
            print("[ARDUINO] Falling back to simulation mode")
            return False
        except Exception as e:
            print(f"[ARDUINO] Connection failed on {self.port}: {e}")
            print("[ARDUINO] Falling back to simulation mode")
            return False

    def disconnect(self):
        """Close serial connection."""
        if self._serial and self._serial.is_open:
            self.stop_ultrasonic()
            self._serial.close()
            self._connected = False
            print("[ARDUINO] Disconnected")

    def trigger_ultrasonic(self):
        """Send trigger command to Arduino."""
        if not self._connected:
            print("[ARDUINO-SIM] ULTRASONIC ON (simulated — no hardware)")
            self._emitting = True
            return True

        with self._lock:
            try:
                self._serial.write(f"{ARDUINO_CMD_TRIGGER}\n".encode())
                self._serial.flush()
                self._emitting = True
                print(f"[ARDUINO] Sent: {ARDUINO_CMD_TRIGGER}")
                return True
            except Exception as e:
                print(f"[ARDUINO] Send error: {e}")
                return False

    def stop_ultrasonic(self):
        """Send stop command to Arduino."""
        if not self._connected:
            self._emitting = False
            return True

        with self._lock:
            try:
                self._serial.write(f"{ARDUINO_CMD_STOP}\n".encode())
                self._serial.flush()
                self._emitting = False
                print(f"[ARDUINO] Sent: {ARDUINO_CMD_STOP}")
                return True
            except Exception as e:
                print(f"[ARDUINO] Send error: {e}")
                return False

    def get_status(self):
        """Query Arduino status."""
        if not self._connected:
            return "SIMULATED"

        with self._lock:
            try:
                self._serial.write(f"{ARDUINO_CMD_STATUS}\n".encode())
                self._serial.flush()
                response = self._serial.readline().decode().strip()
                return response or "NO_RESPONSE"
            except Exception as e:
                return f"ERROR: {e}"

    @property
    def is_connected(self):
        return self._connected

    @property
    def is_emitting(self):
        return self._emitting

    def __del__(self):
        self.disconnect()
