"""
utils/logger.py — CSV event logger for threat detection events.
Logs timestamp, threat level, confidence, audio state, and actions taken.
"""

import csv
import os
from datetime import datetime
from pathlib import Path

from config import LOG_DIR, LOG_FILE_PREFIX, LOG_EVENTS


class EventLogger:
    """Logs detection events to a CSV file with auto-rotating daily files."""

    HEADERS = [
        "timestamp",
        "frame_num",
        "num_dogs",
        "num_humans",
        "threat_class",
        "threat_label",
        "confidence",
        "audio_bark",
        "audio_growl",
        "audio_scream",
        "ultrasonic_triggered",
        "notes",
    ]

    def __init__(self):
        self.enabled = LOG_EVENTS
        self.log_dir = Path(LOG_DIR)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._current_date = None
        self._writer = None
        self._file = None
        self._frame_count = 0

    def _get_log_path(self) -> Path:
        date_str = datetime.now().strftime("%Y-%m-%d")
        return self.log_dir / f"{LOG_FILE_PREFIX}_{date_str}.csv"

    def _ensure_writer(self):
        today = datetime.now().date()
        if self._current_date != today:
            self.close()
            self._current_date = today
            log_path = self._get_log_path()
            file_exists = log_path.exists()
            self._file = open(log_path, "a", newline="", encoding="utf-8")
            self._writer = csv.writer(self._file)
            if not file_exists:
                self._writer.writerow(self.HEADERS)

    def log(
        self,
        num_dogs: int,
        num_humans: int,
        threat_class: int,
        threat_label: str,
        confidence: float,
        audio_bark: bool = False,
        audio_growl: bool = False,
        audio_scream: bool = False,
        ultrasonic_triggered: bool = False,
        notes: str = "",
    ):
        if not self.enabled:
            return

        self._ensure_writer()
        self._frame_count += 1
        self._writer.writerow([
            datetime.now().isoformat(),
            self._frame_count,
            num_dogs,
            num_humans,
            threat_class,
            threat_label,
            f"{confidence:.4f}",
            int(audio_bark),
            int(audio_growl),
            int(audio_scream),
            int(ultrasonic_triggered),
            notes,
        ])
        self._file.flush()

    def close(self):
        if self._file and not self._file.closed:
            self._file.close()
        self._file = None
        self._writer = None

    def __del__(self):
        self.close()
