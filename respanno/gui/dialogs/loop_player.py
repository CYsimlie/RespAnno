"""Loop-playback dialog for a selected audio segment."""

import time
import numpy as np
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QSlider, QPushButton, QLabel
from PyQt5.QtCore import Qt, QTimer

_sd = None


def _get_sd():
    """Lazy import sounddevice — PortAudio may be absent in headless CI."""
    global _sd
    if _sd is None:
        import sounddevice as _sd
    return _sd

class LoopPlayer(QDialog):
    def __init__(self, audio_data, sr, start_sec, end_sec, region_item, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Loop Playback: {start_sec:.2f}s - {end_sec:.2f}s")
        self.audio_data = audio_data
        self.sr = sr
        self.start = start_sec
        self.end = end_sec
        self.region_item = region_item
        self.viewer = parent
        self.setFixedSize(300, 140)

        self.duration_ms = int((self.end - self.start) * 1000)

        layout = QVBoxLayout()
        self.label = QLabel(f"Playing: {start_sec:.2f}s ~ {end_sec:.2f}s")
        layout.addWidget(self.label)

        self.progress = QSlider(Qt.Horizontal)
        self.progress.setRange(0, self.duration_ms)
        self.progress.setValue(0)
        self.progress.setEnabled(False)
        layout.addWidget(self.progress)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop)
        layout.addWidget(self.btn_stop)

        self.setLayout(layout)

        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self.play_loop)
        self.play_timer.start(self.duration_ms)

        self.progress_timer = QTimer(self)
        self.progress_timer.timeout.connect(self.update_progress)
        self.progress_timer.start(30)

        self.start_time = time.time()
        self.play_loop()

    def play_loop(self):
        start_sample = int(self.start * self.sr)
        end_sample = int(self.end * self.sr)
        _get_sd().stop()
        _get_sd().play(self.audio_data[start_sample:end_sample], self.sr)
        self.start_time = time.time()

    def update_progress(self):
        elapsed = (time.time() - self.start_time) * 1000
        val = int(elapsed) % self.duration_ms
        self.progress.setValue(val)

    def stop(self):
        self.play_timer.stop()
        self.progress_timer.stop()
        _get_sd().stop()
        self.close()



