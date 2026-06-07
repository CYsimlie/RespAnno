"""Clickable QSlider that responds to clicks anywhere on the track."""

from PyQt5.QtWidgets import QSlider, QStyle
from PyQt5.QtCore import Qt

class ClickableSlider(QSlider):
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            val = QStyle.sliderValueFromPosition(
                self.minimum(), self.maximum(), event.pos().x(), self.width())
            self.setValue(val)
            self.sliderMoved.emit(val)  # triggerslider移动signal
        super().mousePressEvent(event)





