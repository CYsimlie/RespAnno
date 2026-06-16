"""Custom ViewBox for the waveform panel with drag-to-mark."""

import pyqtgraph as pg
from PyQt5.QtCore import Qt

class WaveViewBox(pg.ViewBox):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setMouseMode(self.PanMode)
        self.is_marking = False
        self.start_pos = None
        self.temp_region = None

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            # When starting a new annotation from the waveform, deselect any current annotation.
            self.parent._selected_span = None
            self.is_marking = True
            self.start_pos = self.mapToView(ev.pos()).x()
            self.parent.plot_waveform_highlight(None, None)
            if self.temp_region:
                self.parent.wave_plot.removeItem(self.temp_region)
            self.temp_region = pg.LinearRegionItem([self.start_pos, self.start_pos], brush=(255, 0, 0, 50))
            self.temp_region.setZValue(10)
            self.temp_region.setMovable(False)
            self.parent.wave_plot.addItem(self.temp_region)
            ev.accept()
        else:
            ev.ignore()

    def mouseMoveEvent(self, ev):
        if self.is_marking and self.temp_region:
            current_x = self.mapToView(ev.pos()).x()
            self.temp_region.setRegion([self.start_pos, current_x])
            self.parent.plot_waveform_highlight(self.start_pos, current_x)
            ev.accept()
        else:
            ev.ignore()

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.LeftButton and self.is_marking:
            end_pos = self.mapToView(ev.pos()).x()
            self.is_marking = False
            if self.temp_region:
                self.parent.wave_plot.removeItem(self.temp_region)
                self.temp_region = None
            if abs(end_pos - self.start_pos) < 0.001:
                return
            start, end = sorted([self.start_pos, end_pos])
            self.parent.finalize_annotation(start, end)
            ev.accept()
        else:
            ev.ignore()



