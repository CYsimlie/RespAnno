"""Custom ViewBox for the annotation panel with click-to-select and drag-to-mark."""

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMenu, QApplication

class AnnotViewBox(pg.ViewBox):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setMouseMode(self.PanMode)  # Disable canvas panning by default.
        self.is_marking = False
        self.start_pos = None
        self.temp_region = None


    def _hit_span_under_cursor(self, ev):
        """If the mouse is over a BoxSpan (or its handle/label), return it; else None."""
        from respanno.gui.spans.box_span import BoxSpan

        try:
            scene = self.scene()
        except Exception:
            scene = None
        if scene is None:
            return None

        # Map to scene coordinates.
        sp = None
        try:
            sp = ev.scenePos()
        except Exception:
            try:
                sp = self.mapToScene(ev.pos())
            except Exception:
                sp = None
        if sp is None:
            return None

        try:
            items = scene.items(sp)
        except Exception:
            items = []

        # Items sorted by Z-value descending; return topmost span.
        for it in items:
            # (1) Direct BoxSpan hit.
            if isinstance(it, BoxSpan):
                return it

            # (2) Label hit (TextItem with _owner_span).
            try:
                spn = getattr(it, "_owner_span", None)
                if isinstance(spn, BoxSpan):
                    return spn
            except Exception:
                pass

            # (3) ROI child hit: walk up parent chain.
            p = it
            for _ in range(6):
                try:
                    p = p.parentItem()
                except Exception:
                    p = None
                if p is None:
                    break
                if isinstance(p, BoxSpan):
                    return p

        return None

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            hit = self._hit_span_under_cursor(ev)
            if hit is not None:
                # Click on existing annotation: select and delegate to BoxSpan.
                self.parent._selected_span = hit
                ev.ignore()
                return
            # Click on empty region: deselect.
            self.parent._selected_span = None
            self.is_marking = True
            self.start_pos = self.mapToView(ev.pos()).x()
            self.parent.plot_waveform_highlight(None, None)
            if self.temp_region:
                self.parent.annot_plot.removeItem(self.temp_region)
            self.temp_region = pg.LinearRegionItem([self.start_pos, self.start_pos], brush=(0, 255, 255, 50))
            self.temp_region.setZValue(10)
            self.parent.annot_plot.addItem(self.temp_region)
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

            start, end = self.start_pos, end_pos
            if abs(end - start) < 0.001:
                if self.temp_region:
                    self.parent.annot_plot.removeItem(self.temp_region)
                    self.temp_region = None
                return
            if end < start:
                start, end = end, start

            # Delegate to main window for the annotation label dialog.
            self.parent.finalize_annotation(start, end)

            if self.temp_region:
                self.parent.annot_plot.removeItem(self.temp_region)
                self.temp_region = None
            ev.accept()

        else:
            ev.ignore()

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() == Qt.RightButton:
            ev.accept()  # disable right-click zoom/pan
        else:
            super().mouseDragEvent(ev, axis)



