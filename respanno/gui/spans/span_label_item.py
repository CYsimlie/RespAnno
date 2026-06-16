"""Text label item placed on annotation span bars."""

import pyqtgraph as pg
from PyQt5.QtCore import Qt

class SpanLabelItem(pg.TextItem):
    # Text label forwarding to BoxSpan for right-click / double-click events.
    def __init__(self, *args, owner_span=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._owner_span = owner_span
        try:
            self.setAcceptedMouseButtons(Qt.LeftButton | Qt.RightButton)
        except Exception:
            pass

    def mouseDoubleClickEvent(self, ev):
        if self._owner_span is not None:
            try:
                self._owner_span.mouseDoubleClickEvent(ev)
            except Exception:
                pass
            ev.accept()
        else:
            ev.ignore()

    def mouseClickEvent(self, ev):
        # Right-click menu delegated to BoxSpan.
        if self._owner_span is not None:
            if ev.button() == Qt.RightButton:
                try:
                    self._owner_span.mouseClickEvent(ev)
                except Exception:
                    pass
            ev.accept()
        else:
            ev.ignore()



