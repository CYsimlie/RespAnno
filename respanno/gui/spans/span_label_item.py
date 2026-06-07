"""Text label item placed on annotation span bars."""

import pyqtgraph as pg
from PyQt5.QtCore import Qt

class SpanLabelItem(pg.TextItem):
    # "\"\"annotation条上的文字label：把右键/双击event转发给所属 BoxSpan，避免被mark拖拽逻辑抢走。\"\"\"
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
        # 右键menu交给 BoxSpan；左键仅用于阻止trigger“拖拽mark”
        if self._owner_span is not None:
            if ev.button() == Qt.RightButton:
                try:
                    self._owner_span.mouseClickEvent(ev)
                except Exception:
                    pass
            ev.accept()
        else:
            ev.ignore()



