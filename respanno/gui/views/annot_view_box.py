"""Custom ViewBox for the annotation panel with click-to-select and drag-to-mark."""

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMenu, QApplication

class AnnotViewBox(pg.ViewBox):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setMouseMode(self.PanMode)  # 默认禁用拖动画布
        self.is_marking = False
        self.start_pos = None
        self.temp_region = None


    def _hit_span_under_cursor(self, ev):
        # "\"\"若鼠标位于某个 BoxSpan (或其把手/文字标签)之上，返回该 BoxSpan；否则返回 None。\"\"\"
        try:
            scene = self.scene()
        except Exception:
            scene = None
        if scene is None:
            return None

        # 取 scene 坐标
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

        # items 按 Z 值从高到低排列；优先返回最上层的 span
        for it in items:
            # 1) 直接命中 BoxSpan
            if isinstance(it, BoxSpan):
                return it

            # 2) 命中 label (TextItem 上挂了 _owner_span)
            try:
                spn = getattr(it, "_owner_span", None)
                if isinstance(spn, BoxSpan):
                    return spn
            except Exception:
                pass

            # 3) 命中 ROI 子对象 (handles 等)：向上找 parentItem
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
                # 点击/双击落在已有标注上：交给 BoxSpan 处理，避免触发“拖拽标记”逻辑
                ev.ignore()
                return
            self.is_marking = True
            self.start_pos = self.mapToView(ev.pos()).x()
            self.parent.plot_waveform_highlight(None, None)  # 清除高亮
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

            # 交由主窗口统一弹出“标记类型选择”对话框
            self.parent.finalize_annotation(start, end)

            if self.temp_region:
                self.parent.annot_plot.removeItem(self.temp_region)
                self.temp_region = None
            ev.accept()

        else:
            ev.ignore()

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() == Qt.RightButton:
            ev.accept()  # 禁用右键缩放/平移
            # 你可以在这里扩展右键拖动的用途
        else:
            super().mouseDragEvent(ev, axis)



