"""Color bar widget showing a vertical gradient, plus histogram widget.

ColorBarWidget: 32px-wide vertical color bar using the viewer's palette.
HistogramWidget: embedded histogram + color bar for STFT display settings.
"""

import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QWidget, QHBoxLayout
from PyQt5.QtGui import QImage, QPainter, QColor

class ColorBarWidget(QWidget):
    """
    右侧的纯Display色条，不可点击，不会生成三角控件。
    直接用 AudioViewer._get_palette_256 画出 0~1 的渐变。
    """

    def __init__(self, parent=None, viewer=None):
        super().__init__(parent)
        self.viewer = viewer  # 主窗口，用来拿 _get_palette_256
        self._img = None
        self.setFixedWidth(32)  # 瘦瘦的一条，类似 colorbar

    def set_cmap(self, name: str):
        import numpy as np

        lut = None
        if self.viewer is not None and hasattr(self.viewer, "_get_palette_256"):
            try:
                lut = self.viewer._get_palette_256(name)
            except Exception:
                lut = None

        if lut is None:
            # 兜底：自己构造一个简单 LUT
            if name == "Heatmap":
                # 粗略 viridis 风格
                pts = np.array([
                    (68, 1, 84),
                    (59, 82, 139),
                    (33, 145, 140),
                    (94, 201, 98),
                    (253, 231, 37),
                ], float) / 255.0
                xs = np.linspace(0, 1, 256)
                lut = np.empty((256, 3), float)
                for k in range(3):
                    lut[:, k] = np.interp(xs, np.linspace(0, 1, len(pts)), pts[:, k])
            else:
                g = np.linspace(0, 1, 256)
                lut = np.stack([g, g, g], axis=1)

        lut = np.clip(np.asarray(lut, float), 0.0, 1.0)
        h = lut.shape[0]
        img = QImage(1, h, QImage.Format_RGB32)
        for i in range(h):
            r = int(lut[i, 0] * 255)
            g = int(lut[i, 1] * 255)
            b = int(lut[i, 2] * 255)
            img.setPixel(0, h - 1 - i, QColor(r, g, b).rgb())  # 低值在底部
        self._img = img
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        rect = self.rect()
        painter.fillRect(rect, Qt.black)
        if self._img is not None:
            painter.drawImage(rect, self._img)

    def __init__(self, parent=None, stft_vals=None, viewer=None):
        super().__init__(parent)
        import numpy as np
        import pyqtgraph as pg

        self.viewer = viewer
        self._vals = stft_vals

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.hist_plot = pg.PlotWidget()
        self.hist_plot.setBackground('k')
        self.hist_plot.showGrid(x=True, y=True, alpha=0.3)
        self.hist_plot.setLabel('bottom', 'Amplitude')
        self.hist_plot.setLabel('left', 'Count')
        layout.addWidget(self.hist_plot, 1)

        self.colorbar = ColorBarWidget(viewer=viewer)
        layout.addWidget(self.colorbar, 0)

        # 上lower limit指示线
        self.min_line = pg.InfiniteLine(angle=90, pen=pg.mkPen('y'))
        self.max_line = pg.InfiniteLine(angle=90, pen=pg.mkPen('y'))
        self.hist_plot.addItem(self.min_line)
        self.hist_plot.addItem(self.max_line)

        self._init_hist()

    def _init_hist(self):
        import numpy as np

        self.hist_plot.clear()
        self.hist_plot.addItem(self.min_line)
        self.hist_plot.addItem(self.max_line)

        if self._vals is None:
            return
        vals = np.asarray(self._vals, float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return

        counts, edges = np.histogram(vals, bins=100)
        centers = 0.5 * (edges[:-1] + edges[1:])
        self.hist_plot.plot(centers, counts, pen='w')

    def update_levels(self, vmin, vmax):
        vmin = float(vmin)
        vmax = float(vmax)
        self.min_line.setPos(vmin)
        self.max_line.setPos(vmax)
        if vmax > vmin:
            self.hist_plot.setXRange(vmin, vmax, padding=0.1)

    def set_cmap(self, name: str):
        self.colorbar.set_cmap(name)


from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QSpinBox,
    QDialogButtonBox, QTabWidget, QWidget, QLabel,
    QSlider, QHBoxLayout, QDoubleSpinBox, QCheckBox
)
from PyQt5.QtWidgets import QSplitter, QStackedWidget, QListWidget, QListWidgetItem, QGraphicsRectItem
from PyQt5.QtWidgets import QComboBox, QDoubleSpinBox, QGroupBox, QLabel, QHBoxLayout, QPushButton
import numpy as np
from pyqtgraph import HistogramLUTWidget, ImageItem, ColorMap
from PyQt5.QtWidgets import QStyledItemDelegate, QStyle
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QPainterPath
from PyQt5.QtCore import QSize, QRect, Qt, QEvent



