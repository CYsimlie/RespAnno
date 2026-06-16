import os
import sys

# 1. 拼接出 plugins File夹的绝对路径
# 注意：根据你的实际安装情况，可能需要调整 'PyQt5', 'Qt5', 'plugins' 这些层级
base_path = os.path.join(sys.base_prefix, "Lib", "site-packages", "PyQt5", "Qt5", "plugins")

# 2. 检查路径是否存在（如果不存在，尝试去掉 'Qt5' 这一层）
if not os.path.exists(base_path):
    base_path = os.path.join(sys.base_prefix, "Lib", "site-packages", "PyQt5", "Qt", "plugins")

# 3. Settings环境变量
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = base_path

import sys
import os
import time
import sounddevice as sd
import librosa
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog,
    QVBoxLayout, QWidget, QLabel, QDialog, QFormLayout,
    QInputDialog, QMessageBox, QComboBox, QLineEdit, QDialogButtonBox
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import (
    confusion_matrix, accuracy_score, balanced_accuracy_score,
    matthews_corrcoef, roc_auc_score, average_precision_score, brier_score_loss
)

from PyQt5.QtWidgets import QMenu, QAction, QToolBar

from PyQt5.QtCore import Qt, QTimer
import pyqtgraph as pg
from PyQt5.QtWidgets import QMenu
from PyQt5.QtWidgets import QAction
from PyQt5.QtGui import QKeySequence
import pyqtgraph as pg
from PyQt5.QtGui import QImage  # ← 顶部 import 里记得加上这一行
from PyQt5.QtWidgets import QShortcut
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QShortcut


class AnnotationLabelDialog(QDialog):
    """Annotation label dialog: supports preset types and custom text"""

    def __init__(self, parent=None, builtin_labels=None, start=None, end=None, default_text=""):
        super().__init__(parent)
        self.setWindowTitle("Add Annotation")
        self._text = None

        layout = QVBoxLayout(self)

        # 顶部Notice：Time段
        if start is not None and end is not None:
            info = QLabel(f"Annotation interval:{start:.3f} - {end:.3f} s")
            layout.addWidget(info)

        form = QFormLayout()
        layout.addLayout(form)

        # 预设类型下拉框：如 Wheeze(Wheeze)、Crackles(Crackles) 等
        self.combo = QComboBox()
        self.combo.addItem("(No preset)", userData=None)
        if builtin_labels:
            for cn, en in builtin_labels:
                self.combo.addItem(f"{cn} ({en})", userData=en)
        form.addRow("Preset type:", self.combo)

        # 文本输入框：最终用于保存的标签文本（通常是英文）
        self.line_edit = QLineEdit()
        if default_text:
            self.line_edit.setText(default_text)
        form.addRow("Label text:", self.line_edit)

        # 选择预设时，自动把英文名写入文本框
        def on_combo_changed(idx):
            en = self.combo.itemData(idx)
            if en:
                self.line_edit.setText(en)

        self.combo.currentIndexChanged.connect(on_combo_changed)

        # 底部按钮
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self._on_accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _on_accept(self):
        text = self.line_edit.text().strip()
        if not text:
            # 不输入则视为取消
            self.reject()
            return
        self._text = text
        self.accept()

    def get_text(self):
        """模态执行并返回最终文本（可能为 None）。"""
        if self.exec_() == QDialog.Accepted:
            return self._text
        return None


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

        # 上下限指示线
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
    QSlider, QHBoxLayout, QDoubleSpinBox
)
from PyQt5.QtWidgets import QSplitter, QStackedWidget, QListWidget, QListWidgetItem, QGraphicsRectItem
from PyQt5.QtWidgets import QComboBox, QDoubleSpinBox, QGroupBox, QLabel, QHBoxLayout, QPushButton
import numpy as np
from pyqtgraph import HistogramLUTWidget, ImageItem, ColorMap
from PyQt5.QtWidgets import QStyledItemDelegate, QStyle
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QPainterPath
from PyQt5.QtCore import QSize, QRect, Qt, QEvent


class LoopPlayer(QDialog):
    def __init__(self, audio_data, sr, start_sec, end_sec, region_item, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"循环Play：{start_sec:.2f}s - {end_sec:.2f}s")
        self.audio_data = audio_data
        self.sr = sr
        self.start = start_sec
        self.end = end_sec
        self.region_item = region_item
        self.viewer = parent
        self.setFixedSize(300, 140)

        self.duration_ms = int((self.end - self.start) * 1000)

        layout = QVBoxLayout()
        self.label = QLabel(f"Play中：{start_sec:.2f}s ~ {end_sec:.2f}s")
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
        sd.stop()
        sd.play(self.audio_data[start_sample:end_sample], self.sr)
        self.start_time = time.time()

    def update_progress(self):
        elapsed = (time.time() - self.start_time) * 1000
        val = int(elapsed) % self.duration_ms
        self.progress.setValue(val)

    def stop(self):
        self.play_timer.stop()
        self.progress_timer.stop()
        sd.stop()
        self.close()


class FrequencyWindow(QMainWindow):
    def __init__(self, signal, sr):
        super().__init__()
        self.setWindowTitle("Frequency Processing and Analysis")
        self.resize(1600, 900)  # 适配大屏幕

        self.original_signal = signal
        self.signal = signal.copy()
        self.sr = sr

        self.init_ui()


class BoxSpan(pg.RectROI):
    def __init__(self, x0, x1, y_base, height, text, owner, label_color=None):
        super().__init__(pos=[x0, y_base], size=[x1 - x0, height],
                         movable=True, resizable=False,  # 禁掉四角把手
                         pen=pg.mkPen(255, 255, 255, 255, width=1))
        # —— 视觉编码：不再强制纯白填充（后面统一由 _apply_visual_style() 控制）——
        try:
            self.setBrush(pg.mkBrush(0, 0, 0, 0))  # 先透明，避免白块遮挡
        except Exception:
            pass

        self.owner = owner
        self.y_base = float(y_base)  # 锁定纵向
        self.h_fix = float(height)  # 锁定高度
        self.text = text
        self.setZValue(6)
        self.setAcceptedMouseButtons(Qt.LeftButton | Qt.RightButton)

        # 独立填充层：RectROI 在部分 pyqtgraph 版本中只明显Display边框，
        # 因此额外用一个 QGraphicsRectItem 负责稳定Display类别填充色。
        self._fill_item = None
        try:
            self._fill_item = QGraphicsRectItem(float(x0), float(y_base), float(x1 - x0), float(height))
            self._fill_item.setPen(QPen(Qt.NoPen))
            self._fill_item.setBrush(QBrush(QColor(255, 255, 255, 80)))
            self._fill_item.setZValue(self.zValue() - 1)
            self._fill_item.setAcceptedMouseButtons(Qt.NoButton)
            self.owner.annot_plot.addItem(self._fill_item)
        except Exception:
            self._fill_item = None

        # ★ 新增：标签文字颜色（不传则默认黑色）
        if label_color is None:
            self.label_color = QColor(0, 0, 0)  # 默认黑字
        else:
            self.label_color = label_color

        # 只保留左右把手 + 中间平移把手
        hL = self.addScaleHandle([0, 0.5], [1, 0.5])
        hR = self.addScaleHandle([1, 0.5], [0, 0.5])

        def _hide_handle(h):
            try:
                h.setOpacity(0.0)
                h.setPen(pg.mkPen(0, 0, 0, 0))
                h.setBrush(pg.mkBrush(0, 0, 0, 0))
                return
            except Exception:
                pass
            try:
                it = h['item']
                it.setOpacity(0.0)
                it.setPen(pg.mkPen(0, 0, 0, 0))
                it.setBrush(pg.mkBrush(0, 0, 0, 0))
            except Exception:
                pass

        for h in (hL, hR):
            _hide_handle(h)

        def _style_handle(h):
            try:
                h.setPen(pg.mkPen(255, 255, 255, 220))
                h.setBrush(pg.mkBrush(0, 0, 0, 160))
                h.setZValue(self.zValue() + 1)
                return
            except Exception:
                pass
            try:
                it = h['item']
                it.setPen(pg.mkPen(255, 255, 255, 220))
                it.setBrush(pg.mkBrush(0, 0, 0, 160))
                it.setZValue(self.zValue() + 1)
            except Exception:
                pass

        for h in (hL, hR):
            _style_handle(h)

        # 备注小白框（Display在条上方）
        self.label = pg.TextItem(anchor=(0.5, 0.0))  # 先建一个空的
        self.owner.annot_plot.addItem(self.label)
        self._update_label_html()  # 根据 label_color & text 更新 HTML
        # 标注条不直接Display文字，标签含义统一通过工具栏“Annotation Legend”查看
        try:
            self.label.hide()
        except Exception:
            pass

        # —— 视觉编码：根据 source(manual/ml) 统一Settings pen/brush/label 样式 ——
        self._apply_visual_style(self._infer_source())

        self.sigRegionChanged.connect(self._on_changed)
        self._on_changed()

    # ==========================
    # 视觉编码：新增的辅助函数
    # ==========================
    def _infer_source(self) -> str:
        """
        从 owner.annotations 里推断当前 span 的 source。
        找不到就默认 manual。仅用于视觉编码，不改变逻辑。
        """
        try:
            m2 = getattr(self.owner, "_span2idx", {})
            if self in m2:
                idx = m2[self]
                if 0 <= idx < len(self.owner.annotations) and self.owner.annotations[idx] is not None:
                    old = self.owner.annotations[idx]
                    if len(old) >= 4:
                        return str(old[3])
        except Exception:
            pass
        return "manual"

    def _apply_visual_style(self, src: str):
        """
        只做视觉编码：
        - manual：实线、边框更粗、填充更明显
        - ml：虚线/点划线、边框略细、填充更淡
        """
        # 标注条颜色按标签类别稳定映射；source 仅决定线型和透明度
        try:
            c = QColor(self.owner.get_annotation_color(getattr(self, "text", "")))
        except Exception:
            c = QColor(255, 255, 255)

        edge_color = QColor(c)
        try:
            edge_color.setAlpha(220)
        except Exception:
            pass

        src_n = str(src).strip().lower()
        is_ml_like = src_n in {"ml", "auto", "machine", "model", "pred"}

        if is_ml_like:
            # 未确认机器标注：虚线、低透明度
            pen = pg.mkPen(edge_color, width=1.5, style=Qt.DashLine)
            fill = QColor(c)
            try:
                fill.setAlpha(115)
            except Exception:
                pass
            label_bg = "rgba(255,255,255,0.80)"
            label_border = "rgba({},{},{},0.55)".format(c.red(), c.green(), c.blue())
        else:
            # 人工或已确认标注：实线、较高透明度
            pen = pg.mkPen(edge_color, width=2.0, style=Qt.SolidLine)
            fill = QColor(c)
            try:
                fill.setAlpha(135)
            except Exception:
                pass
            label_bg = "rgba(255,255,255,0.90)"
            label_border = "rgba({},{},{},0.80)".format(c.red(), c.green(), c.blue())

        # 应用到 ROI 本体
        try:
            self.setPen(pen)
        except Exception:
            try:
                self.pen = pen
                self.update()
            except Exception:
                pass

        # RectROI 自身仍保留淡填充；独立填充层负责主要可见色块。
        try:
            self.setBrush(pg.mkBrush(fill))
        except Exception:
            pass
        try:
            fill_item = getattr(self, "_fill_item", None)
            if fill_item is not None:
                fill_item.setBrush(QBrush(fill))
                fill_item.setPen(QPen(Qt.NoPen))
                fill_item.show()
                self._sync_fill_item()
        except Exception:
            pass

        # 同步 label 的视觉（不改文字内容）
        self._update_label_html(label_bg=label_bg, label_border=label_border)

    def _sync_fill_item(self):
        """同步独立填充矩形的位置与大小。"""
        try:
            fill_item = getattr(self, "_fill_item", None)
            if fill_item is None:
                return
            a, b = self.interval()
            fill_item.setRect(float(a), float(self.y_base), max(0.0, float(b - a)), float(self.h_fix))
            fill_item.setZValue(self.zValue() - 1)
        except Exception:
            pass

    def _update_label_html(self, label_bg: str = None, label_border: str = None):
        """
        根据当前的 self.text 和 self.label_color 更新 label 的 HTML。
        仅视觉编码：支持可选的背景透明度与边框颜色。
        """
        c = self.label_color
        color_str = "#{:02x}{:02x}{:02x}".format(c.red(), c.green(), c.blue())

        if label_bg is None:
            label_bg = "rgba(255,255,255,0.90)"
        if label_border is None:
            label_border = "rgba(120,120,120,0.85)"

        html = (
            f'<div style="background:{label_bg};'
            f'color:{color_str};'
            f'border:1px solid {label_border};border-radius:4px;'
            'padding:1px 4px;white-space:nowrap;font-size:11px">'
            f'{self.text}'
            '</div>'
        )
        try:
            self.label.setHtml(html)
        except Exception:
            pass

    def set_label_color(self, color: QColor):
        """
        外部修改文字颜色的统一入口。
        """
        self.label_color = color
        # 颜色变了，视觉也应一起刷新（manual/ml 不变）
        self._apply_visual_style(self._infer_source())

    def interval(self):
        p = self.pos()
        s = self.size()
        return float(p.x()), float(p.x() + s.x())

    def _on_changed(self):
        # 锁定纵向 & 高度
        p = self.pos()
        s = self.size()
        if float(p.y()) != self.y_base:
            self.setPos([p.x(), self.y_base], update=False)
            p = self.pos()
        if float(s.y()) != self.h_fix:
            self.setSize([s.x(), self.h_fix], update=False)
            s = self.size()

        # 备注框跟随（在条上方一点点）
        cx = float(p.x() + s.x() / 2.0)
        cy = float(self.y_base + s.y()) + 0.05
        self.label.setPos(cx, cy)
        self._sync_fill_item()

        # 同步频谱红条
        m = getattr(self.owner, "_span2spec", {})
        if self in m:
            a, b = self.interval()
            m[self].setRegion([a, b])

        # 同步导出缓存（兼容 3/4 元组；三元组视为人工标注）
        m2 = getattr(self.owner, "_span2idx", {})
        if self in m2:
            idx = m2[self]
            if 0 <= idx < len(self.owner.annotations) and self.owner.annotations[idx] is not None:
                a, b = self.interval()
                s, e = a, b
                old = self.owner.annotations[idx]
                try:
                    if len(old) == 3:
                        _, _, t = old
                        src = "manual"
                    elif len(old) >= 4:
                        _, _, t, src = old[:4]
                    else:
                        t = self.text
                        src = "manual"
                except Exception:
                    t = self.text
                    src = "manual"
                self.owner.annotations[idx] = (s, e, t, src)

                # —— 视觉编码：source 可能为 ml/manual，保持样式同步 ——
                self._apply_visual_style(src)

    def mouseDoubleClickEvent(self, ev):
        s0, s1 = self.interval()
        self.owner.open_loop_player(s0, s1)
        ev.accept()

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.RightButton:
            menu = QMenu()
            play_action = menu.addAction("▶ Play")
            del_action = menu.addAction("🗑 Delete")
            act = menu.exec_(ev.screenPos().toPoint())
            if act == play_action:
                s0, s1 = self.interval()
                self.owner.open_loop_player(s0, s1)
            elif act == del_action:
                self.owner.delete_annotation(self)
            ev.accept()
        else:
            super().mouseClickEvent(ev)

    def cleanup(self):
        try:
            if getattr(self, "_fill_item", None) is not None:
                self.owner.annot_plot.removeItem(self._fill_item)
                self._fill_item = None
        except Exception:
            pass
        try:
            self.owner.annot_plot.removeItem(self)
        except:
            pass
        try:
            self.owner.annot_plot.removeItem(self.label)
        except:
            pass



class AnnotViewBox(pg.ViewBox):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setMouseMode(self.PanMode)  # 默认禁用拖动画布
        self.is_marking = False
        self.start_pos = None
        self.temp_region = None

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
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


class SettingsDialog(QDialog):
    def __init__(self, parent=None, n_fft=512, hop_length=256, f_max=2000,
                 wave_y_range=None, audio_data=None, selected_features=None,
                 stft_last_spec=None, stft_cmap="Heatmap", stft_levels=(None, None),
                 feature_color_map=None):

        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setFixedWidth(480)

        self._stft_vals = stft_last_spec
        self._stft_cmap = stft_cmap if stft_cmap in ("Heatmap", "Grayscale") else "Heatmap"
        self._stft_vmin, self._stft_vmax = stft_levels if stft_levels else (None, None)

        self.audio_data = audio_data
        self.default_ymin = float(np.min(audio_data)) if audio_data is not None else -1.0
        self.default_ymax = float(np.max(audio_data)) if audio_data is not None else 1.0
        self.wave_y_range = wave_y_range or (self.default_ymin, self.default_ymax)

        self._incoming_feat_color_map = feature_color_map or {}

        tabs = QTabWidget()

        # ==== STFT 标签页 ====
        stft_tab = QWidget()
        stft_layout = QFormLayout()

        self.n_fft_box = QSpinBox()
        self.n_fft_box.setRange(64, 8192)
        self.n_fft_box.setValue(n_fft)
        stft_layout.addRow("n_fft (window length)", self.n_fft_box)

        self.hop_box = QSpinBox()
        self.hop_box.setRange(16, 4096)
        self.hop_box.setValue(hop_length)
        stft_layout.addRow("hop_length (step size)", self.hop_box)

        self.f_max_box = QSpinBox()
        self.f_max_box.setRange(500, 40000)
        self.f_max_box.setSingleStep(500)
        self.f_max_box.setValue(f_max)
        stft_layout.addRow("STFT maximum frequency (Hz)", self.f_max_box)

        # ==== STFT Display（直方图 + colorbar + 上下限 + Restore Defaults）====
        stft_group = QGroupBox("STFT Display: Histogram & Color Bar (Editable)")
        stft_vbox = QVBoxLayout(stft_group)

        # 顶部：直方图 + 渐变色条（HistogramLUTWidget）
        self.hist_widget = HistogramLUTWidget()
        self.hist_widget.setMinimumHeight(180)

        # 给 HistogramLUT 提供一份图像数据，便于统计直方图
        if self._stft_vals is not None and np.any(np.isfinite(self._stft_vals)):
            tmp_img = ImageItem(self._stft_vals.T)  # 与主图方向一致：time × freq
        else:
            tmp_img = ImageItem(np.zeros((16, 16), dtype=float))
        self.hist_widget.setImageItem(tmp_img)

        # 生成与主图风格一致的 ColorMap
        def _make_cmap(name: str) -> ColorMap:
            if name == "Heatmap":
                return ColorMap(
                    [0.00, 0.25, 0.50, 0.75, 1.00],
                    np.array([
                        (68 / 255, 1 / 255, 84 / 255),
                        (59 / 255, 82 / 255, 139 / 255),
                        (33 / 255, 145 / 255, 140 / 255),
                        (94 / 255, 201 / 255, 98 / 255),
                        (253 / 255, 231 / 255, 37 / 255),
                    ], float)
                )
            # Grayscale
            return ColorMap(
                [0.0, 1.0],
                np.array([[0, 0, 0], [1, 1, 1]], float)
            )

        # 初始化 colorbar 配色（保留可编辑三角控件）
        self._base_cmap = _make_cmap(self._stft_cmap)
        self.hist_widget.gradient.setColorMap(self._base_cmap)
        # 某些 PyQt5 + 老 pyqtgraph 需要 save/restore 一次才能正确刷新
        try:
            _st = self.hist_widget.gradient.saveState()
            self.hist_widget.gradient.restoreState(_st)
        except Exception:
            pass

        # 中部：配色 + 上下限输入框
        line1 = QHBoxLayout()
        line1.addWidget(QLabel("Color map:"))
        self.cmap_select = QComboBox()
        self.cmap_select.addItems(["Heatmap", "Grayscale"])
        self.cmap_select.setCurrentText(self._stft_cmap)
        line1.addWidget(self.cmap_select)
        line1.addStretch(1)

        line2 = QHBoxLayout()
        self.vmin_edit = QDoubleSpinBox()
        self.vmin_edit.setDecimals(3)
        self.vmin_edit.setMinimum(-1e9)
        self.vmin_edit.setMaximum(1e9)

        self.vmax_edit = QDoubleSpinBox()
        self.vmax_edit.setDecimals(3)
        self.vmax_edit.setMinimum(-1e9)
        self.vmax_edit.setMaximum(1e9)

        line2.addWidget(QLabel("Lower limit:"))
        line2.addWidget(self.vmin_edit)
        line2.addSpacing(12)
        line2.addWidget(QLabel("Upper limit:"))
        line2.addWidget(self.vmax_edit)
        line2.addStretch(1)

        # 初始化上下限：优先用传入值，否则用 1%~99% 分位
        def _init_levels_from_data():
            vals = self._stft_vals
            if vals is not None and np.any(np.isfinite(vals)):
                finite_vals = vals[np.isfinite(vals)]
                vmin = float(np.percentile(finite_vals, 1))
                vmax = float(np.percentile(finite_vals, 99))
            else:
                vmin, vmax = 0.0, 1.0

            if (
                    self._stft_vmin is not None
                    and self._stft_vmax is not None
                    and self._stft_vmax > self._stft_vmin
            ):
                vmin, vmax = float(self._stft_vmin), float(self._stft_vmax)

            # Settings到 HistogramLUTItem：这一步决定开窗区间（与主图共享）
            self.hist_widget.setLevels(vmin, vmax)

            # 同步到输入框
            self.vmin_edit.blockSignals(True)
            self.vmax_edit.blockSignals(True)
            self.vmin_edit.setValue(vmin)
            self.vmax_edit.setValue(vmax)
            self.vmin_edit.blockSignals(False)
            self.vmax_edit.blockSignals(False)

        _init_levels_from_data()

        # 底部：Restore Defaults（回到 1%~99% 分位）
        btn_line = QHBoxLayout()
        self.btn_reset = QPushButton("Restore Defaults")
        btn_line.addStretch(1)
        btn_line.addWidget(self.btn_reset)

        # 组装到 group 里
        stft_vbox.addWidget(QLabel("STFT value histogram (drag both ends of the color bar to set lower/upper limits; the color bar is editable)"))
        stft_vbox.addWidget(self.hist_widget)
        stft_vbox.addLayout(line1)  # 配色
        stft_vbox.addLayout(line2)  # 下限/上限
        stft_vbox.addLayout(btn_line)  # Restore Defaults

        # 组装进 STFT 页
        try:
            stft_layout.addRow(stft_group)  # QFormLayout
        except Exception:
            stft_layout.addWidget(stft_group)  # 其他布局

        # ========= 联动逻辑：levels ↔ 文本框，顺带保证与主图一致 =========

        # 1）拖动 HistogramLUT 条（或编辑 colorbar） → 更新输入框
        def _levels_to_edits():
            try:
                vmin, vmax = self.hist_widget.getLevels()
            except Exception:
                return
            self.vmin_edit.blockSignals(True)
            self.vmax_edit.blockSignals(True)
            self.vmin_edit.setValue(float(vmin))
            self.vmax_edit.setValue(float(vmax))
            self.vmin_edit.blockSignals(False)
            self.vmax_edit.blockSignals(False)

        connected = False
        try:
            self.hist_widget.sigLevelsChanged.connect(_levels_to_edits)
            connected = True
        except Exception:
            pass
        if not connected:
            try:
                self.hist_widget.region.sigRegionChanged.connect(_levels_to_edits)
            except Exception:
                pass

        # 2）改输入框 → 回写到 HistogramLUT（开窗），colorbar 的条和三角会一起动
        def _edits_to_levels():
            vmin = self.vmin_edit.value()
            vmax = self.vmax_edit.value()
            if vmax > vmin:
                self.hist_widget.setLevels(vmin, vmax)

        self.vmin_edit.valueChanged.connect(_edits_to_levels)
        self.vmax_edit.valueChanged.connect(_edits_to_levels)

        # 3）切换配色 → 更新 colorbar（保留颜色编辑功能）
        def _on_cmap_changed_local(txt: str):
            self._stft_cmap = txt
            self._base_cmap = _make_cmap(txt)
            self.hist_widget.gradient.setColorMap(self._base_cmap)
            # 再 save/restore 一次，防止 Qt5 下首次不刷新
            try:
                _st2 = self.hist_widget.gradient.saveState()
                self.hist_widget.gradient.restoreState(_st2)
            except Exception:
                pass

        self.cmap_select.currentTextChanged.connect(_on_cmap_changed_local)

        # 4）Restore Defaults（上下限回到 1%~99% 分位）
        def _reset_defaults():
            self._stft_vmin = None
            self._stft_vmax = None
            _init_levels_from_data()

        self.btn_reset.clicked.connect(_reset_defaults)

        stft_tab.setLayout(stft_layout)
        tabs.addTab(stft_tab, "STFT")

        # ==== Display标签页 ====
        display_tab = QWidget()
        display_layout = QFormLayout()

        # 缩放滑条 + 输入框
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(1, 200)
        self.zoom_slider.setValue(100)

        self.zoom_input = QDoubleSpinBox()
        self.zoom_input.setRange(0.01, 2.0)
        self.zoom_input.setSingleStep(0.01)
        self.zoom_input.setValue(1.0)

        self.zoom_slider.valueChanged.connect(self.on_slider_changed)
        self.zoom_input.valueChanged.connect(self.on_input_changed)

        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(self.zoom_slider)
        zoom_layout.addWidget(self.zoom_input)
        display_layout.addRow("Waveform zoom factor", zoom_layout)

        # 上下限
        self.ymin_box = QDoubleSpinBox()
        self.ymax_box = QDoubleSpinBox()
        self.ymin_box.setRange(-1e3, 1e3)
        self.ymax_box.setRange(-1e3, 1e3)
        self.ymin_box.setDecimals(3)
        self.ymax_box.setDecimals(3)
        self.ymin_box.setValue(self.wave_y_range[0])
        self.ymax_box.setValue(self.wave_y_range[1])
        display_layout.addRow("Y-axis minimum", self.ymin_box)
        display_layout.addRow("Y-axis maximum", self.ymax_box)

        # Restore Defaults按钮
        self.reset_button = QPushButton("Restore Defaults")
        self.reset_button.clicked.connect(self.restore_default_y_range)
        display_layout.addRow(self.reset_button)

        display_tab.setLayout(display_layout)
        tabs.addTab(display_tab, "Display")
        self.tabs = tabs

        # 按钮
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(tabs)
        layout.addWidget(buttons)

        # ==== Short-time Features 标签页 ====
        feat_tab = QWidget()
        feat_layout = QFormLayout()

        hint = QLabel("Note: short-time features use the same window length and hop length as STFT by default.\n"
                      "Frequency-domain features are computed from the current STFT, and time-domain features are computed from the current waveform.\n"
                      "Select up to 5 features for display (normalized overlay).")
        hint.setWordWrap(True)
        feat_layout.addRow(hint)

        self.feat_list = QListWidget()
        self.feat_list.setSelectionMode(QListWidget.NoSelection)
        self.feat_list.setSpacing(6)  # 行间距更舒服
        # 可选：统一放大每一项的 sizeHint（若你想更高）
        for i in range(self.feat_list.count()):
            it = self.feat_list.item(i)
            it.setSizeHint(QSize(it.sizeHint().width(), 32))

        all_feats = [
            "Short-time Energy", "Short-time Mean", "Variance", "Kurtosis", "Skewness", "Zero Crossing Rate", "Teager Energy Operator",

            # —— 频域/时频统计特征（Built with  STFT Amplitude谱）——
            "Spectral Mean", "Spectral Standard Deviation", "Spectral Median", "Spectral Energy", "Spectral RMS", "Spectral Amplitude Sum",
            "Spectral Centroid", "Spectral Bandwidth", "Spectral Skewness", "Spectral Kurtosis", "Spectral Roll-off", "Spectral Flatness", "Spectral Entropy", "Spectral Flux",
            "Maximum Spectral Peak", "Number of Spectral Peaks",
            "Low-frequency Energy Ratio", "Mid-frequency Energy Ratio", "High-frequency Energy Ratio",
            "Spectral IQR", "Spectral MAD", "Spectral Difference ZCR", "Spectral Smoothness", "Primary-to-secondary Peak Ratio", "Spectral Complexity",

            # —— 窄带/少峰检测增强特征 ——
            "Primary Peak Energy Ratio", "Top-3 Peak Energy Ratio", "90% Energy Coverage Bins", "Primary Peak -3 dB Bandwidth", "Primary Peak Q Factor",

            # —— 频移自相关（cor）特征：100–1200 Hz 子带，Built with 每帧谱的“向下频移自相关”曲线 ——
            "cor_dist_ratio_mean", "cor_mean_slope", "cor_max_slope", "cor_std_slope",
            "cor_max_peak", "cor_second_peak", "cor_peak_count", "cor_peak_density",
            "cor_area", "cor_std", "cor_cv", "cor_skewness", "cor_kurtosis",
            "cor_local_max_slope_mean", "cor_local_max_slope_min",
            "cor_local_std_mean", "cor_local_std_max", "cor_local_pk2pk_mean", "cor_local_pk2pk_max"
        ]

        for name in all_feats:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.feat_list.addItem(item)

        # 应用有色对勾委托
        self.feat_list.setItemDelegate(ColorCheckDelegate(self.feat_list))

        # —— 让勾选颜色与曲线一致 ——
        # 颜色分配规则与主窗口一致：按“已选特征”的顺序分配
        sel = list(selected_features or [])
        palette = getattr(parent, "feature_palette", [
            QColor("#e41a1c"), QColor("#377eb8"), QColor("#4daf4a"),
            QColor("#984ea3"), QColor("#ff7f00")
        ])

        # 构建一个 {特征名: QColor} 映射
        color_map = {}
        used = 0
        for nm in sel[:5]:
            color_map[nm] = palette[used % len(palette)]
            used += 1

        # 把颜色写到每个条目的 UserRole，并顺带给未选的分配备用颜色（避免后续点选颜色重复）
        fallback_used = used
        for i in range(self.feat_list.count()):
            it = self.feat_list.item(i)
            nm = it.text()
            if nm in color_map:
                it.setData(Qt.UserRole, color_map[nm])
            else:
                it.setData(Qt.UserRole, palette[fallback_used % len(palette)])
                fallback_used += 1

        # 预勾选（来自主窗口）
        pre = set(selected_features or [])
        for i in range(self.feat_list.count()):
            it = self.feat_list.item(i)
            if it.text() in pre:
                it.setCheckState(Qt.Checked)

        def _limit_to_5(_):
            checked = [self.feat_list.item(i) for i in range(self.feat_list.count())
                       if self.feat_list.item(i).checkState() == Qt.Checked]
            if len(checked) > 5:
                # 取消最新一次勾选（把它设回未选）
                # 注意：此槽在单个 item 改变时触发，找到那个 item
                snd = feat_tab.sender()
                if isinstance(snd, QListWidget):
                    pass  # 兼容性占位
                # 简单策略：从末尾开始把第6个之后的设回未选
                for it in checked[5:]:
                    it.setCheckState(Qt.Unchecked)

        self.feat_list.itemChanged.connect(_limit_to_5)

        feat_layout.addRow(QLabel("Select features to display (up to 5):"), self.feat_list)
        feat_tab.setLayout(feat_layout)
        tabs.addTab(feat_tab, "Short-time Features")

    def on_slider_changed(self, value):
        """滑条改变时更新倍数输入框和上下限"""
        zoom = value / 100
        self.zoom_input.setValue(zoom)
        self.apply_zoom_to_range(zoom)

    def on_input_changed(self, value):
        """手动输入缩放倍数时更新滑条和上下限"""
        self.zoom_slider.setValue(int(value * 100))
        self.apply_zoom_to_range(value)

    def apply_zoom_to_range(self, zoom):
        """根据倍数缩放原始的 min/max 并更新Display范围"""
        center = (self.default_ymin + self.default_ymax) / 2
        half_range = (self.default_ymax - self.default_ymin) / 2 / zoom
        new_ymin = center - half_range
        new_ymax = center + half_range
        self.ymin_box.setValue(new_ymin)
        self.ymax_box.setValue(new_ymax)

    def restore_default_y_range(self):
        """Restore Defaults的自动计算的范围"""
        self.ymin_box.setValue(self.default_ymin)
        self.ymax_box.setValue(self.default_ymax)
        self.zoom_input.setValue(1.0)
        self.zoom_slider.setValue(100)

    def get_values(self):
        return (
            self.n_fft_box.value(),
            self.hop_box.value(),
            self.f_max_box.value(),
            (self.ymin_box.value(), self.ymax_box.value())
        )

    def set_current_tab(self, index):
        self.tabs.setCurrentIndex(index)

    def get_selected_features(self):
        sel = []
        if hasattr(self, "feat_list"):
            for i in range(self.feat_list.count()):
                it = self.feat_list.item(i)
                if it.checkState() == Qt.Checked:
                    sel.append(it.text())
        return sel[:5]

    def get_stft_display_settings(self):
        """
        返回 (cmap_name, vmin, vmax)
        优先从 HistogramLUT 的 levels 读（保证和 colorbar 对齐）
        """
        cmap = self.cmap_select.currentText()
        try:
            vmin, vmax = self.hist_widget.getLevels()
        except Exception:
            vmin = self.vmin_edit.value()
            vmax = self.vmax_edit.value()
        return cmap, float(vmin), float(vmax)


class ColorCheckDelegate(QStyledItemDelegate):
    """勾选后才上色；未勾选Display空框；整行可点击切换；方块尺寸更大"""
    BOX_SIZE = 22  # 放大方块（原来16）
    PADDING_X = 10

    def paint(self, painter, option, index):
        painter.save()
        rect = option.rect

        # 背景
        if option.state & QStyle.State_Selected:
            painter.fillRect(rect, option.palette.highlight())
        else:
            painter.fillRect(rect, option.palette.base())

        # 勾选状态
        checked = (index.data(Qt.CheckStateRole) == Qt.Checked)

        # 颜色（来自 UserRole；未勾选也可以先拿着，但不填充）
        color = index.data(Qt.UserRole)
        if not isinstance(color, QColor):
            try:
                color = QColor(color) if color else QColor("#999999")
            except Exception:
                color = QColor("#999999")

        # 方块区域（更大）
        box = QRect(
            rect.x() + self.PADDING_X,
            rect.y() + (rect.height() - self.BOX_SIZE) // 2,
            self.BOX_SIZE,
            self.BOX_SIZE
        )

        painter.setRenderHint(QPainter.Antialiasing, True)
        # 边框
        border_pen = QPen(color.darker(140) if checked else QColor(160, 160, 160), 1)
        painter.setPen(border_pen)
        # 填充：仅已勾选才填充彩色；未勾选Display空框
        painter.setBrush(QBrush(color) if checked else Qt.NoBrush)
        painter.drawRoundedRect(box, 4, 4)

        # 勾选对号：仅已勾选绘制
        if checked:
            painter.setPen(QPen(Qt.white, 2))
            path = QPainterPath()
            path.moveTo(box.left() + 4, box.center().y())
            path.lineTo(box.center().x() - 1, box.bottom() - 4)
            path.lineTo(box.right() - 4, box.top() + 5)
            painter.drawPath(path)

        # 文本
        text_rect = QRect(box.right() + 10, rect.y(), rect.width() - (box.width() + 20), rect.height())
        painter.setPen(option.palette.text().color())
        painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, index.data())

        painter.restore()

    def sizeHint(self, option, index):
        s = super().sizeHint(option, index)
        # 放大行高，增大点击目标
        return QSize(max(s.width(), 80), max(s.height(), 30))

    def editorEvent(self, event, model, option, index):
        """把整行都变成点击区域：点击行内任意处都切换勾选"""
        if event.type() in (QEvent.MouseButtonRelease, QEvent.MouseButtonDblClick):
            if option.rect.contains(event.pos()):
                state = index.data(Qt.CheckStateRole)
                new_state = Qt.Unchecked if state == Qt.Checked else Qt.Checked
                # 写回 CheckStateRole
                model.setData(index, new_state, Qt.CheckStateRole)
                return True
        return super().editorEvent(event, model, option, index)


class ClickableSlider(QSlider):
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            val = QStyle.sliderValueFromPosition(
                self.minimum(), self.maximum(), event.pos().x(), self.width())
            self.setValue(val)
            self.sliderMoved.emit(val)  # 触发滑块移动信号
        super().mousePressEvent(event)




class MLService:
    """集中管理Machine Learning训练/推理逻辑（Step1：仅搬家，不改算法/输出）。"""
    def __init__(self, owner):
        self.owner = owner

    def clear_ml_annotations_for_label(self, label):
        """
        删除所有标签为 label 且 source != 'manual' 的“机器标注”：
        - 从可视化中移除对应的 BoxSpan 和 STFT 高光
        - 把 self.annotations 里对应条目标记为 None（由 delete_annotation 负责）
        """
        self = self.owner
        if not hasattr(self, "annotations"):
            return

        # 先找出需要删除的 span（用 _span2idx 映射回 annotations）
        spans_to_delete = []

        for span in list(getattr(self, "_spans", [])):
            idx = getattr(self, "_span2idx", {}).get(span, None)
            if idx is None:
                continue

            if idx < 0 or idx >= len(self.annotations):
                continue

            item = self.annotations[idx]
            if item is None:
                continue

            # 兼容 3/4 元组（3 元组一律视为人工标注）
            try:
                if len(item) == 3:
                    s, e, t = item
                    src = "manual"
                elif len(item) >= 4:
                    s, e, t, src = item[:4]
                else:
                    continue
            except Exception:
                continue

            # 非 manual 且标签匹配 → 属于需要删除的机器标注
            if src != "manual" and str(t) == str(label):
                spans_to_delete.append(span)

        # 统一调用已有的删除逻辑
        for sp in spans_to_delete:
            try:
                self.delete_annotation(sp)
            except Exception:
                pass

        # 保险起见：清掉 annotations 中残留的“无可视对象”的机器标注
        for i, item in enumerate(self.annotations):
            if item is None:
                continue
            try:
                if len(item) == 3:
                    # 三元组默认人工，跳过
                    continue
                elif len(item) >= 4:
                    s, e, t, src = item[:4]
                else:
                    continue
            except Exception:
                continue

            if src != "manual" and str(t) == str(label):
                self.annotations[i] = None


    def train_model_for_label(self,
                              label,
                              min_pos_frames=30,
                              neg_pos_ratio=5,
                              random_state=None):
        """
        训练单标签帧级分类器（并输出更完整的训练集报告）：
        - label: 目标标签（如 'Wheeze'）
        - min_pos_frames: 至少需要多少个正样本帧才开始训练
        - neg_pos_ratio: 负样本采样比例（负样本数 ≈ ratio * 正样本数）

        增强项：
        - 训练时进行Feature selection（默认启用：互信息 SelectKBest）
        - 学习完成后除 P/R/F1 外，额外输出 Acc/Spec/BAcc/MCC/AUROC/AUPRC/Brier/Confusion
        """
        self = self.owner
        # ---------- 可调的Feature selection策略（默认开启） ----------
        FS_ENABLE = True
        FS_KBEST = 20  # 互信息 Top-K（若特征维度不足会自动缩小）

        # 1) 准备特征 & 帧标签
        self.ensure_frame_features()
        if self.stft_features is None or self.stft_frame_times is None:
            QMessageBox.information(self, "Machine Learning",
                                    "No available short-time features. The model cannot be trained.")
            return False

        y = self.build_frame_labels_for_tag(label, neg_margin=0.05)
        if y is None:
            QMessageBox.information(self, "Machine Learning",
                                    f"Label {label} has no available frames in the reviewed region and cannot be trained.")
            return False

        y = np.asarray(y, dtype=np.int8)
        idx_pos = np.where(y == 1)[0]
        idx_neg = np.where(y == 0)[0]

        n_pos = int(len(idx_pos))
        n_neg_all = int(len(idx_neg))

        if n_pos < int(min_pos_frames):
            QMessageBox.information(
                self, "Machine Learning",
                f"Label {label} has only {n_pos} positive frames, fewer than the minimum requirement of {min_pos_frames}. Training was skipped.")
            return False

        if n_neg_all == 0:
            QMessageBox.information(
                self, "Machine Learning",
                f"Label {label} has no available safe negative frames. Training was skipped.")
            return False

        # 2) 负样本均匀随机下采样，控制比例
        ratio = max(1, int(neg_pos_ratio))
        n_neg_target = int(min(n_neg_all, ratio * n_pos))

        rng = np.random.default_rng(random_state)
        idx_neg_sample = rng.choice(idx_neg, size=n_neg_target, replace=False)

        idx_all = np.concatenate([idx_pos, idx_neg_sample])
        rng.shuffle(idx_all)

        X = self.stft_features[idx_all, :]
        y_train = (y[idx_all] == 1).astype(int)  # {0,1}

        # 3) Feature selection + 标准化 + LogisticRegression
        D_all = int(X.shape[1])
        use_fs = bool(FS_ENABLE and D_all > 4)
        k_best = int(min(FS_KBEST, max(2, D_all)))

        steps = [("scaler", StandardScaler())]
        if use_fs:
            steps.append(("select", SelectKBest(score_func=mutual_info_classif, k=k_best)))

        clf = LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="lbfgs",
            n_jobs=1
        )
        steps.append(("clf", clf))

        pipe = Pipeline(steps)
        pipe.fit(X, y_train)

        # 4) 在训练集上自动选一个合适的概率阈值（最大化 F1）
        proba = pipe.predict_proba(X)[:, 1]
        best_th = 0.5
        best_f1 = -1.0

        for th in np.linspace(0.3, 0.9, 13):
            y_pred = (proba >= th).astype(int)
            f1 = f1_score(y_train, y_pred)
            if f1 > best_f1:
                best_f1 = float(f1)
                best_th = float(th)

        y_pred_best = (proba >= best_th).astype(int)
        precision, recall, f1_final, _ = precision_recall_fscore_support(
            y_train, y_pred_best, average="binary", zero_division=0
        )

        # 5) 更多训练集指标（便于科研报告与调参）
        tn, fp, fn, tp = confusion_matrix(y_train, y_pred_best, labels=[0, 1]).ravel()
        acc = accuracy_score(y_train, y_pred_best)
        bacc = balanced_accuracy_score(y_train, y_pred_best)
        mcc = matthews_corrcoef(y_train, y_pred_best)
        specificity = float(tn) / (float(tn + fp) + 1e-12)
        npv = float(tn) / (float(tn + fn) + 1e-12)
        auc_roc = roc_auc_score(y_train, proba)
        auc_pr = average_precision_score(y_train, proba)
        brier = brier_score_loss(y_train, proba)

        # 6) Feature selection结果（用于输出与可解释性）
        full_names = list(self.stft_feature_names)
        selected_idx = list(range(D_all))
        if use_fs and "select" in pipe.named_steps:
            selected_idx = pipe.named_steps["select"].get_support(indices=True).tolist()
        selected_names = [full_names[i] for i in selected_idx]

        # Logistic 回归系数（对应筛选后的特征顺序）
        coef = pipe.named_steps["clf"].coef_.reshape(-1)
        top_k_show = int(min(8, len(selected_names)))
        order = np.argsort(np.abs(coef))[::-1][:top_k_show]
        top_feats = [(selected_names[i], float(coef[i])) for i in order]
        top_feat_str = "\n".join([f"  - {n}: {c:+.4f}" for n, c in top_feats])

        # 7) 存到 self.ml_models，方便后续自动标注 / 可视化
        self.ml_models[label] = {
            "clf": pipe,
            "threshold": float(best_th),
            "feature_names": list(self.stft_feature_names),  # 输入特征空间（全量）
            "selected_feature_indices": [int(i) for i in selected_idx],
            "selected_feature_names": list(selected_names),
            "feature_select_method": "mutual_info_kbest" if use_fs else "none",
            "feature_select_k": int(k_best) if use_fs else int(D_all),
            "top_features_by_coef": list(top_feats),
            "n_pos": int(n_pos),
            "n_neg": int(n_neg_target),
            "train_precision": float(precision),
            "train_recall": float(recall),
            "train_f1": float(f1_final),
            "train_accuracy": float(acc),
            "train_specificity": float(specificity),
            "train_npv": float(npv),
            "train_bacc": float(bacc),
            "train_mcc": float(mcc),
            "train_auc_roc": float(auc_roc),
            "train_auc_pr": float(auc_pr),
            "train_brier": float(brier),
            "confusion": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
        }

        QMessageBox.information(
            self, "Machine Learning",
            (f"Trained label '{label}' frame-level model:\n"
             f"Positive frames: {n_pos}, Sampled negative frames: {n_neg_target}\n"
             f"Threshold (best training F1) = {best_th:.2f}\n"
             f"P={precision:.3f}, R={recall:.3f}, F1={f1_final:.3f}, Acc={acc:.3f}, Spec={specificity:.3f}\n"
             f"BAcc={bacc:.3f}, MCC={mcc:.3f}, AUROC={auc_roc:.3f}, AUPRC={auc_pr:.3f}, Brier={brier:.4f}\n"
             f"Confusion: TP={tp}, FP={fp}, TN={tn}, FN={fn}\n"
             f"Feature selection: {'MI-TopK' if use_fs else 'None'} (kept {len(selected_names)}/{D_all})\n"
             f"Top features (|coef|):\n{top_feat_str}")
        )

        return True


    def apply_model_for_label_on_unreviewed(self,
                                            label,
                                            min_dur_sec=0.05):
        """
        使用已经训练好的模型，在“未审阅区域”（帧Time > reviewed_prefix）Auto-labeling指定标签。
        生成的标注统一以 (start, end, label, "ml") 形式写入 self.annotations。
        """
        self = self.owner
        # 1) 检查模型是否存在
        if not hasattr(self, "ml_models") or label not in self.ml_models:
            QMessageBox.information(
                self, "Auto-labeling",
                f"Label {label} has not been trained. Please train this label first."
            )
            return False
        self.clear_ml_annotations_for_label(label)  # 清除上次Machine Learning的标记

        model_info = self.ml_models[label]
        clf = model_info["clf"]
        th = float(model_info.get("threshold", 0.5))

        # 2) 准备帧级特征与Time轴
        self.ensure_frame_features()
        times = getattr(self, "stft_frame_times", None)
        X = getattr(self, "stft_features", None)
        if times is None or X is None or len(times) == 0:
            QMessageBox.information(
                self, "Auto-labeling",
                "No available short-time features. Auto-labeling cannot be performed."
            )
            return False

        times = np.asarray(times, dtype=float)

        # 3) 求“已审阅前缀”，只在未审阅区域应用模型
        T_used = self.get_reviewed_prefix()
        if T_used is None or T_used <= 0:
            QMessageBox.information(
                self, "Auto-labeling",
                "No manual annotations are available, so the unreviewed region cannot be determined."
            )
            return False

        idx_unr = np.where(times > T_used)[0]
        if idx_unr.size == 0:
            QMessageBox.information(
                self, "Auto-labeling",
                "The current recording has been fully reviewed. There are no unreviewed frames."
            )
            return False

        # 4) 在未审阅帧上跑模型，得到帧级 0/1
        X_unr = X[idx_unr, :]
        try:
            proba = clf.predict_proba(X_unr)[:, 1]
        except Exception as e:
            QMessageBox.warning(
                self, "Auto-labeling",
                f"Model prediction failed: {e}"
            )
            return False

        y_hat = (proba >= th).astype(int)

        # 5) 把连续的 1 合并成区间，并过滤掉太短的事件
        new_segments = []
        in_run = False
        start_i = None
        for i, v in enumerate(y_hat):
            if v == 1 and not in_run:
                in_run = True
                start_i = i
            elif v == 0 and in_run:
                frame_idxs = idx_unr[start_i:i]
                s = float(times[frame_idxs[0]])
                e = float(times[frame_idxs[-1]])
                if e - s >= min_dur_sec:
                    new_segments.append((s, e))
                in_run = False
                start_i = None

        # 尾段收尾
        if in_run and start_i is not None:
            frame_idxs = idx_unr[start_i:len(y_hat)]
            s = float(times[frame_idxs[0]])
            e = float(times[frame_idxs[-1]])
            if e - s >= min_dur_sec:
                new_segments.append((s, e))

        if not new_segments:
            QMessageBox.information(
                self, "Auto-labeling",
                "No candidate events were detected in the unreviewed region."
            )
            return False

        # 6) 与已有人工标注做简单去重：同标签且高度重叠的段直接跳过
        manual_segs = self.get_manual_segments_for_label(label)

        def overlap_ratio(seg, base):
            s1, e1 = seg
            s2, e2 = base
            inter = min(e1, e2) - max(s1, s2)
            if inter <= 0:
                return 0.0
            return inter / max(e1 - s1, 1e-6)

        final_segments = []
        for (s, e) in new_segments:
            skip = False
            for (ms, me) in manual_segs:
                if overlap_ratio((s, e), (ms, me)) >= 0.5:
                    skip = True
                    break
            if not skip:
                final_segments.append((s, e))

        if not final_segments:
            QMessageBox.information(
                self, "Auto-labeling",
                "The candidate events highly overlap with existing manual annotations. No new machine annotations were added."
            )
            return False

        # 7) 直接调用 finalize_annotation，画框 + 写入 annotations，标记 source='ml'
        if not hasattr(self, "annotations"):
            self.annotations = []

        for (s, e) in final_segments:
            # 这里传 text=label，所以不会弹出对话框；source 标记为 'ml'
            self.finalize_annotation(s, e, text=label, source="ml")

        # 不需要再手动排序 / 重绘，finalize_annotation 已经做了界面更新
        QMessageBox.information(
            self, "Auto-labeling",
            f"Label '{label}' added {len(final_segments)} machine annotation segment(s) in the unreviewed region."
        )
        return True


'''
        # 8) 刷新界面：这里有两种情况
        #    - 如果你有专门的“从 annotations 重画所有 BoxSpan/红区”的函数，可以在此调用
        #    - 如果没有，而是标注都是通过 finalize_annotation 创建的，可后续考虑加个重绘函数
        try:
            # 如果你有类似的统一刷新函数，可以替换成自己的
            self.plot_waveform()       # 按你项目的实际函数名改
            self.plot_spectrogram()    # 如果有的话
        except Exception:
            # 没有统一刷新函数也没关系，自动标注至少已经写进 self.annotations 了
            pass

        QMessageBox.information(
            self, "Auto-labeling",
            f"Label '{label}' added {len(final_segments)} machine annotation segment(s) in the unreviewed region."
        )
        return True
'''



class AudioViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Time-Frequency Analysis and Annotation")
        self.resize(1200, 800)

        self.n_fft = 512
        self.hop_length = 256
        self.f_max = 2000
        self.sr = None
        self.audio = None
        self.duration = 0

        self.is_playing = False
        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.update_position)

        self.annotations = []  # list of (start, end, label, source)

        # —— 新增：最多 3 行的分轨管理（只用于标注视窗）
        self.MAX_LANES = 3
        self.LANE_H = 0.35  # 单条高度（越小越细）
        self.LANE_GAP = 0.25  # 轨道间距
        self._lanes = [[] for _ in range(self.MAX_LANES)]  # 每行的[(s,e), ...]
        self._spans = []  # 当前所有 BoxSpan
        self._span2spec = {}  # BoxSpan -> 频谱 LinearRegionItem
        self._span2idx = {}  # BoxSpan -> annotations 索引

        # —— STFT Display（仅两种配色 + 开窗上下限）——
        self.stft_cmap = "Heatmap"  # 备选："Heatmap"、"Grayscale"
        self.stft_vmin = None  # None 表示自动：用 1%~99% 分位
        self.stft_vmax = None
        self._last_spec_vals = None  # 最近一次 STFT 的原始 2D 数值（freq×time），用于直方图统计/重着色

        # —— 特征颜色（随便挑的不重复颜色，足够 5 条）——
        self.feature_palette = [
            QColor("#e41a1c"), QColor("#377eb8"), QColor("#4daf4a"),
            QColor("#984ea3"), QColor("#ff7f00"), QColor("#a65628"),
            QColor("#f781bf"), QColor("#999999"), QColor("#66c2a5"),
            QColor("#d95f02")
        ]
        self.feature_color_map = {}  # {特征名: QColor}，由所选特征顺序分配

        # —— 标注类型 & 颜色管理（含预设Wheeze/Crackles等）——
        self.annotation_builtin_labels = [
            ("Wheeze", "Wheeze"),
            ("Crackles", "Crackles"),
            ("Pleural Rub", "Pleural Rub"),
            ("Rhonchi", "Rhonchi"),
            ("Stridor", "Stridor"),
            ("Speech", "Speech"),  # Speech
            ("Cough", "Cough"),  # Cough
            ("Expiration", "Expiration"),  # Expiration
            ("Inspiration", "Inspiration"),  # Inspiration
        ]

        # 预设英文标签对应固定颜色
        self.annotation_color_builtin = {
            "Wheeze": QColor("#e41a1c"),  # 红
            "Crackles": QColor("#377eb8"),  # 蓝
            "Pleural Rub": QColor("#4daf4a"),  # 绿
            "Rhonchi": QColor("#984ea3"),  # 紫
            "Stridor": QColor("#ff7f00"),  # 橙
            "Speech": QColor("#a65628"),  # 棕
            "Cough": QColor("#f781bf"),  # 粉
            "Expiration": QColor("#999999"),  # 灰
            "Inspiration": QColor("#66c2a5"),  # 青绿色
        }

        # 其他任意文本标签的自动配色调色板
        self.annotation_color_palette = [
            QColor("#e41a1c"), QColor("#377eb8"), QColor("#4daf4a"),
            QColor("#984ea3"), QColor("#ff7f00"), QColor("#a65628"),
            QColor("#f781bf"), QColor("#999999"), QColor("#66c2a5"),
            QColor("#d95f02"),
        ]
        self.annotation_color_map = {}  # {标签文本: QColor}
        self._annotation_color_used = 0

        # 先默认 None，等 init_ml_toolbar 里根据下拉框赋值
        self.current_ml_label = None

        self.init_ui()
        # 切换
        self.audio_files = []
        self.current_file_index = -1

        self.last_export_path = None  # 记录上次保存的 CSV 路径（仅用于记住上次导出目录）
        self.default_export_annotation_name = "annotations_events.csv"  # 随当前 WAV 实时更新的默认导出File名

        self.showing_fft = False

        self.wave_y_range = None  # 格式为 (ymin, ymax)

        self.last_settings_tab_index = 0  # 默认打开第一个标签页（STFT）


        # —— 自动导入同名 _events 标注File（可开关）——
        # 规则：<wav_base>_events.(csv|txt) 与 <wav_base>.wav 同目录
        self.auto_import_events_enabled = False
        self._events_index_cache = {}  # {folder(abs): {wav_base_lower: events_path}}
        self._events_parse_cache = {}  # {events_path(abs): (mtime, rows)}
        self.show_y_axis = False  # 初始为Display

        # —— Short-time Features：默认选择（可改）
        self.selected_features = ["Short-time Energy", "Zero Crossing Rate", "Spectral Centroid"]

        # 曲线缓存（特征名 -> pg.PlotDataItem）
        self.feature_curves = {}

        # ML相关参数
        self.stft_frame_times = None  # 1D array, shape (T,)
        self.stft_features = None  # 2D array, shape (T, D)
        self.stft_feature_names = None  # list[str]，特征名顺序
        self.ml_models = {}  # {label: {"clf": ..., "threshold": ..., "feature_names": [...]}}
        self.ml_service = MLService(self)  # Step1: ML 逻辑搬入服务类


        # 临时调试用：Ctrl+T 训练 Wheeze 模型
        shortcut_train_wheeze = QShortcut(QKeySequence("Ctrl+T"), self)
        shortcut_train_wheeze.activated.connect(
            lambda: self.train_model_for_label("Speech")
        )

        # ========= Machine Learning：Auto-labeling Speech 未审阅区域（Ctrl+M） =========
        self.shortcut_auto_speech = QShortcut(QKeySequence("Ctrl+M"), self)
        self.shortcut_auto_speech.activated.connect(
            lambda: self.apply_model_for_label_on_unreviewed("Speech")
        )

    def init_ui(self):
        self.spec_stft_plot = pg.PlotWidget(title="STFT Spectrogram")
        self.spec_fft_plot = pg.PlotWidget(title="FFT Spectrum")
        self.wave_plot = pg.PlotWidget(title="Waveform", viewBox=WaveViewBox(self))
        self.annot_plot = pg.PlotWidget(title="Annotation Track", viewBox=AnnotViewBox(self))
        self.spec_fft_plot.hide()
        self.spec_title_base = "STFT Spectrogram"
        self.spec_stft_plot.setTitle(self.spec_title_base)

        # 监听 STFT 图鼠标移动（用 ViewBox 的矩形来判断是否在绘图区）
        self._stft_proxy = pg.SignalProxy(
            self.spec_stft_plot.scene().sigMouseMoved, rateLimit=60, slot=self._on_stft_mouse_move_title
        )

        y_axis_width = 50
        self.spec_stft_plot.getAxis('left').setWidth(y_axis_width)
        for plot in [self.spec_stft_plot, self.wave_plot, self.annot_plot]:
            plot.setContentsMargins(10, 0, 10, 0)

        self.spec_fft_plot.setLogMode(False, False)
        self.spec_fft_plot.setMouseEnabled(x=True, y=True)
        self.spec_fft_plot.getViewBox().setMouseMode(pg.ViewBox.RectMode)

        self.spec_img = pg.ImageItem()
        self.spec_stft_plot.addItem(self.spec_img)
        self.spec_stft_plot.hideAxis('left')
        self.wave_curve = self.wave_plot.plot([], [])
        self.wave_plot.hideAxis('left')

        self.spec_title_base = "STFT Spectrogram"  # 基础标题
        self.spec_stft_plot.setTitle(self.spec_title_base)
        self._stft_proxy = pg.SignalProxy(  # 监听 STFT 图的鼠标移动
            self.spec_stft_plot.scene().sigMouseMoved, rateLimit=60, slot=self._on_stft_mouse_move_title
        )

        self.annot_plot.setYRange(0, 1)
        self.annot_plot.setLimits(xMin=0, xMax=10)
        self.annot_plot.setMouseEnabled(x=False, y=False)
        self.wave_plot.setMouseEnabled(x=True, y=False)
        self.annot_plot.setLabel('bottom', 'Time', units='s')
        self.annot_plot.hideAxis('left')

        # 标注区 Y 轴固定为 3 行高度
        self.annot_plot.hideAxis('left')
        H = self.MAX_LANES * (self.LANE_H + self.LANE_GAP)
        self.annot_plot.setYRange(0, H)
        self.annot_plot.setLimits(yMin=0, yMax=H)

        self.annotation_items = []
        self.marking = False
        self.mark_start = None
        self.temp_region = None

        self.time_line_spec = pg.InfiniteLine(angle=90, pen=pg.mkPen('r', width=2), movable=False)
        self.time_line_wave = pg.InfiniteLine(angle=90, pen=pg.mkPen('r', width=2), movable=False)
        self.spec_stft_plot.addItem(self.time_line_spec)
        self.wave_plot.addItem(self.time_line_wave)

        btn_load = QPushButton("Import WAV File")
        btn_load.clicked.connect(self.load_audio)
        btn_prev = QPushButton("Previous")
        btn_prev.setShortcut(QKeySequence("Up"))
        btn_prev.clicked.connect(self.load_previous_file)
        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(self.play_pause)
        btn_next = QPushButton("Next")
        btn_next.setShortcut(QKeySequence("Down"))
        btn_next.clicked.connect(self.load_next_file)
        btn_set = QPushButton("Settings")
        btn_set.clicked.connect(self.open_settings)
        btn_export = QPushButton("Export Annotations")
        btn_export.clicked.connect(self.export_annotations)
        btn_import = QPushButton("Import Annotations")
        btn_import.clicked.connect(self.import_annotations)
        self.freq_button = QPushButton("Switch View")
        self.freq_button.clicked.connect(self.toggle_analysis_mode)
        self.legend_button = QPushButton("Annotation Legend")
        self.legend_button.clicked.connect(self.show_annotation_legend)

        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(["Heatmap", "Grayscale"])
        self.cmap_combo.setCurrentText(self.stft_cmap)
        self.cmap_combo.currentTextChanged.connect(self._on_cmap_changed)

        self.slider = ClickableSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.sliderPressed.connect(self.pause_timer)
        self.slider.sliderReleased.connect(self.seek)
        self.slider.valueChanged.connect(self.on_slider_move)
        self.time_label = QLabel("0.000 s")

        # —— 特征页（仅负责展示；计算逻辑在别处） ——
        self.feat_plot = pg.PlotWidget(title="Short-time Features (Normalized)")
        self.feat_plot.setMouseEnabled(x=True, y=False)
        self.feat_plot.setLabel('bottom', 'Time', units='s')
        self.feat_plot.addLegend()
        # —— Short-time Features PlotWidget 基础样式 ——
        self.feat_plot.setMouseEnabled(x=True, y=False)  # 只允许X轴缩放/拖动，禁用Y轴
        pi = self.feat_plot.getPlotItem()

        # 隐藏Y轴（坐标线/刻度/标签都不Display）
        pi.hideAxis('left')
        try:
            pi.hideAxis('right')
        except Exception:
            pass

        # 关掉网格（若你之前开过）
        pi.showGrid(x=False, y=False)

        # Time轴与 STFT 同步缩放/平移（锁定）
        self.feat_plot.setXLink(self.spec_stft_plot)

        # 去掉多余边距，让隐藏轴不留白
        pi.layout.setContentsMargins(0, 0, 0, 0)
        self.feat_plot.setContentsMargins(0, 0, 0, 0)

        self.feat_page = QWidget()
        _feat_layout = QVBoxLayout(self.feat_page)
        _feat_layout.setContentsMargins(0, 0, 0, 0)
        _feat_layout.addWidget(self.feat_plot)

        # —— 页栈：STFT / FFT / Short-time Features ——
        self.spec_stack = QStackedWidget()
        self.spec_stack.addWidget(self.spec_stft_plot)  # 第 0 页：STFT
        self.spec_stack.addWidget(self.spec_fft_plot)  # 第 1 页：FFT
        self.spec_stack.addWidget(self.feat_page)  # 第 2 页：Short-time Features
        self.spec_stack.setCurrentWidget(self.spec_stft_plot)  # 默认Display STFT

        # —— 垂直分割器：上(页栈) | 中(波形) | 下(标注) ——
        self.main_splitter = QSplitter(Qt.Vertical)
        self.main_splitter.addWidget(self.spec_stack)  # 上：页栈（STFT/FFT/特征）
        self.main_splitter.addWidget(self.wave_plot)  # 中：波形
        self.main_splitter.addWidget(self.annot_plot)  # 下：标注

        # 推荐的拉伸比例（可按实际微调）
        self.main_splitter.setStretchFactor(0, 3)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setStretchFactor(2, 1)

        # 防止被完全折叠
        self.main_splitter.setCollapsible(0, False)
        self.main_splitter.setCollapsible(1, False)
        self.main_splitter.setCollapsible(2, False)

        btn_layout = QHBoxLayout()
        for w in (btn_load, btn_prev, btn_next, self.btn_play, btn_set, btn_export, btn_import, self.freq_button,
                  self.legend_button, self.cmap_combo):
            btn_layout.addWidget(w)

        time_layout = QHBoxLayout()
        time_layout.addWidget(QLabel("Progress"))
        time_layout.addWidget(self.slider, stretch=1)
        time_layout.addWidget(self.time_label)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.main_splitter)
        main_layout.addLayout(btn_layout)
        main_layout.addLayout(time_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.wave_plot.setXLink(self.spec_stft_plot)
        self.annot_plot.setXLink(self.spec_stft_plot)

        # 菜单栏
        self.init_menu_bar()
        # Machine Learning工具栏
        self.init_ml_toolbar()

    def init_menu_bar(self):
        # 创建菜单栏
        menu_bar = self.menuBar()

        # ===== File菜单 =====
        file_menu = menu_bar.addMenu("File")

        open_audio_action = QAction("Import Audio", self)
        open_audio_action.setShortcut(QKeySequence("Ctrl+O"))
        open_audio_action.triggered.connect(self.load_audio)
        file_menu.addAction(open_audio_action)

        import_label_action = QAction("Import Annotations", self)
        import_label_action.setShortcut(QKeySequence("Ctrl+I"))
        import_label_action.triggered.connect(self.import_annotations)
        file_menu.addAction(import_label_action)

        export_label_action = QAction("Export Annotations", self)
        export_label_action.setShortcut(QKeySequence("Ctrl+E"))
        export_label_action.triggered.connect(self.export_annotations)
        file_menu.addAction(export_label_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # ===== Settings菜单 =====
        settings_menu = menu_bar.addMenu("Settings")

        param_action = QAction("Settings", self)
        param_action.setShortcut(QKeySequence("Ctrl+P"))
        param_action.triggered.connect(self.open_settings)
        settings_menu.addAction(param_action)


        auto_import_events_action = QAction("Auto-import matching _events annotations", self)
        auto_import_events_action.setCheckable(True)
        auto_import_events_action.setChecked(bool(getattr(self, "auto_import_events_enabled", False)))
        auto_import_events_action.triggered.connect(self.toggle_auto_import_events)
        settings_menu.addAction(auto_import_events_action)
        self.auto_import_events_action = auto_import_events_action
        # ===== Help菜单 =====
        help_menu = menu_bar.addMenu("Help")

        about_action = QAction("About", self)
        about_action.setShortcut(QKeySequence("F1"))
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

    def init_ml_toolbar(self):
        """初始化Machine Learning相关工具栏：标签选择 + 训练 + Auto-labeling"""
        from PyQt5.QtWidgets import QToolBar

        toolbar = QToolBar("Machine Learning", self)
        self.addToolBar(Qt.TopToolBarArea, toolbar)

        # ===== 下拉框：选择要训练/Auto-labeling的Label =====
        self.ml_label_combo = QComboBox(self)

        # 使用内置标签列表的英文名作为Machine Learning标签
        label_en_list = []
        if getattr(self, "annotation_builtin_labels", None):
            label_en_list = [en for (cn, en) in self.annotation_builtin_labels]
            for en in label_en_list:
                self.ml_label_combo.addItem(en)

        # 初始化 current_ml_label
        if label_en_list:
            # 如果之前已经有值并且在列表中，就选中它
            if self.current_ml_label in label_en_list:
                idx = label_en_list.index(self.current_ml_label)
                self.ml_label_combo.setCurrentIndex(idx)
            else:
                self.current_ml_label = label_en_list[0]
                self.ml_label_combo.setCurrentIndex(0)

        self.ml_label_combo.currentTextChanged.connect(self.on_ml_label_changed)

        # 加到工具栏：文字 + 下拉框
        toolbar.addWidget(QLabel("ML Label:"))
        toolbar.addWidget(self.ml_label_combo)

        # ===== 按钮：Train Model =====
        action_train = QAction("Train Model", self)
        action_train.setToolTip("Train a frame-level model for the selected label using current manual annotations")
        action_train.triggered.connect(self.on_ml_train_clicked)
        toolbar.addAction(action_train)

        # ===== 按钮：Auto-label Unreviewed区域 =====
        action_auto = QAction("Auto-label Unreviewed", self)
        action_auto.setToolTip("Use the trained model to auto-label the selected label in unreviewed regions")
        action_auto.triggered.connect(self.on_ml_auto_clicked)
        toolbar.addAction(action_auto)

        # ===== 按钮：标注颜色图例 =====
        action_legend = QAction("Annotation Legend", self)
        action_legend.setToolTip("View the mapping between annotation colors and labels in the third track")
        action_legend.triggered.connect(self.show_annotation_legend)
        toolbar.addAction(action_legend)
        self.action_annotation_legend = action_legend

    # ===== 工具栏相关槽函数 =====
    def on_ml_label_changed(self, text):
        """当工具栏下拉框选择的标签变化时，更新当前 ML 标签"""
        self.current_ml_label = text

    def on_ml_train_clicked(self):
        """工具栏按钮：训练当前标签的模型"""
        label = getattr(self, "current_ml_label", None)
        if not label:
            QMessageBox.information(self, "Machine Learning", "Please select a label in the toolbar first.")
            return
        self.train_model_for_label(label)

    def on_ml_auto_clicked(self):
        """工具栏按钮：用当前标签的模型Auto-label Unreviewed区域"""
        label = getattr(self, "current_ml_label", None)
        if not label:
            QMessageBox.information(self, "Auto-labeling", "Please select a label in the toolbar first.")
            return
        self.apply_model_for_label_on_unreviewed(label)

    def show_annotation_legend(self):
        """按代码中的预设标签和颜色映射生成图例。"""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QDialogButtonBox, QWidget
        from PyQt5.QtCore import Qt

        def _color_for_label_direct(label):
            """优先使用 annotation_color_builtin；大小写不一致时做一次不区分大小写匹配。"""
            lab = str(label).strip()
            cmap = getattr(self, "annotation_color_builtin", {}) or {}
            if lab in cmap:
                return QColor(cmap[lab])
            lab_low = lab.lower()
            for k, v in cmap.items():
                try:
                    if str(k).strip().lower() == lab_low:
                        return QColor(v)
                except Exception:
                    continue
            try:
                return QColor(self.get_annotation_color(lab))
            except Exception:
                return QColor("#999999")

        legend_items = []
        seen = set()

        # 直接按照代码中的预设标签顺序生成图例，不依赖当前是否已有标注。
        for cn, en in getattr(self, "annotation_builtin_labels", []):
            lab = str(en).strip()
            if not lab or lab.lower() in seen:
                continue
            seen.add(lab.lower())
            legend_items.append((str(cn), lab, _color_for_label_direct(lab)))

        # 补充当前File中出现过的Custom标签。
        try:
            for item in getattr(self, "annotations", []):
                if item is None or len(item) < 3:
                    continue
                lab = str(item[2]).strip()
                if not lab or lab.lower() in seen:
                    continue
                seen.add(lab.lower())
                legend_items.append(("Custom", lab, _color_for_label_direct(lab)))
        except Exception:
            pass

        if not legend_items:
            QMessageBox.information(self, "Annotation Legend", "No annotation legend is available.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Annotation Legend")
        dlg.setMinimumWidth(360)

        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(18, 16, 18, 12)
        layout.setSpacing(10)

        hint = QLabel("The annotation track uses colors to distinguish categories; label text is not shown directly on the timeline.")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        list_widget = QWidget(dlg)
        list_layout = QVBoxLayout(list_widget)
        list_layout.setContentsMargins(0, 0, 0, 0)
        list_layout.setSpacing(6)

        for cn, en, color in legend_items:
            qc = QColor(color)
            hex_color = qc.name()

            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(10)

            swatch = QLabel()
            swatch.setFixedSize(30, 14)
            swatch.setStyleSheet(
                f"background-color: {hex_color}; border: 1px solid #444; border-radius: 2px;"
            )
            row.addWidget(swatch, 0, Qt.AlignVCenter)

            cn_label = QLabel(str(cn))
            cn_label.setMinimumWidth(52)
            row.addWidget(cn_label, 0, Qt.AlignVCenter)

            en_label = QLabel(str(en))
            en_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            row.addWidget(en_label, 1, Qt.AlignVCenter)

            list_layout.addLayout(row)

        layout.addWidget(list_widget)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok)
        btn_box.accepted.connect(dlg.accept)
        layout.addWidget(btn_box)

        dlg.exec_()

    def load_audio(self, path=None):
        # 1. 手动导入路径为空时，弹出File选择框
        if not path:
            selected, _ = QFileDialog.getOpenFileName(
                self, "Select WAV File", "", "WAV Files (*.wav)"
            )
            if not selected or not isinstance(selected, str):
                return  # 用户取消选择
            path = selected

        # 2. 加载当前目录下的所有 WAV File
        try:
            folder = os.path.dirname(path)
            all_files = sorted([
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith('.wav')
            ])
            path = os.path.abspath(path)
            all_files = [os.path.abspath(os.path.join(folder, f)) for f in os.listdir(folder) if
                         f.lower().endswith('.wav')]

            if path not in all_files:
                raise FileNotFoundError(f"The selected file was not found in its folder.\n\n{path}\n\nCandidate files:\n" + "\n".join(all_files))

            self.audio_files = all_files
            self.current_file_index = self.audio_files.index(path)
        except Exception as e:
            QMessageBox.critical(self, "Folder Error", f"Failed to get audio files in the same folder:\n{str(e)}")
            return

        # 3. 加载音频数据
        try:
            self.loaded_filename = path
            base = os.path.splitext(os.path.basename(path))[0]
            self.default_export_annotation_name = f"{base}_events.csv"
            self.audio, self.sr = librosa.load(path, sr=None)
            self.duration = len(self.audio) / self.sr
        except Exception as e:
            QMessageBox.critical(self, "Audio Loading Failed", f"Failed to read audio:\n{str(e)}")
            return

        # 4. 更新界面
        self.setWindowTitle(f"Audio Time-Frequency Analyzer — {os.path.basename(path)}")
        self.slider.setMaximum(int(self.duration * 1000))
        self.wave_y_range = (float(np.min(self.audio)), float(np.max(self.audio)))
        self.draw_waveform()
        self.update_spectrogram()
        self.clear_annotations()

        # —— 可选：自动导入同名 _events 标注File（需开关开启）——
        try:
            if getattr(self, "auto_import_events_enabled", False):
                self._auto_import_events_for_wav(path)
        except Exception:
            pass
        self.show_fft()

        # 5. Settings坐标轴范围
        self.spec_stft_plot.setLimits(xMin=0, xMax=self.duration, yMin=0)
        self.wave_plot.setLimits(xMin=0, xMax=self.duration)
        self.annot_plot.setLimits(xMin=0, xMax=self.duration)
        self.annot_plot.setXRange(0, self.duration)
        self.time_line_spec.setPos(0)
        self.time_line_wave.setPos(0)
        # self.wave_y_range = (float(np.min(self.audio)), float(np.max(self.audio)))
        self.spec_stack.setCurrentWidget(self.spec_stft_plot)

        # 回到 STFT 页，准备就绪
        if hasattr(self, "spec_stack"):
            self.spec_stack.setCurrentWidget(self.spec_stft_plot)
        # 预计算一次特征（以便切到第三栏立刻有图）
        self.update_features_plot()

    def update_spectrogram(self):
        if self.audio is None:
            return

        D = librosa.stft(self.audio, n_fft=self.n_fft, hop_length=self.hop_length, center=True, pad_mode='reflect')
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        idx = freqs <= self.f_max
        S_db = S_db[idx, :]
        f_max = freqs[idx][-1] if np.any(idx) else freqs[-1]
        spec = S_db

        # Settings图像内容和映射区域、
        n_time = spec.shape[1]
        t_step = self.hop_length / self.sr
        # 记录本次谱值（freq×time），供直方图与重着色使用
        self._last_spec_vals = spec.copy()

        disp = spec.T  # 与坐标映射一致的转置
        rgb = self._colorize_spec_with_window(disp)

        self.spec_img.resetTransform()
        self.spec_img.setImage(rgb, autoLevels=False)  # 直接像素
        self.spec_img.setRect(pg.QtCore.QRectF(0, 0, n_time * t_step, f_max))

        self.spec_stft_plot.setLabel('bottom', 'Time', units='s')
        self.spec_stft_plot.setLimits(xMin=0, xMax=self.duration, yMin=0, yMax=f_max)
        self.spec_stft_plot.setXRange(0, self.duration, padding=0)

        self.wave_plot.setLimits(xMin=0, xMax=self.duration)
        self.annot_plot.setLimits(xMin=0, xMax=self.duration)
        self.wave_plot.setXRange(0, self.duration, padding=0)
        self.annot_plot.setXRange(0, self.duration, padding=0)

        # STFT/FMAX 变更会影响频域特征
        self.update_features_plot()

    def draw_waveform(self):
        times = np.arange(len(self.audio)) / self.sr
        self.wave_curve.setData(times, self.audio)
        self.wave_plot.setLimits(xMin=0, xMax=self.duration)
        self.wave_plot.setLabel('bottom', 'Time', units='s')

        if self.wave_y_range:
            self.wave_plot.setYRange(*self.wave_y_range)
        else:
            self.wave_plot.setYRange(self.audio.min(), self.audio.max())

    def clear_annotations(self):
        # 1) 删可视对象 —— 新样式 BoxSpan + 其上方文字
        for sp in list(self._spans):
            try:
                sp.cleanup()  # 会把自身和文字从 annot_plot 移除
            except:
                pass
        self._spans.clear()

        # 2) 删 STFT 里对应的红色高亮区
        for reg in list(self._span2spec.values()):
            try:
                self.spec_stft_plot.removeItem(reg)
            except:
                pass
        self._span2spec.clear()
        self._span2idx.clear()

        # 3) 兼容老样式：以前保存在 annotation_items 里的区域和文字
        for item, label in list(self.annotation_items):
            try:
                self.annot_plot.removeItem(item)
            except:
                pass
            if label:
                try:
                    self.annot_plot.removeItem(label)
                except:
                    pass
        self.annotation_items.clear()

        # 4) 临时圈选区域（拖拽时的浅色块）
        if getattr(self, "temp_region", None):
            try:
                self.annot_plot.removeItem(self.temp_region)
            except:
                pass
            self.temp_region = None

        # 5) 数据结构清空
        self.annotations.clear()
        self._lanes = [[] for _ in range(self.MAX_LANES)]

        # 6) 视窗范围复位
        H = self.MAX_LANES * (self.LANE_H + self.LANE_GAP)
        self.annot_plot.setYRange(0, H)
        self.annot_plot.setLimits(yMin=0, yMax=H, xMin=0, xMax=self.duration)
        self.annot_plot.setXRange(0, self.duration)

        # 7) 取消波形高亮
        self.plot_waveform_highlight(None, None)

    def play_pause(self):
        if self.audio is None:
            return
        if not self.is_playing:
            pos = self.slider.value() / 1000.0
            start_sample = int(pos * self.sr)
            sd.stop()
            sd.play(self.audio[start_sample:], self.sr)
            self.start_time = time.time() - pos
            self.timer.start()
            self.btn_play.setText("Pause")
            self.is_playing = True
        else:
            sd.stop()
            self.timer.stop()
            self.btn_play.setText("Resume")
            self.is_playing = False

    def update_position(self):
        elapsed = time.time() - self.start_time
        if elapsed >= self.duration:
            self.timer.stop()
            self.btn_play.setText("Play")
            self.is_playing = False
            elapsed = self.duration
        ms = int(elapsed * 1000)
        self.slider.setValue(ms)
        self.time_label.setText(f"{elapsed:.3f} s")
        self.time_line_spec.setPos(elapsed)
        self.time_line_wave.setPos(elapsed)
        if elapsed >= self.duration:
            self.slider.setValue(0)
            self.time_line_spec.setPos(0)
            self.time_line_wave.setPos(0)

    def pause_timer(self):
        if self.is_playing:
            self.timer.stop()
            sd.stop()

    def seek(self):
        self.time_label.setText(f"{self.slider.value() / 1000.0:.3f} s")
        pos_sec = self.slider.value() / 1000.0
        self.time_line_spec.setPos(pos_sec)
        self.time_line_wave.setPos(pos_sec)
        if self.is_playing:
            sd.stop()
            sd.play(self.audio[int(pos_sec * self.sr):], self.sr)
            self.start_time = time.time() - pos_sec
            self.timer.start()

    def plot_waveform_highlight(self, start=None, end=None):
        if self.audio is None:
            return

        times = np.arange(len(self.audio)) / self.sr
        self.wave_plot.clear()

        if start is None or end is None:
            self.wave_plot.plot(times, self.audio, pen=pg.mkPen('w'))
        else:
            # ✅ 确保 start < end，无论用户从左往右还是从右往左划
            start, end = min(start, end), max(start, end)
            inside = (times >= start) & (times <= end)
            outside = ~inside
            self.wave_plot.plot(times[outside], self.audio[outside], pen=pg.mkPen('w'))
            self.wave_plot.plot(times[inside], self.audio[inside], pen=pg.mkPen('r'))

        # ✅ 保留Time线
        self.wave_plot.addItem(self.time_line_wave)

    def on_slider_move(self, v):
        self.time_label.setText(f"{v / 1000.0:.3f} s")
        self.time_line_spec.setPos(v / 1000.0)
        self.time_line_wave.setPos(v / 1000.0)

    def open_settings(self):
        dlg = SettingsDialog(
            parent=self,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            f_max=self.f_max,
            wave_y_range=self.wave_y_range,
            audio_data=self.audio,
            selected_features=getattr(self, "selected_features", None),
            # ↓↓↓ 新增：把 STFT 相关Display数据传进去
            stft_last_spec=self._last_spec_vals,
            stft_cmap=self.stft_cmap,
            stft_levels=(self.stft_vmin, self.stft_vmax),
            feature_color_map=self.feature_color_map,
        )

        dlg.set_current_tab(self.last_settings_tab_index)  # 💡 Settings上次使用的标签页

        if dlg.exec_():
            self.n_fft, self.hop_length, self.f_max, self.wave_y_range = dlg.get_values()
            self.selected_features = dlg.get_selected_features()  # 新增
            self.last_settings_tab_index = dlg.tabs.currentIndex()
            if self.audio is not None:
                self.update_spectrogram()
                self.draw_waveform()
                self.update_features_plot()

            # 取回 STFT DisplaySettings（配色 + 上下限）
            try:
                cmap, vmin, vmax = dlg.get_stft_display_settings()
                self.stft_cmap = cmap
                self.stft_vmin = vmin
                self.stft_vmax = vmax
            except Exception:
                pass

            # 立即按新Settings重着色（不强制重算 STFT；若前面参数变更导致你重算，那这里会再次渲染也没问题）
            if self._last_spec_vals is not None:
                disp = self._last_spec_vals.T
                rgb = self._colorize_spec_with_window(disp)
                self.spec_img.setImage(rgb, autoLevels=False)

    def export_annotations(self):
        # 过滤掉 None，并兼容 3/4 元组
        rows = [item for item in getattr(self, "annotations", []) if item is not None]
        if not rows:
            QMessageBox.information(self, "Notice", "There are no annotations to export.")
            return

        # 默认路径和File名：
        # - 目录跟随上一次Export Annotations时选择的目录；
        # - File名始终跟随当前 WAV，格式为 <当前File名>_events.csv。
        default_dir = os.getcwd()
        if hasattr(self, 'loaded_filename'):
            default_dir = os.path.dirname(self.loaded_filename)
            base = os.path.splitext(os.path.basename(self.loaded_filename))[0]
            self.default_export_annotation_name = f"{base}_events.csv"

        if getattr(self, "last_export_path", None):
            last_dir = os.path.dirname(os.path.abspath(self.last_export_path))
            if last_dir:
                default_dir = last_dir

        default_name = getattr(self, "default_export_annotation_name", "annotations_events.csv")
        default_path = os.path.join(default_dir, default_name)

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Annotations", default_path, "CSV Files (*.csv)"
        )
        if not path:
            return

        try:
            import csv
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                # 改为导出四列表头：start, end, label, source
                w.writerow(["start", "end", "label", "source"])
                for item in rows:
                    try:
                        start = float(item[0])
                        end = float(item[1])
                        label = str(item[2])
                        # 兼容 3/4 元组：3 元组视为人工标记，source="manual"
                        if len(item) >= 4:
                            source = str(item[3])
                        else:
                            source = "manual"
                    except Exception:
                        # 非法行直接跳过
                        continue
                    w.writerow([f"{start:.4f}", f"{end:.4f}", label, source])

            QMessageBox.information(self, "Export Successful", f"Annotations have been saved to:\n{path}")
            self.last_export_path = path  # 记住路径
        except Exception as e:
            QMessageBox.warning(self, "Export Failed", str(e))

    def open_loop_player(self, start, end):
        if self.audio is None:
            return
        dlg = LoopPlayer(self.audio, self.sr, start, end, self)
        dlg.exec_()

    def on_region_right_click(self, ev, region, start, end):
        if ev.button() != Qt.RightButton:
            return
        menu = QMenu()
        play_action = menu.addAction("▶ Play")
        delete_action = menu.addAction("🗑 Delete")

        action = menu.exec_(ev.screenPos().toPoint())
        if action == play_action:
            self.open_loop_player(start, end)
        elif action == delete_action:
            self.delete_annotation(region, start, end)

    def delete_annotation(self, target):
        # 新：删除 BoxSpan
        if isinstance(target, BoxSpan):
            sp = target
            if sp in self._span2spec:
                try:
                    self.spec_stft_plot.removeItem(self._span2spec[sp])
                except:
                    pass
                self._span2spec.pop(sp, None)
            if sp in self._span2idx:
                idx = self._span2idx.pop(sp)
                if 0 <= idx < len(self.annotations):
                    self.annotations[idx] = None
            sp.cleanup()
            if sp in self._spans:
                self._spans.remove(sp)
            return

        # 兼容：旧 LinearRegionItem 删除路径（如果你代码中还有）
        region_item = target
        try:
            self.annot_plot.removeItem(region_item)
        except:
            pass
        for sp, reg in list(self._span2spec.items()):
            if reg is region_item:
                try:
                    self.spec_stft_plot.removeItem(reg)
                except:
                    pass
                self._span2spec.pop(sp, None)
                break

    def load_previous_file(self):
        if self.audio_files and self.current_file_index > 0:
            self.current_file_index -= 1
            self.load_audio(self.audio_files[self.current_file_index])
        else:
            QMessageBox.information(self, "Notice", "This is already the first file.")

    def load_next_file(self):
        if self.audio_files and self.current_file_index < len(self.audio_files) - 1:
            self.current_file_index += 1
            self.load_audio(self.audio_files[self.current_file_index])
        else:
            QMessageBox.information(self, "Notice", "This is already the last file.")

    def get_annotation_color(self, label_text):
        """根据标签文本返回一个稳定的颜色；预设类型优先使用固定颜色。"""
        if not label_text:
            return QColor(255, 255, 255)

        # 已有映射直接返回
        if label_text in self.annotation_color_map:
            return self.annotation_color_map[label_text]

        # 预设类型（英文名）
        if label_text in self.annotation_color_builtin:
            color = self.annotation_color_builtin[label_text]
            self.annotation_color_map[label_text] = color
            return color

        # 其他任意文本：从调色板顺序取色
        palette = self.annotation_color_palette
        idx = self._annotation_color_used % len(palette)
        color = palette[idx]
        self._annotation_color_used += 1
        self.annotation_color_map[label_text] = color
        return color

    def finalize_annotation(self, start, end, text=None, source="manual"):
        # 若未提供文本（正常交互标注），则弹出带预设类型的对话框
        if text is None:
            dlg = AnnotationLabelDialog(
                parent=self,
                builtin_labels=getattr(self, "annotation_builtin_labels", None),
                start=start,
                end=end,
            )
            text = dlg.get_text()
            if not text:
                return

        # —— 分轨策略...
        lane = self._pick_lane(start, end)
        y0 = lane * (self.LANE_H + self.LANE_GAP)

        # 根据标签拿颜色（用于条的填充、边框、STFT 高光）
        color = self.get_annotation_color(text)
        try:
            pen = pg.mkPen(color, width=1)
            span_brush_color = QColor(color)
            span_brush_color.setAlpha(80)
            brush = pg.mkBrush(span_brush_color)
        except Exception:
            pen = pg.mkPen(255, 255, 255, 255, width=1)
            brush = pg.mkBrush(255, 255, 255, 255)

        # === 根据 source 决定标签文字颜色 ===
        if source == "manual":
            label_color = None  # 用 BoxSpan 默认黑字
        else:
            label_color = QColor(255, 0, 0)  # 机器标注 → 红字

        # —— 画彩色“盒条”
        span = BoxSpan(start, end, y0, self.LANE_H, text, self, label_color=label_color)
        span.setPen(pen)
        try:
            span.setBrush(brush)
        except Exception:
            pass
        self.annot_plot.addItem(span)
        self._spans.append(span)
        span.lane = lane

        # —— 频谱高光：与标签同色
        try:
            spec_color = QColor(color)
            spec_color.setAlpha(50)
            spec_brush = pg.mkBrush(spec_color)
        except Exception:
            spec_brush = pg.mkBrush(255, 0, 0, 50)

        spec = pg.LinearRegionItem([start, end], brush=spec_brush)
        spec.setMovable(False)
        spec.setZValue(1)
        spec.setBounds([start, end])
        self.spec_stft_plot.addItem(spec)
        self._span2spec[span] = spec

        # —— 导出缓存：4 元组 (start, end, text, source)
        if not hasattr(self, "annotations"):
            self.annotations = []
        self.annotations.append((float(start), float(end), str(text), str(source)))
        self._span2idx[span] = len(self.annotations) - 1
        try:
            span._apply_visual_style(str(source))
        except Exception:
            pass

    def import_annotations(self):
        if self.audio is None:
            QMessageBox.warning(self, "Error", "Please import an audio file first.")
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "Import Annotations", "", "Annotation Files (*.csv *.txt);;All Files (*)"
        )
        if not path:
            return

        try:
            # 判断扩展名
            ext = path.split('.')[-1].lower()
            if ext not in ['csv', 'txt']:
                raise ValueError("Unsupported file format. Please import a .csv or .txt file.")

            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            new_annotations = []

            for line in lines[1:]:  # 跳过表头
                line = line.strip()
                if not line:
                    continue

                # 自动选择分隔符
                if ',' in line:
                    parts = line.split(',')
                elif '\t' in line:
                    parts = line.split('\t')
                elif ' ' in line:
                    parts = line.split()
                else:
                    continue  # 不能解析的行跳过

                if len(parts) < 3:
                    continue

                try:
                    start, end = float(parts[0]), float(parts[1])
                    text = parts[2]
                    new_annotations.append((start, end, text))
                except ValueError:
                    continue  # 非法数值跳过

            self.clear_annotations()
            for start, end, text in new_annotations:
                self.finalize_annotation(start, end, text)

            QMessageBox.information(self, "Success", f"Successfully imported {len(new_annotations)} annotation(s).")

        except Exception as e:
            QMessageBox.critical(self, "Import Failed", f"Error while reading the file:\n{str(e)}")


    # =========================
    # 自动导入同名 _events 标注（可开关）
    # =========================
    def toggle_auto_import_events(self, checked: bool):
        """
        开关：自动导入与当前 WAV 同名的 *_events.(csv|txt) 标注File。
        仅在开关开启后，后续 load_audio() 才会自动走导入流程。
        点击开关时允许有一次预索引延迟，以减少后续加载的延迟。
        """
        self.auto_import_events_enabled = bool(checked)

        # 允许开关点击时有一定延迟：预索引当前音频目录
        if self.auto_import_events_enabled:
            wav_path = getattr(self, "loaded_filename", None)
            if wav_path and isinstance(wav_path, str):
                folder = os.path.dirname(os.path.abspath(wav_path))
                self._prepare_events_index(folder)

        # 轻量Notice（不弹窗）
        try:
            msg = "Auto-import matching _events annotations: enabled" if self.auto_import_events_enabled else "Auto-import matching _events annotations: disabled"
            self.statusBar().showMessage(msg, 2000)
        except Exception:
            pass

    def _prepare_events_index(self, folder: str):
        """
        为某个目录建立 *_events.(csv|txt) 的索引缓存，避免每次切歌都扫描磁盘。
        缓存结构：{folder: {wav_base_lower: events_path}}
        优先使用 CSV（若同名同时存在 csv/txt）。
        """
        try:
            folder = os.path.abspath(folder)
        except Exception:
            return

        if not hasattr(self, "_events_index_cache"):
            self._events_index_cache = {}

        if folder in self._events_index_cache:
            return

        mapping = {}
        try:
            for ent in os.scandir(folder):
                if not ent.is_file():
                    continue
                name = ent.name
                low = name.lower()
                if low.endswith("_events.csv"):
                    base = name[:-len("_events.csv")].lower()
                    mapping[base] = ent.path
                elif low.endswith("_events.txt"):
                    base = name[:-len("_events.txt")].lower()
                    # 若同名 csv 已存在，则不覆盖
                    mapping.setdefault(base, ent.path)
        except Exception:
            mapping = {}

        self._events_index_cache[folder] = mapping

    def _resolve_events_path_for_wav(self, wav_path: str):
        """
        解析 wav 对应的 events File路径：
        <wav_base>_events.csv 优先，其次 <wav_base>_events.txt
        若缓存中没有则做一次 O(1) 的 existence check 并更新缓存。
        """
        if not wav_path or not isinstance(wav_path, str):
            return None
        wav_path = os.path.abspath(wav_path)
        folder = os.path.dirname(wav_path)
        wav_base = os.path.splitext(os.path.basename(wav_path))[0]
        key = wav_base.lower()

        self._prepare_events_index(folder)
        mapping = self._events_index_cache.get(folder, {})

        p = mapping.get(key)
        if p and os.path.isfile(p):
            return p

        # 目录索引可能早于File创建；做一次 O(1) check 并回填
        cand_csv = os.path.join(folder, f"{wav_base}_events.csv")
        if os.path.isfile(cand_csv):
            mapping[key] = cand_csv
            self._events_index_cache[folder] = mapping
            return cand_csv

        cand_txt = os.path.join(folder, f"{wav_base}_events.txt")
        if os.path.isfile(cand_txt):
            mapping.setdefault(key, cand_txt)
            self._events_index_cache[folder] = mapping
            return cand_txt

        return None

    def _read_text_file_flexible(self, path: str):
        """
        尝试用常见编码读取文本（utf-8/utf-8-sig/gbk），失败则忽略Error读取。
        """
        for enc in ("utf-8", "utf-8-sig", "gbk"):
            try:
                with open(path, "r", encoding=enc) as f:
                    return f.read()
            except Exception:
                continue
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            return ""

    def _parse_events_file(self, events_path: str):
        """
        解析 events File，返回 rows: [(start, end, label, source), ...]
        - 支持 csv / txt
        - 自动跳过表头/非法行（无法将前两列转为 float）
        - 分隔符支持：逗号、制表符、空格
        """
        content = self._read_text_file_flexible(events_path)
        if not content:
            return []

        rows = []
        for raw in content.splitlines():
            line = raw.strip()
            if not line:
                continue

            # 自动选择分隔符
            if "," in line:
                parts = [p.strip() for p in line.split(",")]
            elif "	" in line:
                parts = [p.strip() for p in line.split("	")]
            else:
                parts = line.split()

            if len(parts) < 3:
                continue

            try:
                s = float(parts[0])
                e = float(parts[1])
            except Exception:
                # 表头或非法行
                continue

            lab = parts[2].strip()
            src = parts[3].strip() if len(parts) >= 4 and parts[3].strip() else "manual"

            rows.append((s, e, lab, src))

        return rows

    def _parse_events_file_cached(self, events_path: str):
        """
        带 mtime 的解析缓存，避免重复 I/O + parse。
        """
        if not hasattr(self, "_events_parse_cache"):
            self._events_parse_cache = {}

        try:
            p = os.path.abspath(events_path)
            mtime = os.path.getmtime(p)
        except Exception:
            return []

        cached = self._events_parse_cache.get(p)
        if cached and isinstance(cached, tuple) and len(cached) == 2 and cached[0] == mtime:
            return cached[1]

        rows = self._parse_events_file(p)
        self._events_parse_cache[p] = (mtime, rows)
        return rows

    def _auto_import_events_for_wav(self, wav_path: str):
        """
        在不弹窗的前提下，自动导入与 wav 同名的 *_events.(csv|txt) File。
        仅负责导入；清空标注由调用方决定（load_audio 已清空）。
        """
        events_path = self._resolve_events_path_for_wav(wav_path)
        if not events_path:
            return  # 静默跳过

        rows = self._parse_events_file_cached(events_path)
        if not rows:
            return

        n_ok = 0
        for s, e, lab, src in rows:
            try:
                if e <= s:
                    continue
                # source 仅用于追溯Display；异常值统一回落到 manual
                if not src:
                    src = "manual"
                self.finalize_annotation(s, e, lab, source=src)
                n_ok += 1
            except Exception:
                continue

        # 轻量Notice（不弹窗）
        try:
            self.statusBar().showMessage(
                f"已自动导入 events 标注：{os.path.basename(events_path)}（{n_ok} 条）", 2500
            )
        except Exception:
            pass
    def toggle_analysis_mode(self):
        cur = self.spec_stack.currentWidget()
        if cur is self.spec_stft_plot:
            # STFT -> FFT
            self.spec_stack.setCurrentWidget(self.spec_fft_plot)
            self.freq_button.setText("View: Short-time Features")
            self.show_fft()
        elif cur is self.spec_fft_plot:
            # FFT -> Short-time Features
            self.spec_stack.setCurrentWidget(self.feat_page)
            self.freq_button.setText("View: STFT")
            self.update_features_plot()
        else:
            # Short-time Features -> STFT
            self.spec_stack.setCurrentWidget(self.spec_stft_plot)
            self.freq_button.setText("View: FFT")

    def show_fft(self):
        if self.audio is None:
            return

        from scipy.fft import rfft, rfftfreq
        N = len(self.audio)
        if N == 0 or self.sr is None or self.sr == 0:
            return

        fft_y = np.abs(rfft(self.audio))
        fft_x = rfftfreq(N, d=1 / self.sr)

        self.spec_fft_plot.clear()
        self.spec_fft_plot.plot(fft_x, fft_y, pen='c')
        self.spec_fft_plot.setLabel('bottom', 'Frequency', units='Hz')
        self.spec_fft_plot.setLabel('left', 'Amplitude')

        x_max = self.sr / 2
        y_max = float(np.max(fft_y)) if fft_y.size else 1.0
        if y_max <= 0:
            y_max = 1.0
        self.spec_fft_plot.setXRange(0, x_max, padding=0.02)
        self.spec_fft_plot.setYRange(0, y_max * 1.05)
        vb = self.spec_fft_plot.getViewBox()
        vb.setLimits(xMin=0, xMax=x_max, yMin=0, yMax=y_max * 2)

    def update_fft(self):
        if self.audio is None:
            return

        freqs = np.fft.rfftfreq(len(self.audio), d=1 / self.sr)
        spectrum = np.abs(np.fft.rfft(self.audio))

        self.fft_curve.setData(freqs, spectrum)
        self.spec_fft_plot.setLabel('left', "Amplitude", units='')
        self.spec_fft_plot.setLabel('bottom', "Frequency", units='Hz')

        # 恢复原始视图范围
        self.spec_fft_plot.setXRange(freqs[0], freqs[-1], padding=0.05)
        self.spec_fft_plot.setYRange(0, np.max(spectrum) * 1.05)
        vb = self.spec_fft_plot.getViewBox()
        vb.setLimits(xMin=0, xMax=self.sr / 2, yMin=0, yMax=np.max(spectrum) * 2)

    def _overlap(self, a1, a2, b1, b2):
        # 是否交叠：只要有交集就算重叠
        return not (a2 <= b1 or a1 >= b2)

    def _pick_lane(self, s, e):
        """根据当前屏幕上的 BoxSpan 动态挑选 0/1/2 行；有空位就用最前的行。"""
        lanes = [[] for _ in range(self.MAX_LANES)]
        gap = (self.LANE_H + self.LANE_GAP)
        # 收集现有每一行的区间
        for sp in list(self._spans):
            try:
                a, b = sp.interval()
                lane = int(round(sp.y_base / gap))
                if 0 <= lane < self.MAX_LANES:
                    lanes[lane].append((a, b))
            except Exception:
                pass
        # 找第一个不重叠的行
        for i in range(self.MAX_LANES):
            if all(not self._overlap(s, e, a, b) for (a, b) in lanes[i]):
                return i
        # 三行都冲突，则放最后一行
        return self.MAX_LANES - 1

    def _on_stft_mouse_move_title(self, evt):
        """鼠标在 STFT 图上移动时：标题Display t / f 坐标；移出绘图区则还原标题"""
        if self.audio is None:
            return
        pos = evt[0] if isinstance(evt, (tuple, list)) else evt

        vb = self.spec_stft_plot.getViewBox()
        # 仅在绘图区内触发（不包含轴刻度/边距）
        if not vb.sceneBoundingRect().contains(pos):
            self.spec_stft_plot.setTitle(self.spec_title_base)
            return

        p = vb.mapSceneToView(pos)
        x = float(p.x())
        y = float(p.y())
        fmax = getattr(self, "_stft_fmax", getattr(self, "f_max", 0.0))
        dur = getattr(self, "duration", 0.0)

        if 0.0 <= x <= (dur or 0.0) and 0.0 <= y <= (fmax or 0.0):
            # 标题右侧Display坐标（可调样式）
            self.spec_stft_plot.setTitle(
                f'{self.spec_title_base}  '
                f'<span style="color:#bbb; font-size:11px;">t={x:.3f}s, f={y:.1f} Hz</span>'
            )
        else:
            self.spec_stft_plot.setTitle(self.spec_title_base)

    def show_about_dialog(self):
        QMessageBox.information(self, "About",
                                "Audio Annotator v1.0\nAuthor: C.Y.Pan\nBuilt with PyQt5 + pyqtgraph\nCurrently only .wav files are supported. Please convert other file types to .wav first.  ")

    def compute_short_time_features(self):
        """
        返回 times(s), feat_dict{name -> 1D array}。
        频域特征：Built with 当前 STFT（n_fft/hop 与 f_max）
        时域特征：Built with 当前波形帧

        对齐原则（与 update_spectrogram 完全一致）：
        - STFT 使用 center=True + pad_mode='reflect'（librosa 默认行为）
        - 所有时域帧级特征使用同样的反射 padding 后再分帧
        - times 使用 STFT 帧索引映射：t_k = k * hop_length / sr
        """
        if self.audio is None or self.sr is None:
            return np.array([]), {}

        import librosa

        x = self.audio
        sr = self.sr
        n = int(self.n_fft)
        h = int(self.hop_length)

        # —— 频域基准（与 update_spectrogram 一致）——
        # 用它来定义“帧数”和“Time轴”，保证所有特征与 STFT 一一对应
        try:
            D = librosa.stft(x, n_fft=n, hop_length=h, center=True, pad_mode='reflect')
        except Exception:
            return np.array([]), {}

        T = D.shape[1]
        times = (np.arange(T) * h) / sr

        # === 时域分帧（使用与 STFT 相同的 padding 语义） ===
        pad = n // 2
        if pad > 0:
            # reflect 需要长度 >= 2；极短信号用 edge 兜底
            if len(x) >= 2:
                x_pad = np.pad(x, (pad, pad), mode='reflect')
            else:
                x_pad = np.pad(x, (pad, pad), mode='edge')
        else:
            x_pad = x

        try:
            frames = librosa.util.frame(x_pad, frame_length=n, hop_length=h)
        except Exception:
            return np.array([]), {}

        # 防御性对齐到 STFT 帧数
        if frames.shape[1] != T:
            T2 = min(frames.shape[1], T)
            frames = frames[:, :T2]
            D = D[:, :T2]
            T = T2
            times = times[:T]

        # —— 时域特征 ——
        f_time = {}
        # 1. Short-time Energy
        f_time["Short-time Energy"] = np.sum(frames ** 2, axis=0)
        # 2. Short-time Mean
        mu = np.mean(frames, axis=0)
        f_time["Short-time Mean"] = mu
        # 3. Variance
        var = np.var(frames, axis=0) + 1e-12
        f_time["Variance"] = var
        # 4/5. Kurtosis/Skewness
        std = np.sqrt(var)
        centered3 = np.mean((frames - mu) ** 3, axis=0)
        centered4 = np.mean((frames - mu) ** 4, axis=0)
        f_time["Skewness"] = centered3 / (std ** 3 + 1e-12)
        f_time["Kurtosis"] = centered4 / (std ** 4 + 1e-12)
        # 6. Zero Crossing Rate（与 STFT 对齐：先对时域信号做同样的 reflect padding，再 center=False 分帧）
        # 说明：部分 librosa 版本的 zero_crossings/zero_crossing_rate 不支持 pad_mode 透传参数；
        # 因此这里采用“手动 padding + center=False”的方式保证与 STFT（center=True,pad_mode=reflect）帧严格一致。
        try:
            zcr = librosa.feature.zero_crossing_rate(
                x_pad, frame_length=n, hop_length=h, center=False
            )[0]
        except TypeError:
            # 极旧版本兼容：若不支持 center 参数，则退化为默认行为（仍Built with  x_pad，帧数通常一致）
            zcr = librosa.feature.zero_crossing_rate(
                x_pad, frame_length=n, hop_length=h
            )[0]

        if zcr.shape[0] != T:
            zcr = zcr[:T] if zcr.shape[0] > T else np.pad(zcr, (0, T - zcr.shape[0]), mode="edge")
        f_time["Zero Crossing Rate"] = zcr
        # 7. Teager 能量算子（帧内平均，Built with 同样 padding 后的信号）
        # ψ(x[n]) = x[n]^2 - x[n-1]*x[n+1]
        xn = x_pad
        teager = np.zeros_like(xn)
        teager[1:-1] = xn[1:-1] ** 2 - xn[:-2] * xn[2:]
        te_frames = librosa.util.frame(teager, frame_length=n, hop_length=h)
        f_time["Teager Energy Operator"] = np.mean(np.abs(te_frames[:, :T]), axis=0)

        # —— 频域特征 ——
        S = np.abs(D)  # Amplitude谱
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n)
        idx = freqs <= self.f_max
        S = S[idx, :]
        freqs = freqs[idx]

        # 子带（例如 200 Hz 以上）
        f_low = 200.0
        idx_sub = (freqs >= f_low) & (freqs <= self.f_max)
        S_sub = S[idx_sub, :]
        freqs_sub = freqs[idx_sub]

        # 概率化功率谱（全带，用于熵/质心等）
        P = (S ** 2)
        P_sum = np.sum(P, axis=0, keepdims=True) + 1e-12
        Pn = P / P_sum  # 每帧归一化为概率

        # 8. Spectral Centroid
        cent = np.sum(freqs[:, None] * Pn, axis=0)
        # 9. Spectral Bandwidth（Variance的平方根）
        var_f = np.sum(((freqs[:, None] - cent[None, :]) ** 2) * Pn, axis=0)
        bw = np.sqrt(var_f)
        # 10/11. Spectral Skewness/Spectral Kurtosis
        m3 = np.sum(((freqs[:, None] - cent[None, :]) ** 3) * Pn, axis=0)
        m4 = np.sum(((freqs[:, None] - cent[None, :]) ** 4) * Pn, axis=0)
        skew_s = m3 / (var_f ** 1.5 + 1e-12)
        kurt_s = m4 / (var_f ** 2 + 1e-12)
        # 12. Spectral Roll-off（85%）
        roll = librosa.feature.spectral_rolloff(
            S=S_sub,
            freq=freqs_sub,
            roll_percent=0.85
        )[0]
        # 13. Spectral Flatness（几何均值/算术均值）
        flat = librosa.feature.spectral_flatness(S=S_sub)[0]
        # 14. Spectral Entropy（0~1）
        ent = -np.sum(Pn * (np.log(Pn + 1e-12)), axis=0) / np.log(Pn.shape[0])
        # 15. Spectral Flux（相邻帧的变化）
        Sn = S / (np.linalg.norm(S, axis=0, keepdims=True) + 1e-12)
        dSn = np.diff(Sn, axis=1)
        flux = np.sqrt(np.sum(np.maximum(dSn, 0.0) ** 2, axis=0))
        flux = np.r_[0.0, flux]  # 与帧数对齐

        # === 补齐频域/时频特征（Built with 子带 S_sub，更适合窄带/少峰检测）===
        # 说明：不改动已有特征定义，仅在此基础上追加更多谱统计/峰值/集中度特征。
        #      这些特征将自动进入 ML 帧特征缓存，并可在“Short-time Features”页勾选Display。

        # 1) 谱统计量
        spec_mean = np.mean(S_sub, axis=0)
        spec_std = np.std(S_sub, axis=0)
        spec_median = np.median(S_sub, axis=0)

        # 2) 能量/Amplitude相关
        spec_energy = np.sum(S_sub ** 2, axis=0)
        spec_rms = np.sqrt(np.mean(S_sub ** 2, axis=0))
        spec_sum = np.sum(S_sub, axis=0)

        # 3) 峰值相关（不依赖 scipy，使用简单局部极大值检测）
        if S_sub.shape[0] >= 3:
            pk_mask = (S_sub[1:-1, :] > S_sub[:-2, :]) & (S_sub[1:-1, :] > S_sub[2:, :])
            peak_count = np.sum(pk_mask, axis=0).astype(float)
            pk_vals = np.where(pk_mask, S_sub[1:-1, :], 0.0)

            if pk_vals.shape[0] >= 2:
                top2 = np.partition(pk_vals, -2, axis=0)[-2:, :]
                pk1 = np.max(top2, axis=0)
                pk2 = np.min(top2, axis=0)
            else:
                pk1 = np.max(pk_vals, axis=0) if pk_vals.size else np.zeros(T)
                pk2 = np.zeros_like(pk1)

            max_peak = pk1
            peak_ratio = np.where(peak_count >= 2, pk1 / (pk2 + 1e-12), 0.0)
        else:
            peak_count = np.zeros(T)
            max_peak = np.zeros(T)
            peak_ratio = np.zeros(T)

        # 4) 频带能量占比（与 MATLAB 版本一致：0-400 / 400-800 / >800）
        E_total = np.sum(P, axis=0) + 1e-12
        low_mask = freqs <= 400.0
        mid_mask = (freqs > 400.0) & (freqs <= 800.0)
        high_mask = freqs > 800.0
        low_ratio = (np.sum(P[low_mask, :], axis=0) / E_total) if np.any(low_mask) else np.zeros(T)
        mid_ratio = (np.sum(P[mid_mask, :], axis=0) / E_total) if np.any(mid_mask) else np.zeros(T)
        high_ratio = (np.sum(P[high_mask, :], axis=0) / E_total) if np.any(high_mask) else np.zeros(T)

        # 5) 分布/形状补充：IQR / MAD / 差分零交叉 / 平滑度 / 复杂度
        q75 = np.percentile(S_sub, 75, axis=0)
        q25 = np.percentile(S_sub, 25, axis=0)
        spec_iqr = q75 - q25
        med0 = np.median(S_sub, axis=0, keepdims=True)
        spec_mad = np.median(np.abs(S_sub - med0), axis=0)

        dS = np.diff(S_sub, axis=0)
        if dS.shape[0] >= 2:
            spec_zc = np.mean((dS[:-1, :] * dS[1:, :]) < 0, axis=0)
        else:
            spec_zc = np.zeros(T)

        spec_smooth = np.mean(np.abs(dS), axis=0) if dS.size else np.zeros(T)
        spec_complex = np.sqrt(np.sum(dS ** 2, axis=0)) / (np.sum(S_sub, axis=0) + 1e-12) if dS.size else np.zeros(T)

        # 6) 窄带/少峰检测增强：Top-K 能量占比、90%能量覆盖bin数、主峰带宽与Q
        P_sub = (S_sub ** 2)
        E_sub = np.sum(P_sub, axis=0) + 1e-12

        top1_energy_ratio = np.max(P_sub, axis=0) / E_sub

        if P_sub.shape[0] >= 3:
            top3_sum = np.sum(np.partition(P_sub, -3, axis=0)[-3:, :], axis=0)
        else:
            top3_sum = np.sum(P_sub, axis=0)
        top3_energy_ratio = top3_sum / E_sub

        n_bins_90pct = np.zeros(T)
        for k in range(T):
            p = P_sub[:, k]
            tot = float(np.sum(p))
            if tot <= 0:
                n_bins_90pct[k] = 0.0
                continue
            sp = np.sort(p)[::-1]
            cs = np.cumsum(sp)
            n_bins_90pct[k] = float(np.searchsorted(cs, 0.90 * tot) + 1)

        dominant_bw_hz = np.zeros(T)
        dominant_Q = np.zeros(T)
        if freqs_sub.size >= 2:
            df_sub = float(np.median(np.diff(freqs_sub)))
        else:
            df_sub = 0.0

        for k in range(T):
            s = S_sub[:, k]
            if s.size < 3:
                continue
            i0 = int(np.argmax(s))
            peak = float(s[i0])
            if peak <= 0:
                continue
            thr = peak / np.sqrt(2.0)  # -3dB (half-power)

            left = i0
            while left > 0 and s[left] >= thr:
                left -= 1
            if left < i0 and s[left] < thr:
                left_in = left + 1
            else:
                left_in = left

            right = i0
            while right < s.size - 1 and s[right] >= thr:
                right += 1
            if right > i0 and s[right] < thr:
                right_in = right - 1
            else:
                right_in = right

            # 注意：不要覆写上方的Spectral Bandwidth向量 bw（用于“Spectral Bandwidth”特征）
            bw0 = float(freqs_sub[right_in] - freqs_sub[left_in])
            if bw0 <= 0.0 and df_sub > 0.0:
                bw0 = df_sub

            dominant_bw_hz[k] = max(0.0, bw0)
            f0 = float(freqs_sub[i0])
            dominant_Q[k] = (f0 / bw0) if bw0 > 0 else 0.0

        # =====================================================================
        # 频移自相关（cor）特征：在 100–1200 Hz 子带上，按 STFT 帧计算“向下频移自相关”曲线
        # - 与你此前 MATLAB 版本一致：每帧先保留Amplitude排序前 50% 的频点，再做频移点积自相关并归一化
        # - 最终从每帧自相关曲线中提取 19 个曲线特征（cor_*）
        # =====================================================================
        cor_low_hz = 100.0
        cor_high_hz = min(1200.0, float(self.f_max))
        cor_mask = (freqs >= cor_low_hz) & (freqs <= cor_high_hz)
        S_cor = S[cor_mask, :]
        freqs_cor = freqs[cor_mask]

        # 初始化输出（若频带不足则全 0）
        cor_dist_ratio_mean = np.zeros(T)
        cor_mean_slope = np.zeros(T)
        cor_max_slope = np.zeros(T)
        cor_std_slope = np.zeros(T)
        cor_max_peak = np.zeros(T)
        cor_second_peak = np.zeros(T)
        cor_peak_count = np.zeros(T)
        cor_peak_density = np.zeros(T)
        cor_area = np.zeros(T)
        cor_std = np.zeros(T)
        cor_cv = np.zeros(T)
        cor_skewness = np.zeros(T)
        cor_kurtosis = np.zeros(T)
        cor_local_max_slope_mean = np.zeros(T)
        cor_local_max_slope_min = np.zeros(T)
        cor_local_std_mean = np.zeros(T)
        cor_local_std_max = np.zeros(T)
        cor_local_pk2pk_mean = np.zeros(T)
        cor_local_pk2pk_max = np.zeros(T)

        if S_cor.shape[0] >= 3 and S_cor.shape[1] == T:
            # Frequency分辨率与最大位移（Hz）
            if freqs_cor.size >= 2:
                df_cor = float(np.median(np.diff(freqs_cor)))
            else:
                df_cor = 0.0

            if df_cor > 0.0:
                max_shift_hz = float(cor_high_hz - cor_low_hz)
                N_f = int(freqs_cor.size)
                N_shift = int(round(max_shift_hz / df_cor))
                N_shift = max(1, min(N_shift, N_f - 1))
                tau_axis = (np.arange(N_shift + 1, dtype=float) * df_cor)

                win_size = 10
                step = 5

                def _curve_features_19(y: np.ndarray, tvec: np.ndarray):
                    """从自相关曲线 y(t) 中提取 19 维特征（与 MATLAB extract_curve_features_local 对齐）。"""
                    y = np.asarray(y, dtype=float).reshape(-1)
                    tvec = np.asarray(tvec, dtype=float).reshape(-1)
                    n0 = y.size
                    if n0 < 2 or tvec.size < 2:
                        return (0.0,) * 19

                    # 1) 全局趋势特征
                    dist = np.sqrt(tvec ** 2 + y ** 2)
                    md = float(np.max(dist))
                    if md > 0:
                        dist_ratio_mean = float(np.mean(dist / md))
                    else:
                        dist_ratio_mean = 0.0

                    denom = float(tvec[-1] - tvec[0])
                    mean_slope = float((y[-1] - y[0]) / denom) if abs(denom) > 1e-12 else 0.0

                    dt = np.diff(tvec)
                    dy = np.diff(y)
                    if np.all(np.abs(dt) > 1e-12):
                        slopes = dy / dt
                        max_slope = float(np.min(slopes))
                        std_slope = float(np.std(slopes))
                    else:
                        max_slope = 0.0
                        std_slope = 0.0

                    # 2) 全局峰值特征（不用 scipy，采用简单局部极大值）
                    if n0 >= 3:
                        pk_mask = (y[1:-1] > y[:-2]) & (y[1:-1] > y[2:])
                        pk_vals = y[1:-1][pk_mask]
                    else:
                        pk_vals = np.array([], dtype=float)

                    if pk_vals.size > 0:
                        spk = np.sort(pk_vals)[::-1]
                        max_peak = float(spk[0])
                        second_peak = float(spk[1]) if spk.size >= 2 else float(spk[0])
                        peak_count = float(pk_vals.size)
                    else:
                        max_peak = float(y[0])
                        second_peak = float(y[0])
                        peak_count = 0.0

                    peak_density = float(peak_count / n0)

                    # 3) 全局波动与统计特征
                    area = float(np.trapz(y, tvec))
                    stdv = float(np.std(y))
                    meanv = float(np.mean(y))
                    cv = float(stdv / (meanv + 1e-12))
                    mu0 = meanv
                    c3 = float(np.mean((y - mu0) ** 3))
                    c4 = float(np.mean((y - mu0) ** 4))
                    skewness = float(c3 / (stdv ** 3 + 1e-12))
                    kurtosis = float(c4 / (stdv ** 4 + 1e-12))

                    # 4) 局部滑动窗口特征
                    win_rows = []
                    start = 0
                    while start < n0:
                        end = min(n0, start + win_size)
                        yw = y[start:end]
                        tw = tvec[start:end]
                        if yw.size > 1:
                            dtw = np.diff(tw)
                            dyw = np.diff(yw)
                            if np.all(np.abs(dtw) > 1e-12):
                                slope_w = dyw / dtw
                                max_slope_w = float(np.min(slope_w))
                            else:
                                max_slope_w = 0.0
                        else:
                            max_slope_w = 0.0

                        std_w = float(np.std(yw))
                        pk2pk_w = float(np.max(yw) - np.min(yw))
                        win_rows.append((max_slope_w, std_w, pk2pk_w))
                        start += step

                    W = np.array(win_rows, dtype=float) if win_rows else np.zeros((1, 3), dtype=float)
                    local_max_slope_mean = float(np.mean(W[:, 0]))
                    local_max_slope_min = float(np.min(W[:, 0]))
                    local_std_mean = float(np.mean(W[:, 1]))
                    local_std_max = float(np.max(W[:, 1]))
                    local_pk2pk_mean = float(np.mean(W[:, 2]))
                    local_pk2pk_max = float(np.max(W[:, 2]))

                    return (
                        dist_ratio_mean, mean_slope, max_slope, std_slope,
                        max_peak, second_peak, peak_count, peak_density,
                        area, stdv, cv, skewness, kurtosis,
                        local_max_slope_mean, local_max_slope_min,
                        local_std_mean, local_std_max,
                        local_pk2pk_mean, local_pk2pk_max,
                    )

                # 逐帧计算 cor 曲线与特征
                N_f = int(freqs_cor.size)
                keep_n = max(1, int(round(N_f * 0.5)))
                for k in range(T):
                    s0 = S_cor[:, k]
                    if s0.size != N_f:
                        continue

                    # STEP1: 保留Amplitude前 50% 的频点
                    if keep_n >= N_f:
                        s = s0.astype(float)
                    else:
                        idx_keep = np.argpartition(s0, -keep_n)[-keep_n:]
                        s = np.zeros_like(s0, dtype=float)
                        s[idx_keep] = s0[idx_keep]

                    # STEP2: 频移自相关（向下位移）并归一化
                    r = np.zeros(N_shift + 1, dtype=float)
                    r[0] = float(np.dot(s, s))
                    for tau_i in range(1, N_shift + 1):
                        r[tau_i] = float(np.dot(s[tau_i:], s[:-tau_i]))

                    r = r - float(np.min(r))
                    mx = float(np.max(r))
                    if mx > 0:
                        r = r / mx

                    (v0, v1, v2, v3,
                     v4, v5, v6, v7,
                     v8, v9, v10, v11, v12,
                     v13, v14, v15, v16, v17, v18) = _curve_features_19(r, tau_axis)

                    cor_dist_ratio_mean[k] = v0
                    cor_mean_slope[k] = v1
                    cor_max_slope[k] = v2
                    cor_std_slope[k] = v3
                    cor_max_peak[k] = v4
                    cor_second_peak[k] = v5
                    cor_peak_count[k] = v6
                    cor_peak_density[k] = v7
                    cor_area[k] = v8
                    cor_std[k] = v9
                    cor_cv[k] = v10
                    cor_skewness[k] = v11
                    cor_kurtosis[k] = v12
                    cor_local_max_slope_mean[k] = v13
                    cor_local_max_slope_min[k] = v14
                    cor_local_std_mean[k] = v15
                    cor_local_std_max[k] = v16
                    cor_local_pk2pk_mean[k] = v17
                    cor_local_pk2pk_max[k] = v18

        f_cor = {
            "cor_dist_ratio_mean": cor_dist_ratio_mean,
            "cor_mean_slope": cor_mean_slope,
            "cor_max_slope": cor_max_slope,
            "cor_std_slope": cor_std_slope,
            "cor_max_peak": cor_max_peak,
            "cor_second_peak": cor_second_peak,
            "cor_peak_count": cor_peak_count,
            "cor_peak_density": cor_peak_density,
            "cor_area": cor_area,
            "cor_std": cor_std,
            "cor_cv": cor_cv,
            "cor_skewness": cor_skewness,
            "cor_kurtosis": cor_kurtosis,
            "cor_local_max_slope_mean": cor_local_max_slope_mean,
            "cor_local_max_slope_min": cor_local_max_slope_min,
            "cor_local_std_mean": cor_local_std_mean,
            "cor_local_std_max": cor_local_std_max,
            "cor_local_pk2pk_mean": cor_local_pk2pk_mean,
            "cor_local_pk2pk_max": cor_local_pk2pk_max,
        }

        f_spec = {
            "Spectral Centroid": cent,
            "Spectral Bandwidth": bw,
            "Spectral Skewness": skew_s,
            "Spectral Kurtosis": kurt_s,
            "Spectral Roll-off": roll,
            "Spectral Flatness": flat,
            "Spectral Entropy": ent,
            "Spectral Flux": flux,
        }

        # 追加补齐的频域/时频特征（不影响既有键的顺序）
        f_spec.update({
            "Spectral Mean": spec_mean,
            "Spectral Standard Deviation": spec_std,
            "Spectral Median": spec_median,
            "Spectral Energy": spec_energy,
            "Spectral RMS": spec_rms,
            "Spectral Amplitude Sum": spec_sum,
            "Maximum Spectral Peak": max_peak,
            "Number of Spectral Peaks": peak_count,
            "Low-frequency Energy Ratio": low_ratio,
            "Mid-frequency Energy Ratio": mid_ratio,
            "High-frequency Energy Ratio": high_ratio,
            "Spectral IQR": spec_iqr,
            "Spectral MAD": spec_mad,
            "Spectral Difference ZCR": spec_zc,
            "Spectral Smoothness": spec_smooth,
            "Primary-to-secondary Peak Ratio": peak_ratio,
            "Spectral Complexity": spec_complex,
            "Primary Peak Energy Ratio": top1_energy_ratio,
            "Top-3 Peak Energy Ratio": top3_energy_ratio,
            "90% Energy Coverage Bins": n_bins_90pct,
            "Primary Peak -3 dB Bandwidth": dominant_bw_hz,
            "Primary Peak Q Factor": dominant_Q,
        })

        # 追加 cor 特征（键名以 cor_ 开头，便于与你的离线分析/论文一致）
        f_spec.update(f_cor)

        feat = {**f_time, **f_spec}
        return times, feat

    def ensure_frame_features(self):
        """
        重新计算帧级特征和Time轴。

        说明：
            - 每次调用都Built with 当前 STFT / Short-time Features参数重新计算；
            - 不再依赖旧的缓存，因此只要窗长、步长发生改变，
              下次调用本函数时特征会自动更新；
            - 后续 build_frame_labels_for_tag() 也会在新的Time轴上重建标签。
        """
        # 1) 调用你现有的Short-time Features计算函数
        times, feat_dict = self.compute_short_time_features()

        # 2) 基本有效性检查
        if times is None:
            self.stft_frame_times = None
            self.stft_features = None
            self.stft_feature_names = None
            return

        times = np.asarray(times, dtype=float)

        if times.size == 0 or not feat_dict:
            self.stft_frame_times = None
            self.stft_features = None
            self.stft_feature_names = None
            return

        # 3) 将特征字典堆成 (T, D) 的矩阵
        #    feat_dict: {name -> 1D array(T,)}
        names = list(feat_dict.keys())
        feat_list = []

        for name in names:
            v = np.asarray(feat_dict[name], dtype=float).reshape(-1)
            feat_list.append(v)

        X = np.vstack(feat_list).T  # (T, D)

        # 防御性对齐：如果 times 与 X 行数有偏差，截到一致
        T_feat, T_time = X.shape[0], times.shape[0]
        if T_feat != T_time:
            T = min(T_feat, T_time)
            X = X[:T, :]
            times = times[:T]

        # 4) 构造“前后帧平滑特征”
        #    X_sm[k] = 0.5 * X[k] + 0.25 * X[k-1] + 0.25 * X[k+1]
        X_sm = 0.5 * X.copy()
        if X.shape[0] > 1:
            X_sm[1:] += 0.25 * X[:-1]
            X_sm[:-1] += 0.25 * X[1:]

        X_full = np.concatenate([X, X_sm], axis=1)

        # 5) 写回缓存
        self.stft_frame_times = times  # (T,)
        self.stft_features = X_full  # (T, 2D)
        self.stft_feature_names = names + [n + "_sm" for n in names]

    def update_features_plot(self):
        """根据 self.selected_features 计算并绘制（0-1 归一化叠加Display）"""
        self.feat_plot.clear()
        self.feat_plot.addLegend()

        times, feat = self.compute_short_time_features()
        if times.size == 0 or not feat:
            return

        # 画最多 5 条
        names = [nm for nm in self.selected_features if nm in feat][:5]
        self._assign_feature_colors(names)  # 选择颜色
        for nm in names:
            y = np.asarray(feat[nm], dtype=float)
            # 0-1 归一化，避免量纲/量级差异
            ymin, ymax = float(np.min(y)), float(np.max(y))
            if abs(ymax - ymin) < 1e-12:
                y_plot = np.zeros_like(y)
            else:
                y_plot = (y - ymin) / (ymax - ymin)
            color = self.feature_color_map.get(nm, QColor("#999999"))
            curve = self.feat_plot.plot(times, y_plot, name=nm, pen=pg.mkPen(color, width=2))
            self.feature_curves[nm] = curve

        # Time轴范围与 STFT 对齐
        self.feat_plot.setLimits(xMin=0, xMax=self.duration)
        self.feat_plot.setXRange(0, self.duration, padding=0)

    def _get_palette_256(self, name: str) -> np.ndarray:
        """返回 (256,3) 的 0~1 RGB 调色板，仅支持 'Heatmap' / 'Grayscale'。"""

        def _interp(points):
            pos = np.array([p[0] for p in points], float)
            col = np.array([p[1] for p in points], float)
            xs = np.linspace(0, 1, 256)
            out = np.empty((256, 3), float)
            for k in range(3):
                out[:, k] = np.interp(xs, pos, col[:, k])
            return np.clip(out, 0.0, 1.0)

        if name == "Heatmap":
            # viridis 风格：紫→靛→青绿→黄绿→亮黄（近似）
            pts = [
                (0.00, (68 / 255, 1 / 255, 84 / 255)),
                (0.25, (59 / 255, 82 / 255, 139 / 255)),
                (0.50, (33 / 255, 145 / 255, 140 / 255)),
                (0.75, (94 / 255, 201 / 255, 98 / 255)),
                (1.00, (253 / 255, 231 / 255, 37 / 255)),
            ]
            return _interp(pts)

        # Grayscale
        g = np.linspace(0, 1, 256)
        return np.stack([g, g, g], axis=1)

    def _colorize_spec_with_window(self, Z2d: np.ndarray) -> np.ndarray:
        """把 2D 谱值按 stft_vmin/vmax 开窗后着色成 uint8 RGB(H,W,3)。"""
        Z = np.asarray(Z2d, float)
        finite = np.isfinite(Z)
        if not np.any(finite):
            return np.zeros((Z.shape[0], Z.shape[1], 3), dtype=np.uint8)

        # 上下限：优先用用户Settings；否则用 1%~99% 分位
        vmin = self.stft_vmin
        vmax = self.stft_vmax
        if vmin is None or vmax is None or not (vmax > vmin):
            vmin = float(np.percentile(Z[finite], 1))
            vmax = float(np.percentile(Z[finite], 99))
            if not (vmax > vmin):
                vmax = vmin + 1.0

        Zn = np.clip((Z - vmin) / (vmax - vmin), 0.0, 1.0)
        lut = (self._get_palette_256(self.stft_cmap) * 255.0).astype(np.uint8)  # (256,3)
        idx = np.clip((Zn * 255.0 + 0.5).astype(np.int16), 0, 255)
        return lut[idx]  # (H,W,3)

    def _on_cmap_changed(self, text: str):
        """配色切换：不重算 STFT，只对缓存谱重着色。"""
        self.stft_cmap = text
        if self._last_spec_vals is not None and hasattr(self, "spec_img"):
            disp = self._last_spec_vals.T  # 与主图方向一致
            rgb = self._colorize_spec_with_window(disp)
            self.spec_img.setImage(rgb, autoLevels=False)

    def _assign_feature_colors(self, selected_names):
        """按 selected_names 的顺序分配不重复颜色"""
        self.feature_color_map = {}
        used = 0
        for name in selected_names[:5]:
            self.feature_color_map[name] = self.feature_palette[used % len(self.feature_palette)]
            used += 1

    def invalidate_ml_cache(self):
        """
        当 STFT / Short-time Features的窗长、步长等参数发生改变时调用：
        - 清空帧级特征缓存
        - 清空已训练的模型
        这样下次训练 / 构造标签时会自动使用新的参数重新计算。
        """
        # 清Short-time Features缓存
        self.stft_frame_times = None
        self.stft_features = None
        self.stft_feature_names = None

        # 已训练的Machine Learning模型不再有效，直接全部丢弃
        if hasattr(self, "ml_models"):
            self.ml_models.clear()

    def _iter_manual_annotations(self):
        """
        统一遍历“人工标注”：
        - 兼容 (start, end, label) 和 (start, end, label, source) 两种形式
        - 三元组一律视为人工标注（source='manual'）
        - 四元组根据 src 是否为 'manual' 决定
        """
        for item in self.annotations:
            if item is None:
                continue

            # 兼容 3 元组 / 4 元组，其它长度直接跳过
            try:
                if len(item) == 3:
                    s, e, t = item
                    src = "manual"  # 三元组直接视为人工标记
                elif len(item) >= 4:
                    s, e, t, src = item[:4]  # 四元组用自带的 source
                else:
                    continue
            except Exception:
                continue

            if src == "manual":
                yield float(s), float(e), str(t)

    def get_manual_segments_for_label(self, label):
        """
        返回当前Label label 的所有“人工标注”区间 [(s, e), ...]。
        - 三元组默认视为人工标注
        - 四元组仅当 source == 'manual' 时才算
        """
        segs = []
        for s, e, t in self._iter_manual_annotations():
            if t == label:
                segs.append((s, e))
        return segs

    def get_reviewed_prefix(self):
        """
        返回“已审阅前缀”的长度 T（秒）：
        - 遍历所有人工标注（包括三元组和 source='manual' 的四元组）
        - 取它们 end 的最大值
        """
        T = 0.0
        for s, e, _ in self._iter_manual_annotations():
            T = max(T, e)
        return float(T)

    def build_frame_labels_for_tag(self, label, neg_margin=0.05):
        """
        针对单一Label label（例如 'Wheeze'）构建帧级标签向量 y。

        返回:
            y: shape (T,), 值 ∈ {1, 0, -1}
               1: 该帧是 label 的正样本（整段 [s, e] 全部算正样本）
               0: 安全负样本（前缀内、且距离任意正样本段 >= neg_margin）
              -1: 忽略（不参与训练）
        """
        # 确保已经有帧级特征和Time轴
        self.ensure_frame_features()
        times = self.stft_frame_times

        # 没有特征，直接放弃
        if times is None or len(times) == 0:
            return None

        times = np.asarray(times, dtype=float)

        # 已审阅前缀（只用前缀内的帧参与训练）
        T_used = self.get_reviewed_prefix()
        if T_used is None or T_used <= 0:
            return None

        # 先全部标为 -1（忽略）
        y = np.full(times.shape, -1, dtype=np.int8)

        # 当前标签的人工标注段
        segs_pos = self.get_manual_segments_for_label(label)

        # ---------- 1) 标记正样本：整段 [s, e] 都算正样本 ----------
        for (s, e) in segs_pos:
            if e <= s:
                continue
            idx = np.where((times >= s) & (times <= e))[0]
            y[idx] = 1

        # ---------- 2) 构造扩展正区域掩码，用于排除“过近”的负样本 ----------
        # mask_pos_ext = True 的地方表示“靠近正样本，不适合作为负样本”
        mask_pos_ext = np.zeros_like(times, dtype=bool)
        for (s, e) in segs_pos:
            if e <= s:
                continue
            ext_start = max(0.0, s - neg_margin)
            ext_end = e + neg_margin
            if ext_end <= ext_start:
                continue
            idx = np.where((times >= ext_start) & (times <= ext_end))[0]
            mask_pos_ext[idx] = True

        # 前缀内帧
        mask_prefix = (times <= T_used)

        # ---------- 3) 标记安全负样本：在前缀内 & 不在扩展正区域内 & 当前不是正样本 ----------
        idx_neg = np.where(mask_prefix & (~mask_pos_ext) & (y != 1))[0]
        y[idx_neg] = 0

        # 若前缀内既没有 1 也没有 0，说明当前无法用于训练，返回 None
        if not np.any(y == 1) and not np.any(y == 0):
            return None

        return y

    def clear_ml_annotations_for_label(self, label):
        return self.ml_service.clear_ml_annotations_for_label(**{k: v for k, v in locals().items() if k != 'self'})
    def train_model_for_label(self,
                              label,
                              min_pos_frames=30,
                              neg_pos_ratio=5,
                              random_state=None):
        return self.ml_service.train_model_for_label(**{k: v for k, v in locals().items() if k != 'self'})
    def apply_model_for_label_on_unreviewed(self,
                                            label,
                                            min_dur_sec=0.05):
        return self.ml_service.apply_model_for_label_on_unreviewed(**{k: v for k, v in locals().items() if k != 'self'})


if __name__ == '__main__':
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    app = QApplication(sys.argv)
    viewer = AudioViewer()
    viewer.show()
    sys.exit(app.exec_())
