import os
import sys

# 1. 拼接出 plugins File夹的绝对路径
# 注意：根据你的实际安装情况，可能需要调整 'PyQt5', 'Qt5', 'plugins' 这些层级
base_path = os.path.join(sys.base_prefix, "Lib", "site-packages", "PyQt5", "Qt5", "plugins")

# 2. 检查路径是否存在 (如果不存在，尝试去掉 'Qt5' 这一层)
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
try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

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

from collections import defaultdict


class AnnotationLabelDialog(QDialog):
    """标注标签选择对话框：支持预设类型 + Custom文本"""

    def __init__(self, parent=None, builtin_labels=None, start=None, end=None, default_text=""):
        super().__init__(parent)
        self.setWindowTitle("Add Annotation")
        self._text = None

        layout = QVBoxLayout(self)

        # 顶部Notice：Time段
        if start is not None and end is not None:
            info = QLabel(f"Annotation interval: {start:.3f} - {end:.3f} s")
            layout.addWidget(info)

        form = QFormLayout()
        layout.addLayout(form)

        # 预设类型下拉框：如 哮鸣音(Wheeze)、爆裂音(Crackles) 等
        self.combo = QComboBox()
        self.combo.addItem("(No preset)", userData=None)
        if builtin_labels:
            for cn, en in builtin_labels:
                self.combo.addItem(f"{cn} ({en})", userData=en)
        form.addRow("Preset type:", self.combo)

        # 文本输入框：最终用于保存的标签文本 (通常是英文)
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
        """模态执行并返回最终文本 (可能为 None)。"""
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
    QSlider, QHBoxLayout, QDoubleSpinBox, QCheckBox
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



class SpanLabelItem(pg.TextItem):
    # "\"\"标注条上的文字标签：把右键/双击事件转发给所属 BoxSpan，避免被标记拖拽逻辑抢走。\"\"\"
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
        # 右键菜单交给 BoxSpan；左键仅用于阻止触发“拖拽标记”
        if self._owner_span is not None:
            if ev.button() == Qt.RightButton:
                try:
                    self._owner_span.mouseClickEvent(ev)
                except Exception:
                    pass
            ev.accept()
        else:
            ev.ignore()


class BoxSpan(pg.RectROI):
    def __init__(self, x0, x1, y_base, height, text, owner, label_color=None):
        super().__init__(pos=[x0, y_base], size=[x1 - x0, height],
                         # 默认“只读”：不允许拖动/改边界；仅在编辑模式下开启
                         movable=False,
                         resizable=False,  # 禁掉四角把手
                         pen=pg.mkPen(255, 255, 255, 255, width=1))
        # —— 视觉编码：不再强制纯白填充 (后面统一由 _apply_visual_style() 控制)——
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

        # —— 编辑模式门控 (默认关闭，双击进入)——
        self._edit_mode = False
        self._orig_interval = None  # (x0, x1)
        self._orig_item = None      # data层快照 (用于编辑提交后可撤销)
        self._handles = []

        # ★ 新增：标签文字颜色 (不传则默认黑色)
        if label_color is None:
            self.label_color = QColor(0, 0, 0)  # 默认黑字
        else:
            self.label_color = label_color

        # 只保留左右把手 (默认隐藏+禁用；仅在编辑模式启用)
        hL = self.addScaleHandle([0, 0.5], [1, 0.5])
        hR = self.addScaleHandle([1, 0.5], [0, 0.5])
        self._handles = [hL, hR]

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
            # 禁用把手的鼠标交互 (避免与“标记拖拽”冲突)
            try:
                h.setAcceptedMouseButtons(Qt.NoButton)
            except Exception:
                try:
                    it = h['item']
                    it.setAcceptedMouseButtons(Qt.NoButton)
                except Exception:
                    pass

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

        # 备注小白框 (Display在条上方)
        self.label = SpanLabelItem(owner_span=self, anchor=(0.5, 0.0))  # 先建一个空的
        self.owner.annot_plot.addItem(self.label)
        try:
            self.label._owner_span = self
        except Exception:
            pass
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
    # 编辑模式：仅在该模式下允许拖动/改边界
    # ==========================
    def enter_edit_mode(self):
        if self._edit_mode:
            return
        self._edit_mode = True
        self._orig_interval = self.interval()

        # 记录 data 层快照 (用于提交后可撤销)
        self._orig_item = None
        try:
            idx = None
            try:
                idx = int(getattr(self.owner, "_span2idx", {}).get(self))
            except Exception:
                idx = None
            if idx is not None and 0 <= idx < len(getattr(self.owner, "annotations", [])):
                it = self.owner.annotations[idx]
                if it is not None:
                    if len(it) == 3:
                        self._orig_item = (float(it[0]), float(it[1]), str(it[2]), "manual")
                    else:
                        self._orig_item = (float(it[0]), float(it[1]), str(it[2]), str(it[3]) if len(it) >= 4 else "manual")
        except Exception:
            self._orig_item = None

        # 允许整体平移
        try:
            self.setMovable(True)
        except Exception:
            pass

        # 启用并Display左右把手 (仅用于水平改边界)
        for h in getattr(self, "_handles", []):
            try:
                h.setOpacity(1.0)
                h.setAcceptedMouseButtons(Qt.LeftButton)
            except Exception:
                try:
                    it = h['item']
                    it.setOpacity(1.0)
                    it.setAcceptedMouseButtons(Qt.LeftButton)
                except Exception:
                    pass

        # 视觉Notice：编辑态用虚线 (不改变 source 的语义；Exit后恢复)
        try:
            pen_now = getattr(self, "pen", None)
        except Exception:
            pen_now = None
        try:
            base = self.owner.get_annotation_color(getattr(self, "text", ""))
            edge_color = QColor(base)
            edge_color.setAlpha(230)
            self.setPen(pg.mkPen(edge_color, width=2.5, style=Qt.DashDotLine))
        except Exception:
            pass

        # 通知 owner：进入编辑态 (用于键盘 Enter/Esc)
        try:
            self.owner.begin_edit_span(self)
        except Exception:
            pass

        # 初次刷新状态栏
        try:
            self.owner._update_edit_status(self)
        except Exception:
            pass

    def exit_edit_mode(self, commit: bool = True):
        if not self._edit_mode:
            return

        old_interval = self._orig_interval
        old_item = getattr(self, "_orig_item", None)

        idx = None
        try:
            idx = int(getattr(self.owner, "_span2idx", {}).get(self))
        except Exception:
            idx = None

        # 取消：恢复原区间 (不入 undo)
        if not commit and old_interval is not None:
            try:
                x0, x1 = old_interval
                self.setPos([float(x0), self.y_base], update=False)
                self.setSize([float(x1 - x0), self.h_fix], update=False)
            except Exception:
                pass

        self._edit_mode = False

        # 禁用移动
        try:
            self.setMovable(False)
        except Exception:
            pass

        # 隐藏并禁用把手
        for h in getattr(self, "_handles", []):
            try:
                h.setOpacity(0.0)
                h.setAcceptedMouseButtons(Qt.NoButton)
            except Exception:
                try:
                    it = h['item']
                    it.setOpacity(0.0)
                    it.setAcceptedMouseButtons(Qt.NoButton)
                except Exception:
                    pass

        # 强制同步一次 (更新 annotations/spec/label + 恢复 source 对应样式)
        try:
            self._on_changed()
        except Exception:
            pass

        # 提交：若区间发生变化，则入栈 undo，支持 Ctrl+Z
        if commit and idx is not None and old_item is not None and old_interval is not None:
            try:
                n0, n1 = self.interval()
                o0, o1 = float(old_interval[0]), float(old_interval[1])
                if abs(n0 - o0) > 1e-9 or abs(n1 - o1) > 1e-9:
                    # new_item 以 data 层为准
                    new_item = None
                    try:
                        if 0 <= idx < len(self.owner.annotations):
                            it = self.owner.annotations[idx]
                            if it is not None:
                                if len(it) == 3:
                                    new_item = (float(it[0]), float(it[1]), str(it[2]), "manual")
                                else:
                                    new_item = (float(it[0]), float(it[1]), str(it[2]), str(it[3]) if len(it) >= 4 else "manual")
                    except Exception:
                        new_item = None

                    if new_item is not None:
                        try:
                            try:
                                _old_src = str(old_item[3]) if (old_item is not None and len(old_item) >= 4) else "manual"
                            except Exception:
                                _old_src = "manual"
                            # 若编辑的是机器/认可标注，则将 source 升级为 auto_edited (便于进入训练与导出追踪)
                            try:
                                _old_src_n = str(_old_src).strip().lower()
                                if _old_src_n in {"ml", "auto", "machine", "model", "pred", "auto_accepted"}:
                                    new_item = (float(new_item[0]), float(new_item[1]), str(new_item[2]), "auto_edited")
                                    try:
                                        if 0 <= idx < len(self.owner.annotations):
                                            self.owner.annotations[idx] = new_item
                                    except Exception:
                                        pass
                                    try:
                                        self._apply_visual_style("auto_edited")
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                            try:
                                _new_src = str(new_item[3]) if (new_item is not None and len(new_item) >= 4) else "manual"
                            except Exception:
                                _new_src = "manual"
                            try:
                                print(f"[VERIFY][EDIT] idx={idx} label={new_item[2] if new_item is not None else ''} "
                                      f"src_old={_old_src} src_new={_new_src} "
                                      f"{o0:.4f}-{o1:.4f} -> {n0:.4f}-{n1:.4f}")
                            except Exception:
                                pass

                            self.owner._push_undo({
                                "op": "edit_interval",
                                "idx": int(idx),
                                "old_item": old_item,
                                "new_item": new_item,
                            })
                        except Exception:
                            pass
            except Exception:
                pass

        # 清理快照
        self._orig_interval = None
        self._orig_item = None

        try:
            self.owner.end_edit_span(self)
        except Exception:
            pass

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
        - 颜色 (边框/填充)：来自标签类别颜色 (owner.get_annotation_color)，避免人工标记编辑后边框变黑/消失
        - source：只决定线型/透明度 (manual 更“权威”，未确认机器标注更“建议”)
        """
        # Box 基色：按标签类别稳定映射 (不要用 label_color；label_color 仅用于文字颜色)
        try:
            base = self.owner.get_annotation_color(getattr(self, "text", ""))
            c = QColor(base)
        except Exception:
            c = QColor(255, 255, 255)

        # 主色 (边框/强调)
        edge_color = QColor(c)
        try:
            edge_color.setAlpha(220)
        except Exception:
            pass

        src_n = str(src).strip().lower()

        # 视觉约定：
        # - 未确认的机器标注：虚线 (ml/auto/pred 等)或 merged_*global*
        # - 已认可/已编辑/局部合并后的标注：实线 (auto_accepted/auto_edited/merged 等)
        is_ml_like = (
            src_n in {"ml", "auto", "machine", "model", "pred"}
            or (src_n.startswith("merged") and ("global" in src_n))
        )

        if is_ml_like:
            pen = pg.mkPen(edge_color, width=1.5, style=Qt.DashLine)
            fill = QColor(c)
            try:
                fill.setAlpha(115)
            except Exception:
                pass
            label_bg = "rgba(255,255,255,0.80)"
            label_border = "rgba({},{},{},0.55)".format(c.red(), c.green(), c.blue())
        else:
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

        # 同步 label 的视觉 (不改文字内容)
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
        # 颜色变了，视觉也应一起刷新 (manual/ml 不变)
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

        # 备注框跟随 (在条上方一点点)
        cx = float(p.x() + s.x() / 2.0)
        cy = float(self.y_base + s.y()) + 0.05
        self.label.setPos(cx, cy)
        self._sync_fill_item()

        # 同步频谱红条
        m = getattr(self.owner, "_span2spec", {})
        if self in m:
            a, b = self.interval()
            m[self].setRegion([a, b])

        # 同步导出缓存 (兼容 3/4 元组；三元组视为人工标注)
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

                # —— 视觉编码：
                # 非编辑态：按 source 同步样式
                # 编辑态：保持“编辑态虚线”Notice，不要被 source 样式覆盖
                if not getattr(self, "_edit_mode", False):
                    self._apply_visual_style(src)

        # 编辑态：实时在状态栏Display起止Time
        if getattr(self, "_edit_mode", False):
            try:
                self.owner._update_edit_status(self)
            except Exception:
                pass

    def mouseDoubleClickEvent(self, ev):
        # 双击切换编辑模式：默认进入；编辑态再双击提交Exit
        try:
            if not self._edit_mode:
                self.enter_edit_mode()
            else:
                self.exit_edit_mode(commit=True)
            ev.accept()
            return
        except Exception:
            pass
        # 兜底：如果出错，仍允许Play
        try:
            s0, s1 = self.interval()
            self.owner.open_loop_player(s0, s1)
        except Exception:
            pass
        ev.accept()

    def mouseDragEvent(self, ev):
        # 非编辑模式下：吞掉拖拽 (不移动/不改边界)，避免与“标记拖拽”冲突
        if not getattr(self, "_edit_mode", False):
            try:
                ev.accept()
            except Exception:
                pass
            return
        # 编辑模式下：交给 ROI 默认实现 (支持平移/改边界)
        return super().mouseDragEvent(ev)

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.RightButton:
            menu = QMenu()

            # —— 机器标注“认可”：将 source 置为 auto_accepted (进入训练/计入已审阅前缀)——
            src = ""
            try:
                src = str(self._infer_source()).strip().lower()
            except Exception:
                src = ""
            ml_like = src in {"ml", "auto", "machine", "model", "pred"}

            accept_action = None
            if ml_like:
                accept_action = menu.addAction("✅ Accept (use for training)")

            play_action = menu.addAction("▶ Play")
            del_action = menu.addAction("🗑 Delete")

            act = menu.exec_(ev.screenPos().toPoint())

            if accept_action is not None and act == accept_action:
                try:
                    self.owner.accept_annotation(self, accepted_source="auto_accepted")
                except Exception:
                    pass
            elif act == play_action:
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
                 feature_color_map=None,
                 preprocessing_enabled=True,
                 resample_enabled=True, resample_target_sr=4000,
                 filter_enabled=False, filter_type="bandpass",
                 filter_lowcut=20.0, filter_highcut=1800.0,
                 filter_order=4, filter_zero_phase=True,
                 auto_label_import_enabled=False, auto_label_import_settings=None):

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

        # —— 新增：读取音频时的预处理设置。默认开启 4000 Hz 重采样；滤波默认关闭，避免改变旧行为。——
        self._preprocessing_enabled = bool(preprocessing_enabled)
        self._resample_enabled = bool(resample_enabled)
        try:
            self._resample_target_sr = int(resample_target_sr)
        except Exception:
            self._resample_target_sr = 4000

        self._filter_enabled = bool(filter_enabled)
        self._filter_type = str(filter_type or "bandpass").lower()
        if self._filter_type not in {"bandpass", "lowpass", "highpass", "bandstop"}:
            self._filter_type = "bandpass"
        try:
            self._filter_lowcut = float(filter_lowcut)
        except Exception:
            self._filter_lowcut = 20.0
        try:
            self._filter_highcut = float(filter_highcut)
        except Exception:
            self._filter_highcut = 1800.0
        try:
            self._filter_order = int(filter_order)
        except Exception:
            self._filter_order = 4
        self._filter_zero_phase = bool(filter_zero_phase)

        # —— 新增：自动读取同名标签文件的解析设置。默认保持旧逻辑：auto/csv/txt、_events 后缀、自动分隔符、前3列为 start/end/label。——
        default_label_settings = {
            "file_format": "auto",
            "file_suffix": "_events",
            "delimiter": "auto",
            "custom_delimiter": "",
            "skip_header_lines": 0,
            "start_col": 1,
            "end_col": 2,
            "label_col": 3,
            "source_col": 4,
            "json_start_key": "start",
            "json_end_key": "end",
            "json_label_key": "label",
            "json_source_key": "source",
        }
        if isinstance(auto_label_import_settings, dict):
            default_label_settings.update(auto_label_import_settings)
        self._auto_label_import_enabled = bool(auto_label_import_enabled)
        self._auto_label_import_settings = default_label_settings

        tabs = QTabWidget()

        # ==== STFT 标签页 ====
        stft_tab = QWidget()
        stft_layout = QFormLayout()

        self.n_fft_box = QSpinBox()
        self.n_fft_box.setRange(64, 8192)
        self.n_fft_box.setValue(n_fft)
        stft_layout.addRow("n_fft (窗长)", self.n_fft_box)

        self.hop_box = QSpinBox()
        self.hop_box.setRange(16, 4096)
        self.hop_box.setValue(hop_length)
        stft_layout.addRow("hop_length (步长)", self.hop_box)

        self.f_max_box = QSpinBox()
        self.f_max_box.setRange(500, 40000)
        self.f_max_box.setSingleStep(500)
        self.f_max_box.setValue(f_max)
        stft_layout.addRow("STFT Max Frequency (Hz)", self.f_max_box)

        # ==== STFT Display (直方图 + colorbar + 上下限 + Reset Defaults)====
        stft_group = QGroupBox("STFT Display: Histogram & ColorBar (Editable)")
        stft_vbox = QVBoxLayout(stft_group)

        # 顶部：直方图 + 渐变色条 (HistogramLUTWidget)
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

        # 初始化 colorbar 配色 (保留可编辑三角控件)
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

        line2.addWidget(QLabel("Lower:"))
        line2.addWidget(self.vmin_edit)
        line2.addSpacing(12)
        line2.addWidget(QLabel("Upper:"))
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

            # Settings到 HistogramLUTItem：这一步决定开窗区间 (与主图共享)
            self.hist_widget.setLevels(vmin, vmax)

            # 同步到输入框
            self.vmin_edit.blockSignals(True)
            self.vmax_edit.blockSignals(True)
            self.vmin_edit.setValue(vmin)
            self.vmax_edit.setValue(vmax)
            self.vmin_edit.blockSignals(False)
            self.vmax_edit.blockSignals(False)

        _init_levels_from_data()

        # 底部：Reset Defaults (回到 1%~99% 分位)
        btn_line = QHBoxLayout()
        self.btn_reset = QPushButton("Reset Defaults")
        btn_line.addStretch(1)
        btn_line.addWidget(self.btn_reset)

        # 组装到 group 里
        stft_vbox.addWidget(QLabel("STFT value histogram (drag the colorbar handles to set limits; colorbar is editable)"))
        stft_vbox.addWidget(self.hist_widget)
        stft_vbox.addLayout(line1)  # 配色
        stft_vbox.addLayout(line2)  # 下限/上限
        stft_vbox.addLayout(btn_line)  # Reset Defaults

        # 组装进 STFT 页
        try:
            stft_layout.addRow(stft_group)  # QFormLayout
        except Exception:
            stft_layout.addWidget(stft_group)  # 其他布局

        # ========= 联动逻辑：levels ↔ 文本框，顺带保证与主图一致 =========

        # 1)拖动 HistogramLUT items (或编辑 colorbar) → 更新输入框
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

        # 2)改输入框 → 回写到 HistogramLUT (开窗)，colorbar 的条和三角会一起动
        def _edits_to_levels():
            vmin = self.vmin_edit.value()
            vmax = self.vmax_edit.value()
            if vmax > vmin:
                self.hist_widget.setLevels(vmin, vmax)

        self.vmin_edit.valueChanged.connect(_edits_to_levels)
        self.vmax_edit.valueChanged.connect(_edits_to_levels)

        # 3)切换配色 → 更新 colorbar (保留颜色编辑功能)
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

        # 4)Reset Defaults (上下限回到 1%~99% 分位)
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

        # Reset Defaults按钮
        self.reset_button = QPushButton("Reset Defaults")
        self.reset_button.clicked.connect(self.restore_default_y_range)
        display_layout.addRow(self.reset_button)

        display_tab.setLayout(display_layout)
        tabs.addTab(display_tab, "Display")

        # ==== Preprocessing 标签页：读取 WAV 时的重采样与可选滤波 ====
        preproc_tab = QWidget()
        preproc_layout = QFormLayout()

        self.preprocessing_enable_check = QCheckBox("Enable preprocessing when loading WAV files")
        self.preprocessing_enable_check.setChecked(bool(self._preprocessing_enabled))
        preproc_layout.addRow(self.preprocessing_enable_check)

        self.resample_enable_check = QCheckBox("Enable resampling")
        self.resample_enable_check.setChecked(bool(self._resample_enabled))
        preproc_layout.addRow(self.resample_enable_check)

        self.resample_sr_box = QSpinBox()
        self.resample_sr_box.setRange(1000, 384000)
        self.resample_sr_box.setSingleStep(1000)
        self.resample_sr_box.setValue(int(max(1000, min(384000, self._resample_target_sr))))
        preproc_layout.addRow("Target sampling rate (Hz):", self.resample_sr_box)

        self.filter_enable_check = QCheckBox("Enable digital filter after loading/resampling")
        self.filter_enable_check.setChecked(bool(self._filter_enabled))
        preproc_layout.addRow(self.filter_enable_check)

        self.filter_type_combo = QComboBox()
        self.filter_type_combo.addItems(["bandpass", "lowpass", "highpass", "bandstop"])
        self.filter_type_combo.setCurrentText(self._filter_type)
        preproc_layout.addRow("Filter type:", self.filter_type_combo)

        self.filter_lowcut_box = QDoubleSpinBox()
        self.filter_lowcut_box.setDecimals(2)
        self.filter_lowcut_box.setRange(0.001, 192000.0)
        self.filter_lowcut_box.setSingleStep(5.0)
        self.filter_lowcut_box.setValue(float(max(0.001, self._filter_lowcut)))
        preproc_layout.addRow("Low cutoff (Hz):", self.filter_lowcut_box)

        self.filter_highcut_box = QDoubleSpinBox()
        self.filter_highcut_box.setDecimals(2)
        self.filter_highcut_box.setRange(0.001, 192000.0)
        self.filter_highcut_box.setSingleStep(50.0)
        self.filter_highcut_box.setValue(float(max(0.001, self._filter_highcut)))
        preproc_layout.addRow("High cutoff (Hz):", self.filter_highcut_box)

        self.filter_order_box = QSpinBox()
        self.filter_order_box.setRange(1, 12)
        self.filter_order_box.setValue(int(max(1, min(12, self._filter_order))))
        preproc_layout.addRow("Butterworth order:", self.filter_order_box)

        self.filter_zero_phase_check = QCheckBox("Use zero-phase filtering (sosfiltfilt)")
        self.filter_zero_phase_check.setChecked(bool(self._filter_zero_phase))
        preproc_layout.addRow(self.filter_zero_phase_check)

        preproc_hint = QLabel(
            "Resampling uses librosa's band-limited resampling, so anti-aliasing is handled internally. "
            "The optional Butterworth filter is applied after loading/resampling and is intended for analysis/display band control. "
            "For 4000 Hz target sampling, set high cutoff below Nyquist (e.g., 1800–1900 Hz). "
            "These preprocessing settings take effect the next time an audio file is loaded."
        )
        preproc_hint.setWordWrap(True)
        preproc_layout.addRow(preproc_hint)

        preproc_tab.setLayout(preproc_layout)
        tabs.addTab(preproc_tab, "Preprocessing")

        # ==== Auto Label Import 标签页：配置自动读取同名标签文件的解析规则 ====
        label_tab = QWidget()
        label_layout = QFormLayout()
        cfg = self._auto_label_import_settings

        self.auto_label_enable_check = QCheckBox("Enable auto-import matching label files when loading WAV")
        self.auto_label_enable_check.setChecked(bool(self._auto_label_import_enabled))
        label_layout.addRow(self.auto_label_enable_check)

        self.auto_label_format_combo = QComboBox()
        self.auto_label_format_combo.addItems(["auto", "csv", "txt", "json"])
        self.auto_label_format_combo.setCurrentText(str(cfg.get("file_format", "auto")).lower() if str(cfg.get("file_format", "auto")).lower() in {"auto", "csv", "txt", "json"} else "auto")
        label_layout.addRow("File format:", self.auto_label_format_combo)

        self.auto_label_suffix_edit = QLineEdit(str(cfg.get("file_suffix", "_events")))
        label_layout.addRow("File suffix before extension:", self.auto_label_suffix_edit)

        self.auto_label_delim_combo = QComboBox()
        self.auto_label_delim_combo.addItems(["auto", "comma ,", "semicolon ;", "tab", "space", "custom"])
        delim_value = str(cfg.get("delimiter", "auto")).lower()
        delim_map = {
            "auto": "auto",
            "comma": "comma ,", ",": "comma ,",
            "semicolon": "semicolon ;", ";": "semicolon ;",
            "tab": "tab", "\t": "tab",
            "space": "space", "whitespace": "space",
            "custom": "custom",
        }
        self.auto_label_delim_combo.setCurrentText(delim_map.get(delim_value, "auto"))
        label_layout.addRow("Delimiter for csv/txt:", self.auto_label_delim_combo)

        self.auto_label_custom_delim_edit = QLineEdit(str(cfg.get("custom_delimiter", "")))
        label_layout.addRow("Custom delimiter:", self.auto_label_custom_delim_edit)

        self.auto_label_skip_header_box = QSpinBox()
        self.auto_label_skip_header_box.setRange(0, 100)
        try:
            self.auto_label_skip_header_box.setValue(int(cfg.get("skip_header_lines", 0)))
        except Exception:
            self.auto_label_skip_header_box.setValue(0)
        label_layout.addRow("Skip header lines:", self.auto_label_skip_header_box)

        # 1-based 列号；source_col=0 表示不读取 source，统一 manual。
        self.auto_label_start_col_box = QSpinBox(); self.auto_label_start_col_box.setRange(1, 100)
        self.auto_label_end_col_box = QSpinBox(); self.auto_label_end_col_box.setRange(1, 100)
        self.auto_label_label_col_box = QSpinBox(); self.auto_label_label_col_box.setRange(1, 100)
        self.auto_label_source_col_box = QSpinBox(); self.auto_label_source_col_box.setRange(0, 100)
        for box, key, default in [
            (self.auto_label_start_col_box, "start_col", 1),
            (self.auto_label_end_col_box, "end_col", 2),
            (self.auto_label_label_col_box, "label_col", 3),
            (self.auto_label_source_col_box, "source_col", 4),
        ]:
            try:
                box.setValue(int(cfg.get(key, default)))
            except Exception:
                box.setValue(default)

        label_layout.addRow("Start time column (1-based):", self.auto_label_start_col_box)
        label_layout.addRow("End time column (1-based):", self.auto_label_end_col_box)
        label_layout.addRow("Label column (1-based):", self.auto_label_label_col_box)
        label_layout.addRow("Source column (0=disabled):", self.auto_label_source_col_box)

        json_hint = QLabel("For JSON, dict keys are used first; list rows still use the column mapping above.")
        json_hint.setWordWrap(True)
        label_layout.addRow(json_hint)

        self.auto_label_json_start_key_edit = QLineEdit(str(cfg.get("json_start_key", "start")))
        self.auto_label_json_end_key_edit = QLineEdit(str(cfg.get("json_end_key", "end")))
        self.auto_label_json_label_key_edit = QLineEdit(str(cfg.get("json_label_key", "label")))
        self.auto_label_json_source_key_edit = QLineEdit(str(cfg.get("json_source_key", "source")))
        label_layout.addRow("JSON start key:", self.auto_label_json_start_key_edit)
        label_layout.addRow("JSON end key:", self.auto_label_json_end_key_edit)
        label_layout.addRow("JSON label key:", self.auto_label_json_label_key_edit)
        label_layout.addRow("JSON source key:", self.auto_label_json_source_key_edit)

        label_tab.setLayout(label_layout)
        tabs.addTab(label_tab, "Auto Label Import")

        self.tabs = tabs

        # 按钮
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(tabs)
        layout.addWidget(buttons)

        # ==== Short-Time Features 标签页 ====
        feat_tab = QWidget()
        feat_layout = QFormLayout()

        hint = QLabel("Note: short-time features use the same window length/hop length as STFT by default;\n"
                      "Frequency-domain features are computed from the current STFT; time-domain features are computed from the current time-domain signal.\n"
                      "Select up to 5 features for display (normalized overlay).")
        hint.setWordWrap(True)
        feat_layout.addRow(hint)

        self.feat_list = QListWidget()
        self.feat_list.setSelectionMode(QListWidget.NoSelection)
        self.feat_list.setSpacing(6)  # 行间距更舒服
        # 可选：统一放大每一项的 sizeHint (若你想更高)
        for i in range(self.feat_list.count()):
            it = self.feat_list.item(i)
            it.setSizeHint(QSize(it.sizeHint().width(), 32))

        all_feats = [
            "短时能量", "短时均值", "方差", "峰度", "偏度", "过零率", "teager能量算子",

            # —— 频域/时频统计特征 (基于 STFT Amplitude谱)——
            "谱均值", "谱标准差", "谱中位数", "谱能量", "谱RMS", "谱幅和",
            "谱质心", "谱带宽", "谱偏度", "谱峰度", "谱滚降", "谱平坦度", "谱熵", "谱通量",
            "最大谱峰值", "谱峰数量",
            "低频能量占比", "中频能量占比", "高频能量占比",
            "谱四分位距", "谱MAD", "谱差分零交叉率", "谱平滑度", "主峰/次峰比", "谱复杂度",

            # —— 窄带/少峰检测增强特征 ——
            "主峰能量占比", "前三峰能量占比", "90%能量覆盖频点数", "主峰-3dB带宽", "主峰Q因子",

            # —— 频移自相关 (cor)特征：100–1200 Hz 子带，基于每帧谱的“向下频移自相关”曲线 ——
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

        # 把颜色写到每个条目的 UserRole，并顺带给未选的分配备用颜色 (避免后续点选颜色重复)
        fallback_used = used
        for i in range(self.feat_list.count()):
            it = self.feat_list.item(i)
            nm = it.text()
            if nm in color_map:
                it.setData(Qt.UserRole, color_map[nm])
            else:
                it.setData(Qt.UserRole, palette[fallback_used % len(palette)])
                fallback_used += 1

        # 预勾选 (来自主窗口)
        pre = set(selected_features or [])
        for i in range(self.feat_list.count()):
            it = self.feat_list.item(i)
            if it.text() in pre:
                it.setCheckState(Qt.Checked)

        def _limit_to_5(_):
            checked = [self.feat_list.item(i) for i in range(self.feat_list.count())
                       if self.feat_list.item(i).checkState() == Qt.Checked]
            if len(checked) > 5:
                # 取消最新一次勾选 (把它设回未选)
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
        tabs.addTab(feat_tab, "Short-Time Features")

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
        """Reset Defaults的自动计算的范围"""
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

    def get_resample_settings(self):
        # Backward-compatible getter: returns only the resampling switch and target sampling rate.
        return bool(self.resample_enable_check.isChecked()), int(self.resample_sr_box.value())

    def get_preprocessing_settings(self):
        return {
            "preprocessing_enabled": bool(self.preprocessing_enable_check.isChecked()),
            "resample_enabled": bool(self.resample_enable_check.isChecked()),
            "resample_target_sr": int(self.resample_sr_box.value()),
            "filter_enabled": bool(self.filter_enable_check.isChecked()),
            "filter_type": str(self.filter_type_combo.currentText()),
            "filter_lowcut": float(self.filter_lowcut_box.value()),
            "filter_highcut": float(self.filter_highcut_box.value()),
            "filter_order": int(self.filter_order_box.value()),
            "filter_zero_phase": bool(self.filter_zero_phase_check.isChecked()),
        }

    def get_auto_label_import_enabled(self):
        return bool(self.auto_label_enable_check.isChecked())

    def get_auto_label_import_settings(self):
        delim_text = self.auto_label_delim_combo.currentText()
        if delim_text.startswith("comma"):
            delim = "comma"
        elif delim_text.startswith("semicolon"):
            delim = "semicolon"
        elif delim_text == "tab":
            delim = "tab"
        elif delim_text == "space":
            delim = "space"
        elif delim_text == "custom":
            delim = "custom"
        else:
            delim = "auto"

        return {
            "file_format": str(self.auto_label_format_combo.currentText()).lower(),
            "file_suffix": str(self.auto_label_suffix_edit.text()).strip() or "_events",
            "delimiter": delim,
            "custom_delimiter": str(self.auto_label_custom_delim_edit.text()),
            "skip_header_lines": int(self.auto_label_skip_header_box.value()),
            "start_col": int(self.auto_label_start_col_box.value()),
            "end_col": int(self.auto_label_end_col_box.value()),
            "label_col": int(self.auto_label_label_col_box.value()),
            "source_col": int(self.auto_label_source_col_box.value()),
            "json_start_key": str(self.auto_label_json_start_key_edit.text()).strip() or "start",
            "json_end_key": str(self.auto_label_json_end_key_edit.text()).strip() or "end",
            "json_label_key": str(self.auto_label_json_label_key_edit.text()).strip() or "label",
            "json_source_key": str(self.auto_label_json_source_key_edit.text()).strip() or "source",
        }

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
        优先从 HistogramLUT 的 levels 读 (保证和 colorbar 对齐)
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
    BOX_SIZE = 22  # 放大方块 (原来16)
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

        # 颜色 (来自 UserRole；未勾选也可以先拿着，但不填充)
        color = index.data(Qt.UserRole)
        if not isinstance(color, QColor):
            try:
                color = QColor(color) if color else QColor("#999999")
            except Exception:
                color = QColor("#999999")

        # 方块区域 (更大)
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
    """集中管理Machine Learning训练/推理逻辑 (Step1：仅搬家，不改算法/输出)。"""
    # --- Label taxonomy routing (phase vs abnormal breath sounds vs other events) ---
    # 1) 呼吸时相 (HSMM 后处理，仅在 Inspiration/Expiration/Pause 管线中使用)
    PHASE_LABELS = {
        "inspiration", "expiration", "pause",
        "insp", "exp", "inhale", "exhale", "exspiration",
        "吸气", "呼气", "Pause", "停顿"
    }
    # 3) 其他异常事件 (说话、咳嗽等非呼吸过程异常音)，目前先与异常音共用同一套二分类训练/后处理，
    #    但通过 model_kind 与 dispatcher 分离，便于后续替换为专用模型与后处理。
    OTHER_EVENT_LABELS = {
        "speech", "talk", "talking", "speaking", "voice", "vocal", "whisper",
        "cough", "coughing", "sneeze", "snore", "laugh", "cry", "swallow", "throat", "说话", "讲话", "咳嗽", "咳",
        "noise", "artifact", "movement", "rub", "stethoscope", "background", "normal"
    }
    # 2) 异常音 (呼吸过程中产生的异常音，如 wheeze/crackle/rhonchi/stridor 等)默认分支
    ABNORMAL_SOUND_KIND = "abnormal_sound"
    OTHER_EVENT_KIND = "other_event"
    PHASE_KIND = "phase"

    def __init__(self, owner):
        self.owner = owner

    def clear_ml_annotations_for_label(self, label):
        """
        删除所有标签为 label 且 source != 'manual' 的“机器标注”：
        - 从可视化中移除对应的 BoxSpan 和 STFT 高光
        - 把 self.annotations 里对应条目标记为 None (由 delete_annotation 负责)
        """
        self = self.owner
        if  hasattr(self, "ml"):
            return

        # 先找出需要删除的 span (用 _span2idx 映射回 annotations)
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

            # 兼容 3/4 元组 (3 元组一律视为人工标注)
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
            if src == "ml" and str(t) == str(label):
                spans_to_delete.append(span)

        # 统一调用已有的删除逻辑
        for sp in spans_to_delete:
            try:
                self.delete_annotation(sp, record_negative=False, push_undo=False)
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

            if src == "ml" and str(t) == str(label):
                self.annotations[i] = None

    # --- Per-label pipeline routing (stepwise integration) ---
    def _label_kind(self, label):
        """
        将标签路由到三类管线：
        - phase: 呼吸时相 (Inspiration/Expiration/Pause)，使用相位模型 + HSMM 后处理
        - abnormal_sound: 呼吸过程中产生的异常音 (默认分支)
        - other_event: 说话、咳嗽等其他非呼吸过程异常事件

        说明：当前版本 abnormal_sound 与 other_event 仍共用同一套二分类训练/后处理，
        但已通过 dispatcher 分开，便于后续替换为不同模型与后处理。
        """
        lab = str(label).strip().lower()
        if lab in getattr(MLService, "PHASE_LABELS", {"inspiration", "expiration"}):
            return MLService.PHASE_KIND
        if lab in getattr(MLService, "OTHER_EVENT_LABELS", set()):
            return MLService.OTHER_EVENT_KIND
        # 默认：异常音
        return MLService.ABNORMAL_SOUND_KIND

    def train_model_for_label(self, label, min_pos_frames=30, neg_pos_ratio=5, random_state=None):
        # Dispatcher entrypoint used by UI
        kind = self._label_kind(label)
        if kind == MLService.PHASE_KIND:
            return self.train_phase_model_for_label(
                label,
                min_pos_frames=min_pos_frames,
                neg_pos_ratio=neg_pos_ratio,
                random_state=random_state,
            )
        if kind == MLService.OTHER_EVENT_KIND:
            return self.train_other_event_model_for_label(
                label,
                min_pos_frames=min_pos_frames,
                neg_pos_ratio=neg_pos_ratio,
                random_state=random_state,
            )
        return self.train_abnormal_sound_model_for_label(
            label,
            min_pos_frames=min_pos_frames,
            neg_pos_ratio=neg_pos_ratio,
            random_state=random_state,
        )

    def train_abnormal_sound_model_for_label(self, label, min_pos_frames=30, neg_pos_ratio=5, random_state=None):
        """
        训练“异常音(呼吸过程中产生)”标签的模型。
        当前版本与 other_event 共用同一套二分类训练逻辑，仅通过 model_kind 区分，便于后续替换。
        """
        return self.train_event_model_for_label(
            label,
            min_pos_frames=min_pos_frames,
            neg_pos_ratio=neg_pos_ratio,
            random_state=random_state,
            model_kind=MLService.ABNORMAL_SOUND_KIND,
        )


    def train_other_event_model_for_label(self, label, min_pos_frames=30, neg_pos_ratio=5, random_state=None):
        """
        训练“其他异常事件(说话/咳嗽等)”标签的模型。
        当前版本与 abnormal_sound 共用同一套二分类训练逻辑，仅通过 model_kind 区分，便于后续替换。
        """
        return self.train_event_model_for_label(
            label,
            min_pos_frames=min_pos_frames,
            neg_pos_ratio=neg_pos_ratio,
            random_state=random_state,
            model_kind=MLService.OTHER_EVENT_KIND,
        )



    def train_phase_model_for_label(self, label, min_pos_frames=30, neg_pos_ratio=5, random_state=None):
        """训练呼吸相位 (Inspiration/Expiration)共享模型，并基于人工标注估计 HSMM 先验。

        规则：
        - 若训练前缀内同时存在 Inspiration 与 Expiration 人工标注：三态 (Inspiration/Expiration/Pause)，空白视为 Pause。
        - 若仅存在一种：Two-state (Inspiration/Expiration)，空白视为另一种 (例如仅有 Inspiration 时，空白视为 Expiration)。

        说明：发射概率 (emission)直接来自分类器的 predict_proba；HSMM 主要提供持续Time与转移约束的后处理。
        """
        ml = self
        viewer = self.owner

        # 1) 准备特征与Time轴
        viewer.ensure_frame_features()
        X_all = getattr(viewer, "stft_features", None)
        times = getattr(viewer, "stft_frame_times", None)
        if X_all is None or times is None or len(times) == 0:
            QMessageBox.information(viewer, "Machine Learning", "No available short-time features; cannot train the phase model.")
            return False

        times = np.asarray(times, dtype=float)

        # 2) 仅使用已审阅前缀 (人工标注覆盖到的最大 end)
        T_used = viewer.get_reviewed_prefix()
        if T_used is None or T_used <= 0:
            QMessageBox.information(viewer, "Machine Learning", "No manual annotations yet; cannot train the phase model.")
            return False

        idx_prefix = np.where(times <= float(T_used))[0]
        if idx_prefix.size == 0:
            QMessageBox.information(viewer, "Machine Learning", "No available frames in the reviewed prefix; cannot train the phase model.")
            return False

        # 3) 收集人工相位标注
        LAB_I = "Inspiration"
        LAB_E = "Expiration"
        seg_I = viewer.get_manual_segments_for_label(LAB_I)
        seg_E = viewer.get_manual_segments_for_label(LAB_E)

        has_I = len(seg_I) > 0
        has_E = len(seg_E) > 0
        if not has_I and not has_E:
            QMessageBox.information(viewer, "Machine Learning", "No manual Inspiration/Expiration annotations in the reviewed prefix; cannot train the phase model.")
            return False

        # 4) 构建多类帧标签 (仅在前缀内有效)
        #    state_id 约定：0=Inspiration, 1=Expiration, 2=Pause (若启用三态)
        y_state = np.full(len(times), -1, dtype=np.int16)

        # 标 Inspiration/Expiration
        for (s, e) in seg_I:
            if e <= s:
                continue
            idx = np.where((times >= float(s)) & (times <= float(e)))[0]
            y_state[idx] = 0
        for (s, e) in seg_E:
            if e <= s:
                continue
            idx = np.where((times >= float(s)) & (times <= float(e)))[0]
            y_state[idx] = 1

        # 空白区域处理
        prefix_mask = np.zeros(len(times), dtype=bool)
        prefix_mask[idx_prefix] = True
        idx_blank = np.where(prefix_mask & (y_state < 0))[0]

        if has_I and has_E:
            # 三态：空白=Pause
            y_state[idx_blank] = 2
            state_id_to_name = {0: LAB_I, 1: LAB_E, 2: "Pause"}
        else:
            # Two-state：空白视为另一相位
            if has_I and not has_E:
                y_state[idx_blank] = 1
            elif has_E and not has_I:
                y_state[idx_blank] = 0
            state_id_to_name = {0: LAB_I, 1: LAB_E}

        y_prefix = y_state[idx_prefix]
        uniq = np.unique(y_prefix)
        if uniq.size < 2:
            QMessageBox.information(viewer, "Machine Learning", "The phase training data has too few valid classes (at least two are required). Please add the other phase or leave blank intervals.")
            return False

        # 5) 训练 LightGBM 分类器 (StandardScaler + 可选互信息 TopK)
        FS_ENABLE = True
        FS_KBEST = 25
        X_prefix = X_all[idx_prefix, :]

        D_all = int(X_prefix.shape[1])
        use_fs = bool(FS_ENABLE and D_all > 6)
        k_best = int(min(FS_KBEST, max(2, D_all)))

        steps = [("scaler", StandardScaler())]
        if use_fs:
            steps.append(("select", SelectKBest(score_func=mutual_info_classif, k=k_best)))

        if LGBMClassifier is None:
            QMessageBox.warning(viewer, "Machine Learning",
                                "lightgbm is not installed; cannot train a LightGBM model. Please run: pip install lightgbm")
            return False

        classes_uniq = np.unique(y_prefix)
        n_classes = int(len(classes_uniq))
        counts = {int(c): int(np.sum(y_prefix == c)) for c in classes_uniq}

        if n_classes <= 2:
            n0 = counts.get(0, 0)
            n1 = counts.get(1, 0)
            scale_pos_weight = float(n0) / float(max(1, n1))

            clf = LGBMClassifier(
                objective="binary",
                n_estimators=400,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                reg_lambda=1.0,
                scale_pos_weight=scale_pos_weight,
                random_state=int(random_state) if random_state is not None else 0,
                n_jobs=1
            )
        else:
            total = float(len(y_prefix))
            class_weight = {
                int(c): float(total) / (float(n_classes) * float(max(1, counts.get(int(c), 1))))
                for c in classes_uniq
            }

            clf = LGBMClassifier(
                objective="multiclass",
                n_estimators=400,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                reg_lambda=1.0,
                class_weight=class_weight,
                random_state=int(random_state) if random_state is not None else 0,
                n_jobs=1
            )

        steps.append(("clf", clf))
        pipe = Pipeline(steps)
        try:
            pipe.fit(X_prefix, y_prefix)
        except Exception as e:
            QMessageBox.warning(viewer, "Machine Learning", f"Phase model training failed: {e}")
            return False

        # 6) 估计 HSMM 先验 (持续Time + 转移约束)
        hop_sec = ml._estimate_hop_sec(times=times, viewer=viewer)
        cycle_sec = ml._estimate_breath_cycle_sec(seg_I, seg_E)
        hsmm_prior = ml._build_hsmm_prior_from_prefix_labels(
            y_prefix=y_prefix,
            classes_=pipe.named_steps["clf"].classes_,
            state_id_to_name=state_id_to_name,
            hop_sec=hop_sec,
            cycle_sec=cycle_sec,
        )

        # 6.1) 注入 HSMM 初始分布 π (方案A：使用前缀末尾概率曲线热启动)
        #      在未审阅区间解码时，log_pi 不再均匀，而由前缀尾部(默认1s)的平均预测概率得到。
        try:
            PI_TAIL_SEC = 1.0
            PI_MIX_LAMBDA = 0.85  # 与均匀先验混合，避免过度自信/数值问题
            proba_prefix = pipe.predict_proba(X_prefix)  # (T_prefix, S) columns align to clf.classes_
            hop_eff = float(hop_sec) if (hop_sec and hop_sec > 1e-6) else 0.05
            n_tail = int(max(1, round(float(PI_TAIL_SEC) / hop_eff)))
            n_tail = int(min(n_tail, proba_prefix.shape[0]))
            pi_hat = np.mean(proba_prefix[-n_tail:, :], axis=0)
            S_pi = int(pi_hat.size)
            pi_uniform = np.full((S_pi,), 1.0 / float(max(1, S_pi)), dtype=float)
            pi_init = (1.0 - float(PI_MIX_LAMBDA)) * pi_uniform + float(PI_MIX_LAMBDA) * pi_hat
            pi_init = np.clip(pi_init, 1e-12, None)
            pi_init = pi_init / float(np.sum(pi_init) + 1e-12)
            hsmm_prior["pi_init"] = [float(x) for x in pi_init.tolist()]
            hsmm_prior["pi_init_classes"] = [int(c) for c in hsmm_prior.get("classes", [])]
            hsmm_prior["pi_init_tail_sec"] = float(PI_TAIL_SEC)
            hsmm_prior["pi_init_lambda"] = float(PI_MIX_LAMBDA)
        except Exception:
            # π 注入失败不应影响主流程 (回退到均匀初始化)
            pass


        # 7) Feature selection信息
        full_names = list(getattr(viewer, "stft_feature_names", []))
        selected_idx = list(range(D_all))
        if use_fs and "select" in pipe.named_steps:
            try:
                selected_idx = pipe.named_steps["select"].get_support(indices=True).tolist()
            except Exception:
                selected_idx = list(range(D_all))

        model_info = {
            "model_kind": "phase",
            "clf": pipe,
            "classes": [int(x) for x in pipe.named_steps["clf"].classes_.tolist()],
            "state_id_to_name": dict((int(k), str(v)) for k, v in state_id_to_name.items()),
            "hsmm_prior": hsmm_prior,
            "feature_names": list(full_names),
            "selected_feature_indices": [int(i) for i in selected_idx],
            "feature_select_method": "mutual_info_kbest" if use_fs else "none",
            "feature_select_k": int(k_best) if use_fs else int(D_all),
            "train_prefix_sec": float(T_used),
        }

        # 8) 共享挂载到 Inspiration / Expiration 两个入口 (便于Two-state情况下也能从另一按钮调用)
        if not hasattr(viewer, "ml_models"):
            viewer.ml_models = {}
        viewer.ml_models[LAB_I] = model_info
        viewer.ml_models[LAB_E] = model_info

        # 9) 汇报
        counts = {int(c): int(np.sum(y_prefix == c)) for c in np.unique(y_prefix)}
        scheme = "Three-state (with Pause)" if ("Pause" in model_info["state_id_to_name"].values()) else "Two-state"
        msg = (
            f"Phase model trained ({scheme}):\n"
            f"Prefix length: {float(T_used):.2f}s, hop≈{hop_sec:.4f}s, estimated cycle≈{cycle_sec:.2f}s\n"
            f"Class frame counts: " + ", ".join([f"{model_info['state_id_to_name'][k]}={v}" for k, v in sorted(counts.items())]) + "\n"
            f"Feature selection: {'MI-TopK' if use_fs else 'None'} (kept {len(selected_idx)}/{D_all})\n"
            f"HSMM duration (frames): " + ", ".join([f"{model_info['state_id_to_name'][sid]}[{hsmm_prior['dmin_frames'][i]}..{hsmm_prior['dmax_frames'][i]}]" for i, sid in enumerate(hsmm_prior['classes'])])
        )
        QMessageBox.information(viewer, "Machine Learning", msg)
        return True


    def apply_model_for_label_on_unreviewed(self, label, min_dur_sec=0.05):
        # Dispatcher entrypoint used by UI
        kind = self._label_kind(label)
        if kind == MLService.PHASE_KIND:
            return self.apply_phase_model_for_label_on_unreviewed(label, min_dur_sec=min_dur_sec)
        if kind == MLService.OTHER_EVENT_KIND:
            return self.apply_other_event_model_for_label_on_unreviewed(label, min_dur_sec=min_dur_sec)
        return self.apply_abnormal_sound_model_for_label_on_unreviewed(label, min_dur_sec=min_dur_sec)

    def apply_abnormal_sound_model_for_label_on_unreviewed(self, label, min_dur_sec=0.05):
        """
        在未审阅区域对“异常音(呼吸过程中产生)”标签Auto Annotation。
        当前版本与 other_event 共用同一套二分类后处理，仅通过 model_kind 区分，便于后续替换。
        """
        return self.apply_event_model_for_label_on_unreviewed(
            label,
            min_dur_sec=min_dur_sec,
            expected_model_kinds={MLService.ABNORMAL_SOUND_KIND, "event"},
        )


    def apply_other_event_model_for_label_on_unreviewed(self, label, min_dur_sec=0.05):
        """
        在未审阅区域对“其他异常事件(说话/咳嗽等)”标签Auto Annotation。
        当前版本与 abnormal_sound 共用同一套二分类后处理，仅通过 model_kind 区分，便于后续替换。
        """
        return self.apply_event_model_for_label_on_unreviewed(
            label,
            min_dur_sec=min_dur_sec,
            expected_model_kinds={MLService.OTHER_EVENT_KIND, "event"},
        )



    def apply_phase_model_for_label_on_unreviewed(self, label, min_dur_sec=0.05):
        """对呼吸相位标签做 HSMM 后处理，并只输出当前 label 对应的区间 (source='ml')。"""
        ml = self
        viewer = self.owner

        LAB_I = "Inspiration"
        LAB_E = "Expiration"
        lab = str(label).strip().lower()
        if lab == "inspiration":
            target_label = LAB_I
        elif lab == "expiration":
            target_label = LAB_E
        else:
            target_label = str(label)

        # 1) 检查模型
        if (not hasattr(viewer, "ml_models")) or (LAB_I not in viewer.ml_models):
            QMessageBox.information(viewer, "Auto Annotation", "The phase model has not been trained. Please train the Inspiration/Expiration phase model first.")
            return False

        model_info = viewer.ml_models.get(target_label, viewer.ml_models.get(LAB_I, None))
        if not model_info or model_info.get("model_kind") != "phase":
            QMessageBox.information(viewer, "Auto Annotation", "The phase model is missing or has a mismatched type. Please retrain the phase model.")
            return False

        viewer.clear_ml_annotations_for_label(target_label)

        # 2) 准备特征与Time
        viewer.ensure_frame_features()
        times = getattr(viewer, "stft_frame_times", None)
        X = getattr(viewer, "stft_features", None)
        if times is None or X is None or len(times) == 0:
            QMessageBox.information(viewer, "Auto Annotation", "No available short-time features; cannot auto-label.")
            return False

        times = np.asarray(times, dtype=float)

        # 3) 未审阅区域
        T_used = viewer.get_reviewed_prefix()
        if T_used is None or T_used <= 0:
            QMessageBox.information(viewer, "Auto Annotation", "No manual annotations yet; cannot determine the unreviewed region.")
            return False

        idx_unr = np.where(times > float(T_used))[0]
        if idx_unr.size == 0:
            QMessageBox.information(viewer, "Auto Annotation", "The current record has already been fully reviewed; there are no unreviewed frames.")
            return False

        X_unr = X[idx_unr, :]
        clf = model_info["clf"]
        try:
            proba = clf.predict_proba(X_unr)  # (T, S)
        except Exception as e:
            QMessageBox.warning(viewer, "Auto Annotation", f"Phase model prediction failed: {e}")
            return False

        classes = [int(c) for c in model_info.get("classes", [])]
        if not classes:
            # 兜底：从 sklearn 里取
            try:
                classes = [int(c) for c in clf.named_steps["clf"].classes_.tolist()]
            except Exception:
                classes = list(range(proba.shape[1]))

        # proba 的列顺序 = clf.classes_；需要与 classes 对齐
        try:
            clf_classes = [int(c) for c in clf.named_steps["clf"].classes_.tolist()]
        except Exception:
            clf_classes = classes

        if len(clf_classes) != proba.shape[1]:
            QMessageBox.warning(viewer, "Auto Annotation", "The phase model has an abnormal class dimension. Retraining is recommended.")
            return False

        # 构建 log-emission
        eps = 1e-12
        log_emit = np.log(np.clip(proba, eps, 1.0))

        # HSMM 先验
        prior = model_info.get("hsmm_prior", {})
        dmin = prior.get("dmin_frames", None)
        dmax = prior.get("dmax_frames", None)
        if dmin is None or dmax is None:
            QMessageBox.warning(viewer, "Auto Annotation", "The phase model lacks HSMM priors (duration). Please retrain the phase model.")
            return False

        # 转移矩阵 (log)
        state_id_to_name = model_info.get("state_id_to_name", {})
        state_names = [state_id_to_name.get(int(cid), str(cid)) for cid in clf_classes]
        log_trans = ml._build_hsmm_log_trans(state_names)
        # 初始分布 π：优先使用训练阶段从前缀末尾概率曲线估计的 pi_init；否则回退到均匀。
        log_pi = np.full((len(state_names),), -np.log(len(state_names) + 1e-12), dtype=float)
        try:
            pi_init = prior.get("pi_init", None)
            pi_classes = prior.get("pi_init_classes", prior.get("classes", None))
            if (pi_init is not None) and (pi_classes is not None) and (len(pi_init) == len(pi_classes)) and len(pi_init) > 0:
                cls2pi = {int(c): float(p) for c, p in zip(pi_classes, pi_init)}
                S0 = float(len(state_names))
                pi_vec = np.array([cls2pi.get(int(cid), 1.0 / max(1.0, S0)) for cid in clf_classes], dtype=float)
                pi_vec = np.clip(pi_vec, 1e-12, None)
                pi_vec = pi_vec / float(np.sum(pi_vec) + 1e-12)
                log_pi = np.log(pi_vec)
        except Exception:
            pass

        # 4) HSMM 解码
        try:
            z_hat = ml._hsmm_viterbi(log_emit, np.asarray(dmin, dtype=int), np.asarray(dmax, dtype=int), log_trans, log_pi)
        except Exception as e:
            QMessageBox.warning(viewer, "Auto Annotation", f"HSMM decoding failed: {e}")
            return False

        # 5) 生成 target_label 的连续区间
        # 找 target_label 对应的 state id
        name_to_stateid = {str(v): int(k) for k, v in state_id_to_name.items()}
        if target_label not in name_to_stateid:
            QMessageBox.information(viewer, "Auto Annotation", f"The current phase model does not contain state {target_label}; cannot output intervals for this label.")
            return False

        target_state_id = name_to_stateid[target_label]
        # z_hat 是按 clf_classes 的索引输出的状态 index (0..S-1)，需要映射回 state id
        # 先构建 index->state_id
        idx_to_state_id = [int(cid) for cid in clf_classes]
        z_state_ids = np.array([idx_to_state_id[int(k)] for k in z_hat], dtype=int)

        new_segments = ml._state_seq_to_segments(times, idx_unr, z_state_ids, target_state_id, float(min_dur_sec))
        if not new_segments:
            QMessageBox.information(viewer, "Auto Annotation", "No candidate phase intervals were detected in the unreviewed region.")
            return False

        # 6) 与已有人工标注去重 (同标签且高度重叠跳过)
        manual_segs = viewer.get_manual_segments_for_label(target_label)

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
            QMessageBox.information(viewer, "Auto Annotation", "Candidate intervals highly overlap with existing manual annotations; no machine annotations were added.")
            return False

        # 7) 写入 annotations
        if not hasattr(viewer, "annotations"):
            viewer.annotations = []

        for (s, e) in final_segments:
            viewer.finalize_annotation(s, e, text=target_label, source="ml")

        QMessageBox.information(viewer, "Auto Annotation", f"Phase '{target_label}' added {len(final_segments)} machine-annotation segments in the unreviewed region.")
        return True


    # ------------------- HSMM helpers (phase only) -------------------
    def _estimate_hop_sec(self, times, viewer=None):
        """估计帧 hop (秒)。优先使用 viewer.hop_length/sr；否则用Time轴差分。"""
        try:
            if viewer is not None and getattr(viewer, "sr", None) and getattr(viewer, "hop_length", None):
                sr = float(viewer.sr)
                hop = float(viewer.hop_length)
                if sr > 0 and hop > 0:
                    return hop / sr
        except Exception:
            pass
        try:
            dt = np.diff(np.asarray(times, dtype=float))
            dt = dt[np.isfinite(dt) & (dt > 0)]
            if dt.size:
                return float(np.median(dt))
        except Exception:
            pass
        return 0.05

    def _estimate_breath_cycle_sec(self, seg_I, seg_E):
        """用人工标注估计呼吸周期 (秒)：优先使用同一相位连续 start 的差分中位数。"""
        starts = []
        for segs in (seg_I, seg_E):
            ss = sorted([float(s) for (s, e) in segs if float(e) > float(s)])
            if len(ss) >= 2:
                dif = [ss[i+1] - ss[i] for i in range(len(ss)-1)]
                dif = [d for d in dif if d > 0.1]
                if dif:
                    starts.extend(dif)
        if starts:
            return float(np.median(starts))
        return 3.0

    def _build_hsmm_prior_from_prefix_labels(self, y_prefix, classes_, state_id_to_name, hop_sec, cycle_sec):
        """从前缀帧标签统计持续Time范围 (帧)。classes_ 为 sklearn 的 classes_ 顺序。"""
        classes = [int(c) for c in np.asarray(classes_).tolist()]
        y = np.asarray(y_prefix, dtype=int)

        # 统计每个状态的连续 run-length (帧)
        runs = {c: [] for c in classes}
        if y.size:
            cur = int(y[0])
            ln = 1
            for v in y[1:]:
                v = int(v)
                if v == cur:
                    ln += 1
                else:
                    if cur in runs:
                        runs[cur].append(int(ln))
                    cur = v
                    ln = 1
            if cur in runs:
                runs[cur].append(int(ln))

        # 默认范围 (秒)
        cycle = float(cycle_sec) if (cycle_sec and cycle_sec > 0) else 3.0
        def_sec = {}
        # Inspiration / Expiration：按周期给一个较宽的范围
        def_sec["Inspiration"] = (max(0.10, 0.08 * cycle), max(0.40, 1.20 * cycle))
        def_sec["Expiration"]  = (max(0.10, 0.08 * cycle), max(0.40, 1.20 * cycle))
        # Pause：允许更短，也允许更长 (但后面会有 cap)
        def_sec["Pause"] = (0.05, max(0.30, 0.80 * cycle))

        def sec_to_frames(sec):
            return max(1, int(round(float(sec) / max(hop_sec, 1e-6))))

        # 估计 dmin/dmax
        dmin = []
        dmax = []
        DMAX_CAP_SEC = 15.0  # 运行时上限，避免 DP 过慢
        dmax_cap = sec_to_frames(DMAX_CAP_SEC)

        for c in classes:
            name = str(state_id_to_name.get(int(c), str(c)))
            arr = np.asarray(runs.get(int(c), []), dtype=float)
            if arr.size >= 5:
                mn = int(max(1, round(np.percentile(arr, 5))))
                mx = int(max(mn, round(np.percentile(arr, 95))))
            elif arr.size >= 1:
                mn = int(max(1, np.min(arr)))
                mx = int(max(mn, np.max(arr) * 2))
            else:
                a, b = def_sec.get(name, (0.10, 1.20 * cycle))
                mn = sec_to_frames(a)
                mx = sec_to_frames(b)

            mn = int(max(1, mn))
            mx = int(min(max(mn, mx), dmax_cap))
            dmin.append(mn)
            dmax.append(mx)

        return {
            "classes": classes,
            "hop_sec": float(hop_sec),
            "cycle_sec": float(cycle),
            "dmin_frames": [int(x) for x in dmin],
            "dmax_frames": [int(x) for x in dmax],
        }

    def _build_hsmm_log_trans(self, state_names):
        """构造 HSMM 转移矩阵 (log)。持续Time由 HSMM 负责，转移只做结构约束。"""
        names = [str(x) for x in state_names]
        S = len(names)
        logA = np.full((S, S), -np.inf, dtype=float)

        def norm_row(i, js):
            if not js:
                return
            p = 1.0 / float(len(js))
            for j in js:
                logA[i, j] = np.log(p)

        if S == 2:
            # Two-state：允许互转，也允许自环 (用于超出 dmax 的长段拼接)
            for i in range(2):
                norm_row(i, [0, 1])
            return logA

        # 三态 (含 Pause)
        # 约束：Insp <-> Exp 可直接转，也可经 Pause；Pause 可自环。
        # 禁止 Insp->Insp 与 Exp->Exp (避免无意义拆段)，Pause 允许 Pause->Pause。
        # 这里用名称来判定索引
        idx_pause = None
        for i, n in enumerate(names):
            if n.lower() == "pause":
                idx_pause = i
                break

        # 兜底：如果找不到 Pause，退化为全连接
        if idx_pause is None:
            for i in range(S):
                norm_row(i, list(range(S)))
            return logA

        # 识别 Insp/Exp
        idx_insp = None
        idx_exp = None
        for i, n in enumerate(names):
            if n.lower() == "inspiration":
                idx_insp = i
            elif n.lower() == "expiration":
                idx_exp = i

        # 兜底
        if idx_insp is None or idx_exp is None:
            for i in range(S):
                norm_row(i, list(range(S)))
            return logA

        # Insp row
        norm_row(idx_insp, [idx_exp, idx_pause])
        # Exp row
        norm_row(idx_exp, [idx_insp, idx_pause])
        # Pause row
        norm_row(idx_pause, [idx_insp, idx_exp, idx_pause])
        return logA

    def _hsmm_viterbi(self, log_emit, dmin, dmax, log_trans, log_pi):
        """HSMM Viterbi (显式持续Time)，返回每帧的 state-index (0..S-1)。"""
        log_emit = np.asarray(log_emit, dtype=float)
        T, S = log_emit.shape
        dmin = np.asarray(dmin, dtype=int).reshape(-1)
        dmax = np.asarray(dmax, dtype=int).reshape(-1)
        log_trans = np.asarray(log_trans, dtype=float)
        log_pi = np.asarray(log_pi, dtype=float).reshape(-1)

        # cumulative sum for segment likelihood
        cum = np.zeros((T + 1, S), dtype=float)
        cum[1:, :] = np.cumsum(log_emit, axis=0)

        def seg_sum(t_end, t_start, s):
            return cum[t_end, s] - cum[t_start, s]

        neg_inf = -1e300
        dp = np.full((T + 1, S), neg_inf, dtype=float)
        bp_state = np.full((T + 1, S), -1, dtype=int)
        bp_dur = np.full((T + 1, S), 0, dtype=int)

        for t in range(1, T + 1):
            for s in range(S):
                best = neg_inf
                best_p = -1
                best_d = 0
                d_lo = int(max(1, dmin[s] if s < len(dmin) else 1))
                d_hi = int(min(dmax[s] if s < len(dmax) else t, t))
                if d_hi < d_lo:
                    d_lo, d_hi = 1, t

                # uniform duration in [d_lo, d_hi]
                log_dur = -np.log(float(d_hi - d_lo + 1))

                for d in range(d_lo, d_hi + 1):
                    start = t - d
                    seg_ll = seg_sum(t, start, s)

                    if start == 0:
                        score = float(log_pi[s]) + log_dur + seg_ll
                        prev = -1
                    else:
                        prev_scores = dp[start, :] + log_trans[:, s]
                        prev = int(np.argmax(prev_scores))
                        score = float(prev_scores[prev]) + log_dur + seg_ll

                    if score > best:
                        best = score
                        best_p = prev
                        best_d = d

                dp[t, s] = best
                bp_state[t, s] = best_p
                bp_dur[t, s] = best_d

        # backtrack
        z = np.zeros(T, dtype=int)
        s = int(np.argmax(dp[T, :]))
        t = T
        while t > 0:
            d = int(bp_dur[t, s])
            start = t - d
            z[start:t] = s
            s_prev = int(bp_state[t, s])
            t = start
            if t <= 0 or s_prev < 0:
                break
            s = s_prev

        return z

    def _state_seq_to_segments(self, times, idx_unr, z_state_ids, target_state_id, min_dur_sec):
        """把未审阅区间的状态序列转换成 (start,end) 段 (用 times 的首尾帧Time)。"""
        times = np.asarray(times, dtype=float)
        idx_unr = np.asarray(idx_unr, dtype=int)
        z = np.asarray(z_state_ids, dtype=int)
        if idx_unr.size == 0 or z.size == 0:
            return []
        assert idx_unr.size == z.size

        segs = []
        in_run = False
        start_i = 0
        for i, sid in enumerate(z):
            if sid == int(target_state_id) and not in_run:
                in_run = True
                start_i = i
            elif sid != int(target_state_id) and in_run:
                frame_idxs = idx_unr[start_i:i]
                s = float(times[frame_idxs[0]])
                e = float(times[frame_idxs[-1]])
                if e - s >= float(min_dur_sec):
                    segs.append((s, e))
                in_run = False

        if in_run:
            frame_idxs = idx_unr[start_i:len(z)]
            s = float(times[frame_idxs[0]])
            e = float(times[frame_idxs[-1]])
            if e - s >= float(min_dur_sec):
                segs.append((s, e))

        return segs


    def train_event_model_for_label(self,
                              label,
                              min_pos_frames=20,
                              neg_pos_ratio=5,
                              random_state=None,
                              model_kind="event"):

        """
        训练单标签帧级分类器 (并输出更完整的训练集报告)：
        - label: 目标标签 (如 'Wheeze')
        - min_pos_frames: 至少需要多少个正样本帧才开始训练
        - neg_pos_ratio: 负样本采样比例 (负样本数 ≈ ratio * 正样本数)

        增强项：
        - 训练时进行Feature selection (默认启用：互信息 SelectKBest)
        - 学习完成后除 P/R/F1 外，额外输出 Acc/Spec/BAcc/MCC/AUROC/AUPRC/Brier/Confusion
        """
        self = self.owner
        # ---------- 可调的Feature selection策略 (默认开启) ----------
        FS_ENABLE = True
        FS_KBEST = 20  # 互信息 Top-K (若特征维度不足会自动缩小)

        # 1) 准备特征 & 帧标签
        self.ensure_frame_features()
        if self.stft_features is None or self.stft_frame_times is None:
            QMessageBox.information(self, "Machine Learning",
                                    "No available short-time features; cannot train the model.")
            return False

        y = self.build_frame_labels_for_tag(label, neg_margin=0.05)
        if y is None:
            QMessageBox.information(self, "Machine Learning",
                                    f"Label {label}  has no available frames in the reviewed region; cannot train.")
            return False

        y = np.asarray(y, dtype=np.int8)
        idx_pos = np.where(y == 1)[0]
        idx_neg = np.where(y == 0)[0]

        n_pos = int(len(idx_pos))
        n_neg_all = int(len(idx_neg))

        if n_pos < int(min_pos_frames):
            QMessageBox.information(
                self, "Machine Learning",
                f"Label {label} has only {n_pos} positive frames, less than the minimum requirement of {min_pos_frames}; training is skipped.")
            return False

        if n_neg_all == 0:
            QMessageBox.information(
                self, "Machine Learning",
                f"Label {label} has no available safe negative frames; training is skipped.")
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

        # 3) Feature selection + 标准化 + LightGBM
        D_all = int(X.shape[1])
        use_fs = bool(FS_ENABLE and D_all > 4)
        k_best = int(min(FS_KBEST, max(2, D_all)))

        steps = [("scaler", StandardScaler())]
        if use_fs:
            steps.append(("select", SelectKBest(score_func=mutual_info_classif, k=k_best)))

        if LGBMClassifier is None:
            QMessageBox.warning(self, "Machine Learning",
                                "lightgbm is not installed; cannot train a LightGBM model. Please run: pip install lightgbm")
            return False

        n_pos_train = int(np.sum(y_train == 1))
        n_neg_train = int(np.sum(y_train == 0))
        scale_pos_weight = float(n_neg_train) / float(max(1, n_pos_train))

        clf = LGBMClassifier(
            objective="binary",
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            random_state=int(random_state) if random_state is not None else 0,
            n_jobs=1
        )
        steps.append(("clf", clf))
        pipe = Pipeline(steps)
        pipe.fit(X, y_train)

        # 4) 在训练集上自动选一个合适的概率阈值 (最大化 F1)
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

        # 5) 更多训练集指标 (便于科研报告与调参)
        tn, fp, fn, tp = confusion_matrix(y_train, y_pred_best, labels=[0, 1]).ravel()
        acc = accuracy_score(y_train, y_pred_best)
        bacc = balanced_accuracy_score(y_train, y_pred_best)
        mcc = matthews_corrcoef(y_train, y_pred_best)
        specificity = float(tn) / (float(tn + fp) + 1e-12)
        npv = float(tn) / (float(tn + fn) + 1e-12)
        auc_roc = roc_auc_score(y_train, proba)
        auc_pr = average_precision_score(y_train, proba)
        brier = brier_score_loss(y_train, proba)

        # 6) Feature selection结果 (用于输出与可解释性)
        full_names = list(self.stft_feature_names)
        selected_idx = list(range(D_all))
        if use_fs and "select" in pipe.named_steps:
            selected_idx = pipe.named_steps["select"].get_support(indices=True).tolist()
        selected_names = [full_names[i] for i in selected_idx]

        # LightGBM 特征重要性 (对应筛选后的特征顺序)
        top_k_show = int(min(8, len(selected_names)))
        top_feats = []
        importance_type = "gain"
        top_feat_str = ""

        try:
            booster = pipe.named_steps["clf"].booster_
            imp = booster.feature_importance(importance_type="gain")
            order = np.argsort(imp)[::-1][:top_k_show]
            top_feats = [(selected_names[i], float(imp[i])) for i in order]
            top_feat_str = "\n".join([f"  - {n}: {v:.4f}" for n, v in top_feats])
        except Exception:
            try:
                imp = pipe.named_steps["clf"].feature_importances_
                importance_type = "split"
                order = np.argsort(imp)[::-1][:top_k_show]
                top_feats = [(selected_names[i], float(imp[i])) for i in order]
                top_feat_str = "\n".join([f"  - {n}: {v:.4f}" for n, v in top_feats])
            except Exception:
                importance_type = "unknown"
                top_feat_str = "  (无法获取特征重要性)"
        # 7) 存到 self.ml_models，方便后续自动标注 / 可视化
        self.ml_models[label] = {
            "clf": pipe,
            "threshold": float(best_th),
            "model_kind": str(model_kind),
            "feature_names": list(self.stft_feature_names),  # 输入特征空间 (全量)
            "selected_feature_indices": [int(i) for i in selected_idx],
            "selected_feature_names": list(selected_names),
            "feature_select_method": "mutual_info_kbest" if use_fs else "none",
            "feature_select_k": int(k_best) if use_fs else int(D_all),
            "top_features_by_importance": list(top_feats),
            "feature_importance_type": str(importance_type),
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
             f"Positive frames: {n_pos}, sampled negative frames: {n_neg_target}\n"
             f"Threshold (best training F1) = {best_th:.2f}\n"
             f"P={precision:.3f}, R={recall:.3f}, F1={f1_final:.3f}, Acc={acc:.3f}, Spec={specificity:.3f}\n"
             f"BAcc={bacc:.3f}, MCC={mcc:.3f}, AUROC={auc_roc:.3f}, AUPRC={auc_pr:.3f}, Brier={brier:.4f}\n"
             f"Confusion: TP={tp}, FP={fp}, TN={tn}, FN={fn}\n"
             f"Feature selection: {'MI-TopK' if use_fs else 'None'} (kept {len(selected_names)}/{D_all})\n"
             f"Top features (importance-{importance_type}):\n{top_feat_str}")
        )

        return True


    def apply_event_model_for_label_on_unreviewed(self,
                                            label,
                                            min_dur_sec=0.05,
                                            expected_model_kinds=None):
        """
        使用已经训练好的模型，在“未审阅区域” (帧Time > reviewed_prefix)Auto Annotation指定标签。
        生成的标注统一以 (start, end, label, "ml") 形式写入 self.annotations。
        """
        self = self.owner
        # 1) 检查模型是否存在
        if not hasattr(self, "ml_models") or label not in self.ml_models:
            QMessageBox.information(
                self, "Auto Annotation",
                f"Label {label} has no trained model yet. Please train this label model first."
            )
            return False
        self.clear_ml_annotations_for_label(label)  # 清除上次Machine Learning的标记

        model_info = self.ml_models[label]
        # 类型检查：为后续“异常音/其他事件”分离模型与后处理做准备
        if expected_model_kinds is not None:
            try:
                mk = str(model_info.get("model_kind", "event"))
            except Exception:
                mk = "event"
            if mk not in set(expected_model_kinds):
                QMessageBox.information(
                    self, "Auto Annotation",
                    f"Label {label} has a mismatched model type (current: {mk}). Please retrain the model for this label type."
                )
                return False
        clf = model_info["clf"]
        th = float(model_info.get("threshold", 0.5))

        # 2) 准备帧级特征与Time轴
        self.ensure_frame_features()
        times = getattr(self, "stft_frame_times", None)
        X = getattr(self, "stft_features", None)
        if times is None or X is None or len(times) == 0:
            QMessageBox.information(
                self, "Auto Annotation",
                "No available short-time features; cannot auto-label."
            )
            return False

        times = np.asarray(times, dtype=float)

        # 3) 求“已审阅前缀”，只在未审阅区域应用模型
        T_used = self.get_reviewed_prefix()
        if T_used is None or T_used <= 0:
            QMessageBox.information(
                self, "Auto Annotation",
                "No manual annotations yet; cannot determine the unreviewed region."
            )
            return False

        idx_unr = np.where(times > T_used)[0]
        if idx_unr.size == 0:
            QMessageBox.information(
                self, "Auto Annotation",
                "The current record has already been fully reviewed; there are no unreviewed frames."
            )
            return False

        # 4) 在未审阅帧上跑模型，得到帧级 0/1
        X_unr = X[idx_unr, :]
        try:
            proba = clf.predict_proba(X_unr)[:, 1]
        except Exception as e:
            QMessageBox.warning(
                self, "Auto Annotation",
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
                self, "Auto Annotation",
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
                self, "Auto Annotation",
                "Candidate events highly overlap with existing manual annotations; no machine annotations were added."
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
            self, "Auto Annotation",
            f"Label '{label}' added {len(final_segments)} machine-annotation segments in the unreviewed region."
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
            self, "Auto Annotation",
            f"Label '{label}' added {len(final_segments)} machine-annotation segments in the unreviewed region."
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

        # —— Machine Learning硬负样本 (仅用于训练；不导出；与标注可视无关)——
        # 形如：{label: [(start, end, neg_id), ...]}
        self.neg_segments = defaultdict(list)
        self._neg_id_counter = 0

        # —— 最小撤销栈 (用于“删除/认可”等编辑操作的撤销)——
        self._undo_stack = []
        self._undo_maxlen = 100

        # —— 新增：最多 3 行的分轨管理 (只用于标注视窗)
        self.MAX_LANES = 3
        self.LANE_H = 0.35  # 单条高度 (越小越细)
        self.LANE_GAP = 0.25  # 轨道间距
        self._lanes = [[] for _ in range(self.MAX_LANES)]  # 每行的[(s,e), ...]
        self._spans = []  # 当前所有 BoxSpan
        self._span2spec = {}  # BoxSpan -> 频谱 LinearRegionItem
        self._span2idx = {}  # BoxSpan -> annotations 索引

        # —— 编辑模式 (仅 BoxSpan 双击进入)——
        self._editing_span = None  # 当前处于编辑态的 BoxSpan

        # —— STFT Display (仅两种配色 + 开窗上下限)——
        self.stft_cmap = "Heatmap"  # 备选："Heatmap"、"Grayscale"
        self.stft_vmin = None  # None 表示自动：用 1%~99% 分位
        self.stft_vmax = None
        self._last_spec_vals = None  # 最近一次 STFT 的原始 2D 数值 (freq×time)，用于直方图统计/重着色

        # —— 特征颜色 (随便挑的不重复颜色，足够 5 items)——
        self.feature_palette = [
            QColor("#e41a1c"), QColor("#377eb8"), QColor("#4daf4a"),
            QColor("#984ea3"), QColor("#ff7f00"), QColor("#a65628"),
            QColor("#f781bf"), QColor("#999999"), QColor("#66c2a5"),
            QColor("#d95f02")
        ]
        self.feature_color_map = {}  # {特征名: QColor}，由所选特征顺序分配

        # —— 标注类型 & 颜色管理 (含预设哮鸣音/爆裂音等)——
        self.annotation_builtin_labels = [
            ("哮鸣音", "wheeze"),
            ("爆裂音", "Crackles"),
            ("摩擦音", "Pleural Rub"),
            ("哼鸣音", "Rhonchi"),
            ("喘息音", "Stridor"),
            ("语音", "Speech"),  # 语音
            ("咳嗽", "Cough"),  # 咳嗽
            ("呼气", "Expiration"),  # 呼气
            ("吸气", "Inspiration"),  # 吸气
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

        self.last_export_path = None  # 记录上次保存的 CSV 路径 (仅用于沿用上次导出目录)
        self.default_export_annotation_name = "annotations_events.csv"  # 当前 WAV 对应的默认导出File名

        self.showing_fft = False

        self.wave_y_range = None  # 格式为 (ymin, ymax)

        self.last_settings_tab_index = 0  # 默认打开第一个标签页 (STFT)

        # —— 读取音频时的预处理：默认启用 4000 Hz 重采样；滤波默认关闭，避免改变旧版音频内容。——
        self.preprocessing_enabled = True
        self.resample_on_load_enabled = True
        self.resample_target_sr = 4000
        self.filter_on_load_enabled = False
        self.filter_type = "bandpass"
        self.filter_lowcut = 20.0
        self.filter_highcut = 1800.0
        self.filter_order = 4
        self.filter_zero_phase = True
        self.audio_original_sr = None
        self.audio_preprocess_summary = ""


        # —— 自动导入同名 _events 标注File (可开关)——
        # 规则：<wav_base>_events.(csv|txt) 与 <wav_base>.wav 同目录
        self.auto_import_events_enabled = False
        self._events_index_cache = {}  # {folder(abs): {wav_base_lower: events_path}}
        self._events_parse_cache = {}  # {events_path(abs): (mtime, rows)}
        self.auto_label_import_settings = {
            "file_format": "auto",
            "file_suffix": "_events",
            "delimiter": "auto",
            "custom_delimiter": "",
            "skip_header_lines": 0,
            "start_col": 1,
            "end_col": 2,
            "label_col": 3,
            "source_col": 4,
            "json_start_key": "start",
            "json_end_key": "end",
            "json_label_key": "label",
            "json_source_key": "source",
        }
        self.show_y_axis = False  # 初始为Display

        # —— Short-Time Features：默认选择 (可改)
        self.selected_features = ["短时能量", "过零率", "谱质心"]

        # 曲线缓存 (特征名 -> pg.PlotDataItem)
        self.feature_curves = {}

        # —— 性能优化状态：切换文件时先显示波形/STFT，FFT 与短时特征改为懒加载。——
        self.features_dirty = True
        self.fft_dirty = True
        self._display_wave_max_points = 50000
        self._spec_display_max_time_bins = 2500
        self._spec_display_max_freq_bins = 512

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

        # ========= Machine Learning：Auto Annotation Speech 未审阅区域 (Ctrl+M) =========
        self.shortcut_auto_speech = QShortcut(QKeySequence("Ctrl+M"), self)
        self.shortcut_auto_speech.activated.connect(
            lambda: self.apply_model_for_label_on_unreviewed("Speech")
        )

        # ========= 撤销 (Ctrl+Z) =========
        self.shortcut_undo = QShortcut(QKeySequence("Ctrl+Z"), self)
        # 让 Ctrl+Z 在任意子控件获得焦点时也可用 (避免被输入框自身的 undo 吞掉)
        self.shortcut_undo.setContext(Qt.ApplicationShortcut)
        self.shortcut_undo.activated.connect(self.undo_last_action)

        # 同时支持系统标准 Undo 快捷键 (Windows/Linux=Ctrl+Z，macOS=Cmd+Z)
        self.shortcut_undo2 = QShortcut(QKeySequence.Undo, self)
        self.shortcut_undo2.setContext(Qt.ApplicationShortcut)
        self.shortcut_undo2.activated.connect(self.undo_last_action)

        # 如果存在其他控件/动作也绑定了 Ctrl+Z，Qt 可能判定为“快捷键歧义”，此时会触发 activatedAmbiguously 而不是 activated
        try:
            self.shortcut_undo.activatedAmbiguously.connect(self.undo_last_action)
            self.shortcut_undo2.activatedAmbiguously.connect(self.undo_last_action)
        except Exception:
            pass

        # 再增加一个 QAction 形式的 Undo (更贴近 Qt 标准动作系统，某些情况下比 QShortcut 更稳)
        try:
            self.action_undo = QAction("Undo", self)
            self.action_undo.setShortcut(QKeySequence.Undo)
            self.action_undo.setShortcutContext(Qt.ApplicationShortcut)
            self.action_undo.triggered.connect(self.undo_last_action)
            self.addAction(self.action_undo)
        except Exception:
            self.action_undo = None

        # 最后兜底：安装应用级事件过滤器，强制捕获 Ctrl+Z (避免被 GraphicsView/输入控件吞掉)
        try:
            app = QApplication.instance()
            if app is not None:
                app.installEventFilter(self)
        except Exception:
            pass

    def init_ui(self):
        self.spec_stft_plot = pg.PlotWidget(title="STFT Spectrogram")
        self.spec_fft_plot = pg.PlotWidget(title="FFT Spectrum")
        self.wave_plot = pg.PlotWidget(title="Waveform", viewBox=WaveViewBox(self))
        self.annot_plot = pg.PlotWidget(title="Annotation Area", viewBox=AnnotViewBox(self))
        self.spec_fft_plot.hide()
        self.spec_title_base = "STFT Spectrogram"
        self.spec_stft_plot.setTitle(self.spec_title_base)

        # 监听 STFT 图鼠标移动 (用 ViewBox 的矩形来判断是否在绘图区)
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
        self.freq_button = QPushButton("Switch Spectrum")
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

        # —— 特征页 (仅负责展示；计算逻辑在别处) ——
        self.feat_plot = pg.PlotWidget(title="Short-Time Features (Normalized Display)")
        self.feat_plot.setMouseEnabled(x=True, y=False)
        self.feat_plot.setLabel('bottom', 'Time', units='s')
        self.feat_plot.addLegend()
        # —— Short-Time Features PlotWidget 基础样式 ——
        self.feat_plot.setMouseEnabled(x=True, y=False)  # 只允许X轴缩放/拖动，禁用Y轴
        pi = self.feat_plot.getPlotItem()

        # 隐藏Y轴 (坐标线/刻度/标签都不Display)
        pi.hideAxis('left')
        try:
            pi.hideAxis('right')
        except Exception:
            pass

        # 关掉网格 (若你之前开过)
        pi.showGrid(x=False, y=False)

        # Time轴与 STFT 同步缩放/平移 (锁定)
        self.feat_plot.setXLink(self.spec_stft_plot)

        # 去掉多余边距，让隐藏轴不留白
        pi.layout.setContentsMargins(0, 0, 0, 0)
        self.feat_plot.setContentsMargins(0, 0, 0, 0)

        self.feat_page = QWidget()
        _feat_layout = QVBoxLayout(self.feat_page)
        _feat_layout.setContentsMargins(0, 0, 0, 0)
        _feat_layout.addWidget(self.feat_plot)

        # —— 页栈：STFT / FFT / Short-Time Features ——
        self.spec_stack = QStackedWidget()
        self.spec_stack.addWidget(self.spec_stft_plot)  # 第 0 页：STFT
        self.spec_stack.addWidget(self.spec_fft_plot)  # 第 1 页：FFT
        self.spec_stack.addWidget(self.feat_page)  # 第 2 页：Short-Time Features
        self.spec_stack.setCurrentWidget(self.spec_stft_plot)  # 默认Display STFT

        # —— 垂直分割器：上(页栈) | 中(波形) | 下(标注) ——
        self.main_splitter = QSplitter(Qt.Vertical)
        self.main_splitter.addWidget(self.spec_stack)  # 上：页栈 (STFT/FFT/特征)
        self.main_splitter.addWidget(self.wave_plot)  # 中：波形
        self.main_splitter.addWidget(self.annot_plot)  # 下：标注

        # 推荐的拉伸比例 (可按实际微调)
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
        """初始化Machine Learning相关工具栏：标签选择 + 训练 + Auto Annotation"""
        from PyQt5.QtWidgets import QToolBar

        toolbar = QToolBar("Machine Learning", self)
        self.addToolBar(Qt.TopToolBarArea, toolbar)

        # ===== 下拉框：选择要训练/Auto Annotation的Label =====
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
        action_legend.setToolTip("View the mapping between annotation colors and labels in the third panel")
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
            QMessageBox.information(self, "Auto Annotation", "Please select a label in the toolbar first.")
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

        hint = QLabel("Annotation Legend")
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

    def _build_preprocessing_config(self):
        """Build a preprocessing config dict from GUI attributes.

        Bridges naming differences: the GUI stores ``resample_on_load_enabled``
        and ``filter_on_load_enabled``, while the preprocessing module expects
        ``resample_enabled`` and ``filter_enabled``.
        """
        return {
            "preprocessing_enabled": bool(getattr(self, "preprocessing_enabled", True)),
            "resample_enabled": bool(getattr(self, "resample_on_load_enabled", False)),
            "resample_target_sr": int(getattr(self, "resample_target_sr", 4000)),
            "filter_enabled": bool(getattr(self, "filter_on_load_enabled", False)),
            "filter_type": str(getattr(self, "filter_type", "bandpass") or "bandpass"),
            "filter_lowcut": float(getattr(self, "filter_lowcut", 20.0) or 20.0),
            "filter_highcut": float(getattr(self, "filter_highcut", 1800.0) or 1800.0),
            "filter_order": int(getattr(self, "filter_order", 4) or 4),
            "filter_zero_phase": bool(getattr(self, "filter_zero_phase", True)),
        }

    def _get_load_audio_target_sr(self):
        """Return target sample rate for audio loading, or None to preserve original."""
        from respanno.audio.preprocessing import compute_target_sr

        cfg = self._build_preprocessing_config()
        return compute_target_sr(cfg)

    def _apply_butter_filter_for_preprocessing(self, audio, sr):
        """Apply optional Butterworth filtering, delegating to preprocessing module.

        GUI-level wrapper: handles enable-check and status-bar error reporting;
        the core DSP is in respanno.audio.preprocessing.apply_butter_filter.
        """
        if audio is None or sr is None:
            return audio

        cfg = self._build_preprocessing_config()
        if not cfg["preprocessing_enabled"] or not cfg["filter_enabled"]:
            return audio

        from respanno.audio.preprocessing import apply_butter_filter

        try:
            return apply_butter_filter(
                audio,
                sr=sr,
                filter_type=cfg["filter_type"],
                lowcut=cfg["filter_lowcut"],
                highcut=cfg["filter_highcut"],
                order=cfg["filter_order"],
                zero_phase=cfg["filter_zero_phase"],
            )
        except Exception as e:
            try:
                self.statusBar().showMessage(
                    f"Filter skipped: {e}", 3000,
                )
            except Exception:
                pass
            return audio

    def _summarize_preprocessing(self):
        """Human-readable preprocessing summary, delegating to preprocessing module."""
        from respanno.audio.preprocessing import summarize_preprocessing

        cfg = self._build_preprocessing_config()
        return summarize_preprocessing(cfg)

    def load_audio(self, path=None):
        """Load one WAV file.

        重构原则：
        1) 以 4000 Hz 重采样为默认分析基础；
        2) 切换上一首/下一首时只做必要显示：读音频、画抽稀波形、画 STFT、导入标签；
        3) FFT 和 Short-Time Features 不在切换时预计算，改为进入对应页面时懒加载。
        """
        # 0. 切歌前停止播放，避免旧音频仍在声卡播放。
        try:
            sd.stop()
            self.timer.stop()
            self.is_playing = False
            if hasattr(self, "btn_play"):
                self.btn_play.setText("Play")
        except Exception:
            pass

        # 1. 手动导入路径为空时，弹出File选择框
        if not path:
            selected, _ = QFileDialog.getOpenFileName(
                self, "Select WAV File", "", "WAV Files (*.wav)"
            )
            if not selected or not isinstance(selected, str):
                return
            path = selected

        # 2. 建立同目录 WAV 列表，用于上一首/下一首。
        try:
            path = os.path.abspath(path)
            folder = os.path.dirname(path)
            all_files = sorted([
                os.path.abspath(os.path.join(folder, f))
                for f in os.listdir(folder)
                if f.lower().endswith('.wav')
            ])
            if path not in all_files:
                raise FileNotFoundError(
                    "The selected file was not found in the folder.\n\n"
                    + path + "\n\nCandidate files:\n" + "\n".join(all_files)
                )
            self.audio_files = all_files
            self.current_file_index = self.audio_files.index(path)
        except Exception as e:
            QMessageBox.critical(self, "Folder Error", f"Failed to get audio files from the same folder:\n{str(e)}")
            return

        # 3. 读取音频。默认使用 Settings 中的重采样采样率；关闭重采样时保留原始采样率。
        try:
            self.statusBar().showMessage("Loading audio...", 0)
            QApplication.processEvents()

            self.loaded_filename = path
            self._update_default_export_annotation_name(path)

            self.audio_original_sr = None
            try:
                from respanno.audio.preprocessing import get_original_sr

                self.audio_original_sr = int(get_original_sr(path))
            except Exception:
                self.audio_original_sr = None

            target_sr = self._get_load_audio_target_sr()
            # librosa.load(..., sr=target_sr) uses band-limited resampling internally,
            # so anti-aliasing is handled by librosa when downsampling.
            from respanno.audio.preprocessing import load_audio_file

            self.audio, self.sr = load_audio_file(path, target_sr=target_sr)

            # Optional analysis/display band filtering after loading/resampling.
            self.audio = self._apply_butter_filter_for_preprocessing(self.audio, self.sr)
            self.audio_preprocess_summary = self._summarize_preprocessing()
            self.duration = len(self.audio) / float(self.sr)
        except Exception as e:
            QMessageBox.critical(self, "Audio Loading Failed", f"Failed to read audio:\n{str(e)}")
            self.statusBar().clearMessage()
            return

        # 4. 切换文件时，先清空旧缓存和旧标注，避免旧图/旧特征误用。
        self.stft_frame_times = None
        self.stft_features = None
        self.stft_feature_names = None
        self.features_dirty = True
        self.fft_dirty = True
        self.feature_curves = {}
        self._last_spec_vals = None
        # 新音频默认重置自动色阶，避免上一条文件的色阶影响当前文件。
        self.stft_vmin = None
        self.stft_vmax = None

        title_suffix = ""
        try:
            summary = str(getattr(self, "audio_preprocess_summary", "") or "")
            if summary:
                title_suffix = f" ({summary})"
            elif self.audio_original_sr and self.audio_original_sr != self.sr:
                title_suffix = f" (resampled {self.audio_original_sr}→{self.sr} Hz)"
        except Exception:
            title_suffix = ""
        self.setWindowTitle(f"Audio Time-Frequency Analyzer — {os.path.basename(path)}{title_suffix}")

        self.slider.setMaximum(int(self.duration * 1000))
        self.slider.setValue(0)
        self.time_label.setText("0.000 s")
        self.time_line_spec.setPos(0)
        self.time_line_wave.setPos(0)

        self.wave_y_range = (float(np.min(self.audio)), float(np.max(self.audio)))

        # 先清旧标注，再重绘新图，防止旧 STFT 高亮残留。
        self.clear_annotations()

        # 5. 只做必要显示：抽稀波形 + STFT。FFT/特征延后到用户切换页面时计算。
        try:
            self.statusBar().showMessage("Drawing waveform...", 0)
            QApplication.processEvents()
            self.draw_waveform()
        except Exception:
            pass

        try:
            self.statusBar().showMessage("Computing STFT display...", 0)
            QApplication.processEvents()
            self.update_spectrogram()
        except Exception as e:
            QMessageBox.warning(self, "Spectrogram Warning", f"Failed to update spectrogram:\n{str(e)}")

        # 6. 自动导入标签。标签导入很轻，仍保留在 load_audio 里。
        try:
            if getattr(self, "auto_import_events_enabled", False):
                self._auto_import_events_for_wav(path)
        except Exception:
            pass

        # 7. 坐标范围与页面状态。
        try:
            fmax_eff = min(float(getattr(self, "f_max", 2000)), float(self.sr) / 2.0)
        except Exception:
            fmax_eff = float(getattr(self, "f_max", 2000))
        self.spec_stft_plot.setLimits(xMin=0, xMax=self.duration, yMin=0, yMax=fmax_eff)
        self.wave_plot.setLimits(xMin=0, xMax=self.duration)
        self.annot_plot.setLimits(xMin=0, xMax=self.duration)
        self.annot_plot.setXRange(0, self.duration, padding=0)
        self.wave_plot.setXRange(0, self.duration, padding=0)
        self.spec_stft_plot.setXRange(0, self.duration, padding=0)
        if hasattr(self, "spec_stack"):
            self.spec_stack.setCurrentWidget(self.spec_stft_plot)
        if hasattr(self, "freq_button"):
            self.freq_button.setText("Switch: FFT")

        self.statusBar().showMessage("Audio loaded. FFT and short-time features will be computed on demand.", 2500)

    def _decimate_spec_for_display(self, spec):
        """Downsample (freq × time) spectrogram for display, delegating to module."""
        from respanno.dsp.spectrogram import decimate_spec_for_display

        return decimate_spec_for_display(
            spec,
            max_time_bins=int(getattr(self, "_spec_display_max_time_bins", 2500)),
            max_freq_bins=int(getattr(self, "_spec_display_max_freq_bins", 512)),
        )

    def update_spectrogram(self):
        if self.audio is None:
            return

        from respanno.dsp.spectrogram import compute_stft_db

        spec, freqs = compute_stft_db(
            self.audio, self.sr,
            n_fft=self.n_fft, hop_length=self.hop_length, f_max=self.f_max,
        )
        f_max = float(freqs[-1]) if len(freqs) else float(min(self.f_max, self.sr / 2.0))

        # 保存完整谱值用于 Settings 直方图；显示用谱图可以抽稀。
        self._last_spec_vals = spec.copy()
        spec_disp = self._decimate_spec_for_display(spec)

        disp = spec_disp.T  # time × freq
        rgb = self._colorize_spec_with_window(disp)

        self.spec_img.resetTransform()
        self.spec_img.setImage(rgb, autoLevels=False)
        # 抽稀后的图像仍映射到完整音频时长，保证标注时间轴一致。
        self.spec_img.setRect(pg.QtCore.QRectF(0, 0, float(self.duration), float(f_max)))

        self.spec_stft_plot.setLabel('bottom', 'Time', units='s')
        self.spec_stft_plot.setLimits(xMin=0, xMax=self.duration, yMin=0, yMax=f_max)
        self.spec_stft_plot.setXRange(0, self.duration, padding=0)

        self.wave_plot.setLimits(xMin=0, xMax=self.duration)
        self.annot_plot.setLimits(xMin=0, xMax=self.duration)
        self.wave_plot.setXRange(0, self.duration, padding=0)
        self.annot_plot.setXRange(0, self.duration, padding=0)

        # STFT/FMAX 变化会影响频域特征，但不在 STFT 刷新时立即计算特征。
        self.features_dirty = True
        self.fft_dirty = True

    def _get_display_waveform(self, max_points=None):
        """Return downsampled waveform for display only.

        The full self.audio is still used for playback, STFT, ML, and export.
        """
        if self.audio is None or self.sr is None:
            return np.array([]), np.array([])
        x = np.asarray(self.audio)
        n = int(len(x))
        if n <= 0:
            return np.array([]), np.array([])
        if max_points is None:
            max_points = int(getattr(self, "_display_wave_max_points", 50000))
        max_points = max(1000, int(max_points))
        if n <= max_points:
            idx = np.arange(n)
        else:
            idx = np.linspace(0, n - 1, max_points).astype(int)
        return idx.astype(float) / float(self.sr), x[idx]

    def draw_waveform(self):
        times, y = self._get_display_waveform()
        try:
            self.wave_plot.clear()
        except Exception:
            pass
        if times.size:
            self.wave_curve = self.wave_plot.plot(times, y, pen=pg.mkPen('w'))
        try:
            self.wave_plot.addItem(self.time_line_wave)
        except Exception:
            pass
        self.wave_plot.setLimits(xMin=0, xMax=self.duration)
        self.wave_plot.setLabel('bottom', 'Time', units='s')

        if self.wave_y_range:
            self.wave_plot.setYRange(*self.wave_y_range)
        elif y.size:
            self.wave_plot.setYRange(float(np.min(y)), float(np.max(y)))

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

        # 4) 临时圈选区域 (拖拽时的浅色块)
        if getattr(self, "temp_region", None):
            try:
                self.annot_plot.removeItem(self.temp_region)
            except:
                pass
            self.temp_region = None

        # 5) 数据结构清空
        self.annotations.clear()
        try:
            self.neg_segments.clear()
        except Exception:
            self.neg_segments = defaultdict(list)
        self._neg_id_counter = 0

        try:
            self._undo_stack.clear()
        except Exception:
            self._undo_stack = []
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

        times, y = self._get_display_waveform()
        self.wave_plot.clear()

        if start is None or end is None:
            if times.size:
                self.wave_plot.plot(times, y, pen=pg.mkPen('w'))
        else:
            start, end = min(start, end), max(start, end)
            inside = (times >= start) & (times <= end)
            outside = ~inside
            if np.any(outside):
                self.wave_plot.plot(times[outside], y[outside], pen=pg.mkPen('w'))
            if np.any(inside):
                self.wave_plot.plot(times[inside], y[inside], pen=pg.mkPen('r'))

        try:
            self.wave_plot.addItem(self.time_line_wave)
        except Exception:
            pass

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
            preprocessing_enabled=getattr(self, "preprocessing_enabled", True),
            resample_enabled=getattr(self, "resample_on_load_enabled", False),
            resample_target_sr=getattr(self, "resample_target_sr", 4000),
            filter_enabled=getattr(self, "filter_on_load_enabled", False),
            filter_type=getattr(self, "filter_type", "bandpass"),
            filter_lowcut=getattr(self, "filter_lowcut", 20.0),
            filter_highcut=getattr(self, "filter_highcut", 1800.0),
            filter_order=getattr(self, "filter_order", 4),
            filter_zero_phase=getattr(self, "filter_zero_phase", True),
            auto_label_import_enabled=getattr(self, "auto_import_events_enabled", False),
            auto_label_import_settings=getattr(self, "auto_label_import_settings", None),
        )

        dlg.set_current_tab(self.last_settings_tab_index)  # 💡 Settings上次使用的标签页

        if dlg.exec_():
            self.n_fft, self.hop_length, self.f_max, self.wave_y_range = dlg.get_values()
            self.selected_features = dlg.get_selected_features()  # 新增
            self.last_settings_tab_index = dlg.tabs.currentIndex()

            # 新增：保存“读取时预处理”和“自动标签读取参数”。不立即重读当前音频，避免影响现有编辑状态；下次 load_audio 生效。
            try:
                pp = dlg.get_preprocessing_settings()
                self.preprocessing_enabled = bool(pp.get("preprocessing_enabled", True))
                self.resample_on_load_enabled = bool(pp.get("resample_enabled", True))
                self.resample_target_sr = int(pp.get("resample_target_sr", 4000))
                self.filter_on_load_enabled = bool(pp.get("filter_enabled", False))
                self.filter_type = str(pp.get("filter_type", "bandpass"))
                self.filter_lowcut = float(pp.get("filter_lowcut", 20.0))
                self.filter_highcut = float(pp.get("filter_highcut", 1800.0))
                self.filter_order = int(pp.get("filter_order", 4))
                self.filter_zero_phase = bool(pp.get("filter_zero_phase", True))
            except Exception:
                # Backward-compatible fallback
                try:
                    self.resample_on_load_enabled, self.resample_target_sr = dlg.get_resample_settings()
                except Exception:
                    pass

            try:
                old_auto_cfg = dict(getattr(self, "auto_label_import_settings", {}) or {})
                self.auto_import_events_enabled = bool(dlg.get_auto_label_import_enabled())
                self.auto_label_import_settings = dlg.get_auto_label_import_settings()
                if hasattr(self, "auto_import_events_action"):
                    self.auto_import_events_action.blockSignals(True)
                    self.auto_import_events_action.setChecked(bool(self.auto_import_events_enabled))
                    self.auto_import_events_action.blockSignals(False)
                if old_auto_cfg != self.auto_label_import_settings:
                    try:
                        self._events_index_cache.clear()
                        self._events_parse_cache.clear()
                    except Exception:
                        pass
            except Exception:
                pass

            # 取回 STFT DisplaySettings (配色 + 上下限)，先保存再重绘，避免使用旧色阶。
            try:
                cmap, vmin, vmax = dlg.get_stft_display_settings()
                self.stft_cmap = cmap
                self.stft_vmin = vmin
                self.stft_vmax = vmax
            except Exception:
                pass

            if self.audio is not None:
                # Settings 改变后只刷新波形/STFT，并标记 FFT/特征失效；不立即计算短时特征。
                self.draw_waveform()
                self.update_spectrogram()
                self.features_dirty = True
                self.fft_dirty = True

            # 若只改了色阶而未重算，也可以直接按当前谱图重着色。
            if self._last_spec_vals is not None:
                spec_disp = self._decimate_spec_for_display(self._last_spec_vals)
                disp = spec_disp.T
                rgb = self._colorize_spec_with_window(disp)
                self.spec_img.setImage(rgb, autoLevels=False)

    def _update_default_export_annotation_name(self, wav_path=None):
        """根据当前 WAV 实时更新默认导出File名。

        只更新File名，不决定目录。目录在导出时优先沿用上一次Export Annotations的目录。
        命名规则：<当前wavFile名不含扩展名>_events.csv
        """
        try:
            if wav_path is None:
                wav_path = getattr(self, "loaded_filename", None)
            if wav_path:
                base = os.path.splitext(os.path.basename(str(wav_path)))[0]
                self.default_export_annotation_name = f"{base}_events.csv"
            else:
                self.default_export_annotation_name = "annotations_events.csv"
        except Exception:
            self.default_export_annotation_name = "annotations_events.csv"
        return self.default_export_annotation_name

    def _get_default_export_annotation_path(self):
        """生成导出对话框默认路径：目录跟随上次导出，File名跟随当前 WAV。"""
        default_name = self._update_default_export_annotation_name(
            getattr(self, "loaded_filename", None)
        )

        default_dir = None
        last_path = getattr(self, "last_export_path", None)
        if last_path:
            try:
                last_dir = os.path.dirname(str(last_path))
                if last_dir:
                    default_dir = last_dir
            except Exception:
                default_dir = None

        if not default_dir and getattr(self, "loaded_filename", None):
            try:
                default_dir = os.path.dirname(str(self.loaded_filename))
            except Exception:
                default_dir = None

        if not default_dir:
            default_dir = os.getcwd()

        return os.path.join(default_dir, default_name)

    def export_annotations(self):
        # 过滤掉 None，并兼容 3/4 元组。
        # 注意：后续会引入更多 source (如 auto_accepted/auto_edited/merged/...)。
        # 这里先把“已归档/隐藏”的标注 (source='archived')排除在导出之外。
        rows = []
        for item in getattr(self, "annotations", []):
            if item is None:
                continue
            try:
                if len(item) >= 4:
                    src = str(item[3]).strip().lower()
                    if src == "archived":
                        continue
                rows.append(item)
            except Exception:
                continue
        if not rows:
            QMessageBox.information(self, "Notice", "There are no annotations to export.")
            return

        # 默认路径和File名：目录沿用上一次导出位置，File名始终跟随当前 WAV。
        # 例如当前音频为 1008.wav，则默认File名为 1008_events.csv；
        # 若上一次导出到 D:/labels/，则默认路径为 D:/labels/1008_events.csv。
        default_path = self._get_default_export_annotation_path()

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Annotations", default_path, "CSV Files (*.csv)"
        )
        if not path:
            return

        try:
            from respanno.labels.annotation_io import write_annotations

            ann_dicts = []
            for item in rows:
                try:
                    s, e, lab = float(item[0]), float(item[1]), str(item[2])
                    src = str(item[3]) if len(item) >= 4 else "manual"
                    ann_dicts.append({"start": s, "end": e, "label": lab, "source": src})
                except Exception:
                    continue

            write_annotations(path, ann_dicts)

            # === VERIFY: 导出 source 校验 (可在控制台查看) ===
            try:
                mem_src = {}
                for d in ann_dicts:
                    mem_src[d["source"]] = mem_src.get(d["source"], 0) + 1
                print("[VERIFY][EXPORT] sources in memory:", mem_src)
            except Exception as _e:
                print("[VERIFY][EXPORT] source check failed:", _e)

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

    def delete_annotation(self, target, *args, record_negative=True, push_undo=True):
        """删除一个标注。

        - 交互删除 BoxSpan 时：默认会把该段加入 neg_segments (仅用于训练的硬负样本)，
          且支持 Ctrl+Z 撤销。
        - 程序内部清理 (例如清理 ML 预测)可传 record_negative=False, push_undo=False。
        """
        # 新：删除 BoxSpan
        if isinstance(target, BoxSpan):
            sp = target

            idx = None
            old_item = None
            try:
                if sp in self._span2idx:
                    idx = int(self._span2idx.get(sp))
                    if 0 <= idx < len(self.annotations):
                        old_item = self.annotations[idx]
            except Exception:
                idx = None
                old_item = None

            # 先记录硬负样本 (按“被删除的标签”记负样本，而不是当前训练类型)
            neg_item = None
            if record_negative and old_item is not None:
                try:
                    s, e, lab = float(old_item[0]), float(old_item[1]), str(old_item[2])
                    neg_item = self._add_neg_segment(lab, s, e)
                except Exception:
                    neg_item = None

            # 真正删除 UI 组件
            if sp in self._span2spec:
                try:
                    self.spec_stft_plot.removeItem(self._span2spec[sp])
                except Exception:
                    pass
                self._span2spec.pop(sp, None)

            if sp in self._span2idx:
                try:
                    idx2 = self._span2idx.pop(sp)
                except Exception:
                    idx2 = None
                if idx2 is not None and 0 <= int(idx2) < len(self.annotations):
                    self.annotations[int(idx2)] = None

            try:
                sp.cleanup()
            except Exception:
                pass
            if sp in self._spans:
                try:
                    self._spans.remove(sp)
                except Exception:
                    pass

            # 记录撤销信息
            if push_undo and old_item is not None and idx is not None:
                self._push_undo({
                    "op": "delete",
                    "idx": int(idx),
                    "item": old_item,
                    "neg_item": neg_item,
                })
            return

        # 兼容：旧 LinearRegionItem 删除路径 (如果你代码中还有)
        # 若传入的是 STFT 高亮区域 (LinearRegionItem)，尝试反查对应的 BoxSpan，
        # 统一走 BoxSpan 删除逻辑，以便支持硬负样本与 Ctrl+Z 撤销。
        region_item = target
        try:
            for _sp, _reg in list(getattr(self, "_span2spec", {}).items()):
                if _reg is region_item:
                    # 复用 BoxSpan 删除逻辑
                    return self.delete_annotation(_sp, record_negative=record_negative, push_undo=push_undo)
        except Exception:
            pass
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

    # ==========================
    # Machine Learning硬负样本 & 撤销栈
    # ==========================
    def _add_neg_segment(self, label: str, s: float, e: float):
        """加入硬负样本段 (仅用于训练)。返回 (s,e,neg_id) 便于撤销。"""
        try:
            label = str(label)
        except Exception:
            label = ""
        if not label:
            return None
        self._neg_id_counter += 1
        item = (float(s), float(e), int(self._neg_id_counter))
        try:
            self.neg_segments[label].append(item)
        except Exception:
            if not hasattr(self, "neg_segments"):
                self.neg_segments = defaultdict(list)
            self.neg_segments[label].append(item)
        return item

    def _remove_neg_segment(self, label: str, neg_id: int):
        """按 id 删除硬负样本段。"""
        try:
            lst = self.neg_segments.get(str(label), [])
        except Exception:
            return
        for i, it in enumerate(list(lst)):
            try:
                if int(it[2]) == int(neg_id):
                    lst.pop(i)
                    break
            except Exception:
                continue

    def _push_undo(self, rec: dict):
        try:
            self._undo_stack.append(rec)
            if len(self._undo_stack) > int(getattr(self, "_undo_maxlen", 100)):
                self._undo_stack = self._undo_stack[-int(getattr(self, "_undo_maxlen", 100)) :]
        except Exception:
            self._undo_stack = [rec]

    def _find_span_by_idx(self, idx: int):
        try:
            for sp, i in list(getattr(self, "_span2idx", {}).items()):
                try:
                    if int(i) == int(idx):
                        return sp
                except Exception:
                    continue
        except Exception:
            pass
        return None

    def _restore_span_from_annotation(self, idx: int, item):
        """仅依据 annotations[idx] 的四元组重建可视化 BoxSpan (撤销用)，不改变 annotations。"""
        # 若已经有同 idx 的 span (理论上不该发生)，直接返回避免重复
        try:
            if self._find_span_by_idx(int(idx)) is not None:
                return
        except Exception:
            pass

        try:
            start, end, text = float(item[0]), float(item[1]), str(item[2])
            source = str(item[3]) if len(item) >= 4 else "manual"
        except Exception:
            return
        if end <= start:
            return

        # —— 分轨策略 (与 finalize_annotation 一致)——
        lane = self._pick_lane(start, end)
        y0 = lane * (self.LANE_H + self.LANE_GAP)

        # —— 颜色/画笔 (与 finalize_annotation 一致)——
        color = self.get_annotation_color(text)
        try:
            pen = pg.mkPen(color, width=1)
            span_brush_color = QColor(color)
            span_brush_color.setAlpha(80)
            brush = pg.mkBrush(span_brush_color)
        except Exception:
            pen = pg.mkPen(255, 255, 255, 255, width=1)
            brush = pg.mkBrush(255, 255, 255, 255)

        # source != manual → 红字 (保持现有逻辑)
        label_color = None if source == "manual" else QColor(255, 0, 0)

        # —— 重建 BoxSpan —— (注意：BoxSpan 的签名与 finalize_annotation 完全一致)
        try:
            sp = BoxSpan(start, end, y0, self.LANE_H, text, self, label_color=label_color)
        except Exception:
            # 若 BoxSpan 初始化失败，撤销无法Resume；不要静默吞掉 (避免“看起来不触发”)
            return

        sp.setPen(pen)
        try:
            sp.setBrush(brush)
        except Exception:
            pass

        self.annot_plot.addItem(sp)
        self._spans.append(sp)
        sp.lane = lane
        self._span2idx[sp] = int(idx)

        # —— 频谱高光 (与 finalize_annotation 一致)——
        try:
            spec_color = QColor(color)
            spec_color.setAlpha(50)
            spec_brush = pg.mkBrush(spec_color)
        except Exception:
            spec_brush = pg.mkBrush(255, 0, 0, 50)

        try:
            spec = pg.LinearRegionItem([start, end], brush=spec_brush)
            spec.setMovable(False)
            spec.setZValue(1)
            # 允许 STFT 高光随编辑扩展到音频全时长 (不要把 bounds 锁死在初始区间)
            _xmax = float(getattr(self, "duration", 0.0) or max(end, start, 0.0))
            spec.setBounds([0.0, _xmax])
            self.spec_stft_plot.addItem(spec)
            self._span2spec[sp] = spec
        except Exception:
            pass



    def _clear_annotation_view_only(self):
        """仅清空“可视化对象” (BoxSpan / STFT高光等)，不清空 annotations/neg/undo 等数据。
        用于视图与数据不同步时的重建兜底。"""
        # 1) BoxSpan 及其文字
        try:
            for sp in list(getattr(self, "_spans", [])):
                try:
                    sp.cleanup()
                except Exception:
                    # 兜底：至少从图里移除
                    try:
                        self.annot_plot.removeItem(sp)
                    except Exception:
                        pass
            getattr(self, "_spans", []).clear()
        except Exception:
            pass

        # 2) STFT 高光
        try:
            for reg in list(getattr(self, "_span2spec", {}).values()):
                try:
                    self.spec_stft_plot.removeItem(reg)
                except Exception:
                    pass
            getattr(self, "_span2spec", {}).clear()
        except Exception:
            pass

        # 3) idx 映射
        try:
            getattr(self, "_span2idx", {}).clear()
        except Exception:
            pass

        # 4) 兼容老样式 (如果还有残留)
        try:
            for item, label in list(getattr(self, "annotation_items", [])):
                try:
                    self.annot_plot.removeItem(item)
                except Exception:
                    pass
                if label:
                    try:
                        self.annot_plot.removeItem(label)
                    except Exception:
                        pass
            getattr(self, "annotation_items", []).clear()
        except Exception:
            pass

        # 5) 临时圈选区域
        if getattr(self, "temp_region", None) is not None:
            try:
                self.annot_plot.removeItem(self.temp_region)
            except Exception:
                pass
            self.temp_region = None

        # 6) lane 缓存重置 (不影响数据)
        try:
            self._lanes = [[] for _ in range(self.MAX_LANES)]
        except Exception:
            pass

    def rebuild_annotation_view_from_data(self):
        """从 self.annotations 重新渲染所有 BoxSpan (不改动 annotations 数据)。"""
        # 清空视图
        self._clear_annotation_view_only()

        # 复位 Y 轴Display范围 (避免“画出来但看不到”)
        try:
            H = self.MAX_LANES * (self.LANE_H + self.LANE_GAP)
            self.annot_plot.setYRange(0, H)
            self.annot_plot.setLimits(yMin=0, yMax=H, xMin=0, xMax=getattr(self, "duration", 0.0))
        except Exception:
            pass

        # 重建
        try:
            for idx, item in enumerate(list(getattr(self, "annotations", []))):
                if item is None:
                    continue
                try:
                    # archived 不Display也不导出 (合并/删除前的旧 span)
                    if len(item) >= 4 and str(item[3]) == "archived":
                        continue
                except Exception:
                    pass
                self._restore_span_from_annotation(int(idx), item)
        except Exception:
            # 不要在这里抛异常影响主流程
            pass


    def set_annotation_source(self, target, new_source: str, push_undo=True):
        """将某条标注的 source 改为 new_source，并同步 UI；可撤销。"""
        idx = None
        if isinstance(target, BoxSpan):
            try:
                idx = int(self._span2idx.get(target))
            except Exception:
                idx = None
        else:
            try:
                idx = int(target)
            except Exception:
                idx = None
        if idx is None or not (0 <= idx < len(self.annotations)):
            return False
        old_item = self.annotations[idx]
        if old_item is None:
            return False
        try:
            s, e, lab = float(old_item[0]), float(old_item[1]), str(old_item[2])
        except Exception:
            return False
        new_item = (s, e, lab, str(new_source))
        self.annotations[idx] = new_item

        sp = self._find_span_by_idx(idx)
        if sp is not None:
            try:
                sp._apply_visual_style(str(new_source))
            except Exception:
                pass

        if push_undo:
            self._push_undo({
                "op": "set_source",
                "idx": int(idx),
                "old_item": old_item,
                "new_item": new_item,
            })
        return True

    def undo_last_action(self):
        """撤销最后一次“删除/改 source”等编辑动作。"""
        try:
            if not self._undo_stack:
                return
            rec = self._undo_stack.pop()
        except Exception:
            return

        op = rec.get("op")
        if op == "delete":
            idx = rec.get("idx")
            item = rec.get("item")
            neg_item = rec.get("neg_item")
            if idx is None or item is None:
                return
            try:
                if 0 <= int(idx) < len(self.annotations):
                    self.annotations[int(idx)] = item
                    self._restore_span_from_annotation(int(idx), item)
                    # 兜底：若删除时映射未清理干净或恢复失败，强制重建整个标注视图
                    try:
                        if self._find_span_by_idx(int(idx)) is None:
                            self.rebuild_annotation_view_from_data()
                    except Exception:
                        pass
            except Exception:
                pass
            if neg_item is not None:
                try:
                    lab = str(item[2])
                    neg_id = int(neg_item[2])
                    self._remove_neg_segment(lab, neg_id)
                except Exception:
                    pass
            return

        if op == "edit_interval":
            idx = rec.get("idx")
            old_item = rec.get("old_item")
            if idx is None or old_item is None:
                return
            try:
                idx = int(idx)
            except Exception:
                return
            try:
                if 0 <= idx < len(self.annotations):
                    self.annotations[idx] = old_item
            except Exception:
                pass

            sp = None
            try:
                sp = self._find_span_by_idx(idx)
            except Exception:
                sp = None

            if sp is not None:
                try:
                    s0, s1 = float(old_item[0]), float(old_item[1])
                    lab = str(old_item[2])
                    src0 = str(old_item[3]) if len(old_item) >= 4 else "manual"

                    # 几何回滚
                    sp.setPos([s0, sp.y_base], update=False)
                    sp.setSize([s1 - s0, sp.h_fix], update=False)

                    # 文本/样式回滚
                    sp.text = lab
                    try:
                        sp._update_label_html()
                    except Exception:
                        pass
                    try:
                        sp._apply_visual_style(src0)
                    except Exception:
                        pass

                    # STFT 高光同步
                    try:
                        reg = getattr(self, "_span2spec", {}).get(sp)
                        if reg is not None:
                            reg.setRegion([s0, s1])
                    except Exception:
                        pass

                    # 触发一次内部同步 (不会改变数据语义)
                    try:
                        sp._on_changed()
                    except Exception:
                        pass
                except Exception:
                    # 兜底：回滚失败则重建视图
                    try:
                        self.rebuild_annotation_view_from_data()
                    except Exception:
                        pass
            else:
                # span 丢失：重建整个视图
                try:
                    self.rebuild_annotation_view_from_data()
                except Exception:
                    pass
            return

        if op == "set_source":
            idx = rec.get("idx")
            old_item = rec.get("old_item")
            if idx is None or old_item is None:
                return
            try:
                if 0 <= int(idx) < len(self.annotations):
                    self.annotations[int(idx)] = old_item
                    sp = self._find_span_by_idx(int(idx))
                    if sp is not None:
                        try:
                            src = str(old_item[3]) if len(old_item) >= 4 else "manual"
                            sp._apply_visual_style(src)
                        except Exception:
                            pass
            except Exception:
                pass
            return

    def accept_annotation(self, target, accepted_source: str = "auto_accepted"):
        """把机器标记置为“已认可”，默认 source=auto_accepted (可撤销)。"""
        return self.set_annotation_source(target, accepted_source, push_undo=True)

    # ==========================
    # BoxSpan 编辑模式：双击进入；Enter 提交；Esc 取消
    # ==========================
    def begin_edit_span(self, span):
        """记录当前编辑中的 BoxSpan (若已有其它在编辑，先提交Exit)。"""
        try:
            cur = getattr(self, "_editing_span", None)
            if cur is not None and cur is not span:
                try:
                    cur.exit_edit_mode(commit=True)
                except Exception:
                    pass
        except Exception:
            pass

        self._editing_span = span
        try:
            self._update_edit_status(span)
        except Exception:
            pass

    def end_edit_span(self, span):
        """Exit编辑模式后清空状态。"""
        try:
            if getattr(self, "_editing_span", None) is span:
                self._editing_span = None
        except Exception:
            self._editing_span = None

        # Exit编辑后清理状态栏Notice
        try:
            self.statusBar().showMessage("", 1000)
        except Exception:
            pass

    def _update_edit_status(self, span):
        """编辑态：在状态栏Display选中标记的起止Time。"""
        try:
            s0, s1 = span.interval()
            dur = float(s1) - float(s0)
            lab = getattr(span, "text", "")
            self.statusBar().showMessage(
                f"[Edit] {lab}  start={s0:.3f}s  end={s1:.3f}s  dur={dur:.3f}s   (Enter to commit / Esc to cancel / double-click to commit)",
                0,
            )
        except Exception:
            pass

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

        # 预设类型 (英文名)
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
        # 若未提供文本 (正常交互标注)，则弹出带预设类型的对话框
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

        # 根据标签拿颜色 (用于条的填充、边框、STFT 高光)
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
        # 允许 STFT 高光随编辑扩展到音频全时长 (不要把 bounds 锁死在初始区间)
        _xmax = float(getattr(self, "duration", 0.0) or max(end, start, 0.0))
        spec.setBounds([0.0, _xmax])
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
            self, "Import Annotations", "", "Annotation Files (*.csv *.txt *.json);;All Files (*)"
        )
        if not path:
            return

        try:
            from respanno.labels.annotation_io import read_annotations

            cfg = self._get_auto_label_import_settings()
            rows = read_annotations(path, cfg)

            if not rows:
                QMessageBox.information(self, "Notice", "No annotations found in the file.")
                return

            self.clear_annotations()
            for ann in rows:
                self.finalize_annotation(
                    ann["start"], ann["end"], ann["label"],
                    source=ann.get("source", "manual"),
                )

            QMessageBox.information(self, "Success", f"Successfully imported {len(rows)} annotations.")

        except Exception as e:
            QMessageBox.critical(self, "Import Failed", f"An error occurred while reading the file:\n{str(e)}")


    # =========================
    # 自动导入同名 _events 标注 (可开关)
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

        # 轻量Notice (不弹窗)
        try:
            msg = "Auto-import matching _events annotations: enabled" if self.auto_import_events_enabled else "Auto-import matching _events annotations: disabled"
            self.statusBar().showMessage(msg, 2000)
        except Exception:
            pass

    def _get_auto_label_import_settings(self):
        """返回自动标签读取配置，并补齐默认值。"""
        default_cfg = {
            "file_format": "auto",
            "file_suffix": "_events",
            "delimiter": "auto",
            "custom_delimiter": "",
            "skip_header_lines": 0,
            "start_col": 1,
            "end_col": 2,
            "label_col": 3,
            "source_col": 4,
            "json_start_key": "start",
            "json_end_key": "end",
            "json_label_key": "label",
            "json_source_key": "source",
        }
        try:
            cfg = dict(getattr(self, "auto_label_import_settings", {}) or {})
        except Exception:
            cfg = {}
        default_cfg.update(cfg)
        return default_cfg

    def _auto_label_candidate_extensions(self):
        """根据 Settings 中的 file_format 返回自动匹配标签文件的扩展名优先级。"""
        cfg = self._get_auto_label_import_settings()
        fmt = str(cfg.get("file_format", "auto")).strip().lower()
        if fmt in {"csv", "txt", "json"}:
            return [fmt]
        # 保持旧逻辑优先 csv/txt；新增 json 作为第三优先级。
        return ["csv", "txt", "json"]

    def _prepare_events_index(self, folder: str):
        """
        为某个目录建立自动标签文件索引。
        新增支持：Settings 中可配置文件格式(csv/txt/json)、文件后缀，默认仍为 <wav_base>_events.csv/txt。
        """
        try:
            folder = os.path.abspath(folder)
        except Exception:
            return

        if not hasattr(self, "_events_index_cache"):
            self._events_index_cache = {}

        if folder in self._events_index_cache:
            return

        cfg = self._get_auto_label_import_settings()
        suffix = str(cfg.get("file_suffix", "_events") or "_events")
        suffix_low = suffix.lower()
        exts = self._auto_label_candidate_extensions()
        ext_rank = {ext: i for i, ext in enumerate(exts)}

        mapping = {}
        try:
            for ent in os.scandir(folder):
                if not ent.is_file():
                    continue
                name = ent.name
                low = name.lower()
                for ext in exts:
                    tail = f"{suffix_low}.{ext}"
                    if low.endswith(tail):
                        base = name[:-len(tail)].lower()
                        old_path = mapping.get(base)
                        if old_path is None:
                            mapping[base] = ent.path
                        else:
                            try:
                                old_ext = os.path.splitext(old_path)[1].lower().lstrip(".")
                                if ext_rank.get(ext, 999) < ext_rank.get(old_ext, 999):
                                    mapping[base] = ent.path
                            except Exception:
                                pass
                        break
        except Exception:
            mapping = {}

        self._events_index_cache[folder] = mapping

    def _resolve_events_path_for_wav(self, wav_path: str):
        """
        解析 wav 对应的自动标签文件路径。
        默认仍查找 <wav_base>_events.csv / .txt；也可在 Settings 中指定 json 或自定义后缀。
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

        # 目录索引可能早于文件创建；按当前 Settings 做一次 O(扩展名数) existence check 并回填。
        cfg = self._get_auto_label_import_settings()
        suffix = str(cfg.get("file_suffix", "_events") or "_events")
        for ext in self._auto_label_candidate_extensions():
            cand = os.path.join(folder, f"{wav_base}{suffix}.{ext}")
            if os.path.isfile(cand):
                mapping[key] = cand
                self._events_index_cache[folder] = mapping
                return cand

        return None

    def _read_text_file_flexible(self, path: str):
        """
        尝试用常见编码读取文本 (utf-8/utf-8-sig/gbk)，失败则忽略Error读取。
        """
        for enc in ("utf-8-sig", "utf-8", "gbk"):
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

    def _split_label_line_by_settings(self, line: str):
        """按 Settings 中的分隔符配置拆分一行 csv/txt 标签。"""
        import csv

        cfg = self._get_auto_label_import_settings()
        delim = str(cfg.get("delimiter", "auto")).strip().lower()

        def _read_with(delimiter):
            try:
                return [p.strip() for p in next(csv.reader([line], delimiter=delimiter))]
            except Exception:
                return [p.strip() for p in line.split(delimiter)]

        if delim in {"comma", ","}:
            return _read_with(",")
        if delim in {"semicolon", ";"}:
            return _read_with(";")
        if delim in {"tab", "\\t"}:
            return _read_with("\t")
        if delim in {"space", "whitespace"}:
            return line.split()
        if delim == "custom":
            custom = str(cfg.get("custom_delimiter", ""))
            if custom:
                return [p.strip() for p in line.split(custom)]
            return line.split()

        # auto：兼容旧逻辑，同时新增分号。
        if "," in line:
            return _read_with(",")
        if "\t" in line:
            return _read_with("\t")
        if ";" in line:
            return _read_with(";")
        return line.split()

    def _annotation_row_from_sequence(self, parts):
        """按 1-based 列号把一行拆分结果转换为 (start,end,label,source)。source_col=0 表示禁用。"""
        cfg = self._get_auto_label_import_settings()

        def _col(key, default):
            try:
                return int(cfg.get(key, default)) - 1
            except Exception:
                return int(default) - 1

        i_s = _col("start_col", 1)
        i_e = _col("end_col", 2)
        i_l = _col("label_col", 3)
        try:
            src_col = int(cfg.get("source_col", 4))
        except Exception:
            src_col = 4
        i_src = src_col - 1

        if not isinstance(parts, (list, tuple)):
            return None
        if min(i_s, i_e, i_l) < 0:
            return None
        if max(i_s, i_e, i_l) >= len(parts):
            return None

        try:
            s = float(str(parts[i_s]).strip())
            e = float(str(parts[i_e]).strip())
        except Exception:
            return None

        lab = str(parts[i_l]).strip()
        if not lab:
            return None
        src = "manual"
        if i_src >= 0 and i_src < len(parts):
            src0 = str(parts[i_src]).strip()
            if src0:
                src = src0
        return (s, e, lab, src)

    def _annotation_row_from_dict(self, item: dict):
        """解析 JSON dict 行。优先使用 Settings 中的键名，找不到时使用常见同义键兜底。"""
        cfg = self._get_auto_label_import_settings()
        if not isinstance(item, dict):
            return None

        def _get_by_keys(primary, fallbacks):
            keys = [primary] + list(fallbacks)
            # 先精确匹配，再不区分大小写匹配。
            for k in keys:
                if k in item:
                    return item.get(k)
            lower_map = {str(k).lower(): k for k in item.keys()}
            for k in keys:
                kk = lower_map.get(str(k).lower())
                if kk is not None:
                    return item.get(kk)
            return None

        start_v = _get_by_keys(str(cfg.get("json_start_key", "start")), ["start_time", "start_sec", "onset", "begin", "s"])
        end_v = _get_by_keys(str(cfg.get("json_end_key", "end")), ["end_time", "end_sec", "offset", "finish", "e"])
        label_v = _get_by_keys(str(cfg.get("json_label_key", "label")), ["text", "tag", "type", "class", "name"])
        source_v = _get_by_keys(str(cfg.get("json_source_key", "source")), ["src", "origin"])

        try:
            s = float(start_v)
            e = float(end_v)
        except Exception:
            return None
        lab = str(label_v).strip() if label_v is not None else ""
        if not lab:
            return None
        src = str(source_v).strip() if source_v is not None and str(source_v).strip() else "manual"
        return (s, e, lab, src)

    def _flatten_json_annotation_records(self, obj):
        """从 JSON 对象中提取可能的 annotation records。"""
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            # 如果自身就是一条记录。
            if self._annotation_row_from_dict(obj) is not None:
                return [obj]
            for key in ("annotations", "events", "labels", "data", "items", "records"):
                val = obj.get(key)
                if isinstance(val, list):
                    return val
        return []

    def _parse_events_json_content(self, content: str):
        import json
        try:
            obj = json.loads(content)
        except Exception:
            return []

        rows = []
        for rec in self._flatten_json_annotation_records(obj):
            if isinstance(rec, dict):
                row = self._annotation_row_from_dict(rec)
            elif isinstance(rec, (list, tuple)):
                row = self._annotation_row_from_sequence(rec)
            else:
                row = None
            if row is not None:
                rows.append(row)
        return rows

    def _parse_events_file(self, events_path: str):
        """解析自动导入的 events 文件，返回标准 annotation dict 列表。

        委托 respanno.labels.annotation_io.read_annotations，支持 csv / txt / json，
        默认配置保持旧行为：自动分隔符，前 3 列 start/end/label，第 4 列 source 可选。
        """
        from respanno.labels.annotation_io import read_annotations

        cfg = self._get_auto_label_import_settings()
        return read_annotations(events_path, cfg)


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
        仅负责导入；清空标注由调用方决定 (load_audio 已清空)。
        """
        events_path = self._resolve_events_path_for_wav(wav_path)
        if not events_path:
            return  # 静默跳过

        rows = self._parse_events_file_cached(events_path)
        if not rows:
            return

        n_ok = 0
        for ann in rows:
            try:
                s, e, lab = float(ann["start"]), float(ann["end"]), str(ann["label"])
                src = str(ann.get("source", "manual") or "manual")
                if e <= s:
                    continue
                self.finalize_annotation(s, e, lab, source=src)
                n_ok += 1
            except Exception:
                continue

        # 轻量Notice (不弹窗)
        try:
            self.statusBar().showMessage(
                f"Auto-imported events annotations: {os.path.basename(events_path)} ({n_ok} items)", 2500
            )
        except Exception:
            pass
    def toggle_analysis_mode(self):
        cur = self.spec_stack.currentWidget()
        if cur is self.spec_stft_plot:
            # STFT -> FFT：懒加载 FFT
            self.spec_stack.setCurrentWidget(self.spec_fft_plot)
            self.freq_button.setText("Switch: Short-Time Features")
            if getattr(self, "fft_dirty", True):
                self.show_fft()
        elif cur is self.spec_fft_plot:
            # FFT -> Short-Time Features：懒加载短时特征
            self.spec_stack.setCurrentWidget(self.feat_page)
            self.freq_button.setText("Switch: STFT")
            if getattr(self, "features_dirty", True) or not getattr(self, "feature_curves", {}):
                self.update_features_plot()
        else:
            # Short-Time Features -> STFT
            self.spec_stack.setCurrentWidget(self.spec_stft_plot)
            self.freq_button.setText("Switch: FFT")

    def show_fft(self):
        if self.audio is None:
            return

        from scipy.fft import rfft, rfftfreq
        N = len(self.audio)
        if N == 0 or self.sr is None or self.sr == 0:
            return

        fft_y = np.abs(rfft(self.audio))
        fft_x = rfftfreq(N, d=1 / self.sr)

        # FFT 仅用于显示时也做抽稀，避免长音频频谱曲线点数过多。
        try:
            max_fft_points = 50000
            if fft_x.size > max_fft_points:
                idx_fft = np.linspace(0, fft_x.size - 1, max_fft_points).astype(int)
                fft_x_plot = fft_x[idx_fft]
                fft_y_plot = fft_y[idx_fft]
            else:
                fft_x_plot = fft_x
                fft_y_plot = fft_y
        except Exception:
            fft_x_plot = fft_x
            fft_y_plot = fft_y

        self.spec_fft_plot.clear()
        self.spec_fft_plot.plot(fft_x_plot, fft_y_plot, pen='c')
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
        self.fft_dirty = False

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
        # 仅在绘图区内触发 (不包含轴刻度/边距)
        if not vb.sceneBoundingRect().contains(pos):
            self.spec_stft_plot.setTitle(self.spec_title_base)
            return

        p = vb.mapSceneToView(pos)
        x = float(p.x())
        y = float(p.y())
        fmax = getattr(self, "_stft_fmax", getattr(self, "f_max", 0.0))
        dur = getattr(self, "duration", 0.0)

        if 0.0 <= x <= (dur or 0.0) and 0.0 <= y <= (fmax or 0.0):
            # 标题右侧Display坐标 (可调样式)
            self.spec_stft_plot.setTitle(
                f'{self.spec_title_base}  '
                f'<span style="color:#bbb; font-size:11px;">t={x:.3f}s, f={y:.1f} Hz</span>'
            )
        else:
            self.spec_stft_plot.setTitle(self.spec_title_base)

    def show_about_dialog(self):
        QMessageBox.information(self, "About",
                                "Audio Annotator v1.0\nAuthor: C.Y.Pan\nBuilt with PyQt5 + pyqtgraph\nCurrently only .wav files are supported. Please convert other file types to .wav.")

    def compute_short_time_features(self):
        """
        返回 times(s), feat_dict{name -> 1D array}。
        频域特征：基于当前 STFT (n_fft/hop 与 f_max)
        时域特征：基于当前波形帧

        对齐原则 (与 update_spectrogram 完全一致)：
        - STFT 使用 center=True + pad_mode='reflect' (librosa 默认行为)
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

        # —— 频域基准 (与 update_spectrogram 一致)——
        # 用它来定义“帧数”和“Time轴”，保证所有特征与 STFT 一一对应
        try:
            D = librosa.stft(x, n_fft=n, hop_length=h, center=True, pad_mode='reflect')
        except Exception:
            return np.array([]), {}

        T = D.shape[1]
        times = (np.arange(T) * h) / sr

        # === 时域分帧 (使用与 STFT 相同的 padding 语义) ===
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
        # 1. 短时能量
        f_time["短时能量"] = np.sum(frames ** 2, axis=0)
        # 2. 短时均值
        mu = np.mean(frames, axis=0)
        f_time["短时均值"] = mu
        # 3. 方差
        var = np.var(frames, axis=0) + 1e-12
        f_time["方差"] = var
        # 4/5. 峰度/偏度
        std = np.sqrt(var)
        centered3 = np.mean((frames - mu) ** 3, axis=0)
        centered4 = np.mean((frames - mu) ** 4, axis=0)
        f_time["偏度"] = centered3 / (std ** 3 + 1e-12)
        f_time["峰度"] = centered4 / (std ** 4 + 1e-12)
        # 6. 过零率 (与 STFT 对齐：先对时域信号做同样的 reflect padding，再 center=False 分帧)
        # 说明：部分 librosa 版本的 zero_crossings/zero_crossing_rate 不支持 pad_mode 透传参数；
        # 因此这里采用“手动 padding + center=False”的方式保证与 STFT (center=True,pad_mode=reflect)帧严格一致。
        try:
            zcr = librosa.feature.zero_crossing_rate(
                x_pad, frame_length=n, hop_length=h, center=False
            )[0]
        except TypeError:
            # 极旧版本兼容：若不支持 center 参数，则退化为默认行为 (仍基于 x_pad，帧数通常一致)
            zcr = librosa.feature.zero_crossing_rate(
                x_pad, frame_length=n, hop_length=h
            )[0]

        if zcr.shape[0] != T:
            zcr = zcr[:T] if zcr.shape[0] > T else np.pad(zcr, (0, T - zcr.shape[0]), mode="edge")
        f_time["过零率"] = zcr
        # 7. Teager 能量算子 (帧内平均，基于同样 padding 后的信号)
        # ψ(x[n]) = x[n]^2 - x[n-1]*x[n+1]
        xn = x_pad
        teager = np.zeros_like(xn)
        teager[1:-1] = xn[1:-1] ** 2 - xn[:-2] * xn[2:]
        te_frames = librosa.util.frame(teager, frame_length=n, hop_length=h)
        f_time["teager能量算子"] = np.mean(np.abs(te_frames[:, :T]), axis=0)

        # —— 频域特征 ——
        S = np.abs(D)  # Amplitude谱
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n)
        idx = freqs <= self.f_max
        S = S[idx, :]
        freqs = freqs[idx]

        # 子带 (例如 200 Hz 以上)
        f_low = 200.0
        idx_sub = (freqs >= f_low) & (freqs <= self.f_max)
        S_sub = S[idx_sub, :]
        freqs_sub = freqs[idx_sub]

        # 概率化功率谱 (全带，用于熵/质心等)
        P = (S ** 2)
        P_sum = np.sum(P, axis=0, keepdims=True) + 1e-12
        Pn = P / P_sum  # 每帧归一化为概率

        # 8. 谱质心
        cent = np.sum(freqs[:, None] * Pn, axis=0)
        # 9. 谱带宽 (方差的平方根)
        var_f = np.sum(((freqs[:, None] - cent[None, :]) ** 2) * Pn, axis=0)
        bw = np.sqrt(var_f)
        # 10/11. 谱偏度/谱峰度
        m3 = np.sum(((freqs[:, None] - cent[None, :]) ** 3) * Pn, axis=0)
        m4 = np.sum(((freqs[:, None] - cent[None, :]) ** 4) * Pn, axis=0)
        skew_s = m3 / (var_f ** 1.5 + 1e-12)
        kurt_s = m4 / (var_f ** 2 + 1e-12)
        # 12. 谱滚降 (85%)
        roll = librosa.feature.spectral_rolloff(
            S=S_sub,
            freq=freqs_sub,
            roll_percent=0.85
        )[0]
        # 13. 谱平坦度 (几何均值/算术均值)
        flat = librosa.feature.spectral_flatness(S=S_sub)[0]
        # 14. 谱熵 (0~1)
        ent = -np.sum(Pn * (np.log(Pn + 1e-12)), axis=0) / np.log(Pn.shape[0])
        # 15. 谱通量 (相邻帧的变化)
        Sn = S / (np.linalg.norm(S, axis=0, keepdims=True) + 1e-12)
        dSn = np.diff(Sn, axis=1)
        flux = np.sqrt(np.sum(np.maximum(dSn, 0.0) ** 2, axis=0))
        flux = np.r_[0.0, flux]  # 与帧数对齐

        # === 补齐频域/时频特征 (基于子带 S_sub，更适合窄带/少峰检测)===
        # 说明：不改动已有特征定义，仅在此基础上追加更多谱统计/峰值/集中度特征。
        #      这些特征将自动进入 ML 帧特征缓存，并可在“Short-Time Features”页勾选Display。

        # 1) 谱统计量
        spec_mean = np.mean(S_sub, axis=0)
        spec_std = np.std(S_sub, axis=0)
        spec_median = np.median(S_sub, axis=0)

        # 2) 能量/Amplitude相关
        spec_energy = np.sum(S_sub ** 2, axis=0)
        spec_rms = np.sqrt(np.mean(S_sub ** 2, axis=0))
        spec_sum = np.sum(S_sub, axis=0)

        # 3) 峰值相关 (不依赖 scipy，使用简单局部极大值检测)
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

        # 4) 频带能量占比 (与 MATLAB 版本一致：0-400 / 400-800 / >800)
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

            # 注意：不要覆写上方的谱带宽向量 bw (用于“谱带宽”特征)
            bw0 = float(freqs_sub[right_in] - freqs_sub[left_in])
            if bw0 <= 0.0 and df_sub > 0.0:
                bw0 = df_sub

            dominant_bw_hz[k] = max(0.0, bw0)
            f0 = float(freqs_sub[i0])
            dominant_Q[k] = (f0 / bw0) if bw0 > 0 else 0.0

        # =====================================================================
        # 频移自相关 (cor)特征：在 100–1200 Hz 子带上，按 STFT 帧计算“向下频移自相关”曲线
        # - 与你此前 MATLAB 版本一致：每帧先保留Amplitude排序前 50% 的频点，再做频移点积自相关并归一化
        # - 最终从每帧自相关曲线中提取 19 个曲线特征 (cor_*)
        # =====================================================================
        cor_low_hz = 100.0
        cor_high_hz = min(1200.0, float(self.f_max))
        cor_mask = (freqs >= cor_low_hz) & (freqs <= cor_high_hz)
        S_cor = S[cor_mask, :]
        freqs_cor = freqs[cor_mask]

        # 初始化输出 (若频带不足则全 0)
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
            # Frequency分辨率与最大位移 (Hz)
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
                    """从自相关曲线 y(t) 中提取 19 维特征 (与 MATLAB extract_curve_features_local 对齐)。"""
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

                    # 2) 全局峰值特征 (不用 scipy，采用简单局部极大值)
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

                    # STEP2: 频移自相关 (向下位移)并归一化
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
            "谱质心": cent,
            "谱带宽": bw,
            "谱偏度": skew_s,
            "谱峰度": kurt_s,
            "谱滚降": roll,
            "谱平坦度": flat,
            "谱熵": ent,
            "谱通量": flux,
        }

        # 追加补齐的频域/时频特征 (不影响既有键的顺序)
        f_spec.update({
            "谱均值": spec_mean,
            "谱标准差": spec_std,
            "谱中位数": spec_median,
            "谱能量": spec_energy,
            "谱RMS": spec_rms,
            "谱幅和": spec_sum,
            "最大谱峰值": max_peak,
            "谱峰数量": peak_count,
            "低频能量占比": low_ratio,
            "中频能量占比": mid_ratio,
            "高频能量占比": high_ratio,
            "谱四分位距": spec_iqr,
            "谱MAD": spec_mad,
            "谱差分零交叉率": spec_zc,
            "谱平滑度": spec_smooth,
            "主峰/次峰比": peak_ratio,
            "谱复杂度": spec_complex,
            "主峰能量占比": top1_energy_ratio,
            "前三峰能量占比": top3_energy_ratio,
            "90%能量覆盖频点数": n_bins_90pct,
            "主峰-3dB带宽": dominant_bw_hz,
            "主峰Q因子": dominant_Q,
        })

        # 追加 cor 特征 (键名以 cor_ 开头，便于与你的离线分析/论文一致)
        f_spec.update(f_cor)

        feat = {**f_time, **f_spec}
        return times, feat

    def ensure_frame_features(self):
        """
        重新计算帧级特征和Time轴。

        说明：
            - 每次调用都基于当前 STFT / Short-Time Features参数重新计算；
            - 不再依赖旧的缓存，因此只要窗长、步长发生改变，
              下次调用本函数时特征会自动更新；
            - 后续 build_frame_labels_for_tag() 也会在新的Time轴上重建标签。
        """
        # 1) 调用你现有的Short-Time Features计算函数
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
        """根据 self.selected_features 计算并绘制 (0-1 归一化叠加Display)。

        注意：该函数仍在主线程计算，用于切换到特征页时懒加载；
        load_audio / 上一首 / 下一首 不再主动调用它。
        """
        self.feat_plot.clear()
        self.feat_plot.addLegend()

        try:
            self.statusBar().showMessage("Computing short-time features...", 0)
            QApplication.processEvents()
        except Exception:
            pass

        times, feat = self.compute_short_time_features()
        if times.size == 0 or not feat:
            return

        # 画最多 5 items
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
        self.features_dirty = False
        try:
            self.statusBar().showMessage("Short-time features ready.", 2000)
        except Exception:
            pass

    def _get_palette_256(self, name: str) -> np.ndarray:
        """Return (256,3) float32 RGB palette, delegating to spectrogram module."""
        from respanno.dsp.spectrogram import get_palette_256

        return get_palette_256(name)

    def _colorize_spec_with_window(self, Z2d: np.ndarray) -> np.ndarray:
        """Map 2-D spectrogram to uint8 RGB, delegating to spectrogram module."""
        from respanno.dsp.spectrogram import colorize_spectrogram

        return colorize_spectrogram(
            Z2d, cmap=self.stft_cmap, vmin=self.stft_vmin, vmax=self.stft_vmax,
        )

    def _on_cmap_changed(self, text: str):
        """配色切换：不重算 STFT，只对显示用谱图重着色。"""
        self.stft_cmap = text
        if self._last_spec_vals is not None and hasattr(self, "spec_img"):
            spec_disp = self._decimate_spec_for_display(self._last_spec_vals)
            disp = spec_disp.T  # 与主图方向一致：time × freq
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
        当 STFT / Short-Time Features的窗长、步长等参数发生改变时调用：
        - 清空帧级特征缓存
        - 清空已训练的模型
        这样下次训练 / 构造标签时会自动使用新的参数重新计算。
        """
        # 清Short-Time Features缓存
        self.stft_frame_times = None
        self.stft_features = None
        self.stft_feature_names = None

        # 已训练的Machine Learning模型不再有效，直接全部丢弃
        if hasattr(self, "ml_models"):
            self.ml_models.clear()

    def _iter_manual_annotations(self):
        """
        统一遍历“可用于训练/视为已审阅”的标注区间：
        - 兼容 (start, end, label) 和 (start, end, label, source) 两种形式
        - 三元组一律视为人工标注 (source='manual')
        - 四元组根据 source 是否属于“已审阅集合”决定

        说明：本函数名为了兼容旧代码保留为 _iter_manual_annotations，
        但其语义已升级为“reviewed annotations”。
        """

        def _norm_src(x):
            try:
                return str(x).strip().lower()
            except Exception:
                return ""

        # 先搭“source 状态机”的基础：后续会逐步引入更多状态。
        # reviewed: 参与“已审阅前缀”统计；trainable: 可进入下一轮训练的正样本标注。
        reviewed_sources = {
            "manual",
            "auto_accepted",
            "auto_edited",
            "merged",
            "merged_thresh_ctx",
        }
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

            src_n = _norm_src(src)
            if src_n in reviewed_sources:
                yield float(s), float(e), str(t)

    def get_manual_segments_for_label(self, label):
        """
        返回当前Label label 的所有“已审阅/可训练 (正样本)”区间 [(s, e), ...]。
        - 三元组默认视为人工标注 (视为已审阅)
        - 四元组若 source 属于 reviewed_sources，则计入
        """
        segs = []
        for s, e, t in self._iter_manual_annotations():
            if t == label:
                segs.append((s, e))
        return segs

    def get_reviewed_prefix(self):
        """
        返回“已审阅前缀”的长度 T (秒)：
        - 遍历所有“已审阅标注” (包括三元组和 reviewed_sources 中的四元组)
        - 取它们 end 的最大值
        """
        T = 0.0
        for s, e, _ in self._iter_manual_annotations():
            T = max(T, e)
        return float(T)

    def build_frame_labels_for_tag(self, label, neg_margin=0.05):
        """
        针对单一Label label (例如 'Wheeze')构建帧级标签向量 y。

        返回:
            y: shape (T,), 值 ∈ {1, 0, -1}
               1: 该帧是 label 的正样本 (整段 [s, e] 全部算正样本)
               0: 安全负样本 (前缀内、且距离任意正样本段 >= neg_margin)
              -1: 忽略 (不参与训练)
        """
        # 确保已经有帧级特征和Time轴
        self.ensure_frame_features()
        times = self.stft_frame_times

        # 没有特征，直接放弃
        if times is None or len(times) == 0:
            return None

        times = np.asarray(times, dtype=float)

        # 已审阅前缀 (只用前缀内的帧参与训练)
        T_used = self.get_reviewed_prefix()
        if T_used is None or T_used <= 0:
            return None

        # 先全部标为 -1 (忽略)
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

        # ---------- 3.5) Machine Learning硬负样本：由“删除/纠正”产生，仅对被删除的标签生效 ----------
        # 规则：只覆盖 -1/0，不覆盖正样本(1)；仅对前缀内帧生效。
        try:
            neg_list = getattr(self, "neg_segments", {}).get(label, [])
        except Exception:
            neg_list = []
        if neg_list:
            for it in neg_list:
                try:
                    s, e = float(it[0]), float(it[1])
                except Exception:
                    continue
                if e <= s:
                    continue
                idx = np.where(mask_prefix & (times >= s) & (times <= e) & (y != 1))[0]
                y[idx] = 0

        # 若前缀内既没有 1 也没有 0，说明当前无法用于训练，返回 None
        if not np.any(y == 1) and not np.any(y == 0):
            return None

        return y

    def clear_ml_annotations_for_label(self, label):
        return self.ml_service.clear_ml_annotations_for_label(**{k: v for k, v in locals().items() if k != 'self'})
    def train_model_for_label(self,
                              label,
                              min_pos_frames=20,
                              neg_pos_ratio=5,
                              random_state=None):
        return self.ml_service.train_model_for_label(**{k: v for k, v in locals().items() if k != 'self'})
    def apply_model_for_label_on_unreviewed(self,
                                            label,
                                            min_dur_sec=0.05):
        return self.ml_service.apply_model_for_label_on_unreviewed(**{k: v for k, v in locals().items() if k != 'self'})


    def eventFilter(self, obj, event):
        """应用级兜底：强制捕获 Ctrl+Z 触发标注撤销 (仅当撤销栈非空时)。
        说明：我们把 QShortcut/QAction 都挂上了，但在部分焦点/GraphicsView 场景下仍可能不触发；
        这里作为最后保险，不改变其它行为 (撤销栈为空时放行给控件自身处理)。
        """
        try:
            et = event.type()
            if et in (QEvent.KeyPress, QEvent.ShortcutOverride):
                key = event.key() if hasattr(event, "key") else None
                mods = event.modifiers() if hasattr(event, "modifiers") else Qt.NoModifier

                # Ctrl+Z
                if (mods & Qt.ControlModifier) and key == Qt.Key_Z:
                    if getattr(self, "_undo_stack", None):
                        self.undo_last_action()
                        try:
                            event.accept()
                        except Exception:
                            pass
                        return True

                # 编辑模式键盘：Enter 提交 / Esc 取消
                cur = getattr(self, "_editing_span", None)
                if cur is not None:
                    if key in (Qt.Key_Return, Qt.Key_Enter):
                        try:
                            cur.exit_edit_mode(commit=True)
                        except Exception:
                            pass
                        try:
                            event.accept()
                        except Exception:
                            pass
                        return True
                    if key == Qt.Key_Escape:
                        try:
                            cur.exit_edit_mode(commit=False)
                        except Exception:
                            pass
                        try:
                            event.accept()
                        except Exception:
                            pass
                        return True
        except Exception:
            pass

        return super().eventFilter(obj, event)


if __name__ == '__main__':
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    app = QApplication(sys.argv)
    viewer = AudioViewer()
    viewer.show()
    sys.exit(app.exec_())
