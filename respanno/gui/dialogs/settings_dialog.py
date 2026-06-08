"""Settings dialog with STFT / Display / Preprocessing / Auto Label Import / Features tabs.

Source: 1.0.0.py lines 79-762 (class SettingsDialog)
"""

import numpy as np

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QDialog, QTabWidget, QFormLayout, QSpinBox, QGroupBox, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QDoubleSpinBox,
    QPushButton, QSlider, QCheckBox, QLineEdit, QListWidget,
    QListWidgetItem, QDialogButtonBox,
)

from pyqtgraph import HistogramLUTWidget, ImageItem, ColorMap

from respanno.gui.widgets.color_check_delegate import ColorCheckDelegate


class SettingsDialog(QDialog):
    def __init__(self, parent=None, n_fft=256, hop_length=64, f_max=2000,
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

        # —— New：读取audio时的预handleset。default开启 4000 Hz resampling；filteringdefault关闭，避免改变旧行为。——
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

        # —— New：auto读取同名labelfile的parseset。default保持旧逻辑：auto/csv/txt、_events suffix、autodelimiter、前3列为 start/end/label。——
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

        # ==== STFT label页 ====
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

        # ==== STFT Display (直方图 + colorbar + 上lower limit + Reset Defaults)====
        stft_group = QGroupBox("STFT Display: Histogram & ColorBar (Editable)")
        stft_vbox = QVBoxLayout(stft_group)

        # 顶部：直方图 + 渐变色条 (HistogramLUTWidget)
        self.hist_widget = HistogramLUTWidget()
        self.hist_widget.setMinimumHeight(180)

        # 给 HistogramLUT 提供一份图像data，便于统计直方图
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

        # initialize colorbar Color scheme (保留可edit三角控件)
        self._base_cmap = _make_cmap(self._stft_cmap)
        self.hist_widget.gradient.setColorMap(self._base_cmap)
        # 某些 PyQt5 + 老 pyqtgraph 需要 save/restore 一次才能正确刷新
        try:
            _st = self.hist_widget.gradient.saveState()
            self.hist_widget.gradient.restoreState(_st)
        except Exception:
            pass

        # 中部：Color scheme + 上lower limitinput框
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

        # initialize上lower limit：优先用传入值，否则用 1%~99% 分位
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

            # Settings到 HistogramLUTItem：这一步决定开窗interval (与主图共享)
            self.hist_widget.setLevels(vmin, vmax)

            # synchronize到input框
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

        # Assemble into group
        stft_vbox.addWidget(QLabel("STFT value histogram (drag the colorbar handles to set limits; colorbar is editable)"))
        stft_vbox.addWidget(self.hist_widget)
        stft_vbox.addLayout(line1)  # 配色
        stft_vbox.addLayout(line2)  # 下限/上限
        stft_vbox.addLayout(btn_line)  # Reset Defaults

        # Assemble into STFT page
        try:
            stft_layout.addRow(stft_group)  # QFormLayout
        except Exception:
            stft_layout.addWidget(stft_group)  # 其他布局

        # ========= 联动逻辑：levels ↔ text框，顺带保证与主图一致 =========

        # 1)拖动 HistogramLUT items (或edit colorbar) → updateinput框
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

        # 2)改input框 → 回写到 HistogramLUT (开窗)，colorbar 的条和三角会一起动
        def _edits_to_levels():
            vmin = self.vmin_edit.value()
            vmax = self.vmax_edit.value()
            if vmax > vmin:
                self.hist_widget.setLevels(vmin, vmax)

        self.vmin_edit.valueChanged.connect(_edits_to_levels)
        self.vmax_edit.valueChanged.connect(_edits_to_levels)

        # 3)switchColor scheme → update colorbar (保留coloredit功能)
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

        # 4)Reset Defaults (上lower limit回到 1%~99% 分位)
        def _reset_defaults():
            self._stft_vmin = None
            self._stft_vmax = None
            _init_levels_from_data()

        self.btn_reset.clicked.connect(_reset_defaults)

        stft_tab.setLayout(stft_layout)
        tabs.addTab(stft_tab, "STFT")

        # ==== Displaylabel页 ====
        display_tab = QWidget()
        display_layout = QFormLayout()

        # zoom滑条 + input框
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

        # 上lower limit
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

        # Reset Defaultsbutton
        self.reset_button = QPushButton("Reset Defaults")
        self.reset_button.clicked.connect(self.restore_default_y_range)
        display_layout.addRow(self.reset_button)

        display_tab.setLayout(display_layout)
        tabs.addTab(display_tab, "Display")

        # ==== Preprocessing label页：读取 WAV 时的resampling与可选filtering ====
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

        # ==== Auto Label Import label页：configureauto读取同名labelfile的parse规则 ====
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

        # button
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(tabs)
        layout.addWidget(buttons)

        # ==== Short-Time Features label页 ====
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

            # —— frequency domain/时频统计feature (基于 STFT Amplitude谱)——
            "谱均值", "谱标准差", "谱中位数", "谱能量", "谱RMS", "谱幅和",
            "谱质心", "谱带宽", "谱偏度", "谱峰度", "谱滚降", "谱平坦度", "谱熵", "谱通量",
            "最大谱峰值", "谱峰数量",
            "低频能量占比", "中频能量占比", "高频能量占比",
            "谱四分位距", "谱MAD", "谱差分零交叉率", "谱平滑度", "主峰/次峰比", "谱复杂度",

            # —— 窄带/少crestdetect增强feature ——
            "主峰能量占比", "前三峰能量占比", "90%能量覆盖频点数", "主峰-3dB带宽", "主峰Q因子",

            # —— 频移autocorrelation (cor)feature：100–1200 Hz sub-band，基于每帧谱的“向下频移autocorrelation”曲线 ——
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

        # —— 让勾选color与曲线一致 ——
        # color分配规则与主window一致：按“已选feature”的顺序分配
        sel = list(selected_features or [])
        palette = getattr(parent, "feature_palette", [
            QColor("#e41a1c"), QColor("#377eb8"), QColor("#4daf4a"),
            QColor("#984ea3"), QColor("#ff7f00")
        ])

        # 构建一个 {feature名: QColor} 映射
        color_map = {}
        used = 0
        for nm in sel[:5]:
            color_map[nm] = palette[used % len(palette)]
            used += 1

        # 把color写到每个条目的 UserRole，并顺带给未选的分配备用color (避免后续点选color重复)
        fallback_used = used
        for i in range(self.feat_list.count()):
            it = self.feat_list.item(i)
            nm = it.text()
            if nm in color_map:
                it.setData(Qt.UserRole, color_map[nm])
            else:
                it.setData(Qt.UserRole, palette[fallback_used % len(palette)])
                fallback_used += 1

        # 预勾选 (来自主window)
        pre = set(selected_features or [])
        for i in range(self.feat_list.count()):
            it = self.feat_list.item(i)
            if it.text() in pre:
                it.setCheckState(Qt.Checked)

        def _limit_to_5(_):
            checked = [self.feat_list.item(i) for i in range(self.feat_list.count())
                       if self.feat_list.item(i).checkState() == Qt.Checked]
            if len(checked) > 5:
                # cancel最新一次勾选 (把它设回未选)
                # 注意：此槽在single item 改变时trigger，找到那个 item
                snd = feat_tab.sender()
                if isinstance(snd, QListWidget):
                    pass  # 兼容性占位
                # 简单策略：从末尾start把第6个之后的设回未选
                for it in checked[5:]:
                    it.setCheckState(Qt.Unchecked)

        self.feat_list.itemChanged.connect(_limit_to_5)

        feat_layout.addRow(QLabel("Select features to display (up to 5):"), self.feat_list)
        feat_tab.setLayout(feat_layout)
        tabs.addTab(feat_tab, "Short-Time Features")

    def on_slider_changed(self, value):
        """滑条改变时update倍数input框和上lower limit"""
        zoom = value / 100
        self.zoom_input.setValue(zoom)
        self.apply_zoom_to_range(zoom)

    def on_input_changed(self, value):
        """manualinputzoom倍数时update滑条和上lower limit"""
        self.zoom_slider.setValue(int(value * 100))
        self.apply_zoom_to_range(value)

    def apply_zoom_to_range(self, zoom):
        """根据倍数zoom原始的 min/max 并updateDisplayrange"""
        center = (self.default_ymin + self.default_ymax) / 2
        half_range = (self.default_ymax - self.default_ymin) / 2 / zoom
        new_ymin = center - half_range
        new_ymax = center + half_range
        self.ymin_box.setValue(new_ymin)
        self.ymax_box.setValue(new_ymax)

    def restore_default_y_range(self):
        """Reset Defaults的autocompute的range"""
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


