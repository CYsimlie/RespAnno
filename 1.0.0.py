import os
import sys

# Construct the absolute path to the Qt plugins directory (compatible with different PyQt5 installation layouts)
base_path = os.path.join(sys.base_prefix, "Lib", "site-packages", "PyQt5", "Qt5", "plugins")

# If the Qt5-level directory does not exist, fall back to the path without the Qt5 level
if not os.path.exists(base_path):
    base_path = os.path.join(sys.base_prefix, "Lib", "site-packages", "PyQt5", "Qt", "plugins")

# Set the QT plugin environment variable to ensure the Qt runtime can locate platform plugins
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = base_path

import time
# sounddevice imported lazily via _sd() — not available in headless CI
import librosa
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog,
    QVBoxLayout, QHBoxLayout, QWidget, QLabel, QDialog, QFormLayout,
    QInputDialog, QMessageBox, QComboBox, QLineEdit, QDialogButtonBox,
    QPushButton, QSplitter, QStackedWidget,
)
from PyQt5.QtWidgets import QMenu, QAction, QToolBar, QShortcut
from PyQt5.QtCore import Qt, QTimer, QEvent
import pyqtgraph as pg
from PyQt5.QtGui import QKeySequence, QColor, QImage

import numpy as np


from respanno.gui.dialogs.annotation_label_dialog import AnnotationLabelDialog  # noqa: F401
from respanno.gui.widgets.color_bar import ColorBarWidget  # noqa: F401
from respanno.gui.dialogs.loop_player import LoopPlayer  # noqa: F401
from respanno.gui.dialogs.settings_dialog import SettingsDialog  # noqa: F401
from respanno.gui.spans.span_label_item import SpanLabelItem  # noqa: F401
from respanno.gui.spans.box_span import BoxSpan  # noqa: F401
from respanno.gui.views.annot_view_box import AnnotViewBox  # noqa: F401
from respanno.gui.views.wave_view_box import WaveViewBox  # noqa: F401
from respanno.gui.widgets.clickable_slider import ClickableSlider  # noqa: F401
from respanno.ml.service import MLService  # noqa: F401
from respanno.ml.negatives import NegSampleManager  # noqa: F401


_sd = None


def _get_sd():
    """Lazy import sounddevice — PortAudio may be absent in headless CI."""
    global _sd
    if _sd is None:
        import sounddevice as _sd
    return _sd


class AudioViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Time-Frequency Analysis and Annotation")
        self.resize(1200, 800)

        self.n_fft = 256
        self.hop_length = 64
        self.f_max = 2000
        self.sr = None
        self.audio = None
        self.duration = 0

        self.is_playing = False
        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.update_position)

        self.annotations = []  # list of (start, end, label, source)

        # Machine-learning hard-negative sample manager (training only; not exported; unrelated to annotation display)
        self.neg_manager = NegSampleManager()

        # Undo stack (supports rollback of delete, accept, and other edit operations)
        self._undo_stack = []
        self._undo_maxlen = 100

        # Multi-lane annotation management (max 3 lanes; annotation view only)
        self.MAX_LANES = 3
        self.LANE_H = 0.35          # Single-lane height
        self.LANE_GAP = 0.25        # Lane gap
        self._lanes = [[] for _ in range(self.MAX_LANES)]    # Occupied intervals per lane [(s, e), ...]
        self._spans = []                                      # All current BoxSpan instances
        self._span2spec = {}                                  # BoxSpan to spectrogram LinearRegionItem mapping
        self._span2idx = {}                                   # BoxSpan to annotations index mapping

        # Edit mode (double-click a BoxSpan to enter)
        self._editing_span = None   # BoxSpan currently in edit mode
        self._selected_span = None  # Currently selected BoxSpan (used by Delete and other shortcuts)

        # STFT display parameters (color scheme and display window bounds)
        self.stft_cmap = "Heatmap"  # Options: "Heatmap", "Grayscale"
        self.stft_vmin = None       # None means auto-compute from the 1st-99th percentile
        self.stft_vmax = None
        self._last_spec_vals = None  # Most recent STFT raw data (freq x time), used for histogram statistics and recoloring

        # Short-time feature curve color palette (predefined, non-repeating; supports up to 10 curves)
        self.feature_palette = [
            QColor("#e41a1c"), QColor("#377eb8"), QColor("#4daf4a"),
            QColor("#984ea3"), QColor("#ff7f00"), QColor("#a65628"),
            QColor("#f781bf"), QColor("#999999"), QColor("#66c2a5"),
            QColor("#d95f02")
        ]
        self.feature_color_map = {}  # {feature_name: QColor}, assigned sequentially in selection order

        # Annotation type and color management (built-in respiratory sound labels with corresponding colors)
        self.annotation_builtin_labels = [
            ("哮鸣音", "wheeze"),
            ("爆裂音", "crackles"),
            ("摩擦音", "pleural rub"),
            ("哼鸣音", "rhonchi"),
            ("喘息音", "stridor"),
            ("语音", "speech"),
            ("咳嗽", "cough"),
            ("呼气", "expiration"),
            ("吸气", "inspiration"),
        ]

        # Fixed color mapping for built-in labels
        self.annotation_color_builtin = {
            "wheeze": QColor("#e41a1c"),       # Red
            "crackles": QColor("#377eb8"),      # Blue
            "pleural rub": QColor("#4daf4a"),   # Green
            "rhonchi": QColor("#984ea3"),        # Purple
            "stridor": QColor("#ff7f00"),        # Orange
            "speech": QColor("#a65628"),          # Brown
            "cough": QColor("#f781bf"),           # Pink
            "expiration": QColor("#999999"),      # Gray
            "inspiration": QColor("#66c2a5"),     # Teal
        }

        # Auto-coloring palette for custom labels
        self.annotation_color_palette = [
            QColor("#e41a1c"), QColor("#377eb8"), QColor("#4daf4a"),
            QColor("#984ea3"), QColor("#ff7f00"), QColor("#a65628"),
            QColor("#f781bf"), QColor("#999999"), QColor("#66c2a5"),
            QColor("#d95f02"),
        ]
        self.annotation_color_map = {}  # {label_text: QColor}
        self._annotation_color_used = 0

        # Target label for the current ML operation (assigned by the combo box in init_ml_toolbar)
        self.current_ml_label = None

        self.init_ui()
        self.audio_files = []
        self.current_file_index = -1

        self.last_export_path = None  # Last export CSV path (reused as the default directory for the next export)
        self.default_export_annotation_name = "annotations_events.csv"  # Default export filename for the current WAV

        self.showing_fft = False

        self.wave_y_range = None  # Waveform display range (ymin, ymax)

        self.last_settings_tab_index = 0  # Index of the last-opened Settings tab

        # Audio loading preprocessing parameters (4000 Hz resampling enabled by default; filtering disabled by default to preserve original audio content)
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


        # Auto-import of matching events files (toggleable)
        # Rule: look for <wav_base>_events.(csv|txt) in the same directory as the WAV file
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
        self.show_y_axis = False  # Initially hide the Y-axis

        # Default short-time feature selection (configurable in Settings)
        self.selected_features = ["短时能量", "过零率", "谱质心"]

        # Short-time feature curve cache {feature_name: pg.PlotDataItem}
        self.feature_curves = {}

        # Performance optimization: when switching files, only load waveform and STFT; FFT and short-time features are lazy-loaded on demand
        self.features_dirty = True
        self.fft_dirty = True
        self._display_wave_max_points = 50000
        self._spec_display_max_time_bins = 2500
        self._spec_display_max_freq_bins = 512

        # Machine-learning state
        self.stft_frame_times = None    # Frame timestamps, 1D array (T,)
        self.stft_features = None       # Frame features, 2D array (T, D)
        self.stft_feature_names = None  # Feature name list, list[str]
        self.ml_models = {}             # Trained models {label: {clf, threshold, feature_names, ...}}
        self.ml_service = MLService(self)  # ML training/inference dispatcher

        # ========= Play / Pause (Space) =========
        self.shortcut_play = QShortcut(QKeySequence(Qt.Key_Space), self)
        self.shortcut_play.activated.connect(self.play_pause)

        # ========= Seek backward / forward (Left / Right) =========
        self.shortcut_backward = QShortcut(QKeySequence(Qt.Key_Left), self)
        self.shortcut_backward.activated.connect(lambda: self._seek_delta(-1.0))
        self.shortcut_forward = QShortcut(QKeySequence(Qt.Key_Right), self)
        self.shortcut_forward.activated.connect(lambda: self._seek_delta(1.0))

        # ========= ML training / auto-labeling: uses the label selected in the ML combo box =========
        self.shortcut_train = QShortcut(QKeySequence("Ctrl+T"), self)
        self.shortcut_train.activated.connect(self.on_ml_train_clicked)

        self.shortcut_auto_label = QShortcut(QKeySequence("Ctrl+M"), self)
        self.shortcut_auto_label.activated.connect(self.on_ml_auto_clicked)

        # ========= Undo (Ctrl+Z) =========
        self.shortcut_undo = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.shortcut_undo.setContext(Qt.ApplicationShortcut)
        self.shortcut_undo.activated.connect(self.undo_last_action)

        # Also register the system-standard Undo shortcut (Windows/Linux: Ctrl+Z, macOS: Cmd+Z)
        self.shortcut_undo2 = QShortcut(QKeySequence.Undo, self)
        self.shortcut_undo2.setContext(Qt.ApplicationShortcut)
        self.shortcut_undo2.activated.connect(self.undo_last_action)

        # When shortcut ambiguity exists, also connect activatedAmbiguously to ensure Undo works
        try:
            self.shortcut_undo.activatedAmbiguously.connect(self.undo_last_action)
            self.shortcut_undo2.activatedAmbiguously.connect(self.undo_last_action)
        except Exception:
            pass

        # Additionally register an Undo QAction (closer to the Qt standard action system; more reliable than QShortcut in some scenarios)
        try:
            self.action_undo = QAction("Undo", self)
            self.action_undo.setShortcut(QKeySequence.Undo)
            self.action_undo.setShortcutContext(Qt.ApplicationShortcut)
            self.action_undo.triggered.connect(self.undo_last_action)
            self.addAction(self.action_undo)
        except Exception:
            self.action_undo = None

        # Application-level event filter: handles shortcuts requiring complex logic (Delete, Ctrl+A, edit mode, etc.)
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

        # Listen for mouse movement on the STFT plot; use the ViewBox rectangle to determine if the cursor is inside the plotting area
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

        self.spec_title_base = "STFT Spectrogram"  # Base title
        self.spec_stft_plot.setTitle(self.spec_title_base)
        self._stft_proxy = pg.SignalProxy(  # Listen for mouse movement on the STFT plot
            self.spec_stft_plot.scene().sigMouseMoved, rateLimit=60, slot=self._on_stft_mouse_move_title
        )

        self.annot_plot.setYRange(0, 1)
        self.annot_plot.setLimits(xMin=0, xMax=10)
        self.annot_plot.setMouseEnabled(x=False, y=False)
        self.wave_plot.setMouseEnabled(x=True, y=False)
        self.annot_plot.setLabel('bottom', 'Time', units='s')
        self.annot_plot.hideAxis('left')

        # Annotation area Y-axis is fixed at 3-lane height
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

        # Short-time features page (display only; computation logic is handled by a separate module)
        self.feat_plot = pg.PlotWidget(title="Short-Time Features (Normalized Display)")
        self.feat_plot.setMouseEnabled(x=True, y=False)
        self.feat_plot.setLabel('bottom', 'Time', units='s')
        self.feat_plot.addLegend()
        # Allow only X-axis zoom/pan; disable the Y-axis
        self.feat_plot.setMouseEnabled(x=True, y=False)
        pi = self.feat_plot.getPlotItem()

        # Hide the Y-axis (axis line, ticks, and labels are all hidden)
        pi.hideAxis('left')
        try:
            pi.hideAxis('right')
        except Exception:
            pass

        # Turn off the grid
        pi.showGrid(x=False, y=False)

        # Synchronize time-axis zoom/pan with the STFT plot
        self.feat_plot.setXLink(self.spec_stft_plot)

        # Remove extra margins to avoid whitespace after hiding the axis
        pi.layout.setContentsMargins(0, 0, 0, 0)
        self.feat_plot.setContentsMargins(0, 0, 0, 0)

        self.feat_page = QWidget()
        _feat_layout = QVBoxLayout(self.feat_page)
        _feat_layout.setContentsMargins(0, 0, 0, 0)
        _feat_layout.addWidget(self.feat_plot)

        # Page stack: STFT / FFT / Short-Time Features
        self.spec_stack = QStackedWidget()
        self.spec_stack.addWidget(self.spec_stft_plot)  # Page 0: STFT
        self.spec_stack.addWidget(self.spec_fft_plot)   # Page 1: FFT
        self.spec_stack.addWidget(self.feat_page)        # Page 2: Short-Time Features
        self.spec_stack.setCurrentWidget(self.spec_stft_plot)  # Default to STFT

        # Vertical splitter: top (page stack) | middle (waveform) | bottom (annotations)
        self.main_splitter = QSplitter(Qt.Vertical)
        self.main_splitter.addWidget(self.spec_stack)   # Top: page stack
        self.main_splitter.addWidget(self.wave_plot)    # Middle: waveform
        self.main_splitter.addWidget(self.annot_plot)   # Bottom: annotations

        # Stretch factors (adjustable according to actual needs)
        self.main_splitter.setStretchFactor(0, 3)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setStretchFactor(2, 1)

        # Prevent any panel from being collapsed entirely
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

        self.init_menu_bar()
        self.init_ml_toolbar()

    def init_menu_bar(self):
        """Create the menu bar: File / Settings / Help."""
        menu_bar = self.menuBar()

        # ===== File menu =====
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

        # ===== Settings menu =====
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
        # ===== Help menu =====
        help_menu = menu_bar.addMenu("Help")

        about_action = QAction("About", self)
        about_action.setShortcut(QKeySequence("F1"))
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

    def init_ml_toolbar(self):
        """Initialize the Machine Learning toolbar: label selection + training + auto-labeling + negative sample management."""

        toolbar = QToolBar("Machine Learning", self)
        self.addToolBar(Qt.TopToolBarArea, toolbar)

        # ===== Combo box: select the label to train/auto-label =====
        self.ml_label_combo = QComboBox(self)

        # Use the English names from the built-in label list as ML label options
        label_en_list = []
        if getattr(self, "annotation_builtin_labels", None):
            label_en_list = [en for (cn, en) in self.annotation_builtin_labels]
            for en in label_en_list:
                self.ml_label_combo.addItem(en)

        # Initialize current_ml_label (keep the existing value if it is in the list; otherwise use the first)
        if label_en_list:
            if self.current_ml_label in label_en_list:
                idx = label_en_list.index(self.current_ml_label)
                self.ml_label_combo.setCurrentIndex(idx)
            else:
                self.current_ml_label = label_en_list[0]
                self.ml_label_combo.setCurrentIndex(0)

        self.ml_label_combo.currentTextChanged.connect(self.on_ml_label_changed)

        toolbar.addWidget(QLabel("ML Label:"))
        toolbar.addWidget(self.ml_label_combo)

        # ===== Train button =====
        self.action_train_btn = QAction("Train Model", self)
        self.action_train_btn.setToolTip("Train a frame-level model for the selected label using current manual annotations")
        self.action_train_btn.triggered.connect(self.on_ml_train_clicked)
        toolbar.addAction(self.action_train_btn)

        # ===== Auto-label button =====
        self.action_auto_btn = QAction("Auto-label Unreviewed", self)
        self.action_auto_btn.setToolTip("Use the trained model to auto-label the selected label in unreviewed regions")
        self.action_auto_btn.triggered.connect(self.on_ml_auto_clicked)
        toolbar.addAction(self.action_auto_btn)

        # ===== Clear negatives button =====
        self.action_clear_neg = QAction("Clear Negatives", self)
        self.action_clear_neg.setToolTip("Remove all hard negative samples for the selected label")
        self.action_clear_neg.triggered.connect(self.on_clear_negatives_clicked)
        toolbar.addAction(self.action_clear_neg)
        self._update_neg_count_tip()

        # ===== Annotation color legend button =====
        action_legend = QAction("Annotation Legend", self)
        action_legend.setToolTip("View the mapping between annotation colors and labels in the third panel")
        action_legend.triggered.connect(self.show_annotation_legend)
        toolbar.addAction(action_legend)
        self.action_annotation_legend = action_legend

    # ===== Toolbar slot functions =====
    def _get_current_ml_label(self):
        """Return the label currently selected in the ML combo box.

        Reads directly from the widget so that Ctrl+T / Ctrl+M always use
        the label the user is seeing, regardless of signal timing.
        """
        combo = getattr(self, "ml_label_combo", None)
        if combo is None:
            return None
        text = combo.currentText()
        if not text:
            return None
        # Keep the stored attribute in sync for other consumers
        self.current_ml_label = text
        return text

    def on_ml_label_changed(self, text):
        """Update the current ML operation target when the combo-box label changes."""
        self.current_ml_label = text
        self._update_neg_count_tip()

    def on_ml_train_clicked(self):
        """Train button: train a frame-level model for the currently selected label."""
        label = self._get_current_ml_label()
        if not label:
            QMessageBox.information(self, "Machine Learning", "Please select a label in the toolbar first.")
            return
        self.train_model_for_label(label)

    def on_ml_auto_clicked(self):
        """Auto-label button: use the trained model to auto-label unreviewed regions for the current label."""
        label = self._get_current_ml_label()
        if not label:
            QMessageBox.information(self, "Auto Annotation", "Please select a label in the toolbar first.")
            return
        self.apply_model_for_label_on_unreviewed(label)

    def on_clear_negatives_clicked(self):
        """Clear all hard negative samples for the current label."""
        label = self._get_current_ml_label()
        if not label:
            QMessageBox.information(self, "Clear Negatives", "Please select a label in the toolbar first.")
            return
        cnt = self.neg_manager.count(label)
        if cnt == 0:
            QMessageBox.information(self, "Clear Negatives",
                                    f"No hard negative samples for label '{label}'.")
            return
        reply = QMessageBox.question(
            self, "Clear Negatives",
            f"Are you sure you want to clear all {cnt} hard negative samples "
            f"for label '{label}'?\n\n"
            f"These samples were collected from deleted/corrected annotations "
            f"and help the model avoid false positives.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.neg_manager.clear(label)
            self._update_neg_count_tip()
            QMessageBox.information(self, "Clear Negatives",
                                    f"Cleared {cnt} negative sample(s) for label '{label}'.")

    def _update_neg_count_tip(self):
        """Update the Clear Negatives button tooltip to show the current label's negative sample count."""
        label = self._get_current_ml_label()
        cnt = self.neg_manager.count(label) if label else 0
        act = getattr(self, "action_clear_neg", None)
        if act is not None:
            if cnt > 0:
                act.setToolTip(f"Clear {cnt} hard negative sample(s) for '{label}' "
                               f"(collected from deleted annotations)")
            else:
                act.setToolTip(f"No hard negative samples for '{label}'")

    def show_annotation_legend(self):
        """Generate the annotation color legend from built-in labels and color mappings."""

        def _color_for_label_direct(label):
            """Prefer the built-in color mapping; fall back to case-insensitive matching when casing differs."""
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

        # Generate the legend directly from the preset label order in the code, without depending on whether annotations currently exist.
        for cn, en in getattr(self, "annotation_builtin_labels", []):
            lab = str(en).strip()
            if not lab or lab.lower() in seen:
                continue
            seen.add(lab.lower())
            legend_items.append((str(cn), lab, _color_for_label_direct(lab)))

        # Supplement with any Custom labels that appear in the current file.
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
        """Load a single WAV file.

        Design principles:
        1) Resample to 4000 Hz by default as the analysis base sample rate;
        2) When switching files, perform only essential operations: read audio, draw decimated waveform, draw STFT, import labels;
        3) FFT and short-time features are not pre-computed; they are lazy-loaded when switching to the corresponding page to improve switching speed.
        """
        # 0. 0. Stop playback before switching files to prevent the old audio from still occupying the sound card.
        try:
            _get_sd().stop()
            self.timer.stop()
            self.is_playing = False
            if hasattr(self, "btn_play"):
                self.btn_play.setText("Play")
        except Exception:
            pass

        # 1. If no path is provided, show the file selection dialog
        if not path:
            selected, _ = QFileDialog.getOpenFileName(
                self, "Select WAV File", "", "WAV Files (*.wav)"
            )
            if not selected or not isinstance(selected, str):
                return
            path = selected

        # 2. 2. Build a WAV-file list from the same directory for previous/next navigation.
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

        # 3. 3. Read the audio. Use the resample rate from Settings by default; preserve the original sample rate when resampling is disabled.
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

        # 4. 4. When switching files, clear old caches and old annotations first to avoid reusing stale plots/features.
        self.stft_frame_times = None
        self.stft_features = None
        self.stft_feature_names = None
        self.features_dirty = True
        self.fft_dirty = True
        self.feature_curves = {}
        self._last_spec_vals = None
        # Reset auto-levels so the previous file does not affect the current one.
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

        # Clear old annotations first, then redraw new plots to prevent residual old STFT highlights.
        self.clear_annotations()

        # 5. 5. Show only essentials: decimated waveform + STFT. FFT and features are deferred until the user switches pages.
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

        # 6. 6. Auto-import labels. Label import is lightweight, so it stays inside load_audio.
        try:
            if getattr(self, "auto_import_events_enabled", False):
                self._auto_import_events_for_wav(path)
        except Exception:
            pass

        # 7. 7. Coordinate ranges and page state.
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

        # Save full spectrogram values for the Settings histogram; the display spectrogram may be decimated.
        self._last_spec_vals = spec.copy()
        spec_disp = self._decimate_spec_for_display(spec)

        disp = spec_disp.T  # time × freq
        rgb = self._colorize_spec_with_window(disp)

        self.spec_img.resetTransform()
        self.spec_img.setImage(rgb, autoLevels=False)
        # The decimated image still maps to the full audio duration, ensuring consistent annotation time axes.
        self.spec_img.setRect(pg.QtCore.QRectF(0, 0, float(self.duration), float(f_max)))

        self.spec_stft_plot.setLabel('bottom', 'Time', units='s')
        self.spec_stft_plot.setLimits(xMin=0, xMax=self.duration, yMin=0, yMax=f_max)
        self.spec_stft_plot.setXRange(0, self.duration, padding=0)

        self.wave_plot.setLimits(xMin=0, xMax=self.duration)
        self.annot_plot.setLimits(xMin=0, xMax=self.duration)
        self.wave_plot.setXRange(0, self.duration, padding=0)
        self.annot_plot.setXRange(0, self.duration, padding=0)

        # Changes to STFT/f_max affect frequency-domain features, but features are not recomputed immediately during STFT refresh.
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
        """Clear all annotations: remove visual objects, clear data structures, reset view ranges."""
        # 1) 1) Remove visual objects -- BoxSpan instances and their text labels
        for sp in list(self._spans):
            try:
                sp.cleanup()  # Remove itself and its text label from annot_plot
            except:
                pass
        self._spans.clear()

        # 2) 2) Remove highlight regions from the STFT spectrogram
        for reg in list(self._span2spec.values()):
            try:
                self.spec_stft_plot.removeItem(reg)
            except:
                pass
        self._span2spec.clear()
        self._span2idx.clear()

        # 3) 3) Backward compatibility: clear regions and text stored in annotation_items
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

        # 4) 4) Clear temporary selection regions from an in-progress drag
        if getattr(self, "temp_region", None):
            try:
                self.annot_plot.removeItem(self.temp_region)
            except:
                pass
            self.temp_region = None

        # 5) 5) Clear data structures
        self.annotations.clear()
        try:
            self.neg_manager.clear_all()
        except Exception:
            pass

        try:
            self._undo_stack.clear()
        except Exception:
            self._undo_stack = []
        self._lanes = [[] for _ in range(self.MAX_LANES)]

        # 6) 6) Reset the annotation view range
        H = self.MAX_LANES * (self.LANE_H + self.LANE_GAP)
        self.annot_plot.setYRange(0, H)
        self.annot_plot.setLimits(yMin=0, yMax=H, xMin=0, xMax=self.duration)
        self.annot_plot.setXRange(0, self.duration)

        # 7) 7) Cancel waveform highlight
        self.plot_waveform_highlight(None, None)

    def play_pause(self):
        if self.audio is None:
            return
        if not self.is_playing:
            pos = self.slider.value() / 1000.0
            start_sample = int(pos * self.sr)
            _get_sd().stop()
            _get_sd().play(self.audio[start_sample:], self.sr)
            self.start_time = time.time() - pos
            self.timer.start()
            self.btn_play.setText("Pause")
            self.is_playing = True
        else:
            _get_sd().stop()
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
            _get_sd().stop()

    def seek(self):
        self.time_label.setText(f"{self.slider.value() / 1000.0:.3f} s")
        pos_sec = self.slider.value() / 1000.0
        self.time_line_spec.setPos(pos_sec)
        self.time_line_wave.setPos(pos_sec)
        if self.is_playing:
            _get_sd().stop()
            _get_sd().play(self.audio[int(pos_sec * self.sr):], self.sr)
            self.start_time = time.time() - pos_sec
            self.timer.start()

    def _seek_delta(self, delta_sec):
        """Seek backward/forward by delta_sec seconds (called by the Left/Right shortcut keys)."""
        if self.audio is None or self.duration <= 0:
            return
        pos = self.slider.value() / 1000.0 + delta_sec
        pos = max(0.0, min(pos, self.duration))
        self.slider.setValue(int(pos * 1000))
        self.seek()

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
            # New: pass STFT-related display data in
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

        dlg.set_current_tab(self.last_settings_tab_index)  # 💡 Last-used Settings tab

        if dlg.exec_():
            self.n_fft, self.hop_length, self.f_max, self.wave_y_range = dlg.get_values()
            self.selected_features = dlg.get_selected_features()  # New
            self.last_settings_tab_index = dlg.tabs.currentIndex()

            # New: save "load-time preprocessing" and "auto label import" parameters. Do not immediately reload the current audio to avoid disrupting the current editing state; takes effect on the next load_audio call.
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

            # Retrieve STFT Display settings (color scheme + bounds); save before redrawing to avoid using old color levels.
            try:
                cmap, vmin, vmax = dlg.get_stft_display_settings()
                self.stft_cmap = cmap
                self.stft_vmin = vmin
                self.stft_vmax = vmax
            except Exception:
                pass

            if self.audio is not None:
                # After Settings changes, only refresh the waveform/STFT and mark FFT/features as dirty; do not recompute short-time features immediately.
                self.draw_waveform()
                self.update_spectrogram()
                self.features_dirty = True
                self.fft_dirty = True

            # If only the color levels were changed without recomputing, simply recolor the current spectrogram in-place.
            if self._last_spec_vals is not None:
                spec_disp = self._decimate_spec_for_display(self._last_spec_vals)
                disp = spec_disp.T
                rgb = self._colorize_spec_with_window(disp)
                self.spec_img.setImage(rgb, autoLevels=False)

    def _update_default_export_annotation_name(self, wav_path=None):
        """Update the default export filename in real time based on the current WAV file.

        Only the filename is updated; the directory is not determined here.
        On export, the directory defaults to the last export directory.
        Naming convention: <current_wav_basename_without_extension>_events.csv
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
        """Generate the default path for the export dialog: directory follows the last export, filename follows the current WAV."""
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
        """Export annotation data to a CSV file."""
        # Filter out None entries; support 3-/4-tuple formats; exclude annotations whose source is 'archived' (archived/hidden)
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

        # Default path: directory reuses the last export location; filename follows the current WAV
        # Example: current audio 1008.wav -> default filename 1008_events.csv
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

            # === Verification: print exported source-distribution statistics (viewable in the console)===
            try:
                mem_src = {}
                for d in ann_dicts:
                    mem_src[d["source"]] = mem_src.get(d["source"], 0) + 1
                print("[VERIFY][EXPORT] sources in memory:", mem_src)
            except Exception as _e:
                print("[VERIFY][EXPORT] source check failed:", _e)

            QMessageBox.information(self, "Export Successful", f"Annotations have been saved to:\n{path}")
            self.last_export_path = path  # Remember the path
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
        """Delete an annotation.

        - When interactively deleting a BoxSpan: by default add the segment to neg_segments
          (hard negative samples for training only), and support Ctrl+Z undo.
        - Internal cleanup (e.g. clearing ML predictions) can pass record_negative=False,
          push_undo=False.
        """
        # Remove the BoxSpan object
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

            # Record a hard negative sample (using the deleted annotation's original label, not the current training label)
            neg_item = None
            if record_negative and old_item is not None:
                try:
                    s, e, lab = float(old_item[0]), float(old_item[1]), str(old_item[2])
                    neg_item = self.neg_manager.add(lab, s, e)
                except Exception:
                    neg_item = None

            # Remove UI components
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

            # Record undo information
            if push_undo and old_item is not None and idx is not None:
                self._push_undo({
                    "op": "delete",
                    "idx": int(idx),
                    "item": old_item,
                    "neg_item": neg_item,
                })
            return

        # Backward compatibility: old LinearRegionItem deletion path (if still present in your code)
        # If the input is an STFT highlight region (LinearRegionItem), attempt to reverse-lookup the corresponding BoxSpan,
        # and route through the unified BoxSpan deletion logic to support hard-negative samples and Ctrl+Z undo.
        region_item = target
        try:
            for _sp, _reg in list(getattr(self, "_span2spec", {}).items()):
                if _reg is region_item:
                    # Reuse the BoxSpan deletion logic
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
        """Rebuild the BoxSpan visualization from annotations[idx] (for undo only); does not modify annotations."""
        # If a span with the same index already exists (should not happen in theory), return immediately to prevent duplication
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

        try:
            sp = self._render_annotation_span(start, end, text, source, idx=int(idx))
        except Exception:
            return
        if sp is None:
            return



    def _render_annotation_span(self, start, end, text, source, idx=None):
        """Create and render a BoxSpan plus STFT highlight region for an annotation.

        Shared by ``finalize_annotation`` and ``_restore_span_from_annotation``,
        eliminating duplicated lane-assignment / color / pen / highlight logic.
        """
        # Lane assignment
        lane = self._pick_lane(start, end)
        y0 = lane * (self.LANE_H + self.LANE_GAP)

        # Color and pen
        color = self.get_annotation_color(text)
        try:
            pen = pg.mkPen(color, width=1)
            span_brush_color = QColor(color)
            span_brush_color.setAlpha(80)
            brush = pg.mkBrush(span_brush_color)
        except Exception:
            pen = pg.mkPen(255, 255, 255, 255, width=1)
            brush = pg.mkBrush(255, 255, 255, 255)

        # Non-manual source annotations use red text
        label_color = None if source == "manual" else QColor(255, 0, 0)

        # BoxSpan
        sp = BoxSpan(start, end, y0, self.LANE_H, text, self, label_color=label_color)
        sp.setPen(pen)
        try:
            sp.setBrush(brush)
        except Exception:
            pass
        self.annot_plot.addItem(sp)
        self._spans.append(sp)
        sp.lane = lane
        if idx is not None:
            self._span2idx[sp] = int(idx)

        # STFT highlight
        try:
            spec_color = QColor(color)
            spec_color.setAlpha(50)
            spec_brush = pg.mkBrush(spec_color)
        except Exception:
            spec_brush = pg.mkBrush(255, 0, 0, 50)

        spec = pg.LinearRegionItem([start, end], brush=spec_brush)
        spec.setMovable(False)
        spec.setZValue(1)
        _xmax = float(getattr(self, "duration", 0.0) or max(end, start, 0.0))
        spec.setBounds([0.0, _xmax])
        self.spec_stft_plot.addItem(spec)
        self._span2spec[sp] = spec

        return sp

    def _clear_annotation_view_only(self):
        """Clear only visual objects (BoxSpan, STFT highlights, etc.); do not clear annotations/neg/undo data.

        Used as a rebuild fallback when the view and data are out of sync.
        """
        # 1) 1) BoxSpan instances and their text labels
        try:
            for sp in list(getattr(self, "_spans", [])):
                try:
                    sp.cleanup()
                except Exception:
                    # Fallback: at least remove it from the plot
                    try:
                        self.annot_plot.removeItem(sp)
                    except Exception:
                        pass
            getattr(self, "_spans", []).clear()
        except Exception:
            pass

        # 2) STFT highlight regions
        try:
            for reg in list(getattr(self, "_span2spec", {}).values()):
                try:
                    self.spec_stft_plot.removeItem(reg)
                except Exception:
                    pass
            getattr(self, "_span2spec", {}).clear()
        except Exception:
            pass

        # 3) Index mapping
        try:
            getattr(self, "_span2idx", {}).clear()
        except Exception:
            pass

        # 4) Backward compatibility with legacy format remnants
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

        # 5) 5) Temporary drag-selection region
        if getattr(self, "temp_region", None) is not None:
            try:
                self.annot_plot.removeItem(self.temp_region)
            except Exception:
                pass
            self.temp_region = None

        # 6) Reset lane cache (does not affect data)
        try:
            self._lanes = [[] for _ in range(self.MAX_LANES)]
        except Exception:
            pass

    def rebuild_annotation_view_from_data(self):
        """Rebuild all BoxSpan visualizations from self.annotations (does not modify annotation data)."""
        # Clear the view
        self._clear_annotation_view_only()

        # Reset the Y-axis display range (to avoid invisibility after rendering)
        try:
            H = self.MAX_LANES * (self.LANE_H + self.LANE_GAP)
            self.annot_plot.setYRange(0, H)
            self.annot_plot.setLimits(yMin=0, yMax=H, xMin=0, xMax=getattr(self, "duration", 0.0))
        except Exception:
            pass

        # Rebuild
        try:
            for idx, item in enumerate(list(getattr(self, "annotations", []))):
                if item is None:
                    continue
                try:
                    # archived spans are neither displayed nor exported (old spans before merge/delete)
                    if len(item) >= 4 and str(item[3]) == "archived":
                        continue
                except Exception:
                    pass
                self._restore_span_from_annotation(int(idx), item)
        except Exception:
            # Avoid interrupting the main flow with exceptions
            pass


    def set_annotation_source(self, target, new_source: str, push_undo=True):
        """Change the source of a specified annotation to new_source, sync the UI, and support undo."""
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

    # ---- undo dispatcher -------------------------------------------------
    def undo_last_action(self):
        """Undo the last edit operation (delete, source change, geometry edit)."""
        try:
            if not self._undo_stack:
                return
            rec = self._undo_stack.pop()
        except Exception:
            return

        op = rec.get("op")
        if op == "delete":
            self._undo_delete(rec)
        elif op == "edit_interval":
            self._undo_edit_interval(rec)
        elif op == "set_source":
            self._undo_set_source(rec)

    def _undo_delete(self, rec):
        """Undo a single deletion: restore the annotations entry + rebuild visualization + remove the negative sample."""
        idx = rec.get("idx")
        item = rec.get("item")
        neg_item = rec.get("neg_item")
        if idx is None or item is None:
            return
        try:
            if 0 <= int(idx) < len(self.annotations):
                self.annotations[int(idx)] = item
                self._restore_span_from_annotation(int(idx), item)
                # Fallback: if restoration fails, rebuild the entire view
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
                self.neg_manager.remove(lab, neg_id)
            except Exception:
                pass

    def _undo_edit_interval(self, rec):
        """Undo a single geometry edit: roll back position, text, style, and highlight."""
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

        if sp is None:
            # If the span is lost, rebuild the entire view
            try:
                self.rebuild_annotation_view_from_data()
            except Exception:
                pass
            return

        try:
            s0, s1 = float(old_item[0]), float(old_item[1])
            lab = str(old_item[2])
            src0 = str(old_item[3]) if len(old_item) >= 4 else "manual"

            sp.setPos([s0, sp.y_base], update=False)
            sp.setSize([s1 - s0, sp.h_fix], update=False)
            sp.text = lab
            try:
                sp._update_label_html()
            except Exception:
                pass
            try:
                sp._apply_visual_style(src0)
            except Exception:
                pass

            # Sync STFT highlight
            try:
                reg = getattr(self, "_span2spec", {}).get(sp)
                if reg is not None:
                    reg.setRegion([s0, s1])
            except Exception:
                pass

            try:
                sp._on_changed()
            except Exception:
                pass
        except Exception:
            # Fallback: if rollback fails, rebuild the view
            try:
                self.rebuild_annotation_view_from_data()
            except Exception:
                pass

    def _undo_set_source(self, rec):
        """Undo a source change: restore the old source and redraw the visual style."""
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

    def accept_annotation(self, target, accepted_source: str = "auto_accepted"):
        """Mark a machine annotation as accepted; default source=auto_accepted; undoable."""
        return self.set_annotation_source(target, accepted_source, push_undo=True)

    # ==========================
    # BoxSpan edit mode: double-click to enter; Enter to commit; Esc to cancel
    # ==========================
    def begin_edit_span(self, span):
        """Record the BoxSpan currently entering edit mode (if another span is already in edit mode, commit and exit it first)."""
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
        """Exit edit mode and clear the editing state."""
        try:
            if getattr(self, "_editing_span", None) is span:
                self._editing_span = None
        except Exception:
            self._editing_span = None

        # Clean the status bar notice after exiting edit mode
        try:
            self.statusBar().showMessage("", 1000)
        except Exception:
            pass

    def _update_edit_status(self, span):
        """Edit mode: display the start and end times of the currently selected span in the status bar."""
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
        """Return a stable color for a label text: built-in labels use fixed colors; custom labels use auto-assigned colors."""
        if not label_text:
            return QColor(255, 255, 255)

        # Return directly if already mapped
        if label_text in self.annotation_color_map:
            return self.annotation_color_map[label_text]

        # Preset type (English name)
        if label_text in self.annotation_color_builtin:
            color = self.annotation_color_builtin[label_text]
            self.annotation_color_map[label_text] = color
            return color

        # Any other text: pick a color sequentially from the palette
        palette = self.annotation_color_palette
        idx = self._annotation_color_used % len(palette)
        color = palette[idx]
        self._annotation_color_used += 1
        self.annotation_color_map[label_text] = color
        return color

    def finalize_annotation(self, start, end, text=None, source="manual"):
        # If no text is provided (normal interactive annotation), show the dialog with preset types
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

        # —— Render annotation visualization + spectrogram highlight ——
        span = self._render_annotation_span(start, end, text, source)
        if span is None:
            return

        # —— Register into the annotations data store ——
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
    # Auto-import matching _events annotations (toggleable)
    # =========================

    def _get_events_indexer(self):
        """Lazy-initialize the EventsFileIndexer."""
        if not hasattr(self, "_events_indexer"):
            from respanno.labels.events_importer import EventsFileIndexer

            self._events_indexer = EventsFileIndexer(self)
        return self._events_indexer


    def toggle_auto_import_events(self, checked: bool):
        """Toggle auto-import of matching _events files, delegating to EventsFileIndexer."""
        self.auto_import_events_enabled = bool(checked)

        if self.auto_import_events_enabled:
            wav_path = getattr(self, "loaded_filename", None)
            if wav_path and isinstance(wav_path, str):
                folder = os.path.dirname(os.path.abspath(wav_path))
                self._get_events_indexer().build_index(folder)

        try:
            msg = "Auto-import matching _events annotations: enabled" if self.auto_import_events_enabled else "Auto-import matching _events annotations: disabled"
            self.statusBar().showMessage(msg, 2000)
        except Exception:
            pass



    def _get_auto_label_import_settings(self):
        """Return merged auto-import settings, delegating to EventsFileIndexer."""
        return self._get_events_indexer()._get_settings()



    def _auto_label_candidate_extensions(self):
        """Return candidate file extensions, delegating to EventsFileIndexer."""
        return self._get_events_indexer()._candidate_extensions()



    def _prepare_events_index(self, folder: str):
        """Build events file index, delegating to EventsFileIndexer."""
        self._get_events_indexer().build_index(folder)



    def _resolve_events_path_for_wav(self, wav_path: str):
        """Resolve matching events file path, delegating to EventsFileIndexer."""
        return self._get_events_indexer().resolve_path(wav_path)



    def _parse_events_file(self, events_path: str):
        """Parse events file, delegating to EventsFileIndexer."""
        return self._get_events_indexer().parse_file(events_path)




    def _parse_events_file_cached(self, events_path: str):
        """Parse events file with caching, delegating to EventsFileIndexer."""
        return self._get_events_indexer().parse_file_cached(events_path)



    def _auto_import_events_for_wav(self, wav_path: str):
        """Auto-import matching events file, delegating to EventsFileIndexer."""
        self._get_events_indexer().auto_import(wav_path)


    def toggle_analysis_mode(self):
        cur = self.spec_stack.currentWidget()
        if cur is self.spec_stft_plot:
            # STFT -> FFT: trigger lazy loading
            self.spec_stack.setCurrentWidget(self.spec_fft_plot)
            self.freq_button.setText("Switch: Short-Time Features")
            if getattr(self, "fft_dirty", True):
                self.show_fft()
        elif cur is self.spec_fft_plot:
            # FFT -> Short-Time Features：lazy-load short-time features
            self.spec_stack.setCurrentWidget(self.feat_page)
            self.freq_button.setText("Switch: STFT")
            if getattr(self, "features_dirty", True) or not getattr(self, "feature_curves", {}):
                self.update_features_plot()
        else:
            # Short-Time Features -> STFT
            self.spec_stack.setCurrentWidget(self.spec_stft_plot)
            self.freq_button.setText("Switch: FFT")

    def show_fft(self):
        if self.audio is None or self.sr is None:
            return

        from respanno.dsp.fft import compute_fft

        freqs, mag = compute_fft(self.audio, self.sr, max_points=50000)
        if freqs.size == 0:
            return

        self.spec_fft_plot.clear()
        self.spec_fft_plot.plot(freqs, mag, pen='c')
        self.spec_fft_plot.setLabel('bottom', 'Frequency', units='Hz')
        self.spec_fft_plot.setLabel('left', 'Amplitude')

        x_max = self.sr / 2
        y_max = float(np.max(mag)) if mag.size else 1.0
        if y_max <= 0:
            y_max = 1.0
        self.spec_fft_plot.setXRange(0, x_max, padding=0.02)
        self.spec_fft_plot.setYRange(0, y_max * 1.05)
        vb = self.spec_fft_plot.getViewBox()
        vb.setLimits(xMin=0, xMax=x_max, yMin=0, yMax=y_max * 2)
        self.fft_dirty = False

    def update_fft(self):
        if self.audio is None or self.sr is None:
            return

        from respanno.dsp.fft import compute_fft

        freqs, spectrum = compute_fft(self.audio, self.sr)

        self.fft_curve.setData(freqs, spectrum)
        self.spec_fft_plot.setLabel('left', "Amplitude", units='')
        self.spec_fft_plot.setLabel('bottom', "Frequency", units='Hz')

        self.spec_fft_plot.setXRange(freqs[0], freqs[-1], padding=0.05)
        self.spec_fft_plot.setYRange(0, np.max(spectrum) * 1.05)
        vb = self.spec_fft_plot.getViewBox()
        vb.setLimits(xMin=0, xMax=self.sr / 2, yMin=0, yMax=np.max(spectrum) * 2)

    def _overlap(self, a1, a2, b1, b2):
        # Any temporal intersection is considered overlap
        return not (a2 <= b1 or a1 >= b2)

    def _pick_lane(self, s, e):
        """Dynamically select a free lane based on the currently visible BoxSpan instances; prefer the lane with the smallest index."""
        lanes = [[] for _ in range(self.MAX_LANES)]
        gap = (self.LANE_H + self.LANE_GAP)
        # Collect the intervals of every existing lane
        for sp in list(self._spans):
            try:
                a, b = sp.interval()
                lane = int(round(sp.y_base / gap))
                if 0 <= lane < self.MAX_LANES:
                    lanes[lane].append((a, b))
            except Exception:
                pass
        # Find the first non-overlapping lane
        for i in range(self.MAX_LANES):
            if all(not self._overlap(s, e, a, b) for (a, b) in lanes[i]):
                return i
        # If all three lanes conflict, place it in the last lane
        return self.MAX_LANES - 1

    def _on_stft_mouse_move_title(self, evt):
        """Display time/frequency coordinates when the mouse moves over the STFT plot; restore the title when the mouse leaves the plotting area."""
        if self.audio is None:
            return
        pos = evt[0] if isinstance(evt, (tuple, list)) else evt

        vb = self.spec_stft_plot.getViewBox()
        # Trigger only inside the plotting area (excluding axis ticks/margins)
        if not vb.sceneBoundingRect().contains(pos):
            self.spec_stft_plot.setTitle(self.spec_title_base)
            return

        p = vb.mapSceneToView(pos)
        x = float(p.x())
        y = float(p.y())
        fmax = getattr(self, "_stft_fmax", getattr(self, "f_max", 0.0))
        dur = getattr(self, "duration", 0.0)

        if 0.0 <= x <= (dur or 0.0) and 0.0 <= y <= (fmax or 0.0):
            # Right side of titleDisplayCoordinate (adjustablestyle)
            self.spec_stft_plot.setTitle(
                f'{self.spec_title_base}  '
                f'<span style="color:#bbb; font-size:11px;">t={x:.3f}s, f={y:.1f} Hz</span>'
            )
        else:
            self.spec_stft_plot.setTitle(self.spec_title_base)

    def show_about_dialog(self):
        QMessageBox.information(self, "About",
                                "RespAnno v1.0.0\nAuthor: Chaoyue Pan\nBuilt with PyQt5 + pyqtgraph\nCurrently only .wav files are supported. Please convert other file types to .wav.")

    def compute_short_time_features(self):
        """Compute all short-time features, delegating to features module.

        Returns (times, feat_dict) as (T,) float array and {name: (T,) array}.
        """
        from respanno.dsp.features import compute_short_time_features as _cstf

        return _cstf(
            self.audio, self.sr,
            n_fft=self.n_fft, hop_length=self.hop_length, f_max=self.f_max,
        )

    def ensure_frame_features(self):
        """Recompute frame features and write to ML cache.

        Delegates computation to features module; build_feature_matrix
        handles stacking and smoothing.
        """
        times, feat_dict = self.compute_short_time_features()

        if times is None or times.size == 0 or not feat_dict:
            self.stft_frame_times = None
            self.stft_features = None
            self.stft_feature_names = None
            return

        from respanno.dsp.features import build_feature_matrix

        X_full, times_aligned, full_names = build_feature_matrix(times, feat_dict)

        self.stft_frame_times = times_aligned  # (T,)
        self.stft_features = X_full  # (T, 2D)
        self.stft_feature_names = full_names
    def update_features_plot(self):
        """Compute and plot short-time feature curves based on self.selected_features (0-1 normalized, overlaid display).

        Note: this function computes on the main thread and is only lazy-loaded when switching to
        the features page; it is not called proactively during load_audio / previous / next.
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

        # Plot at most 5 items
        from respanno.dsp.features import normalize_feature_for_display

        names = [nm for nm in self.selected_features if nm in feat][:5]
        self._assign_feature_colors(names)  # Assign colors
        for nm in names:
            y_plot = normalize_feature_for_display(np.asarray(feat[nm], dtype=float))
            color = self.feature_color_map.get(nm, QColor("#999999"))
            curve = self.feat_plot.plot(times, y_plot, name=nm, pen=pg.mkPen(color, width=2))
            self.feature_curves[nm] = curve

        # Align the time-axis range with STFT
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
        """Switch the color scheme: do not recompute the STFT; only recolor the display spectrogram."""
        self.stft_cmap = text
        if self._last_spec_vals is not None and hasattr(self, "spec_img"):
            spec_disp = self._decimate_spec_for_display(self._last_spec_vals)
            disp = spec_disp.T  # Match the main plot orientation: time x freq
            rgb = self._colorize_spec_with_window(disp)
            self.spec_img.setImage(rgb, autoLevels=False)

    def _assign_feature_colors(self, selected_names):
        """Assign a unique color to each feature curve in the order of selected_names."""
        self.feature_color_map = {}
        used = 0
        for name in selected_names[:5]:
            self.feature_color_map[name] = self.feature_palette[used % len(self.feature_palette)]
            used += 1

    def _iter_manual_annotations(self):
        """
        Unified iteration over "trainable / reviewed" annotation intervals:
        - Compatible with both (start, end, label) and (start, end, label, source) formats
        - 3-tuples are always treated as manual annotations (source='manual')
        - 4-tuples are judged based on whether source belongs to the "reviewed set"

        Note: the function name _iter_manual_annotations is retained for backward
        compatibility, but its semantics have been upgraded to "reviewed annotations".
        """

        def _norm_src(x):
            try:
                return str(x).strip().lower()
            except Exception:
                return ""

        # Lay the foundation for the "source state machine": more states will be introduced gradually in the future.
        # reviewed: participates in the "reviewed prefix" statistic; trainable: a positive-sample annotation eligible for the next training round.
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

            # Support 3-tuple / 4-tuple formats; skip other lengths
            try:
                if len(item) == 3:
                    s, e, t = item
                    src = "manual"  # 3-tuples are treated directly as manual labels
                elif len(item) >= 4:
                    s, e, t, src = item[:4]  # 4-tuples use their own source field
                else:
                    continue
            except Exception:
                continue

            src_n = _norm_src(src)
            if src_n in reviewed_sources:
                yield float(s), float(e), str(t)

    def get_manual_segments_for_label(self, label):
        """Return [(s, e), ...] for reviewed annotations matching label, delegating."""
        from respanno.ml.frame_labels import get_manual_segments

        return get_manual_segments(self.annotations, label)



    def get_reviewed_prefix(self):
        """Return max end time among reviewed annotations, delegating."""
        from respanno.ml.frame_labels import get_reviewed_prefix

        return get_reviewed_prefix(self.annotations)



    def build_frame_labels_for_tag(self, label, neg_margin=0.05):
        """Build frame-level labels, delegating to respanno.ml.frame_labels."""
        self.ensure_frame_features()
        from respanno.ml.frame_labels import build_frame_labels

        return build_frame_labels(
            self.annotations,
            self.stft_frame_times,
            label,
            neg_segments=self.neg_manager.to_dict(),
            neg_margin=neg_margin,
        )



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
        """Application-level event filter: handles shortcuts that require complex logic.

        Includes: Ctrl+Z undo (fallback), edit-mode Enter/Esc,
        Delete to remove the selected annotation, Ctrl+A to accept an ML annotation.
        """
        try:
            et = event.type()
            if et in (QEvent.KeyPress, QEvent.ShortcutOverride):
                key = event.key() if hasattr(event, "key") else None
                mods = event.modifiers() if hasattr(event, "modifiers") else Qt.NoModifier

                # Ctrl+Z undo
                if (mods & Qt.ControlModifier) and key == Qt.Key_Z:
                    if getattr(self, "_undo_stack", None):
                        self.undo_last_action()
                        try:
                            event.accept()
                        except Exception:
                            pass
                        return True

                # Ctrl+A accept the selected ML annotation
                if (mods & Qt.ControlModifier) and key == Qt.Key_A:
                    sel = getattr(self, "_selected_span", None)
                    if sel is not None:
                        try:
                            self.accept_annotation(sel)
                        except Exception:
                            pass
                        try:
                            event.accept()
                        except Exception:
                            pass
                        return True

                # Delete key removes the selected annotation
                if key in (Qt.Key_Delete, Qt.Key_Backspace):
                    sel = getattr(self, "_selected_span", None)
                    if sel is not None:
                        try:
                            self.delete_annotation(sel)
                            self._selected_span = None
                        except Exception:
                            pass
                        try:
                            event.accept()
                        except Exception:
                            pass
                        return True

                # Edit-mode keyboard: Enter to commit / Esc to cancel
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
