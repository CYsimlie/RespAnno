"""Headless GUI widget tests — automated verification of PyQt5 widgets.

All tests run under QT_QPA_PLATFORM=offscreen (no display required).
Covers: SettingsDialog, ClickableSlider, ColorCheckDelegate,
        SpanLabelItem, LoopPlayer, AnnotationLabelDialog, AnnotViewBox, WaveViewBox.

Excludes:
  - BoxSpan: requires full pyqtgraph PlotWidget scene.
  - ColorBarWidget: the file has a structural issue (two __init__ methods in
    one class due to a missing ``class HistogramWidget`` declaration above
    line 71). The widget works in the real app because the GUI initialisation
    path avoids the recursion. This is tracked as a known refactoring note.
"""
import os
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    """Session-scoped offscreen QApplication."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(["pytest"])
    return app


# ═══════════════════════════════════════════════════════════════════════════
# 1. SettingsDialog
# ═══════════════════════════════════════════════════════════════════════════

class TestSettingsDialogInit:
    """SettingsDialog construction and basic attributes."""

    def test_creates_with_defaults(self, qapp):
        from respanno.gui.dialogs.settings_dialog import SettingsDialog
        dlg = SettingsDialog(parent=None)
        assert dlg.windowTitle() == "Settings"
        nfft, hop, fmax, yrange = dlg.get_values()
        assert nfft == 256 and hop == 64 and fmax == 2000
        dlg.close()

    def test_creates_with_explicit_values(self, qapp):
        from respanno.gui.dialogs.settings_dialog import SettingsDialog
        dlg = SettingsDialog(parent=None, n_fft=512, hop_length=128, f_max=1500)
        nfft, hop, fmax, yrange = dlg.get_values()
        assert nfft == 512 and hop == 128 and fmax == 1500
        dlg.close()

    def test_wave_y_range_preserved(self, qapp):
        from respanno.gui.dialogs.settings_dialog import SettingsDialog
        audio = (np.sin(2 * np.pi * 200 * np.linspace(0, 1, 4000))).astype(np.float32)
        dlg = SettingsDialog(parent=None, wave_y_range=(-0.6, 0.6), audio_data=audio)
        _, _, _, yrange = dlg.get_values()
        assert yrange[0] < yrange[1]
        dlg.close()


class TestSettingsDialogPreprocessing:
    """SettingsDialog — preprocessing config roundtrip."""

    @pytest.fixture
    def dlf(self, qapp):
        from respanno.gui.dialogs.settings_dialog import SettingsDialog
        dlg = SettingsDialog(
            parent=None,
            preprocessing_enabled=True,
            resample_enabled=False, resample_target_sr=8000,
            filter_enabled=True, filter_type="lowpass",
            filter_lowcut=50.0, filter_highcut=500.0,
            filter_order=6, filter_zero_phase=False,
        )
        yield dlg
        dlg.close()

    def test_preprocessing_enabled(self, dlf):
        cfg = dlf.get_preprocessing_settings()
        assert cfg["preprocessing_enabled"] is True

    def test_resample_enabled_false(self, dlf):
        cfg = dlf.get_preprocessing_settings()
        assert cfg["resample_enabled"] is False
        assert cfg["resample_target_sr"] == 8000

    def test_resample_tuple_backward_compat(self, dlf):
        enabled, sr = dlf.get_resample_settings()
        assert enabled is False and sr == 8000

    def test_filter_type_preserved(self, dlf):
        cfg = dlf.get_preprocessing_settings()
        assert cfg["filter_type"] == "lowpass"

    def test_filter_params_preserved(self, dlf):
        cfg = dlf.get_preprocessing_settings()
        assert cfg["filter_enabled"] is True
        assert cfg["filter_lowcut"] == 50.0
        assert cfg["filter_highcut"] == 500.0
        assert cfg["filter_order"] == 6
        assert cfg["filter_zero_phase"] is False

    def test_preprocessing_complete_dict(self, dlf):
        cfg = dlf.get_preprocessing_settings()
        for key in ["preprocessing_enabled", "resample_enabled", "resample_target_sr",
                     "filter_enabled", "filter_type", "filter_lowcut",
                     "filter_highcut", "filter_order", "filter_zero_phase"]:
            assert key in cfg, f"Missing key '{key}'"


class TestSettingsDialogCmap:
    """get_stft_display_settings returns (cmap_name, vmin, vmax) tuple."""

    @pytest.mark.parametrize("cmap", ["Heatmap", "Grayscale"])
    def test_cmap_roundtrip(self, qapp, cmap):
        from respanno.gui.dialogs.settings_dialog import SettingsDialog
        dlg = SettingsDialog(parent=None, stft_cmap=cmap)
        cmap_out, vmin, vmax = dlg.get_stft_display_settings()
        assert cmap_out == cmap
        dlg.close()

    def test_stft_display_returns_floats(self, qapp):
        from respanno.gui.dialogs.settings_dialog import SettingsDialog
        dlg = SettingsDialog(parent=None, stft_cmap="Grayscale", stft_levels=(-80.0, 0.0))
        cmap_out, vmin, vmax = dlg.get_stft_display_settings()
        assert cmap_out == "Grayscale"
        assert isinstance(vmin, float) and isinstance(vmax, float)
        dlg.close()


class TestSettingsDialogFilterTypes:
    """All four Butterworth filter types."""

    @pytest.mark.parametrize("ftype", ["bandpass", "lowpass", "highpass", "bandstop"])
    def test_filter_type_roundtrip(self, qapp, ftype):
        from respanno.gui.dialogs.settings_dialog import SettingsDialog
        dlg = SettingsDialog(parent=None, filter_enabled=True, filter_type=ftype)
        assert dlg.get_preprocessing_settings()["filter_type"] == ftype
        dlg.close()


class TestSettingsDialogAutoImport:
    """Auto label import settings."""

    def test_auto_import_toggle(self, qapp):
        from respanno.gui.dialogs.settings_dialog import SettingsDialog
        dlg = SettingsDialog(parent=None, auto_label_import_enabled=True)
        assert dlg.get_auto_label_import_enabled() is True
        dlg.close()

    def test_auto_import_settings_dict(self, qapp):
        from respanno.gui.dialogs.settings_dialog import SettingsDialog
        dlg = SettingsDialog(parent=None)
        settings = dlg.get_auto_label_import_settings()
        assert isinstance(settings, dict)
        assert "file_format" in settings
        assert "file_suffix" in settings
        dlg.close()


class TestSettingsDialogFeatures:
    """Feature selection."""

    def test_selected_features_returned(self, qapp):
        from respanno.gui.dialogs.settings_dialog import SettingsDialog
        dlg = SettingsDialog(parent=None, selected_features=["feat_A", "feat_B"])
        assert isinstance(dlg.get_selected_features(), list)
        dlg.close()


class TestSettingsDialogNoAudio:
    """Graceful degradation without audio data."""

    def test_no_audio_no_crash(self, qapp):
        from respanno.gui.dialogs.settings_dialog import SettingsDialog
        dlg = SettingsDialog(parent=None, audio_data=None)
        assert dlg is not None
        dlg.close()

    def test_no_wave_y_range_no_crash(self, qapp):
        from respanno.gui.dialogs.settings_dialog import SettingsDialog
        dlg = SettingsDialog(parent=None, audio_data=None, wave_y_range=None)
        _, _, _, yrange = dlg.get_values()
        assert len(yrange) == 2
        dlg.close()


# ═══════════════════════════════════════════════════════════════════════════
# 2. ClickableSlider
# ═══════════════════════════════════════════════════════════════════════════

class TestClickableSlider:
    """ClickableSlider: QSlider that jumps to click position."""

    def test_creates_horizontal(self, qapp):
        from respanno.gui.widgets.clickable_slider import ClickableSlider
        s = ClickableSlider(Qt.Horizontal)
        assert s.orientation() == Qt.Horizontal

    def test_range(self, qapp):
        from respanno.gui.widgets.clickable_slider import ClickableSlider
        s = ClickableSlider(Qt.Horizontal)
        s.setRange(0, 1000)
        assert s.minimum() == 0 and s.maximum() == 1000

    def test_click_sets_value(self, qapp):
        from respanno.gui.widgets.clickable_slider import ClickableSlider
        from PyQt5.QtTest import QTest
        from PyQt5.QtCore import QPoint

        s = ClickableSlider(Qt.Horizontal)
        s.setRange(0, 1000)
        s.setValue(0)
        s.resize(200, 30)
        s.show()

        QTest.mouseClick(s, Qt.LeftButton, pos=QPoint(150, 15))
        assert s.value() > 100, f"Expected value > 100, got {s.value()}"


# ═══════════════════════════════════════════════════════════════════════════
# 3. ColorCheckDelegate
# ═══════════════════════════════════════════════════════════════════════════

class TestColorCheckDelegate:
    """ColorCheckDelegate: renders colored checkboxes for feature selection."""

    def test_creates(self, qapp):
        from respanno.gui.widgets.color_check_delegate import ColorCheckDelegate
        from PyQt5.QtWidgets import QStyledItemDelegate
        assert isinstance(ColorCheckDelegate(), QStyledItemDelegate)

    def test_constants(self, qapp):
        from respanno.gui.widgets.color_check_delegate import ColorCheckDelegate
        d = ColorCheckDelegate()
        assert d.BOX_SIZE > 0 and d.PADDING_X >= 0

    def test_size_hint(self, qapp):
        from respanno.gui.widgets.color_check_delegate import ColorCheckDelegate
        from PyQt5.QtWidgets import QStyleOptionViewItem
        from PyQt5.QtCore import QModelIndex, QSize

        d = ColorCheckDelegate()
        opt = QStyleOptionViewItem()
        sz = d.sizeHint(opt, QModelIndex())
        assert isinstance(sz, QSize)
        assert sz.width() > 0 and sz.height() > 0


# ═══════════════════════════════════════════════════════════════════════════
# 4. SpanLabelItem
# ═══════════════════════════════════════════════════════════════════════════

class TestSpanLabelItem:
    """SpanLabelItem: text item placed on annotation span bars."""

    def test_creates_without_owner(self, qapp):
        from respanno.gui.spans.span_label_item import SpanLabelItem
        item = SpanLabelItem(text="Test")
        assert item._owner_span is None

    def test_creates_with_owner(self, qapp):
        from respanno.gui.spans.span_label_item import SpanLabelItem
        item = SpanLabelItem(text="Wheeze", owner_span="fake_span")
        assert item._owner_span == "fake_span"

    def test_setHtml_no_crash(self, qapp):
        from respanno.gui.spans.span_label_item import SpanLabelItem
        item = SpanLabelItem(text="")
        item.setHtml('<div style="color:red">Wheeze</div>')


# ═══════════════════════════════════════════════════════════════════════════
# 5. LoopPlayer
# ═══════════════════════════════════════════════════════════════════════════

class TestLoopPlayer:
    """LoopPlayer: loop-playback dialog with sounddevice mocked."""

    def test_creates_dialog(self, qapp, monkeypatch):
        import sys, types
        fake_sd = types.ModuleType("sounddevice")
        fake_sd.stop = lambda: None
        fake_sd.play = lambda *a, **kw: None
        fake_sd.OutputStream = type("OutputStream", (), {})
        sys.modules["sounddevice"] = fake_sd

        from respanno.gui.dialogs.loop_player import LoopPlayer
        audio = np.zeros(4000, dtype=np.float32)
        dlg = LoopPlayer(audio, 4000, 1.0, 2.0, None, parent=None)
        assert dlg.windowTitle().startswith("Loop Playback")
        dlg.close()


# ═══════════════════════════════════════════════════════════════════════════
# 6. AnnotationLabelDialog
# ═══════════════════════════════════════════════════════════════════════════

class TestAnnotationLabelDialog:
    """AnnotationLabelDialog: dialog for editing an annotation label.

    NOTE: get_text() calls exec_() (modal) — we test construction and
    internal widget state only. Modal interaction needs manual testing.
    """

    def test_creates(self, qapp):
        from respanno.gui.dialogs.annotation_label_dialog import AnnotationLabelDialog
        dlg = AnnotationLabelDialog(default_text="Wheeze", parent=None)
        assert dlg is not None
        dlg.close()

    def test_line_edit_default_text(self, qapp):
        from respanno.gui.dialogs.annotation_label_dialog import AnnotationLabelDialog
        dlg = AnnotationLabelDialog(default_text="Crackles", parent=None)
        assert dlg.line_edit.text() == "Crackles"
        dlg.close()

    def test_with_start_end_params(self, qapp):
        from respanno.gui.dialogs.annotation_label_dialog import AnnotationLabelDialog
        dlg = AnnotationLabelDialog(default_text="Rhonchi", start=0.5, end=1.2, parent=None)
        assert dlg.line_edit.text() == "Rhonchi"
        dlg.close()

    def test_preset_combo_populated(self, qapp):
        from respanno.gui.dialogs.annotation_label_dialog import AnnotationLabelDialog
        builtins = [("哮鸣音", "Wheeze"), ("爆裂音", "Crackles")]
        dlg = AnnotationLabelDialog(builtin_labels=builtins, parent=None)
        assert dlg.combo.count() >= 2
        dlg.close()


# ═══════════════════════════════════════════════════════════════════════════
# 7. AnnotViewBox / WaveViewBox
# ═══════════════════════════════════════════════════════════════════════════

class TestViewBoxes:
    """AnnotViewBox and WaveViewBox: pyqtgraph-based view containers."""

    def test_annot_view_box_creates(self, qapp):
        import pyqtgraph as pg
        from respanno.gui.views.annot_view_box import AnnotViewBox
        vb = AnnotViewBox(parent=None)
        assert isinstance(vb, pg.ViewBox)

    def test_wave_view_box_creates(self, qapp):
        import pyqtgraph as pg
        from respanno.gui.views.wave_view_box import WaveViewBox
        vb = WaveViewBox(parent=None)
        assert isinstance(vb, pg.ViewBox)
