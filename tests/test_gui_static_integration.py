"""Static checks: verify that 1.6.6.py delegates to respanno modules.

Stages 1-5:  all five pure modules connected (annotation_io, preprocessing,
             spectrogram, features, hsmm).
Phase 2:     nine GUI widget classes extracted to respanno/gui/.

These tests do NOT import 1.6.6.py (which would trigger PyQt5 / sounddevice).
They only scan the source text.
"""

from __future__ import annotations

import ast
import os
import tokenize


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUI_FILE = os.path.join(ROOT, "1.6.6.py")
LEGACY_FILE = os.path.join(ROOT, "legacy", "1.6.6.py")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _parse_ast(path: str) -> ast.AST:
    with open(path, "r", encoding="utf-8") as f:
        return ast.parse(f.read(), filename=path)


def _tokenized_source(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return tokenize.untokenize(tokenize.generate_tokens(f.readline))


# ---------------------------------------------------------------------------
# 1. legacy/1.6.6.py is untouched
# ---------------------------------------------------------------------------

def test_legacy_is_unchanged():
    text = _read_text(LEGACY_FILE)
    assert "respanno" not in text, (
        "legacy/1.6.6.py must remain frozen"
    )


# ---------------------------------------------------------------------------
# 2. Stage 1: annotation_io
# ---------------------------------------------------------------------------

def test_gui_imports_annotation_io():
    tree = _parse_ast(GUI_FILE)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append(f"{module}.{alias.name}")
    target = "respanno.labels.annotation_io"
    found = any(target in imp or imp.startswith(target) for imp in imports)
    assert found, f"1.6.6.py must import from {target}"


def test_gui_calls_read_annotations():
    assert "read_annotations" in _tokenized_source(GUI_FILE)


def test_gui_calls_write_annotations():
    assert "write_annotations" in _tokenized_source(GUI_FILE)


# ---------------------------------------------------------------------------
# 3. Stage 2: preprocessing
# ---------------------------------------------------------------------------

def test_gui_imports_preprocessing():
    tree = _parse_ast(GUI_FILE)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append(f"{module}.{alias.name}")
    assert any("respanno.audio.preprocessing" in imp for imp in imports)


def test_gui_calls_preprocessing_functions():
    src = _tokenized_source(GUI_FILE)
    required = ["apply_butter_filter", "summarize_preprocessing",
                 "compute_target_sr", "load_audio_file", "get_original_sr"]
    assert not [fn for fn in required if fn not in src]


# ---------------------------------------------------------------------------
# 4. Stage 3: spectrogram
# ---------------------------------------------------------------------------

def test_gui_imports_spectrogram():
    tree = _parse_ast(GUI_FILE)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append(f"{module}.{alias.name}")
    assert any("respanno.dsp.spectrogram" in imp for imp in imports)


def test_gui_calls_spectrogram_functions():
    src = _tokenized_source(GUI_FILE)
    required = ["compute_stft_db", "decimate_spec_for_display",
                 "get_palette_256", "colorize_spectrogram"]
    assert not [fn for fn in required if fn not in src]


# ---------------------------------------------------------------------------
# 5. Stage 4: features
# ---------------------------------------------------------------------------

def test_gui_imports_features():
    tree = _parse_ast(GUI_FILE)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append(f"{module}.{alias.name}")
    assert any("respanno.dsp.features" in imp for imp in imports)


def test_gui_calls_features_functions():
    src = _tokenized_source(GUI_FILE)
    required = ["compute_short_time_features", "build_feature_matrix",
                 "normalize_feature_for_display"]
    assert not [fn for fn in required if fn not in src]


# ---------------------------------------------------------------------------
# 6. Stage 5: hsmm
# ---------------------------------------------------------------------------

def test_gui_imports_hsmm():
    tree = _parse_ast(GUI_FILE)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append(f"{module}.{alias.name}")
    assert any("respanno.ml.hsmm" in imp for imp in imports), (
        f"1.6.6.py must import from respanno.ml.hsmm\n"
        f"Found: {[i for i in imports if 'respanno' in i]}"
    )


def test_gui_calls_hsmm_functions():
    src = _tokenized_source(GUI_FILE)
    required = ["estimate_hop_sec", "estimate_breath_cycle_sec",
                 "build_hsmm_prior_from_prefix_labels", "build_hsmm_log_trans",
                 "hsmm_viterbi", "state_seq_to_segments"]
    missing = [fn for fn in required if fn not in src]
    assert not missing, f"Missing hsmm calls: {missing}"


# ---------------------------------------------------------------------------
# 6b. Stage 5b: label_taxonomy
# ---------------------------------------------------------------------------

def test_gui_imports_label_taxonomy():
    tree = _parse_ast(GUI_FILE)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append(f"{module}.{alias.name}")
    assert any("respanno.ml.label_taxonomy" in imp for imp in imports), (
        f"1.6.6.py must import from respanno.ml.label_taxonomy\n"
        f"Found: {[i for i in imports if 'respanno' in i]}"
    )


def test_gui_calls_label_taxonomy_functions():
    src = _tokenized_source(GUI_FILE)
    required = ["label_kind", "clear_ml_annotations"]
    missing = [fn for fn in required if fn not in src]
    assert not missing, f"Missing label_taxonomy calls: {missing}"


# ---------------------------------------------------------------------------
# 6c. Stage 5c: phase_model
# ---------------------------------------------------------------------------

def test_gui_imports_phase_model():
    tree = _parse_ast(GUI_FILE)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append(f"{module}.{alias.name}")
    assert any("respanno.ml.phase_model" in imp for imp in imports), (
        f"1.6.6.py must import from respanno.ml.phase_model\n"
        f"Found: {[i for i in imports if 'respanno' in i]}"
    )


def test_gui_calls_phase_model_functions():
    src = _tokenized_source(GUI_FILE)
    required = ["train_phase_model", "apply_phase_model"]
    missing = [fn for fn in required if fn not in src]
    assert not missing, f"Missing phase_model calls: {missing}"


# ---------------------------------------------------------------------------
# 6d. Stage 5d: classifier
# ---------------------------------------------------------------------------

def test_gui_imports_classifier():
    tree = _parse_ast(GUI_FILE)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append(f"{module}.{alias.name}")
    assert any("respanno.ml.classifier" in imp for imp in imports), (
        f"1.6.6.py must import from respanno.ml.classifier\n"
        f"Found: {[i for i in imports if 'respanno' in i]}"
    )


def test_gui_calls_classifier_functions():
    src = _tokenized_source(GUI_FILE)
    required = ["train_event_model", "apply_event_model"]
    missing = [fn for fn in required if fn not in src]
    assert not missing, f"Missing classifier calls: {missing}"


# ---------------------------------------------------------------------------
# 7. MLService, BoxSpan, eventFilter are untouched
# ---------------------------------------------------------------------------

ALLOWED_RESPANNO = {"annotation_io", "preprocessing", "spectrogram",
                     "features", "hsmm", "label_taxonomy",
                     "phase_model", "classifier"}


def _respanno_refs_in_class(tree: ast.AST, class_name: str) -> list:
    refs = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in ast.walk(node):
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    module = getattr(child, "module", "") or ""
                    for alias in child.names:
                        full = f"{module}.{alias.name}"
                        if "respanno" in full:
                            refs.append(full)
    return refs


# ---------------------------------------------------------------------------
# 7. Phase 2: GUI widget modules exist and compile
# ---------------------------------------------------------------------------

GUI_MODULES = [
    "respanno/gui/widgets/clickable_slider.py",
    "respanno/gui/widgets/color_bar.py",
    "respanno/gui/widgets/color_check_delegate.py",
    "respanno/gui/dialogs/annotation_label_dialog.py",
    "respanno/gui/dialogs/loop_player.py",
    "respanno/gui/dialogs/settings_dialog.py",
    "respanno/gui/spans/span_label_item.py",
    "respanno/gui/spans/box_span.py",
    "respanno/gui/views/annot_view_box.py",
    "respanno/gui/views/wave_view_box.py",
]


def test_gui_module_files_exist():
    for rel_path in GUI_MODULES:
        path = os.path.join(ROOT, rel_path)
        assert os.path.isfile(path), f"Missing module: {rel_path}"


def test_gui_modules_compile():
    import py_compile
    for rel_path in GUI_MODULES:
        path = os.path.join(ROOT, rel_path)
        try:
            py_compile.compile(path, doraise=True)
        except py_compile.PyCompileError as e:
            raise AssertionError(f"Module does not compile: {rel_path}\n{e}")


def test_gui_imports_widget_classes():
    """1.6.6.py must import the 9 extracted classes from respanno.gui."""
    tree = _parse_ast(GUI_FILE)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append(f"{module}.{alias.name}")
    required = {
        "respanno.gui.widgets.clickable_slider.ClickableSlider",
        "respanno.gui.widgets.color_bar.ColorBarWidget",
        "respanno.gui.dialogs.annotation_label_dialog.AnnotationLabelDialog",
        "respanno.gui.dialogs.loop_player.LoopPlayer",
        "respanno.gui.dialogs.settings_dialog.SettingsDialog",
        "respanno.gui.spans.span_label_item.SpanLabelItem",
        "respanno.gui.spans.box_span.BoxSpan",
        "respanno.gui.views.annot_view_box.AnnotViewBox",
        "respanno.gui.views.wave_view_box.WaveViewBox",
    }
    missing = required - set(imports)
    assert not missing, f"1.6.6.py missing GUI imports: {missing}"


def test_mlservice_not_rewired():
    tree = _parse_ast(GUI_FILE)
    refs = _respanno_refs_in_class(tree, "MLService")
    for full in refs:
        assert any(allowed in full for allowed in ALLOWED_RESPANNO), (
            f"MLService must not import {full}"
        )


def test_boxspan_not_rewired():
    tree = _parse_ast(GUI_FILE)
    refs = _respanno_refs_in_class(tree, "BoxSpan")
    assert not refs, f"BoxSpan must not import respanno modules, found: {refs}"


# ---------------------------------------------------------------------------
# 8. Phase 4: SettingsDialog extracted to respanno/gui/dialogs/
# ---------------------------------------------------------------------------

def test_settings_dialog_imports_color_check_delegate():
    """ColorCheckDelegate is no longer imported in 1.6.6.py — it must be
    imported internally by settings_dialog.py instead."""
    import ast as ast_m
    settings_path = os.path.join(
        ROOT, "respanno", "gui", "dialogs", "settings_dialog.py")
    with open(settings_path, "r", encoding="utf-8") as f:
        tree = ast_m.parse(f.read(), filename=settings_path)
    imports = []
    for node in ast_m.walk(tree):
        if isinstance(node, ast_m.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append(f"{module}.{alias.name}")
    assert "respanno.gui.widgets.color_check_delegate.ColorCheckDelegate" in imports, (
        "settings_dialog.py must import ColorCheckDelegate internally"
    )


def test_settings_dialog_basic_getters(qapp):
    """Instantiate SettingsDialog with defaults and verify all 7 getters
    return correct types and reasonable values."""
    from respanno.gui.dialogs.settings_dialog import SettingsDialog

    dlg = SettingsDialog(
        n_fft=1024, hop_length=512, f_max=2000,
        wave_y_range=(-0.5, 0.8),
        selected_features=["短时能量", "过零率"],
        stft_cmap="Grayscale",
        preprocessing_enabled=False,
    )

    # 1. get_values()
    n_fft, hop, fmax, (ymin, ymax) = dlg.get_values()
    assert n_fft == 1024
    assert hop == 512
    assert fmax == 2000
    assert abs(ymin - (-0.5)) < 1e-9
    assert abs(ymax - 0.8) < 1e-9

    # 2. get_resample_settings()
    res_enabled, target_sr = dlg.get_resample_settings()
    assert isinstance(res_enabled, bool)
    assert isinstance(target_sr, int)
    assert target_sr == 4000

    # 3. get_preprocessing_settings()
    pp = dlg.get_preprocessing_settings()
    assert isinstance(pp, dict)
    assert "preprocessing_enabled" in pp
    assert "resample_enabled" in pp
    assert "filter_enabled" in pp
    assert pp["preprocessing_enabled"] is False  # we set it False above

    # 4. get_auto_label_import_enabled()
    auto_enabled = dlg.get_auto_label_import_enabled()
    assert isinstance(auto_enabled, bool)
    assert auto_enabled is False  # default

    # 5. get_auto_label_import_settings()
    cfg = dlg.get_auto_label_import_settings()
    assert isinstance(cfg, dict)
    assert cfg["file_format"] == "auto"
    assert cfg["file_suffix"] == "_events"
    assert cfg["start_col"] == 1

    # 6. get_selected_features()
    feats = dlg.get_selected_features()
    assert isinstance(feats, list)
    assert "短时能量" in feats
    assert "过零率" in feats
    assert len(feats) <= 5

    # 7. get_stft_display_settings()
    cmap, vmin, vmax = dlg.get_stft_display_settings()
    assert cmap == "Grayscale"
    assert isinstance(vmin, float)
    assert isinstance(vmax, float)
    assert vmax > vmin


def test_settings_dialog_preprocessing_settings_default(qapp):
    """Default preprocessing settings match expected values."""
    from respanno.gui.dialogs.settings_dialog import SettingsDialog

    dlg = SettingsDialog()
    pp = dlg.get_preprocessing_settings()
    assert pp["preprocessing_enabled"] is True
    assert pp["resample_enabled"] is True
    assert pp["resample_target_sr"] == 4000
    assert pp["filter_enabled"] is False
    assert pp["filter_type"] == "bandpass"
    assert pp["filter_lowcut"] == 20.0
    assert pp["filter_highcut"] == 1800.0
    assert pp["filter_order"] == 4
    assert pp["filter_zero_phase"] is True
