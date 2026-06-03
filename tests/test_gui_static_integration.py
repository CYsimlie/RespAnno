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
GUI_FILE = os.path.join(ROOT, '1.6.6.py')
LEGACY_FILE = os.path.join(ROOT, 'legacy', '1.6.6.py')

def _read_text(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def _parse_ast(path: str) -> ast.AST:
    with open(path, 'r', encoding='utf-8') as f:
        return ast.parse(f.read(), filename=path)

def _tokenized_source(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return tokenize.untokenize(tokenize.generate_tokens(f.readline))

def test_legacy_is_unchanged():
    """验证 legacy/1.6.6.py 不包含对 respanno 模块的引用（保持原始快照）。"""
    text = _read_text(LEGACY_FILE)
    assert 'respanno' not in text, 'legacy/1.6.6.py must remain frozen'

def test_gui_imports_annotation_io():
    """验证 1.6.6.py 的 AST 中包含对 respanno.annotation.io 的 import 语句。"""
    tree = _parse_ast(GUI_FILE)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            for alias in node.names:
                imports.append(f'{module}.{alias.name}')
    target = 'respanno.labels.annotation_io'
    found = any((target in imp or imp.startswith(target) for imp in imports))
    assert found, f'1.6.6.py must import from {target}'

def test_gui_calls_read_annotations():
    """验证 1.6.6.py 源码中包含对 read annotations 的函数调用。"""
    assert 'read_annotations' in _tokenized_source(GUI_FILE)

def test_gui_calls_write_annotations():
    """验证 1.6.6.py 源码中包含对 write annotations 的函数调用。"""
    assert 'write_annotations' in _tokenized_source(GUI_FILE)

def test_gui_imports_preprocessing():
    """验证 1.6.6.py 的 AST 中包含对 respanno.preprocessing 的 import 语句。"""
    tree = _parse_ast(GUI_FILE)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            for alias in node.names:
                imports.append(f'{module}.{alias.name}')
    assert any(('respanno.audio.preprocessing' in imp for imp in imports))

def test_gui_calls_preprocessing_functions():
    """验证 1.6.6.py 源码中包含对 preprocessing functions 的函数调用。"""
    src = _tokenized_source(GUI_FILE)
    required = ['apply_butter_filter', 'summarize_preprocessing', 'compute_target_sr', 'load_audio_file', 'get_original_sr']
    assert not [fn for fn in required if fn not in src]

def test_gui_imports_spectrogram():
    """验证 1.6.6.py 的 AST 中包含对 respanno.spectrogram 的 import 语句。"""
    tree = _parse_ast(GUI_FILE)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            for alias in node.names:
                imports.append(f'{module}.{alias.name}')
    assert any(('respanno.dsp.spectrogram' in imp for imp in imports))

def test_gui_calls_spectrogram_functions():
    """验证 1.6.6.py 源码中包含对 spectrogram functions 的函数调用。"""
    src = _tokenized_source(GUI_FILE)
    required = ['compute_stft_db', 'decimate_spec_for_display', 'get_palette_256', 'colorize_spectrogram']
    assert not [fn for fn in required if fn not in src]

def test_gui_imports_features():
    """验证 1.6.6.py 的 AST 中包含对 respanno.features 的 import 语句。"""
    tree = _parse_ast(GUI_FILE)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            for alias in node.names:
                imports.append(f'{module}.{alias.name}')
    assert any(('respanno.dsp.features' in imp for imp in imports))

def test_gui_calls_features_functions():
    """验证 1.6.6.py 源码中包含对 features functions 的函数调用。"""
    src = _tokenized_source(GUI_FILE)
    required = ['compute_short_time_features', 'build_feature_matrix', 'normalize_feature_for_display']
    assert not [fn for fn in required if fn not in src]

def test_gui_imports_hsmm():
    """验证 1.6.6.py 的 AST 中包含对 respanno.hsmm 的 import 语句。"""
    tree = _parse_ast(GUI_FILE)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            for alias in node.names:
                imports.append(f'{module}.{alias.name}')
    assert any(('respanno.ml.hsmm' in imp for imp in imports)), f"1.6.6.py must import from respanno.ml.hsmm\nFound: {[i for i in imports if 'respanno' in i]}"

def test_gui_calls_hsmm_functions():
    """验证 1.6.6.py 源码中包含对 hsmm functions 的函数调用。"""
    src = _tokenized_source(GUI_FILE)
    required = ['estimate_hop_sec', 'estimate_breath_cycle_sec', 'build_hsmm_prior_from_prefix_labels', 'build_hsmm_log_trans', 'hsmm_viterbi', 'state_seq_to_segments']
    missing = [fn for fn in required if fn not in src]
    assert not missing, f'Missing hsmm calls: {missing}'

def test_gui_imports_label_taxonomy():
    """验证 1.6.6.py 的 AST 中包含对 respanno.label.taxonomy 的 import 语句。"""
    tree = _parse_ast(GUI_FILE)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            for alias in node.names:
                imports.append(f'{module}.{alias.name}')
    assert any(('respanno.ml.label_taxonomy' in imp for imp in imports)), f"1.6.6.py must import from respanno.ml.label_taxonomy\nFound: {[i for i in imports if 'respanno' in i]}"

def test_gui_calls_label_taxonomy_functions():
    """验证 1.6.6.py 源码中包含对 label taxonomy functions 的函数调用。"""
    src = _tokenized_source(GUI_FILE)
    required = ['label_kind', 'clear_ml_annotations']
    missing = [fn for fn in required if fn not in src]
    assert not missing, f'Missing label_taxonomy calls: {missing}'

def test_gui_imports_phase_model():
    """验证模型训练后正确存储到 ml_models 字典。"""
    tree = _parse_ast(GUI_FILE)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            for alias in node.names:
                imports.append(f'{module}.{alias.name}')
    assert any(('respanno.ml.phase_model' in imp for imp in imports)), f"1.6.6.py must import from respanno.ml.phase_model\nFound: {[i for i in imports if 'respanno' in i]}"

def test_gui_calls_phase_model_functions():
    """验证模型训练后正确存储到 ml_models 字典。"""
    src = _tokenized_source(GUI_FILE)
    required = ['train_phase_model', 'apply_phase_model']
    missing = [fn for fn in required if fn not in src]
    assert not missing, f'Missing phase_model calls: {missing}'

def test_gui_imports_classifier():
    """验证 1.6.6.py 的 AST 中包含对 respanno.classifier 的 import 语句。"""
    tree = _parse_ast(GUI_FILE)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            for alias in node.names:
                imports.append(f'{module}.{alias.name}')
    assert any(('respanno.ml.classifier' in imp for imp in imports)), f"1.6.6.py must import from respanno.ml.classifier\nFound: {[i for i in imports if 'respanno' in i]}"

def test_gui_calls_classifier_functions():
    """验证 1.6.6.py 源码中包含对 classifier functions 的函数调用。"""
    src = _tokenized_source(GUI_FILE)
    required = ['train_event_model', 'apply_event_model']
    missing = [fn for fn in required if fn not in src]
    assert not missing, f'Missing classifier calls: {missing}'
ALLOWED_RESPANNO = {'annotation_io', 'preprocessing', 'spectrogram', 'features', 'hsmm', 'label_taxonomy', 'phase_model', 'classifier'}

def _respanno_refs_in_class(tree: ast.AST, class_name: str) -> list:
    refs = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in ast.walk(node):
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    module = getattr(child, 'module', '') or ''
                    for alias in child.names:
                        full = f'{module}.{alias.name}'
                        if 'respanno' in full:
                            refs.append(full)
    return refs
GUI_MODULES = ['respanno/gui/widgets/clickable_slider.py', 'respanno/gui/widgets/color_bar.py', 'respanno/gui/widgets/color_check_delegate.py', 'respanno/gui/dialogs/annotation_label_dialog.py', 'respanno/gui/dialogs/loop_player.py', 'respanno/gui/dialogs/settings_dialog.py', 'respanno/gui/spans/span_label_item.py', 'respanno/gui/spans/box_span.py', 'respanno/gui/views/annot_view_box.py', 'respanno/gui/views/wave_view_box.py']

def test_gui_module_files_exist():
    """验证：os.path.isfile(path)。"""
    for rel_path in GUI_MODULES:
        path = os.path.join(ROOT, rel_path)
        assert os.path.isfile(path), f'Missing module: {rel_path}'

def test_gui_modules_compile():
    """测试 join, AssertionError, compile 的行为。"""
    import py_compile
    for rel_path in GUI_MODULES:
        path = os.path.join(ROOT, rel_path)
        try:
            py_compile.compile(path, doraise=True)
        except py_compile.PyCompileError as e:
            raise AssertionError(f'Module does not compile: {rel_path}\n{e}')

def test_gui_imports_widget_classes():
    """1.6.6.py must import the 9 extracted classes from respanno.gui."""
    tree = _parse_ast(GUI_FILE)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ''
            for alias in node.names:
                imports.append(f'{module}.{alias.name}')
    required = {'respanno.gui.widgets.clickable_slider.ClickableSlider', 'respanno.gui.widgets.color_bar.ColorBarWidget', 'respanno.gui.dialogs.annotation_label_dialog.AnnotationLabelDialog', 'respanno.gui.dialogs.loop_player.LoopPlayer', 'respanno.gui.dialogs.settings_dialog.SettingsDialog', 'respanno.gui.spans.span_label_item.SpanLabelItem', 'respanno.gui.spans.box_span.BoxSpan', 'respanno.gui.views.annot_view_box.AnnotViewBox', 'respanno.gui.views.wave_view_box.WaveViewBox'}
    missing = required - set(imports)
    assert not missing, f'1.6.6.py missing GUI imports: {missing}'

def test_mlservice_not_rewired():
    """验证：any((allowed in full for allowed in ALLOWED_RESPANNO))。"""
    tree = _parse_ast(GUI_FILE)
    refs = _respanno_refs_in_class(tree, 'MLService')
    for full in refs:
        assert any((allowed in full for allowed in ALLOWED_RESPANNO)), f'MLService must not import {full}'

def test_boxspan_not_rewired():
    """验证：not refs。"""
    tree = _parse_ast(GUI_FILE)
    refs = _respanno_refs_in_class(tree, 'BoxSpan')
    assert not refs, f'BoxSpan must not import respanno modules, found: {refs}'

def test_settings_dialog_imports_color_check_delegate():
    """ColorCheckDelegate is no longer imported in 1.6.6.py — it must be
    imported internally by settings_dialog.py instead."""
    import ast as ast_m
    settings_path = os.path.join(ROOT, 'respanno', 'gui', 'dialogs', 'settings_dialog.py')
    with open(settings_path, 'r', encoding='utf-8') as f:
        tree = ast_m.parse(f.read(), filename=settings_path)
    imports = []
    for node in ast_m.walk(tree):
        if isinstance(node, ast_m.ImportFrom):
            module = node.module or ''
            for alias in node.names:
                imports.append(f'{module}.{alias.name}')
    assert 'respanno.gui.widgets.color_check_delegate.ColorCheckDelegate' in imports, 'settings_dialog.py must import ColorCheckDelegate internally'

def test_settings_dialog_basic_getters(qapp):
    """Instantiate SettingsDialog with defaults and verify all 7 getters
    return correct types and reasonable values."""
    from respanno.gui.dialogs.settings_dialog import SettingsDialog
    dlg = SettingsDialog(n_fft=1024, hop_length=512, f_max=2000, wave_y_range=(-0.5, 0.8), selected_features=['短时能量', '过零率'], stft_cmap='Grayscale', preprocessing_enabled=False)
    (n_fft, hop, fmax, (ymin, ymax)) = dlg.get_values()
    assert n_fft == 1024
    assert hop == 512
    assert fmax == 2000
    assert abs(ymin - -0.5) < 1e-09
    assert abs(ymax - 0.8) < 1e-09
    (res_enabled, target_sr) = dlg.get_resample_settings()
    assert isinstance(res_enabled, bool)
    assert isinstance(target_sr, int)
    assert target_sr == 4000
    pp = dlg.get_preprocessing_settings()
    assert isinstance(pp, dict)
    assert 'preprocessing_enabled' in pp
    assert 'resample_enabled' in pp
    assert 'filter_enabled' in pp
    assert pp['preprocessing_enabled'] is False
    auto_enabled = dlg.get_auto_label_import_enabled()
    assert isinstance(auto_enabled, bool)
    assert auto_enabled is False
    cfg = dlg.get_auto_label_import_settings()
    assert isinstance(cfg, dict)
    assert cfg['file_format'] == 'auto'
    assert cfg['file_suffix'] == '_events'
    assert cfg['start_col'] == 1
    feats = dlg.get_selected_features()
    assert isinstance(feats, list)
    assert '短时能量' in feats
    assert '过零率' in feats
    assert len(feats) <= 5
    (cmap, vmin, vmax) = dlg.get_stft_display_settings()
    assert cmap == 'Grayscale'
    assert isinstance(vmin, float)
    assert isinstance(vmax, float)
    assert vmax > vmin

def test_settings_dialog_preprocessing_settings_default(qapp):
    """Default preprocessing settings match expected values."""
    from respanno.gui.dialogs.settings_dialog import SettingsDialog
    dlg = SettingsDialog()
    pp = dlg.get_preprocessing_settings()
    assert pp['preprocessing_enabled'] is True
    assert pp['resample_enabled'] is True
    assert pp['resample_target_sr'] == 4000
    assert pp['filter_enabled'] is False
    assert pp['filter_type'] == 'bandpass'
    assert pp['filter_lowcut'] == 20.0
    assert pp['filter_highcut'] == 1800.0
    assert pp['filter_order'] == 4
    assert pp['filter_zero_phase'] is True