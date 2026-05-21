"""Static checks: verify that 1.6.6.py delegates to respanno modules.

Stage 1: annotation_io   (import/export)
Stage 2: preprocessing   (audio load/filter/summary)
Stage 3: spectrogram     (STFT / decimate / palette / colorize)
Stage 4: features        (short-time features / build matrix / normalize)
Stage 5: hsmm            (Viterbi / duration priors / state-to-segment)

ALL FIVE pure modules are now connected.

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
# 7. MLService, BoxSpan, eventFilter are untouched
# ---------------------------------------------------------------------------

ALLOWED_RESPANNO = {"annotation_io", "preprocessing", "spectrogram",
                     "features", "hsmm"}


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
