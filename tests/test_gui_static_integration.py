"""Static checks: verify that 1.6.6.py delegates to respanno modules.

Stage 1: annotation_io   (import/export)
Stage 2: preprocessing   (audio load/filter/summary)
Stage 3: spectrogram     (STFT / decimate / palette / colorize)

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
    """legacy/1.6.6.py must NOT import from respanno."""
    text = _read_text(LEGACY_FILE)
    assert "respanno" not in text, (
        "legacy/1.6.6.py must remain frozen — remove any respanno import"
    )


# ---------------------------------------------------------------------------
# 2. Stage 1: annotation_io is still connected
# ---------------------------------------------------------------------------

def test_gui_imports_annotation_io():
    """1.6.6.py must import respanno.labels.annotation_io (or its symbols)."""
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
    found = any(
        target in imp or imp.startswith("respanno.labels.annotation_io")
        for imp in imports
    )
    assert found, (
        f"1.6.6.py must import from {target}\n"
        f"Found respanno-related imports: {[i for i in imports if 'respanno' in i]}"
    )


def test_gui_calls_read_annotations():
    """import_annotations or _parse_events_file must call read_annotations."""
    src = _tokenized_source(GUI_FILE)
    assert "read_annotations" in src, (
        "1.6.6.py must call read_annotations (from annotation_io)"
    )


def test_gui_calls_write_annotations():
    """export_annotations must call write_annotations."""
    src = _tokenized_source(GUI_FILE)
    assert "write_annotations" in src, (
        "1.6.6.py must call write_annotations (from annotation_io)"
    )


# ---------------------------------------------------------------------------
# 3. Stage 2: preprocessing is still connected
# ---------------------------------------------------------------------------

def test_gui_imports_preprocessing():
    """1.6.6.py must import from respanno.audio.preprocessing."""
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

    target = "respanno.audio.preprocessing"
    found = any(
        target in imp or imp.startswith("respanno.audio.preprocessing")
        for imp in imports
    )
    assert found, (
        f"1.6.6.py must import from {target}\n"
        f"Found respanno-related imports: {[i for i in imports if 'respanno' in i]}"
    )


def test_gui_calls_preprocessing_functions():
    """1.6.6.py must call key preprocessing module functions."""
    src = _tokenized_source(GUI_FILE)
    required = [
        "apply_butter_filter",
        "summarize_preprocessing",
        "compute_target_sr",
        "load_audio_file",
        "get_original_sr",
    ]
    missing = [fn for fn in required if fn not in src]
    assert not missing, (
        f"1.6.6.py must call these preprocessing functions: {missing}"
    )


# ---------------------------------------------------------------------------
# 4. Stage 3: spectrogram is connected
# ---------------------------------------------------------------------------

def test_gui_imports_spectrogram():
    """1.6.6.py must import from respanno.dsp.spectrogram."""
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

    target = "respanno.dsp.spectrogram"
    found = any(
        target in imp or imp.startswith("respanno.dsp.spectrogram")
        for imp in imports
    )
    assert found, (
        f"1.6.6.py must import from {target}\n"
        f"Found respanno-related imports: {[i for i in imports if 'respanno' in i]}"
    )


def test_gui_calls_spectrogram_functions():
    """1.6.6.py must call key spectrogram module functions."""
    src = _tokenized_source(GUI_FILE)
    required = [
        "compute_stft_db",
        "decimate_spec_for_display",
        "get_palette_256",
        "colorize_spectrogram",
    ]
    missing = [fn for fn in required if fn not in src]
    assert not missing, (
        f"1.6.6.py must call these spectrogram functions: {missing}"
    )


# ---------------------------------------------------------------------------
# 5. features / hsmm are NOT connected yet
# ---------------------------------------------------------------------------

def test_no_early_module_leak():
    """1.6.6.py must NOT import from features or hsmm yet."""
    tree = _parse_ast(GUI_FILE)
    forbidden = {
        "respanno.dsp.features",
        "respanno.ml.hsmm",
    }
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = getattr(node, "module", "") or ""
            for forbidden_module in forbidden:
                if module == forbidden_module or module.startswith(forbidden_module + "."):
                    raise AssertionError(
                        f"1.6.6.py must not import {forbidden_module} yet — "
                        f"only annotation_io, preprocessing, and spectrogram are connected"
                    )


# ---------------------------------------------------------------------------
# 6. MLService, BoxSpan, eventFilter are untouched
# ---------------------------------------------------------------------------

ALLOWED_RESPANNO = {"annotation_io", "preprocessing", "spectrogram"}


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
    """MLService class must NOT reference respanno beyond allowed modules."""
    tree = _parse_ast(GUI_FILE)
    refs = _respanno_refs_in_class(tree, "MLService")
    for full in refs:
        assert any(
            allowed in full for allowed in ALLOWED_RESPANNO
        ), f"MLService must not import {full} — only {ALLOWED_RESPANNO} are allowed"


def test_boxspan_not_rewired():
    """BoxSpan class must NOT reference respanno."""
    tree = _parse_ast(GUI_FILE)
    refs = _respanno_refs_in_class(tree, "BoxSpan")
    assert not refs, f"BoxSpan must not import respanno modules, found: {refs}"
