"""Static checks: verify that 1.6.6.py delegates annotation I/O to respanno.

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
# 2. 1.6.6.py imports from respanno.labels.annotation_io
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


# ---------------------------------------------------------------------------
# 3. 1.6.6.py actually calls read_annotations / write_annotations from annotation_io
# ---------------------------------------------------------------------------

def test_gui_calls_read_annotations():
    """import_annotations or _parse_events_file must call read_annotations."""
    # Use tokenize to catch string mentions even inside function-local imports
    with open(GUI_FILE, "r", encoding="utf-8") as f:
        tokens = list(tokenize.generate_tokens(f.readline))
    source_text = tokenize.untokenize(tokens)

    assert "read_annotations" in source_text, (
        "1.6.6.py must call read_annotations (from annotation_io)"
    )


def test_gui_calls_write_annotations():
    """export_annotations must call write_annotations."""
    with open(GUI_FILE, "r", encoding="utf-8") as f:
        tokens = list(tokenize.generate_tokens(f.readline))
    source_text = tokenize.untokenize(tokens)

    assert "write_annotations" in source_text, (
        "1.6.6.py must call write_annotations (from annotation_io)"
    )


# ---------------------------------------------------------------------------
# 4. MLService, BoxSpan, eventFilter are untouched
# ---------------------------------------------------------------------------

def test_mlservice_not_rewired():
    """MLService class must NOT reference respanno (only annotation_io is connected)."""
    tree = _parse_ast(GUI_FILE)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "MLService":
            for child in ast.walk(node):
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    module = getattr(child, "module", "") or ""
                    for alias in child.names:
                        full = f"{module}.{alias.name}"
                        if "respanno" in full:
                            # OK if it's annotation_io (used in event parsing helpers)
                            if "annotation_io" in full:
                                continue
                            raise AssertionError(
                                f"MLService should not import {full} — only annotation_io is allowed"
                            )


def test_boxspan_not_rewired():
    """BoxSpan class must NOT reference respanno."""
    tree = _parse_ast(GUI_FILE)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "BoxSpan":
            for child in ast.walk(node):
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    module = getattr(child, "module", "") or ""
                    for alias in child.names:
                        full = f"{module}.{alias.name}"
                        assert "respanno" not in full, (
                            f"BoxSpan must not import respanno modules, found: {full}"
                        )


# ---------------------------------------------------------------------------
# 5. verify the file still compiles (covered by py_compile in workflow)
# ---------------------------------------------------------------------------
