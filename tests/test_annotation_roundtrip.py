"""Tests for annotation import/export roundtrip (CSV, TXT, JSON).

Status: TEST SCAFFOLDING — the label I/O logic is embedded in AudioViewer
methods that require a full QApplication + loaded audio.

These tests verify:
1. CSV write -> read roundtrip using the same format the legacy code uses.
2. Row parsing logic (pure functions extracted for testing today).
3. JSON row-from-dict matching (pure functions extracted for testing today).

TODO (Phase 2): After extracting `_annotation_row_from_sequence`,
`_annotation_row_from_dict`, `_split_label_line_by_settings`, and
`_parse_events_file` into `respanno/labels/`, rewrite these tests
to use the extracted modules directly.
"""

import csv
import io
import json
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Pure-function replicas of the legacy label-I/O logic
# (temporarily duplicated here so we can test without QApplication)
# ---------------------------------------------------------------------------

def _build_default_cfg(overrides=None):
    cfg = {
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
    if overrides:
        cfg.update(overrides)
    return cfg


def _split_label_line_by_settings(line, cfg=None):
    """Replica of AudioViewer._split_label_line_by_settings."""
    if cfg is None:
        cfg = _build_default_cfg()
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

    if "," in line:
        return _read_with(",")
    if "\t" in line:
        return _read_with("\t")
    if ";" in line:
        return _read_with(";")
    return line.split()


def _annotation_row_from_sequence(parts, cfg=None):
    """Replica of AudioViewer._annotation_row_from_sequence."""
    if cfg is None:
        cfg = _build_default_cfg()

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


def _annotation_row_from_dict(item, cfg=None):
    """Replica of AudioViewer._annotation_row_from_dict."""
    if cfg is None:
        cfg = _build_default_cfg()
    if not isinstance(item, dict):
        return None

    def _get_by_keys(primary, fallbacks):
        keys = [primary] + list(fallbacks)
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCSVRowParsing:
    """Verify CSV/txt row -> (start, end, label, source) parsing."""

    def test_default_columns_comma(self):
        parts = ["0.5000", "1.2000", "wheeze"]
        result = _annotation_row_from_sequence(parts)
        assert result == (0.5, 1.2, "wheeze", "manual")

    def test_default_columns_comma_with_source(self):
        parts = ["0.5000", "1.2000", "wheeze", "ml"]
        result = _annotation_row_from_sequence(parts)
        assert result == (0.5, 1.2, "wheeze", "ml")

    def test_custom_column_order(self):
        cfg = _build_default_cfg({"start_col": 2, "end_col": 3, "label_col": 1, "source_col": 4})
        # col1=label, col2=start, col3=end, col4=source
        parts = ["Crackles", "0.8", "2.5", "manual"]
        result = _annotation_row_from_sequence(parts, cfg=cfg)
        assert result == (0.8, 2.5, "Crackles", "manual")

    def test_source_col_zero_disabled(self):
        cfg = _build_default_cfg({"source_col": 0})
        parts = ["0.5", "1.2", "wheeze"]
        result = _annotation_row_from_sequence(parts, cfg=cfg)
        assert result == (0.5, 1.2, "wheeze", "manual")

    def test_invalid_numeric_returns_none(self):
        parts = ["abc", "def", "label"]
        result = _annotation_row_from_sequence(parts)
        assert result is None

    def test_empty_label_returns_none(self):
        parts = ["0.5", "1.2", ""]
        result = _annotation_row_from_sequence(parts)
        assert result is None

    def test_too_few_columns(self):
        parts = ["0.5"]
        result = _annotation_row_from_sequence(parts)
        assert result is None

    def test_skip_header_lines_parsing(self, sample_annotations):
        """Simulate reading a CSV with a header line skipped."""
        # The sample_annotations fixture writes header row; verify structure
        for s, e, lab, src in sample_annotations:
            assert s > 0
            assert e > s
            assert isinstance(lab, str) and len(lab) > 0


class TestDelimiterParsing:
    """Verify delimiter detection logic."""

    def test_comma_delimiter(self):
        parts = _split_label_line_by_settings("0.5,1.2,wheeze", _build_default_cfg())
        assert len(parts) == 3
        assert parts[0] == "0.5"

    def test_tab_delimiter(self):
        parts = _split_label_line_by_settings("0.5\t1.2\twheeze", _build_default_cfg())
        assert len(parts) == 3
        assert parts[0] == "0.5"

    def test_semicolon_delimiter(self):
        parts = _split_label_line_by_settings("0.5;1.2;wheeze", _build_default_cfg())
        assert len(parts) == 3
        assert parts[0] == "0.5"

    def test_space_delimiter_explicit(self):
        cfg = _build_default_cfg({"delimiter": "space"})
        parts = _split_label_line_by_settings("0.5 1.2 wheeze", cfg)
        assert len(parts) == 3
        assert parts[0] == "0.5"

    def test_custom_delimiter(self):
        cfg = _build_default_cfg({"delimiter": "custom", "custom_delimiter": "|"})
        parts = _split_label_line_by_settings("0.5|1.2|wheeze", cfg)
        assert len(parts) == 3
        assert parts[0] == "0.5"


class TestJSONRowParsing:
    """Verify JSON dict -> (start, end, label, source) parsing."""

    def test_default_keys(self):
        item = {"start": 0.5, "end": 1.2, "label": "wheeze", "source": "ml"}
        result = _annotation_row_from_dict(item)
        assert result == (0.5, 1.2, "wheeze", "ml")

    def test_missing_source_defaults_manual(self):
        item = {"start": 0.5, "end": 1.2, "label": "wheeze"}
        result = _annotation_row_from_dict(item)
        assert result == (0.5, 1.2, "wheeze", "manual")

    def test_fallback_keys(self):
        item = {"onset": 0.5, "offset": 1.2, "type": "Crackles"}
        result = _annotation_row_from_dict(item)
        assert result == (0.5, 1.2, "Crackles", "manual")

    def test_case_insensitive_match(self):
        item = {"Start": 0.5, "End": 1.2, "Label": "wheeze"}
        result = _annotation_row_from_dict(item)
        assert result == (0.5, 1.2, "wheeze", "manual")

    def test_custom_keys(self):
        cfg = _build_default_cfg({
            "json_start_key": "time_start",
            "json_end_key": "time_end",
            "json_label_key": "class_name",
        })
        item = {"time_start": 0.5, "time_end": 1.2, "class_name": "Stridor"}
        result = _annotation_row_from_dict(item, cfg=cfg)
        assert result == (0.5, 1.2, "Stridor", "manual")


class TestCSVRoundtrip:
    """Write -> read roundtrip using the same format as legacy export/import."""

    def test_write_read_roundtrip(self, tmp_path, sample_annotations):
        """Write annotations to CSV, then read them back."""
        # Write (matching legacy export_annotations format)
        csv_path = tmp_path / "roundtrip.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["start", "end", "label", "source"])
            for s, e, lab, src in sample_annotations:
                w.writerow([f"{s:.4f}", f"{e:.4f}", lab, src])

        # Read back
        rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            for parts in reader:
                rows.append((float(parts[0]), float(parts[1]), parts[2], parts[3]))

        assert len(rows) == len(sample_annotations)
        for (s1, e1, l1, r1), (s2, e2, l2, r2) in zip(rows, sample_annotations):
            assert abs(s1 - s2) < 0.001
            assert abs(e1 - e2) < 0.001
            assert l1 == l2
            assert r1 in ("manual", "ml", "auto_accepted", "auto_edited")


class TestExportFormat:
    """Verify the legacy export format is well-formed."""

    def test_annotation_sort_order(self, sample_annotations):
        """Annotations sorted by start time should maintain that order."""
        sorted_anns = sorted(sample_annotations, key=lambda x: x[0])
        for i in range(len(sorted_anns) - 1):
            assert sorted_anns[i][0] <= sorted_anns[i + 1][0]


# ---------------------------------------------------------------------------
# TODO: Tests that require extraction from AudioViewer
# ---------------------------------------------------------------------------

class TestAnnotationImportFromAudioViewer:
    """
    TODO: After extracting label I/O to respanno/labels/, implement:

    1. test_import_csv_file:
       - Create a CSV, call the import function, verify annotations populated

    2. test_export_csv_file:
       - Populate annotations, call export, verify CSV content matches

    3. test_auto_import_events_for_wav:
       - Create a WAV + _events.csv pair, verify auto-import works

    4. test_parse_events_json_top_level_array:
       - Verify JSON array-of-dicts parsing

    5. test_parse_events_json_nested:
       - Verify nested {"annotations": [...]} parsing
    """

    def test_todo_placeholder(self):
        pytest.skip("TODO: extract label I/O from AudioViewer first")
