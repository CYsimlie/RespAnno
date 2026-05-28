"""Tests for respanno.labels.events_importer — auto-import of _events files.

Uses a mock viewer to avoid PyQt5 / real WAV loading.
"""

import os
import tempfile
import pytest

from respanno.labels.events_importer import EventsFileIndexer, DEFAULT_AUTO_IMPORT_CFG


# ---------------------------------------------------------------------------
# Minimal mock viewer
# ---------------------------------------------------------------------------


class MockViewer:
    """Minimal viewer stub for EventsFileIndexer tests."""

    def __init__(self, auto_label_import_settings=None):
        self.auto_label_import_settings = auto_label_import_settings or {}
        self.imported = []       # (start, end, label, source)
        self.status_messages = []

    def finalize_annotation(self, start, end, text, source="manual"):
        self.imported.append((start, end, text, source))

    def statusBar(self):
        return self

    def showMessage(self, msg, timeout=0):
        self.status_messages.append((msg, timeout))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# ---------------------------------------------------------------------------
# resolve_path
# ---------------------------------------------------------------------------


class TestResolvePath:
    def test_finds_matching_csv(self):
        with tempfile.TemporaryDirectory() as d:
            wav = os.path.join(d, "recording.wav")
            evt = os.path.join(d, "recording_events.csv")
            _write(wav, "")
            _write(evt, "start,end,label\n1,2,Wheeze\n")

            indexer = EventsFileIndexer(MockViewer())
            assert indexer.resolve_path(wav) == os.path.abspath(evt)

    def test_finds_matching_txt(self):
        with tempfile.TemporaryDirectory() as d:
            wav = os.path.join(d, "rec.wav")
            evt = os.path.join(d, "rec_events.txt")
            _write(wav, "")
            _write(evt, "1\t2\tWheeze\n")

            indexer = EventsFileIndexer(MockViewer())
            assert indexer.resolve_path(wav) == os.path.abspath(evt)

    def test_finds_matching_json(self):
        with tempfile.TemporaryDirectory() as d:
            wav = os.path.join(d, "r.wav")
            evt = os.path.join(d, "r_events.json")
            _write(wav, "")
            _write(evt, '[{"start":1,"end":2,"label":"X"}]')

            indexer = EventsFileIndexer(MockViewer())
            assert indexer.resolve_path(wav) == os.path.abspath(evt)

    def test_no_match_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            wav = os.path.join(d, "x.wav")
            _write(wav, "")

            indexer = EventsFileIndexer(MockViewer())
            assert indexer.resolve_path(wav) is None

    def test_none_path_returns_none(self):
        indexer = EventsFileIndexer(MockViewer())
        assert indexer.resolve_path(None) is None

    def test_non_string_path_returns_none(self):
        indexer = EventsFileIndexer(MockViewer())
        assert indexer.resolve_path(123) is None

    def test_prefers_csv_over_txt_when_both_exist(self):
        with tempfile.TemporaryDirectory() as d:
            wav = os.path.join(d, "r.wav")
            csv = os.path.join(d, "r_events.csv")
            txt = os.path.join(d, "r_events.txt")
            _write(wav, "")
            _write(csv, "start,end,label\n1,2,W\n")
            _write(txt, "1\t2\tT\n")

            indexer = EventsFileIndexer(MockViewer())
            # csv is priority 0, txt is 1 → csv wins
            assert indexer.resolve_path(wav) == os.path.abspath(csv)

    def test_fallback_to_existence_check(self):
        """Late-created file found via existence fallback."""
        with tempfile.TemporaryDirectory() as d:
            wav = os.path.join(d, "late.wav")
            _write(wav, "")

            indexer = EventsFileIndexer(MockViewer())
            assert indexer.resolve_path(wav) is None

            # create file after initial scan
            evt = os.path.join(d, "late_events.csv")
            _write(evt, "start,end,label\n1,2,X\n")
            assert indexer.resolve_path(wav) == os.path.abspath(evt)


# ---------------------------------------------------------------------------
# build_index
# ---------------------------------------------------------------------------


class TestBuildIndex:
    def test_builds_and_caches(self):
        with tempfile.TemporaryDirectory() as d:
            _write(os.path.join(d, "a.wav"), "")
            _write(os.path.join(d, "a_events.csv"), "start,end,label\n1,2,X\n")

            indexer = EventsFileIndexer(MockViewer())
            indexer.build_index(d)
            assert d in indexer._index
            assert "a" in indexer._index[d]


# ---------------------------------------------------------------------------
# parse_file_cached
# ---------------------------------------------------------------------------


class TestParseFileCached:
    def test_caches_by_mtime(self):
        with tempfile.TemporaryDirectory() as d:
            evt = os.path.join(d, "events.csv")
            _write(evt, "start,end,label\n1.0,2.0,Wheeze\n")

            indexer = EventsFileIndexer(MockViewer())
            rows1 = indexer.parse_file_cached(evt)
            rows2 = indexer.parse_file_cached(evt)
            assert rows1 == rows2
            # cache hit: _parse_cache has an entry
            assert os.path.abspath(evt) in indexer._parse_cache

    def test_non_string_returns_empty(self):
        indexer = EventsFileIndexer(MockViewer())
        assert indexer.parse_file_cached(None) == []
        assert indexer.parse_file_cached(123) == []


# ---------------------------------------------------------------------------
# auto_import
# ---------------------------------------------------------------------------


class TestAutoImport:
    def test_imports_valid_rows(self):
        viewer = MockViewer()
        with tempfile.TemporaryDirectory() as d:
            wav = os.path.join(d, "test.wav")
            evt = os.path.join(d, "test_events.csv")
            _write(wav, "")
            _write(evt, "start,end,label\n1.0,2.0,Wheeze\n3.0,4.0,Crackles\n")

            indexer = EventsFileIndexer(viewer)
            indexer.auto_import(wav)
            assert len(viewer.imported) == 2
            assert viewer.imported[0][2] == "Wheeze"

    def test_no_events_file_no_import(self):
        viewer = MockViewer()
        with tempfile.TemporaryDirectory() as d:
            wav = os.path.join(d, "test.wav")
            _write(wav, "")

            indexer = EventsFileIndexer(viewer)
            indexer.auto_import(wav)
            assert len(viewer.imported) == 0

    def test_skips_end_before_start(self):
        viewer = MockViewer()
        with tempfile.TemporaryDirectory() as d:
            wav = os.path.join(d, "test.wav")
            evt = os.path.join(d, "test_events.csv")
            _write(wav, "")
            _write(evt, "start,end,label\n5.0,3.0,Bad\n1.0,2.0,Good\n")

            indexer = EventsFileIndexer(viewer)
            indexer.auto_import(wav)
            assert len(viewer.imported) == 1
            assert viewer.imported[0][2] == "Good"


# ---------------------------------------------------------------------------
# DEFAULT_AUTO_IMPORT_CFG
# ---------------------------------------------------------------------------


class TestDefaultConfig:
    def test_all_required_keys(self):
        required = {
            "file_format", "file_suffix", "delimiter", "custom_delimiter",
            "skip_header_lines", "start_col", "end_col", "label_col",
            "source_col", "json_start_key", "json_end_key",
            "json_label_key", "json_source_key",
        }
        assert set(DEFAULT_AUTO_IMPORT_CFG.keys()) >= required
