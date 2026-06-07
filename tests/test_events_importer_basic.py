"""Tests for respanno.labels.events_importer — auto-import of _events files.

Uses a mock viewer to avoid PyQt5 / real WAV loading.
"""
import os
import tempfile
import pytest
from respanno.labels.events_importer import EventsFileIndexer, DEFAULT_AUTO_IMPORT_CFG

class MockViewer:
    """Minimal viewer stub for EventsFileIndexer tests."""

    def __init__(self, auto_label_import_settings=None):
        self.auto_label_import_settings = auto_label_import_settings or {}
        self.imported = []
        self.status_messages = []

    def finalize_annotation(self, start, end, text, source='manual'):
        self.imported.append((start, end, text, source))

    def statusBar(self):
        return self

    def showMessage(self, msg, timeout=0):
        self.status_messages.append((msg, timeout))

def _write(path, content):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

class TestResolvePath:

    def test_finds_matching_csv(self):
        """Verifyauto匹配同名 _csv eventfile。"""
        with tempfile.TemporaryDirectory() as d:
            wav = os.path.join(d, 'recording.wav')
            evt = os.path.join(d, 'recording_events.csv')
            _write(wav, '')
            _write(evt, 'start,end,label\n1,2,Wheeze\n')
            indexer = EventsFileIndexer(MockViewer())
            assert indexer.resolve_path(wav) == os.path.abspath(evt)

    def test_finds_matching_txt(self):
        """Verifyauto匹配同名 _txt eventfile。"""
        with tempfile.TemporaryDirectory() as d:
            wav = os.path.join(d, 'rec.wav')
            evt = os.path.join(d, 'rec_events.txt')
            _write(wav, '')
            _write(evt, '1\t2\tWheeze\n')
            indexer = EventsFileIndexer(MockViewer())
            assert indexer.resolve_path(wav) == os.path.abspath(evt)

    def test_finds_matching_json(self):
        """Verifyauto匹配同名 _json eventfile。"""
        with tempfile.TemporaryDirectory() as d:
            wav = os.path.join(d, 'r.wav')
            evt = os.path.join(d, 'r_events.json')
            _write(wav, '')
            _write(evt, '[{"start":1,"end":2,"label":"X"}]')
            indexer = EventsFileIndexer(MockViewer())
            assert indexer.resolve_path(wav) == os.path.abspath(evt)

    def test_no_match_returns_none(self):
        """Verify空input或 None input时的行为。"""
        with tempfile.TemporaryDirectory() as d:
            wav = os.path.join(d, 'x.wav')
            _write(wav, '')
            indexer = EventsFileIndexer(MockViewer())
            assert indexer.resolve_path(wav) is None

    def test_none_path_returns_none(self):
        """Verify空input或 None input时的行为。"""
        indexer = EventsFileIndexer(MockViewer())
        assert indexer.resolve_path(None) is None

    def test_non_string_path_returns_none(self):
        """Verify空input或 None input时的行为。"""
        indexer = EventsFileIndexer(MockViewer())
        assert indexer.resolve_path(123) is None

    def test_prefers_csv_over_txt_when_both_exist(self):
        """Verify：indexer.resolve_path(wav) == os.path.abspath(csv)。"""
        with tempfile.TemporaryDirectory() as d:
            wav = os.path.join(d, 'r.wav')
            csv = os.path.join(d, 'r_events.csv')
            txt = os.path.join(d, 'r_events.txt')
            _write(wav, '')
            _write(csv, 'start,end,label\n1,2,W\n')
            _write(txt, '1\t2\tT\n')
            indexer = EventsFileIndexer(MockViewer())
            assert indexer.resolve_path(wav) == os.path.abspath(csv)

    def test_fallback_to_existence_check(self):
        """Late-created file found via existence fallback."""
        with tempfile.TemporaryDirectory() as d:
            wav = os.path.join(d, 'late.wav')
            _write(wav, '')
            indexer = EventsFileIndexer(MockViewer())
            assert indexer.resolve_path(wav) is None
            evt = os.path.join(d, 'late_events.csv')
            _write(evt, 'start,end,label\n1,2,X\n')
            assert indexer.resolve_path(wav) == os.path.abspath(evt)

class TestBuildIndex:

    def test_builds_and_caches(self):
        """Verify：d in indexer._index。"""
        with tempfile.TemporaryDirectory() as d:
            _write(os.path.join(d, 'a.wav'), '')
            _write(os.path.join(d, 'a_events.csv'), 'start,end,label\n1,2,X\n')
            indexer = EventsFileIndexer(MockViewer())
            indexer.build_index(d)
            assert d in indexer._index
            assert 'a' in indexer._index[d]

class TestParseFileCached:

    def test_caches_by_mtime(self):
        """Verify：rows1 == rows2。"""
        with tempfile.TemporaryDirectory() as d:
            evt = os.path.join(d, 'events.csv')
            _write(evt, 'start,end,label\n1.0,2.0,Wheeze\n')
            indexer = EventsFileIndexer(MockViewer())
            rows1 = indexer.parse_file_cached(evt)
            rows2 = indexer.parse_file_cached(evt)
            assert rows1 == rows2
            assert os.path.abspath(evt) in indexer._parse_cache

    def test_non_string_returns_empty(self):
        """Verify空input或 None input时的行为。"""
        indexer = EventsFileIndexer(MockViewer())
        assert indexer.parse_file_cached(None) == []
        assert indexer.parse_file_cached(123) == []

class TestAutoImport:

    def test_imports_valid_rows(self):
        """Verify respanno 子包可被 importlib.import_module 正常import。"""
        viewer = MockViewer()
        with tempfile.TemporaryDirectory() as d:
            wav = os.path.join(d, 'test.wav')
            evt = os.path.join(d, 'test_events.csv')
            _write(wav, '')
            _write(evt, 'start,end,label\n1.0,2.0,Wheeze\n3.0,4.0,Crackles\n')
            indexer = EventsFileIndexer(viewer)
            indexer.auto_import(wav)
            assert len(viewer.imported) == 2
            assert viewer.imported[0][2] == 'Wheeze'

    def test_no_events_file_no_import(self):
        """Verify：len(viewer.imported) == 0。"""
        viewer = MockViewer()
        with tempfile.TemporaryDirectory() as d:
            wav = os.path.join(d, 'test.wav')
            _write(wav, '')
            indexer = EventsFileIndexer(viewer)
            indexer.auto_import(wav)
            assert len(viewer.imported) == 0

    def test_skips_end_before_start(self):
        """Verify：len(viewer.imported) == 1。"""
        viewer = MockViewer()
        with tempfile.TemporaryDirectory() as d:
            wav = os.path.join(d, 'test.wav')
            evt = os.path.join(d, 'test_events.csv')
            _write(wav, '')
            _write(evt, 'start,end,label\n5.0,3.0,Bad\n1.0,2.0,Good\n')
            indexer = EventsFileIndexer(viewer)
            indexer.auto_import(wav)
            assert len(viewer.imported) == 1
            assert viewer.imported[0][2] == 'Good'

class TestDefaultConfig:

    def test_all_required_keys(self):
        """Verify：set(DEFAULT_AUTO_IMPORT_CFG.keys()) >= required。"""
        required = {'file_format', 'file_suffix', 'delimiter', 'custom_delimiter', 'skip_header_lines', 'start_col', 'end_col', 'label_col', 'source_col', 'json_start_key', 'json_end_key', 'json_label_key', 'json_source_key'}
        assert set(DEFAULT_AUTO_IMPORT_CFG.keys()) >= required