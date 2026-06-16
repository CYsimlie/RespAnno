"""Tests for annotation data-quality invariants.

Validates that annotations—whether manual, imported, or ML-generated—
satisfy basic quality constraints.
"""
import json
import os
import tempfile
import numpy as np
import pytest
from respanno.labels.annotation_io import normalize_annotation, read_annotations, write_annotations
FIXTURE_DIR = os.path.join(os.path.dirname(__file__), 'fixtures', 'annotations')

class TestNormalizeInvariants:

    def test_start_before_end(self):
        """Verify annotations with start >= end are rejected and return None."""
        ann = normalize_annotation((1.5, 0.5, 'X'))
        assert ann is None

    def test_empty_label(self):
        """Verify behaviour on empty or None input。"""
        ann = normalize_annotation((0.5, 1.0, ''))
        assert ann is None

    def test_always_has_source(self):
        """Verify annotation source provenance is preserved."""
        ann = normalize_annotation((0.5, 1.0, 'Wheeze'))
        assert 'source' in ann

    def test_default_source_is_manual(self):
        """Verify default parameter values are correct."""
        ann = normalize_annotation((0.5, 1.0, 'Wheeze'))
        assert ann['source'] == 'manual'

    def test_source_preserved(self):
        """Verify annotation source provenance is preserved."""
        ann = normalize_annotation((0.5, 1.0, 'Wheeze', 'ml'))
        assert ann['source'] == 'ml'

    def test_none_returns_none(self):
        """Verify behaviour on empty or None input。"""
        assert normalize_annotation(None) is None

class TestRoundtrip:

    def test_csv_roundtrip_preserves_all_fields(self):
        """Verify CSV read/write roundtrip preserves all fields exactly."""
        original = [(0.5, 1.2, 'Wheeze', 'manual'), (1.8, 2.4, 'Crackles', 'ml'), (3.0, 4.1, 'Inspiration', 'auto_accepted')]
        anns = [normalize_annotation(a) for a in original]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            write_annotations(f.name, anns)
            path = f.name
        try:
            loaded = read_annotations(path)
            assert len(loaded) == 3
            for (o, l) in zip(original, loaded):
                assert o[0] == pytest.approx(l['start'])
                assert o[1] == pytest.approx(l['end'])
                assert o[2] == l['label']
                assert (o[3] if len(o) >= 4 else 'manual') == l['source']
        finally:
            os.unlink(path)

    def test_json_roundtrip_preserves_all_fields(self):
        """Verify JSON read/write roundtrip preserves all fields exactly."""
        original = [(0.5, 1.2, 'Wheeze', 'manual'), (1.8, 2.4, 'Crackles', 'auto_edited'), (3.0, 4.1, 'Inspiration', 'merged')]
        anns = [normalize_annotation(a) for a in original]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            write_annotations(f.name, anns)
            path = f.name
        try:
            loaded = read_annotations(path)
            assert len(loaded) == 3
            for (o, l) in zip(original, loaded):
                assert o[0] == pytest.approx(l['start'])
                assert o[1] == pytest.approx(l['end'])
                assert o[2] == l['label']
                assert (o[3] if len(o) >= 4 else 'manual') == l['source']
        finally:
            os.unlink(path)

class TestSourceProvenance:
    """Each annotation MUST retain its source field through any transform."""
    VALID_SOURCES = {'manual', 'ml', 'auto_accepted', 'auto_edited', 'merged'}

    def test_all_sources_survive_normalize(self):
        """Verify annotation source provenance is preserved."""
        for src in self.VALID_SOURCES:
            ann = normalize_annotation((0.5, 1.0, 'X', src))
            assert ann is not None
            assert ann['source'] == src

    def test_all_sources_survive_csv_roundtrip(self):
        """Verify annotation source provenance is preserved."""
        original = [(0.5, 1.0, 'X', src) for src in self.VALID_SOURCES]
        anns = [normalize_annotation(a) for a in original]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            write_annotations(f.name, anns)
            path = f.name
        try:
            loaded = read_annotations(path)
            assert len(loaded) == len(self.VALID_SOURCES)
            for (a, b) in zip(original, loaded):
                src = a[3] if len(a) >= 4 else 'manual'
                assert b['source'] == src
        finally:
            os.unlink(path)

def overlap_ratio(seg, base):
    """Calculates the IoU-style overlap ratio between two segments."""
    (s1, e1) = seg
    (s2, e2) = base
    inter = min(e1, e2) - max(s1, s2)
    if inter <= 0:
        return 0.0
    return inter / max(e1 - s1, 1e-06)

class TestOverlapRatio:

    def test_no_overlap(self):
        """Verify：overlap_ratio((0, 1), (2, 3)) == 0.0。"""
        assert overlap_ratio((0, 1), (2, 3)) == 0.0

    def test_full_overlap(self):
        """Verify：overlap_ratio((0, 2), (0, 2)) == pytest.approx(1.0)。"""
        assert overlap_ratio((0, 2), (0, 2)) == pytest.approx(1.0)

    def test_partial_overlap(self):
        """Verify：overlap_ratio((0, 2), (1, 3)) == pytest.approx(0.5)。"""
        assert overlap_ratio((0, 2), (1, 3)) == pytest.approx(0.5)

    def test_contained(self):
        """Verify：overlap_ratio((0.5, 1.5), (0, 2)) == pytest.approx(1.0)。"""
        assert overlap_ratio((0.5, 1.5), (0, 2)) == pytest.approx(1.0)

    def test_threshold_50_percent(self):
        """Segments with ≥50% overlap should be dedup'd."""
        assert overlap_ratio((0, 2), (1, 3)) >= 0.5
        assert overlap_ratio((0, 2), (1.5, 3)) < 0.5

class TestFixtureFiles:

    def test_simple_csv_readable(self):
        """Verify：len(rows) == 5。"""
        path = os.path.join(FIXTURE_DIR, 'simple_5rows.csv')
        rows = read_annotations(path)
        assert len(rows) == 5

    def test_simple_json_readable(self):
        """Verify：len(rows) == 5。"""
        path = os.path.join(FIXTURE_DIR, 'simple_5rows.json')
        rows = read_annotations(path)
        assert len(rows) == 5

    def test_simple_txt_readable(self):
        """Verify：len(rows) == 5。"""
        path = os.path.join(FIXTURE_DIR, 'simple_5rows_tab.txt')
        rows = read_annotations(path)
        assert len(rows) == 5

    def test_with_source_col(self):
        """Verify annotation source provenance is preserved."""
        path = os.path.join(FIXTURE_DIR, 'with_source_col.csv')
        rows = read_annotations(path)
        sources = {r['source'] for r in rows}
        assert 'ml' in sources
        assert 'auto_accepted' in sources
        assert 'merged' in sources

    def test_empty_file(self):
        """Verify behaviour on empty or None input。"""
        path = os.path.join(FIXTURE_DIR, 'empty.csv')
        rows = read_annotations(path)
        assert rows == []

    def test_bad_rows_tolerated(self):
        """Verify：isinstance(rows, list)。"""
        path = os.path.join(FIXTURE_DIR, 'bad_rows.csv')
        rows = read_annotations(path)
        assert isinstance(rows, list)