"""Tests for respanno.ml.frame_labels — frame-level label builder."""

import numpy as np
import pytest

from respanno.ml.frame_labels import (
    build_frame_labels,
    get_manual_segments,
    get_reviewed_prefix,
    _iter_reviewed_annotations,
)


def _make_times(start, end, hop=0.064):
    return np.arange(start, end, hop, dtype=float)


# ---------------------------------------------------------------------------
# _iter_reviewed_annotations
# ---------------------------------------------------------------------------


class TestIterReviewedAnnotations:
    def test_3tuple_yields_manual(self):
        anns = [(0.5, 1.2, "Wheeze")]
        items = list(_iter_reviewed_annotations(anns))
        assert len(items) == 1
        assert items[0] == (0.5, 1.2, "Wheeze")

    def test_4tuple_yields_with_source(self):
        anns = [(0.5, 1.2, "Wheeze", "manual")]
        items = list(_iter_reviewed_annotations(anns))
        assert len(items) == 1

    def test_skips_non_reviewed_source(self):
        """ml and archived sources are NOT reviewed."""
        anns = [
            (0.1, 0.5, "Wheeze", "ml"),
            (0.6, 1.0, "Crackles", "archived"),
            (1.1, 1.5, "Rhonchi", "manual"),
        ]
        items = list(_iter_reviewed_annotations(anns))
        assert len(items) == 1
        assert items[0][2] == "Rhonchi"

    def test_skips_none(self):
        anns = [None, (0.1, 0.5, "Wheeze"), None]
        items = list(_iter_reviewed_annotations(anns))
        assert len(items) == 1

    def test_all_reviewed_sources_accepted(self):
        for src in ("manual", "auto_accepted", "auto_edited", "merged", "merged_thresh_ctx"):
            items = list(_iter_reviewed_annotations([(0.1, 0.5, "X", src)]))
            assert len(items) == 1, f"source '{src}' should be reviewed"

    def test_case_insensitive_source(self):
        items = list(_iter_reviewed_annotations([(0.1, 0.5, "X", "MANUAL")]))
        assert len(items) == 1


# ---------------------------------------------------------------------------
# get_manual_segments
# ---------------------------------------------------------------------------


class TestGetManualSegments:
    def test_returns_matching_label(self):
        anns = [
            (0.1, 0.5, "Wheeze"),
            (0.6, 1.0, "Crackles"),
            (1.1, 1.5, "Wheeze"),
        ]
        segs = get_manual_segments(anns, "Wheeze")
        assert segs == [(0.1, 0.5), (1.1, 1.5)]

    def test_empty_when_no_match(self):
        anns = [(0.1, 0.5, "Crackles")]
        assert get_manual_segments(anns, "Wheeze") == []

    def test_empty_annotations(self):
        assert get_manual_segments([], "Wheeze") == []


# ---------------------------------------------------------------------------
# get_reviewed_prefix
# ---------------------------------------------------------------------------


class TestGetReviewedPrefix:
    def test_max_end_time(self):
        anns = [(0.1, 0.5, "A"), (0.6, 2.0, "B"), (1.0, 1.5, "C")]
        assert get_reviewed_prefix(anns) == pytest.approx(2.0)

    def test_zero_when_empty(self):
        assert get_reviewed_prefix([]) == 0.0


# ---------------------------------------------------------------------------
# build_frame_labels
# ---------------------------------------------------------------------------


class TestBuildFrameLabels:
    def test_empty_frame_times_returns_none(self):
        y = build_frame_labels([(0.1, 1.0, "Wheeze")], np.array([]), "Wheeze")
        assert y is None

    def test_empty_annotations_returns_none(self):
        times = _make_times(0, 2.0)
        y = build_frame_labels([], times, "Wheeze")
        assert y is None

    def test_positive_frames_marked_one(self):
        """Frames inside a Wheeze segment → label 1."""
        anns = [(0.5, 1.0, "Wheeze")]
        times = _make_times(0, 2.0)
        y = build_frame_labels(anns, times, "Wheeze")
        assert y is not None
        pos_idx = np.where((times >= 0.5) & (times <= 1.0))[0]
        assert np.all(y[pos_idx] == 1)

    def test_prefix_boundary(self):
        """Frames beyond reviewed prefix → label -1 (ignored)."""
        anns = [(0.5, 1.0, "Wheeze")]
        times = _make_times(0, 3.0)
        y = build_frame_labels(anns, times, "Wheeze")
        assert y is not None
        beyond = np.where(times > 1.0)[0]
        assert np.all(y[beyond] == -1)

    def test_negative_zone_around_positive(self):
        """Frames within neg_margin of positive segment NOT marked negative."""
        anns = [(0.5, 1.0, "Wheeze")]
        times = _make_times(0, 2.0)
        y = build_frame_labels(anns, times, "Wheeze", neg_margin=0.1)
        assert y is not None
        # 0.1s before 0.5s = 0.4s; within margin → should NOT be 0
        near = np.where((times >= 0.40) & (times <= 0.50))[0]
        for idx in near:
            assert y[idx] != 0, f"frame at {times[idx]:.3f}s within margin should not be 0"

    def test_safe_negatives_marked_zero(self):
        """Frames outside all positive segments → label 0."""
        anns = [(0.5, 1.0, "Wheeze")]
        times = _make_times(0, 2.0)
        y = build_frame_labels(anns, times, "Wheeze", neg_margin=0.05)
        assert y is not None
        # well before the segment
        early = np.where(times < 0.4)[0]
        assert len(early) > 0
        assert np.any(y[early] == 0)

    def test_hard_negative_segments(self):
        """neg_segments explicitly marks certain frames as 0."""
        anns = [(0.5, 2.0, "Wheeze")]
        times = _make_times(0, 2.0)
        # place hard-neg outside the positive region (0.5-2.0)
        # but within the prefix → frames 0.1-0.4 are safe negatives
        neg_segs = {"Wheeze": [(0.1, 0.4)]}
        y = build_frame_labels(anns, times, "Wheeze", neg_segments=neg_segs)
        assert y is not None
        idx = np.where((times >= 0.15) & (times <= 0.35))[0]
        assert np.all(y[idx] == 0), "hard-negative frames should be 0"

    def test_returns_none_when_no_pos_and_no_neg(self):
        """No usable frames for training → None."""
        times = _make_times(0, 5.0)  # times beyond the short annotation
        anns = [(0.1, 0.2, "Wheeze")]  # prefix = 0.2
        # All frames after 0.2 are -1, no pos (times 0-0.064 maybe? actually 0-0.064 is in prefix and not pos, could be neg)
        # Let me make it clear: annotation prefix at 0.2, times start at 1.0
        times2 = _make_times(1.0, 3.0)
        y = build_frame_labels([(0.1, 0.2, "Wheeze")], times2, "Wheeze")
        assert y is None
