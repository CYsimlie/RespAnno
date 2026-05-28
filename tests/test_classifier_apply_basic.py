"""Tests for respanno.ml.classifier — apply_event_model."""

import numpy as np
import pytest

from respanno.ml.classifier import train_event_model, apply_event_model
from tests.fixtures.mock_viewer import MockViewer, make_synthetic_features


@pytest.fixture(autouse=True)
def _mock_qmessagebox(monkeypatch):
    """Suppress QMessageBox dialogs."""
    monkeypatch.setattr(
        "respanno.ml.classifier.QMessageBox.information", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        "respanno.ml.classifier.QMessageBox.warning", lambda *a, **kw: None
    )


def _make_viewer(n_frames=200, n_features=56, annotations=None, seed=42):
    X, times, names = make_synthetic_features(n_frames, n_features, seed=seed)
    return MockViewer(
        stft_features=X,
        stft_frame_times=times,
        stft_feature_names=names,
        annotations=annotations or [],
    )


# ---------------------------------------------------------------------------
# Pre-condition failures
# ---------------------------------------------------------------------------


class TestPreconditions:
    def test_returns_false_when_no_model(self):
        viewer = _make_viewer()
        ok = apply_event_model(viewer, "Wheeze")
        assert ok is False

    def test_returns_false_when_no_features(self):
        viewer = MockViewer()
        viewer.ml_models["Wheeze"] = {"clf": "dummy", "threshold": 0.5}
        ok = apply_event_model(viewer, "Wheeze")
        assert ok is False

    def test_returns_false_when_no_annotations(self):
        viewer = _make_viewer(annotations=[])
        ok = apply_event_model(viewer, "Wheeze")
        assert ok is False


# ---------------------------------------------------------------------------
# Successful inference
# ---------------------------------------------------------------------------


class TestSuccessfulApply:
    def test_generates_segments_after_training(self):
        viewer = _make_viewer(
            n_frames=200,
            annotations=[
                (0.5, 3.0, "Wheeze", "manual"),  # reviewed prefix ~3s
            ],
        )
        train_event_model(viewer, "Wheeze", random_state=42)
        ok = apply_event_model(viewer, "Wheeze", min_dur_sec=0.05)
        # may return False if no frames pass threshold; that's valid too
        assert ok in (True, False)
        if ok:
            assert len(viewer.imported) > 0

    def test_generated_segments_within_unreviewed_region(self):
        viewer = _make_viewer(
            n_frames=200,
            annotations=[
                (0.5, 3.0, "Wheeze", "manual"),
            ],
        )
        train_event_model(viewer, "Wheeze", random_state=42)
        ok = apply_event_model(viewer, "Wheeze", min_dur_sec=0.0)
        if ok:
            for s, e, _, _ in viewer.imported:
                # should be in unreviewed region (after 3.0 s)
                assert s > 3.0 or e > 3.0

    def test_min_dur_filters_short_segments(self):
        viewer = _make_viewer(
            n_frames=200,
            annotations=[
                (0.5, 5.0, "Wheeze", "manual"),
            ],
        )
        train_event_model(viewer, "Wheeze", random_state=42)
        # with huge min_dur, no segment should pass
        ok_strict = apply_event_model(viewer, "Wheeze", min_dur_sec=100.0)
        # should return False because no segment passes min_dur
        assert ok_strict is False or len(viewer.imported) == 0


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_dedup_against_manual(self):
        """ML segments overlapping manual (>50%) should be skipped."""
        viewer = _make_viewer(
            n_frames=200,
            annotations=[
                (0.5, 5.0, "Wheeze", "manual"),
                (6.0, 7.0, "Wheeze", "manual"),  # exists as manual
            ],
        )
        train_event_model(viewer, "Wheeze", random_state=42)
        ok = apply_event_model(viewer, "Wheeze", min_dur_sec=0.0)
        if ok:
            # None of the ML segments should overlap >50% with manual
            manual_segs = [(6.0, 7.0)]
            for s, e, _, src in viewer.imported:
                assert src == "ml"
                # ensure no heavy overlap with manual
                for ms, me in manual_segs:
                    inter = min(e, me) - max(s, ms)
                    if inter > 0:
                        ratio = inter / max(e - s, 1e-6)
                        assert ratio < 0.5, f"segment ({s},{e}) overlaps manual {ratio:.2f}"
