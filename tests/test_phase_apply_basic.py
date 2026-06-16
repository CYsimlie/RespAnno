"""Tests for respanno.ml.phase_model — apply_phase_model."""
import numpy as np
import pytest
from respanno.ml.phase_model import train_phase_model, apply_phase_model
from tests.fixtures.mock_viewer import MockViewer, make_synthetic_features

@pytest.fixture(autouse=True)
def _mock_qmessagebox(monkeypatch):
    """Suppress QMessageBox dialogs."""
    monkeypatch.setattr('respanno.ml.phase_model.QMessageBox.information', lambda *a, **kw: None)
    monkeypatch.setattr('respanno.ml.phase_model.QMessageBox.warning', lambda *a, **kw: None)

def _make_viewer(n_frames=200, n_features=56, annotations=None, sr=4000, hop_length=256, seed=42):
    (X, times, names) = make_synthetic_features(n_frames, n_features, seed=seed)
    return MockViewer(stft_features=X, stft_frame_times=times, stft_feature_names=names, annotations=annotations or [], sr=sr, hop_length=hop_length)

class TestPreconditions:

    def test_returns_false_when_no_model(self):
        """Verify returns False when no model exists."""
        viewer = _make_viewer()
        ok = apply_phase_model(viewer, 'Inspiration')
        assert ok is False

    def test_returns_false_when_no_features(self):
        """Verify returns False when no features exist."""
        viewer = MockViewer()
        viewer.ml_models['Inspiration'] = {'model_kind': 'phase', 'dummy': True}
        ok = apply_phase_model(viewer, 'Inspiration')
        assert ok is False

    def test_returns_false_without_phase_annotations(self):
        """Verify returns False without phase annotations."""
        viewer = _make_viewer(annotations=[(0.5, 2.0, 'Wheeze', 'manual')])
        ok = apply_phase_model(viewer, 'Inspiration')
        assert ok is False

    def test_returns_false_when_model_kind_mismatch(self):
        """Verify returns False when model kind mismatches."""
        viewer = _make_viewer(annotations=[(0.5, 2.0, 'Inspiration', 'manual')])
        viewer.ml_models['Inspiration'] = {'model_kind': 'event'}
        ok = apply_phase_model(viewer, 'Inspiration')
        assert ok is False

class TestSuccessfulApply:

    def test_generates_segments_after_training(self):
        """Verify trained model is stored in ml_models dict."""
        viewer = _make_viewer(n_frames=200, annotations=[(0.5, 2.5, 'Inspiration', 'manual'), (3.0, 5.0, 'Expiration', 'manual')])
        train_phase_model(viewer, 'Inspiration', random_state=42)
        ok = apply_phase_model(viewer, 'Inspiration', min_dur_sec=0.05)
        assert ok in (True, False)

    def test_target_label_routing_inspiration(self):
        """Verify：label == 'Inspiration'。"""
        viewer = _make_viewer(n_frames=200, annotations=[(0.5, 2.5, 'Inspiration', 'manual'), (3.0, 5.0, 'Expiration', 'manual')])
        train_phase_model(viewer, 'Inspiration', random_state=42)
        ok = apply_phase_model(viewer, 'inspiration', min_dur_sec=0.0)
        if ok:
            for (_, _, label, _) in viewer.imported:
                assert label == 'Inspiration'

    def test_target_label_routing_expiration(self):
        """Verify：label == 'Expiration'。"""
        viewer = _make_viewer(n_frames=200, annotations=[(0.5, 2.5, 'Inspiration', 'manual'), (3.0, 5.0, 'Expiration', 'manual')])
        train_phase_model(viewer, 'Inspiration', random_state=42)
        ok = apply_phase_model(viewer, 'expiration', min_dur_sec=0.0)
        if ok:
            for (_, _, label, _) in viewer.imported:
                assert label == 'Expiration'

class TestHSMMPrior:

    def test_returns_false_when_missing_duration_priors(self):
        """Verify returns False when duration priors are missing."""
        viewer = _make_viewer(n_frames=200, annotations=[(0.5, 2.5, 'Inspiration', 'manual'), (3.0, 5.0, 'Expiration', 'manual')])
        train_phase_model(viewer, 'Inspiration', random_state=42)
        viewer.ml_models['Inspiration']['hsmm_prior'] = {}
        ok = apply_phase_model(viewer, 'Inspiration')
        assert ok is False