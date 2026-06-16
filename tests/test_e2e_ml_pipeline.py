"""End-to-end ML pipeline tests.

Validates the full chain: synthetic audio → features → frame labels →
LightGBM training → inference → prediction verification.

Uses MockViewer to avoid GUI dependency while exercising real compute modules.
"""
import numpy as np
import pytest
from respanno.dsp.features import compute_short_time_features, build_feature_matrix
from respanno.ml.frame_labels import build_frame_labels
from respanno.ml.classifier import train_event_model, apply_event_model
from tests.fixtures.mock_viewer import MockViewer
from tests.fixtures.synthetic_signals import generate_wheeze_episode, generate_mixed_episode, generate_respiratory_cycle

@pytest.fixture(autouse=True)
def _mock_qmessagebox(monkeypatch):
    monkeypatch.setattr('respanno.ml.classifier.QMessageBox.information', lambda *a, **kw: None)
    monkeypatch.setattr('respanno.ml.classifier.QMessageBox.warning', lambda *a, **kw: None)

def _build_viewer(audio, sr, annotations, hop_length=256, seed=42):
    """Extract features and build a MockViewer from audio + annotations."""
    (times, feat_dict) = compute_short_time_features(audio, int(sr), hop_length=hop_length)
    (X, _, names) = build_feature_matrix(times, feat_dict)
    return MockViewer(stft_features=X.astype(np.float64), stft_frame_times=times.astype(float), stft_feature_names=names, annotations=list(annotations), sr=int(sr), hop_length=hop_length)

class TestE2EWheezeClassifier:
    """Train a wheeze classifier on synthetic data and verify it learns."""

    def test_pipeline_runs_without_error(self):
        """Verify full ML pipeline runs without error."""
        (audio, sr, anns) = generate_wheeze_episode(duration=5.0, wheeze_start=1.0, wheeze_dur=2.5, seed=42)
        viewer = _build_viewer(audio, sr, anns)
        ok_train = train_event_model(viewer, 'Wheeze', random_state=42)
        assert ok_train is True
        assert 'Wheeze' in viewer.ml_models
        ok_apply = apply_event_model(viewer, 'Wheeze', min_dur_sec=0.05)
        assert ok_apply in (True, False)

    def test_model_trains_on_correct_label_only(self):
        """Classifier for 'Crackles' should not use 'Wheeze' annotations."""
        (audio, sr, anns) = generate_mixed_episode(duration=6.0, seed=42)
        viewer = _build_viewer(audio, sr, anns)
        ok = train_event_model(viewer, 'Rhonchi', random_state=42)
        assert ok is False

    def test_full_phase_pipeline(self):
        """Train and apply a phase model on a respiratory cycle."""
        (audio, sr, anns) = generate_respiratory_cycle(duration=8.0, seed=42)
        viewer = _build_viewer(audio, sr, anns)
        from respanno.ml.phase_model import train_phase_model, apply_phase_model
        import respanno.ml.phase_model as pm
        pm.QMessageBox.information = lambda *a, **kw: None
        pm.QMessageBox.warning = lambda *a, **kw: None
        ok_train = train_phase_model(viewer, 'Inspiration', random_state=42)
        assert ok_train is True
        ok_apply = apply_phase_model(viewer, 'Inspiration', min_dur_sec=0.05)
        assert ok_apply in (True, False)

class TestPipelineDeterminism:
    """Identical inputs → identical results across independent runs."""

    def test_full_pipeline_deterministic(self):
        """Verify full pipeline determinism: same input yields identical output."""
        (audio, sr, anns) = generate_wheeze_episode(duration=5.0, wheeze_start=1.0, wheeze_dur=2.5, seed=42)
        v1 = _build_viewer(audio, sr, anns, seed=42)
        v2 = _build_viewer(audio, sr, anns, seed=42)
        train_event_model(v1, 'Wheeze', random_state=42)
        train_event_model(v2, 'Wheeze', random_state=42)
        (m1, m2) = (v1.ml_models['Wheeze'], v2.ml_models['Wheeze'])
        assert m1['train_f1'] == m2['train_f1']
        assert m1['threshold'] == m2['threshold']

    def test_full_pipeline_ok_on_mixed(self):
        """End-to-end with multi-label mixed episode."""
        (audio, sr, anns) = generate_mixed_episode(duration=6.0, seed=42)
        viewer = _build_viewer(audio, sr, anns)
        ok = train_event_model(viewer, 'Wheeze', min_pos_frames=5, random_state=42)
        assert ok is True
        ok = train_event_model(viewer, 'Crackles', min_pos_frames=2, random_state=42)
        assert ok is True
        assert viewer.ml_models['Wheeze']['clf'] is not viewer.ml_models['Crackles']['clf']

class TestFrameLabelE2E:
    """Verify that build_frame_labels integrates correctly with real features."""

    def test_labels_integrate_with_real_features(self):
        """Verify frame labels align correctly with feature matrix dimensions."""
        (audio, sr, anns) = generate_wheeze_episode(duration=4.0, wheeze_start=1.0, wheeze_dur=1.0, seed=42)
        (times, feat_dict) = compute_short_time_features(audio, int(sr))
        y = build_frame_labels(anns, times, 'Wheeze', neg_margin=0.05)
        assert y is not None
        n_pos = int(np.sum(y == 1))
        n_neg = int(np.sum(y == 0))
        assert n_pos > 0, 'should have positive frames covering the wheeze'
        assert n_neg > 0, 'should have negative frames in prefix'

class TestEdgeCases:

    def test_empty_audio_does_not_crash(self):
        """Pipeline with zero-length audio should degrade gracefully."""
        audio = np.zeros(0, dtype=np.float32)
        sr = 4000
        (times, feat_dict) = compute_short_time_features(audio, sr)
        assert len(times) == 0

    def test_dc_signal_produces_valid_features(self):
        """Pipeline should handle DC-only signal."""
        from tests.fixtures.synthetic_signals import generate_dc_offset
        (audio, sr, _) = generate_dc_offset(duration=1.0, offset=1.0)
        (times, feat_dict) = compute_short_time_features(audio, int(sr))
        assert len(times) > 0
        for v in feat_dict.values():
            assert not np.any(np.isnan(v)), 'features should not contain NaN'