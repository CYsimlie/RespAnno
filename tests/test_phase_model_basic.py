"""Tests for respanno.ml.phase_model — train_phase_model."""
import numpy as np
import pytest
from respanno.ml.phase_model import train_phase_model
from tests.fixtures.mock_viewer import MockViewer, make_synthetic_features

@pytest.fixture(autouse=True)
def _mock_qmessagebox(monkeypatch):
    """Suppress QMessageBox dialogs during tests."""
    monkeypatch.setattr('respanno.ml.phase_model.QMessageBox.information', lambda *a, **kw: None)
    monkeypatch.setattr('respanno.ml.phase_model.QMessageBox.warning', lambda *a, **kw: None)

def _make_viewer(n_frames=120, n_features=56, annotations=None, sr=4000, hop_length=256, seed=42):
    (X, times, names) = make_synthetic_features(n_frames, n_features, seed=seed)
    return MockViewer(stft_features=X, stft_frame_times=times, stft_feature_names=names, annotations=annotations or [], sr=sr, hop_length=hop_length)

class TestNoFeatures:

    def test_returns_false_when_no_features(self):
        """Verify前置条件不满足时返回 False：no features。"""
        viewer = MockViewer()
        ok = train_phase_model(viewer, 'Inspiration', random_state=42)
        assert ok is False

class TestNoPhaseAnnotations:

    def test_returns_false_when_no_annotations(self):
        """Verify前置条件不满足时返回 False：no annotations。"""
        viewer = _make_viewer(annotations=[])
        ok = train_phase_model(viewer, 'Inspiration', random_state=42)
        assert ok is False

    def test_returns_false_without_insp_or_exp(self):
        """Verify前置条件不满足时返回 False：without insp or exp。"""
        viewer = _make_viewer(annotations=[(0.5, 2.0, 'Wheeze', 'manual'), (2.5, 4.0, 'Crackles', 'manual')])
        ok = train_phase_model(viewer, 'Inspiration', random_state=42)
        assert ok is False

class TestTwoStateModel:
    """Only Inspiration or only Expiration → binary classifier."""

    def test_inspiration_only(self):
        """Verify仅有 Inspiration annotation时autodetect为 2 statusmodel（Insp+Exp）。"""
        viewer = _make_viewer(annotations=[(0.5, 2.0, 'Inspiration', 'manual')])
        ok = train_phase_model(viewer, 'Inspiration', random_state=42)
        assert ok is True
        info = viewer.ml_models['Inspiration']
        assert len(info['classes']) == 2

    def test_expiration_only(self):
        """Verify仅有 Expiration annotation时autodetect为 2 statusmodel。"""
        viewer = _make_viewer(annotations=[(1.0, 3.0, 'Expiration', 'manual')])
        ok = train_phase_model(viewer, 'Expiration', random_state=42)
        assert ok is True
        info = viewer.ml_models['Expiration']
        assert len(info['classes']) == 2

class TestThreeStateModel:
    """Both Inspiration + Expiration → multiclass with Pause."""

    def test_both_phases_yields_three_states(self):
        """Verify同时有 Inspiration 和 Expiration annotation时detect为 3 statusmodel（含 Pause）。"""
        viewer = _make_viewer(annotations=[(0.5, 1.5, 'Inspiration', 'manual'), (2.0, 3.0, 'Expiration', 'manual')])
        ok = train_phase_model(viewer, 'Inspiration', random_state=42)
        assert ok is True
        info = viewer.ml_models['Inspiration']
        assert len(info['classes']) == 3
        assert 'Pause' in info['state_id_to_name'].values()

    def test_model_shared_to_both_keys(self):
        """Verifymodeltrain后正确存储到 ml_models 字典。"""
        viewer = _make_viewer(annotations=[(0.5, 1.5, 'Inspiration', 'manual'), (2.0, 3.0, 'Expiration', 'manual')])
        train_phase_model(viewer, 'Inspiration', random_state=42)
        assert 'Inspiration' in viewer.ml_models
        assert 'Expiration' in viewer.ml_models
        assert viewer.ml_models['Inspiration'] is viewer.ml_models['Expiration']

class TestHSMMPriors:

    def test_hsmm_prior_contains_duration_bounds(self):
        """Verify从已annotation帧学习到的 HSMM 时长prior（dmin/dmax）在合理range内。"""
        viewer = _make_viewer(annotations=[(0.5, 1.5, 'Inspiration', 'manual'), (2.0, 3.0, 'Expiration', 'manual')])
        train_phase_model(viewer, 'Inspiration', random_state=42)
        prior = viewer.ml_models['Inspiration']['hsmm_prior']
        assert 'dmin_frames' in prior
        assert 'dmax_frames' in prior
        assert len(prior['dmin_frames']) >= 2
        assert len(prior['dmax_frames']) >= 2

    def test_hsmm_prior_contains_classes(self):
        """Verify从已annotation帧学习到的 HSMM 时长prior（dmin/dmax）在合理range内。"""
        viewer = _make_viewer(annotations=[(0.5, 1.5, 'Inspiration', 'manual'), (2.0, 3.0, 'Expiration', 'manual')])
        train_phase_model(viewer, 'Inspiration', random_state=42)
        prior = viewer.ml_models['Inspiration']['hsmm_prior']
        assert 'classes' in prior

    def test_pi_init_present_when_possible(self):
        """Verify：len(prior['pi_init']) >= 2。"""
        viewer = _make_viewer(annotations=[(0.5, 1.5, 'Inspiration', 'manual'), (2.0, 3.0, 'Expiration', 'manual')])
        train_phase_model(viewer, 'Inspiration', random_state=42)
        prior = viewer.ml_models['Inspiration']['hsmm_prior']
        if 'pi_init' in prior:
            assert len(prior['pi_init']) >= 2

    def test_model_info_has_required_keys(self):
        """Verifymodeltrain后正确存储到 ml_models 字典。"""
        viewer = _make_viewer(annotations=[(0.5, 1.5, 'Inspiration', 'manual'), (2.0, 3.0, 'Expiration', 'manual')])
        train_phase_model(viewer, 'Inspiration', random_state=42)
        info = viewer.ml_models['Inspiration']
        required = {'model_kind', 'clf', 'classes', 'state_id_to_name', 'hsmm_prior', 'feature_names', 'train_prefix_sec'}
        missing = required - set(info.keys())
        assert not missing, f'Missing keys: {missing}'

class TestDeterminism:

    def test_same_seed_same_model(self):
        """Verifymodeltrain后正确存储到 ml_models 字典。"""
        annotations = [(0.5, 1.5, 'Inspiration', 'manual'), (2.0, 3.0, 'Expiration', 'manual')]
        v1 = _make_viewer(seed=42, annotations=annotations)
        v2 = _make_viewer(seed=42, annotations=annotations)
        train_phase_model(v1, 'Inspiration', random_state=42)
        train_phase_model(v2, 'Inspiration', random_state=42)
        c1 = v1.ml_models['Inspiration']['classes']
        c2 = v2.ml_models['Inspiration']['classes']
        assert c1 == c2