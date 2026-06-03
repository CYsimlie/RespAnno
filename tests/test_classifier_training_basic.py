"""Tests for respanno.ml.classifier — train_event_model."""
import numpy as np
import pytest
from respanno.ml.classifier import train_event_model
from tests.fixtures.mock_viewer import MockViewer, make_synthetic_features

@pytest.fixture(autouse=True)
def _mock_qmessagebox(monkeypatch):
    """Suppress QMessageBox dialogs during tests."""
    monkeypatch.setattr('respanno.ml.classifier.QMessageBox.information', lambda *a, **kw: None)
    monkeypatch.setattr('respanno.ml.classifier.QMessageBox.warning', lambda *a, **kw: None)

def _make_viewer(n_frames=120, n_features=56, annotations=None, seed=42):
    """Build a MockViewer with synthetic features and optional annotations."""
    (X, times, names) = make_synthetic_features(n_frames, n_features, seed=seed)
    return MockViewer(stft_features=X, stft_frame_times=times, stft_feature_names=names, annotations=annotations or [])

class TestNoFeatures:

    def test_returns_false_when_no_features(self):
        """验证前置条件不满足时返回 False：no features。"""
        viewer = MockViewer()
        ok = train_event_model(viewer, 'Wheeze', random_state=42)
        assert ok is False

class TestNoValidFrames:

    def test_returns_false_when_no_annotations(self):
        """验证前置条件不满足时返回 False：no annotations。"""
        viewer = _make_viewer(annotations=[])
        ok = train_event_model(viewer, 'Wheeze', random_state=42)
        assert ok is False

    def test_returns_false_when_no_match_for_label(self):
        """验证前置条件不满足时返回 False：no match for label。"""
        viewer = _make_viewer(annotations=[(0.5, 2.0, 'Crackles', 'manual')])
        ok = train_event_model(viewer, 'Wheeze', random_state=42)
        assert ok is False

class TestInsufficientSamples:

    def test_returns_false_when_too_few_positives(self):
        """Only 2 frames positive → below min_pos_frames=20."""
        viewer = _make_viewer(annotations=[(0.05, 0.15, 'Wheeze', 'manual')])
        ok = train_event_model(viewer, 'Wheeze', min_pos_frames=20, random_state=42)
        assert ok is False

class TestSuccessfulTraining:

    def test_returns_true_on_success(self):
        """验证正常训练流程成功返回 True：on success。"""
        viewer = _make_viewer(annotations=[(0.5, 3.0, 'Wheeze', 'manual')])
        ok = train_event_model(viewer, 'Wheeze', random_state=42)
        assert ok is True

    def test_model_stored_in_ml_models(self):
        """验证模型训练后正确存储到 ml_models 字典。"""
        viewer = _make_viewer(annotations=[(0.5, 3.0, 'Wheeze', 'manual')])
        train_event_model(viewer, 'Wheeze', random_state=42)
        assert 'Wheeze' in viewer.ml_models

    def test_model_info_has_required_keys(self):
        """验证模型训练后正确存储到 ml_models 字典。"""
        viewer = _make_viewer(annotations=[(0.5, 3.0, 'Wheeze', 'manual')])
        train_event_model(viewer, 'Wheeze', random_state=42)
        info = viewer.ml_models['Wheeze']
        required = {'clf', 'threshold', 'model_kind', 'feature_names', 'selected_feature_indices', 'train_f1', 'train_accuracy', 'train_auc_roc', 'n_pos', 'n_neg', 'confusion'}
        missing = required - set(info.keys())
        assert not missing, f'Missing keys: {missing}'

    def test_threshold_in_expected_range(self):
        """验证：0.2 <= th <= 0.95。"""
        viewer = _make_viewer(annotations=[(0.5, 3.0, 'Wheeze', 'manual')])
        train_event_model(viewer, 'Wheeze', random_state=42)
        th = viewer.ml_models['Wheeze']['threshold']
        assert 0.2 <= th <= 0.95

    def test_confusion_matrix_counts(self):
        """验证：c['tp'] + c['fn'] > 0。"""
        viewer = _make_viewer(annotations=[(0.5, 3.0, 'Wheeze', 'manual')])
        train_event_model(viewer, 'Wheeze', random_state=42)
        c = viewer.ml_models['Wheeze']['confusion']
        assert c['tp'] + c['fn'] > 0, 'should have positive samples'
        assert c['tn'] + c['fp'] > 0, 'should have negative samples'

    def test_feature_selection_results(self):
        """验证：len(info['selected_feature_indices']) > 0。"""
        viewer = _make_viewer(annotations=[(0.5, 3.0, 'Wheeze', 'manual')])
        train_event_model(viewer, 'Wheeze', random_state=42)
        info = viewer.ml_models['Wheeze']
        assert len(info['selected_feature_indices']) > 0
        assert info['feature_select_method'] in ('mutual_info_kbest', 'none')

    def test_top_features_present(self):
        """验证：len(top) > 0。"""
        viewer = _make_viewer(annotations=[(0.5, 3.0, 'Wheeze', 'manual')])
        train_event_model(viewer, 'Wheeze', random_state=42)
        top = viewer.ml_models['Wheeze'].get('top_features_by_importance', [])
        assert len(top) > 0

    def test_multiple_labels_independent_models(self):
        """验证模型训练后正确存储到 ml_models 字典。"""
        viewer = _make_viewer(annotations=[(0.5, 2.0, 'Wheeze', 'manual'), (2.5, 4.0, 'Crackles', 'manual')])
        ok1 = train_event_model(viewer, 'Wheeze', random_state=42)
        ok2 = train_event_model(viewer, 'Crackles', random_state=42)
        assert ok1 and ok2
        assert 'Wheeze' in viewer.ml_models
        assert 'Crackles' in viewer.ml_models
        assert viewer.ml_models['Wheeze']['clf'] is not viewer.ml_models['Crackles']['clf']

class TestDeterminism:

    def test_same_seed_same_model(self):
        """验证模型训练后正确存储到 ml_models 字典。"""
        v1 = _make_viewer(seed=42, annotations=[(0.5, 3.0, 'Wheeze', 'manual')])
        v2 = _make_viewer(seed=42, annotations=[(0.5, 3.0, 'Wheeze', 'manual')])
        train_event_model(v1, 'Wheeze', random_state=42)
        train_event_model(v2, 'Wheeze', random_state=42)
        assert v1.ml_models['Wheeze']['threshold'] == v2.ml_models['Wheeze']['threshold']
        assert v1.ml_models['Wheeze']['train_f1'] == v2.ml_models['Wheeze']['train_f1']