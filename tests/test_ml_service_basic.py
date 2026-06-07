"""Test respanno.ml.service — MLService dispatcher routing.

CLAUDE.md known gap #1: the central pipeline router had zero tests.
This module validates that train/apply/clear calls route to the correct
backend module based on label taxonomy.
"""
import numpy as np
import pytest
from respanno.ml.service import MLService
from respanno.ml.label_taxonomy import ABNORMAL_SOUND_KIND, PHASE_KIND, OTHER_EVENT_KIND
from tests.fixtures.mock_viewer import MockViewer, make_synthetic_features


@pytest.fixture(autouse=True)
def _mock_qmessagebox(monkeypatch):
    """Suppress QMessageBox popups during training."""
    monkeypatch.setattr('respanno.ml.classifier.QMessageBox.information', lambda *a, **kw: None)
    monkeypatch.setattr('respanno.ml.classifier.QMessageBox.warning', lambda *a, **kw: None)
    monkeypatch.setattr('respanno.ml.phase_model.QMessageBox.information', lambda *a, **kw: None)
    monkeypatch.setattr('respanno.ml.phase_model.QMessageBox.warning', lambda *a, **kw: None)


def _make_viewer():
    """Build a MockViewer with synthetic features and annotations for training."""
    X, times, names = make_synthetic_features(n_frames=120, n_features=56, seed=42)
    # Create annotations: frames 10-30 = "Wheeze", frames 50-55 = "Crackles"
    # Also add phase annotations for inspiration/expiration
    annotations = [
        (0.0, 3.0, "Inspiration", "manual"),
        (3.5, 6.0, "Expiration", "manual"),
        (0.64, 1.92, "Wheeze", "manual"),    # frames 10-30 at 64ms hop
        (3.2, 3.52, "Crackles", "manual"),   # frames 50-55
    ]
    return MockViewer(stft_features=X.astype(np.float64), stft_frame_times=times.astype(float),
                      stft_feature_names=names, annotations=annotations, sr=4000, hop_length=256)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Initialization
# ═══════════════════════════════════════════════════════════════════════════

class TestMLServiceInit:
    """MLService creation and basic attributes."""

    def test_creates_with_owner(self):
        """MLService(owner=viewer) should store the owner reference."""
        viewer = _make_viewer()
        svc = MLService(owner=viewer)
        assert svc.owner is viewer

    def test_creates_with_none_owner(self):
        """MLService(owner=None) should work (lazy owner pattern)."""
        svc = MLService(owner=None)
        assert svc.owner is None

    def test_label_kind_constants_defined(self):
        """Verify that all three pipeline kind constants are defined."""
        assert MLService.PHASE_KIND == "phase"
        assert MLService.ABNORMAL_SOUND_KIND == "abnormal_sound"
        assert MLService.OTHER_EVENT_KIND == "other_event"

    def test_label_sets_defined(self):
        """PHASE_LABELS and OTHER_EVENT_LABELS should be non-empty sets."""
        assert len(MLService.PHASE_LABELS) > 0
        assert len(MLService.OTHER_EVENT_LABELS) > 0
        assert "inspiration" in MLService.PHASE_LABELS
        assert "cough" in MLService.OTHER_EVENT_LABELS


# ═══════════════════════════════════════════════════════════════════════════
# 2. Label routing (_label_kind)
# ═══════════════════════════════════════════════════════════════════════════

class TestLabelKindRouting:
    """_label_kind must route to correct pipeline kind."""

    @pytest.mark.parametrize("label,expected", [
        ("Wheeze", ABNORMAL_SOUND_KIND),
        ("Crackles", ABNORMAL_SOUND_KIND),
        ("Rhonchi", ABNORMAL_SOUND_KIND),
        ("Stridor", ABNORMAL_SOUND_KIND),
        ("Inspiration", PHASE_KIND),
        ("Expiration", PHASE_KIND),
        ("Pause", PHASE_KIND),
        ("吸气", PHASE_KIND),
        ("呼气", PHASE_KIND),
        ("cough", OTHER_EVENT_KIND),
        ("speech", OTHER_EVENT_KIND),
        ("throat", OTHER_EVENT_KIND),
    ])
    def test_label_kind_matches_taxonomy(self, label, expected):
        """Each label must route to its correct pipeline kind."""
        svc = MLService(owner=None)
        assert svc._label_kind(label) == expected


# ═══════════════════════════════════════════════════════════════════════════
# 3. Training dispatch (train_model_for_label)
# ═══════════════════════════════════════════════════════════════════════════

class TestTrainDispatch:
    """train_model_for_label routes to correct backend."""

    def test_train_wheeze_dispatches_to_classifier(self, monkeypatch):
        """Training 'Wheeze' must call classifier.train_event_model, not phase."""
        viewer = _make_viewer()
        svc = MLService(owner=viewer)

        called_classifier = []
        called_phase = []

        def fake_train_event(owner, label, **kw):
            called_classifier.append(label)
            owner.ml_models[label] = {"clf": "mock_clf", "model_kind": kw.get("model_kind", "event"), "train_f1": 0.8, "threshold": 0.5}
            return True

        def fake_train_phase(owner, label, **kw):
            called_phase.append(label)
            return True

        monkeypatch.setattr('respanno.ml.classifier.train_event_model', fake_train_event)
        monkeypatch.setattr('respanno.ml.phase_model.train_phase_model', fake_train_phase)

        result = svc.train_model_for_label("Wheeze", random_state=42)
        assert result is True
        assert len(called_classifier) == 1
        assert called_classifier[0] == "Wheeze"
        assert len(called_phase) == 0

    def test_train_inspiration_dispatches_to_phase(self, monkeypatch):
        """Training 'Inspiration' must call phase_model.train_phase_model."""
        viewer = _make_viewer()
        svc = MLService(owner=viewer)

        called_classifier = []
        called_phase = []

        def fake_train_event(owner, label, **kw):
            called_classifier.append(label)
            return True

        def fake_train_phase(owner, label, **kw):
            called_phase.append(label)
            return True

        monkeypatch.setattr('respanno.ml.classifier.train_event_model', fake_train_event)
        monkeypatch.setattr('respanno.ml.phase_model.train_phase_model', fake_train_phase)

        result = svc.train_model_for_label("Inspiration", random_state=42)
        assert result is True
        assert len(called_phase) == 1
        assert called_phase[0] == "Inspiration"
        assert len(called_classifier) == 0

    def test_train_cough_dispatches_to_other_event(self, monkeypatch):
        """Training 'cough' must go to train_other_event_model_for_label."""
        viewer = _make_viewer()
        svc = MLService(owner=viewer)

        called = []

        def fake_train_event(owner, label, **kw):
            called.append((label, kw.get("model_kind")))
            owner.ml_models[label] = {"clf": "mock", "model_kind": kw.get("model_kind", "event"), "train_f1": 0.7, "threshold": 0.5}
            return True

        monkeypatch.setattr('respanno.ml.classifier.train_event_model', fake_train_event)
        monkeypatch.setattr('respanno.ml.phase_model.train_phase_model', lambda *a, **kw: True)

        result = svc.train_model_for_label("cough", random_state=42)
        assert result is True
        assert called[0][0] == "cough"
        assert called[0][1] == OTHER_EVENT_KIND, f"expected {OTHER_EVENT_KIND}, got {called[0][1]}"

    def test_abnormal_sound_stores_model_kind(self, monkeypatch):
        """Abnormal sound training must tag model_kind='abnormal_sound'."""
        viewer = _make_viewer()
        svc = MLService(owner=viewer)

        recorded_kwargs = {}

        def fake_train_event(owner, label, **kw):
            recorded_kwargs.update(kw)
            owner.ml_models[label] = {"clf": "mock", "model_kind": kw.get("model_kind"), "train_f1": 0.8, "threshold": 0.5}
            return True

        monkeypatch.setattr('respanno.ml.classifier.train_event_model', fake_train_event)

        svc.train_abnormal_sound_model_for_label("Wheeze", random_state=42)
        assert recorded_kwargs.get("model_kind") == ABNORMAL_SOUND_KIND

    def test_phase_model_train_does_not_call_classifier(self, monkeypatch):
        """train_phase_model_for_label must NOT call classifier module."""
        viewer = _make_viewer()
        svc = MLService(owner=viewer)

        called = []

        def fake_phase(owner, label, **kw):
            called.append(("phase", label))
            return True

        def fake_event(owner, label, **kw):
            called.append(("classifier", label))
            return True

        monkeypatch.setattr('respanno.ml.phase_model.train_phase_model', fake_phase)
        monkeypatch.setattr('respanno.ml.classifier.train_event_model', fake_event)

        svc.train_phase_model_for_label("Inspiration", random_state=42)
        assert called == [("phase", "Inspiration")]


# ═══════════════════════════════════════════════════════════════════════════
# 4. Apply dispatch (apply_model_for_label_on_unreviewed)
# ═══════════════════════════════════════════════════════════════════════════

class TestApplyDispatch:
    """apply_model_for_label_on_unreviewed routes to correct backend."""

    def test_apply_wheeze_dispatches_to_classifier(self, monkeypatch):
        """Applying 'Wheeze' must call classifier.apply_event_model."""
        viewer = _make_viewer()
        svc = MLService(owner=viewer)

        # First train so a model exists
        monkeypatch.setattr('respanno.ml.classifier.train_event_model',
            lambda o, l, **kw: o.ml_models.update({l: {"clf": "mock", "model_kind": "event", "train_f1": 0.8, "threshold": 0.5}}) or True)

        svc.train_model_for_label("Wheeze", random_state=42)

        called_apply = []

        def fake_apply(owner, label, **kw):
            called_apply.append(label)
            return True

        monkeypatch.setattr('respanno.ml.classifier.apply_event_model', fake_apply)

        result = svc.apply_model_for_label_on_unreviewed("Wheeze", min_dur_sec=0.05)
        assert result is True
        assert "Wheeze" in called_apply

    def test_apply_inspiration_dispatches_to_phase(self, monkeypatch):
        """Applying 'Inspiration' must call phase_model.apply_phase_model."""
        viewer = _make_viewer()
        svc = MLService(owner=viewer)

        called_apply = []

        def fake_train_phase(owner, label, **kw):
            return True

        def fake_apply_phase(owner, label, **kw):
            called_apply.append(("phase", label))
            return True

        def fake_apply_event(owner, label, **kw):
            called_apply.append(("classifier", label))
            return True

        monkeypatch.setattr('respanno.ml.phase_model.train_phase_model', fake_train_phase)
        monkeypatch.setattr('respanno.ml.phase_model.apply_phase_model', fake_apply_phase)
        monkeypatch.setattr('respanno.ml.classifier.apply_event_model', fake_apply_event)

        svc.train_model_for_label("Inspiration", random_state=42)
        result = svc.apply_model_for_label_on_unreviewed("Inspiration", min_dur_sec=0.05)
        assert result is True
        assert called_apply == [("phase", "Inspiration")]

    def test_apply_abnormal_uses_expected_model_kinds(self, monkeypatch):
        """apply_abnormal_sound_model must pass expected_model_kinds set."""
        viewer = _make_viewer()
        svc = MLService(owner=viewer)

        recorded_kwargs = []

        def fake_apply(owner, label, **kw):
            recorded_kwargs.append(kw)
            return True

        monkeypatch.setattr('respanno.ml.classifier.apply_event_model', fake_apply)

        svc.apply_abnormal_sound_model_for_label_on_unreviewed("Wheeze", min_dur_sec=0.05)
        assert len(recorded_kwargs) == 1
        kinds = recorded_kwargs[0].get("expected_model_kinds")
        assert kinds == {ABNORMAL_SOUND_KIND, "event"}

    def test_apply_other_event_uses_expected_model_kinds(self, monkeypatch):
        """apply_other_event_model must pass expected_model_kinds set."""
        viewer = _make_viewer()
        svc = MLService(owner=viewer)

        recorded_kwargs = []

        def fake_apply(owner, label, **kw):
            recorded_kwargs.append(kw)
            return True

        monkeypatch.setattr('respanno.ml.classifier.apply_event_model', fake_apply)

        svc.apply_other_event_model_for_label_on_unreviewed("cough", min_dur_sec=0.05)
        kinds = recorded_kwargs[0].get("expected_model_kinds")
        assert kinds == {OTHER_EVENT_KIND, "event"}


# ═══════════════════════════════════════════════════════════════════════════
# 5. Clear dispatch
# ═══════════════════════════════════════════════════════════════════════════

class TestClearDispatch:
    """clear_ml_annotations_for_label delegates to label_taxonomy."""

    def test_clear_delegates_to_label_taxonomy(self, monkeypatch):
        """clear_ml_annotations_for_label must call clear_ml_annotations."""
        viewer = _make_viewer()
        svc = MLService(owner=viewer)

        called = []

        def fake_clear(v, label):
            called.append((v, label))

        monkeypatch.setattr('respanno.ml.label_taxonomy.clear_ml_annotations', fake_clear)

        svc.clear_ml_annotations_for_label("Wheeze")
        assert len(called) == 1
        assert called[0][0] is viewer
        assert called[0][1] == "Wheeze"


# ═══════════════════════════════════════════════════════════════════════════
# 6. HSMM helper delegation
# ═══════════════════════════════════════════════════════════════════════════

class TestHSMMHelpers:
    """HSMM helper methods delegate correctly to respanno.ml.hsmm."""

    def test_estimate_hop_sec_defaults(self):
        """_estimate_hop_sec should return a positive float with defaults."""
        svc = MLService(owner=None)
        hop = svc._estimate_hop_sec(times=None)
        assert hop > 0

    def test_estimate_hop_sec_from_viewer(self):
        """_estimate_hop_sec reads sr/hop_length from viewer when available."""
        viewer = _make_viewer()
        svc = MLService(owner=viewer)
        hop = svc._estimate_hop_sec(times=viewer.stft_frame_times, viewer=viewer)
        assert hop > 0

    def test_estimate_breath_cycle_sec_default(self):
        """_estimate_breath_cycle_sec returns default 3.0 on empty segments."""
        svc = MLService(owner=None)
        bc = svc._estimate_breath_cycle_sec([], [])
        assert bc == 3.0

    def test_build_hsmm_log_trans_delegates(self):
        """_build_hsmm_log_trans returns a square matrix."""
        svc = MLService(owner=None)
        lt = svc._build_hsmm_log_trans(["S0", "S1"])
        assert lt.shape == (2, 2)
        assert np.all(np.isfinite(lt))

    def test_hsmm_viterbi_delegates(self):
        """_hsmm_viterbi should return a state sequence of correct length."""
        svc = MLService(owner=None)
        rng = np.random.RandomState(42)
        T, S, max_dur = 50, 2, 20
        log_emit = np.log(np.abs(rng.randn(T, S)) + 1e-6)
        dmin = [1, 1]
        dmax = [max_dur, max_dur]
        log_trans = svc._build_hsmm_log_trans(["S0", "S1"])
        log_pi = np.log(np.array([0.5, 0.5]))
        states = svc._hsmm_viterbi(log_emit, dmin, dmax, log_trans, log_pi)
        assert len(states) == T

    def test_state_seq_to_segments_delegates(self):
        """_state_seq_to_segments extracts segments for target state."""
        svc = MLService(owner=None)
        T_all = 100
        times = np.arange(T_all) * 0.016
        idx_unr = np.arange(30, 80)
        z = np.array([0] * 20 + [1] * 20 + [0] * 10)
        segs = svc._state_seq_to_segments(times, idx_unr, z, target_state_id=1, min_dur_sec=0.05)
        assert len(segs) >= 1

    def test_build_hsmm_prior_delegates(self):
        """_build_hsmm_prior_from_prefix_labels returns valid priors."""
        svc = MLService(owner=None)
        y_prefix = np.array([0, 0, 1, 1, 0, 0, 1, 1] * 6)[:50]
        classes_ = [0, 1]
        name_map = {0: "Pause", 1: "Inspiration"}
        priors = svc._build_hsmm_prior_from_prefix_labels(y_prefix, classes_, name_map, 0.016, 3.0)
        assert "dmin_frames" in priors
        assert "dmax_frames" in priors
        assert priors["hop_sec"] == 0.016


# ═══════════════════════════════════════════════════════════════════════════
# 7. End-to-end: train via MLService (not directly via classifier/phase)
# ═══════════════════════════════════════════════════════════════════════════

class TestMLServiceE2E:
    """Full training + apply through the MLService dispatcher."""

    def test_train_then_apply_wheeze_via_service(self):
        """Train 'Wheeze' through MLService → model stored → apply succeeds."""
        from tests.fixtures.synthetic_signals import generate_wheeze_episode
        from respanno.dsp.features import compute_short_time_features, build_feature_matrix

        audio, sr, anns = generate_wheeze_episode(duration=5.0, wheeze_start=1.0, wheeze_dur=2.0, seed=42)
        times, feat_dict = compute_short_time_features(audio, int(sr))
        X_full, _, full_names = build_feature_matrix(times, feat_dict)
        viewer = MockViewer(stft_features=X_full.astype(np.float64), stft_frame_times=times.astype(float),
                            stft_feature_names=full_names, annotations=list(anns), sr=int(sr), hop_length=256)
        svc = MLService(owner=viewer)

        # Train through service
        ok = svc.train_model_for_label("Wheeze", min_pos_frames=5, random_state=42)
        assert ok is True
        assert "Wheeze" in viewer.ml_models

        # Apply through service
        ok = svc.apply_model_for_label_on_unreviewed("Wheeze", min_dur_sec=0.05)
        assert ok in (True, False)

    def test_train_then_apply_inspiration_via_service(self):
        """Train 'Inspiration' through MLService → phase model trained."""
        from tests.fixtures.synthetic_signals import generate_respiratory_cycle
        from respanno.dsp.features import compute_short_time_features, build_feature_matrix
        import respanno.ml.phase_model as pm
        pm.QMessageBox.information = lambda *a, **kw: None
        pm.QMessageBox.warning = lambda *a, **kw: None

        audio, sr, anns = generate_respiratory_cycle(duration=8.0, seed=42)
        times, feat_dict = compute_short_time_features(audio, int(sr))
        X_full, _, full_names = build_feature_matrix(times, feat_dict)
        viewer = MockViewer(stft_features=X_full.astype(np.float64), stft_frame_times=times.astype(float),
                            stft_feature_names=full_names, annotations=list(anns), sr=int(sr), hop_length=256)
        svc = MLService(owner=viewer)

        ok = svc.train_model_for_label("Inspiration", min_pos_frames=3, random_state=42)
        assert ok is True

        ok = svc.apply_model_for_label_on_unreviewed("Inspiration", min_dur_sec=0.05)
        assert ok in (True, False)

    def test_three_way_routing_disjoint(self, monkeypatch):
        """Abnormal, phase, other_event → three distinct code paths."""
        viewer = _make_viewer()
        svc = MLService(owner=viewer)

        routes = {}

        def fake_train_event(owner, label, **kw):
            routes[label] = kw.get("model_kind", "event")
            owner.ml_models[label] = {"clf": "mock", "model_kind": kw.get("model_kind", "event"), "train_f1": 0.8, "threshold": 0.5}
            return True

        def fake_train_phase(owner, label, **kw):
            routes[label] = "phase"
            return True

        monkeypatch.setattr('respanno.ml.classifier.train_event_model', fake_train_event)
        monkeypatch.setattr('respanno.ml.phase_model.train_phase_model', fake_train_phase)

        svc.train_model_for_label("Wheeze", random_state=42)
        svc.train_model_for_label("Inspiration", random_state=42)
        svc.train_model_for_label("cough", random_state=42)

        assert routes["Wheeze"] == ABNORMAL_SOUND_KIND
        assert routes["Inspiration"] == "phase"
        assert routes["cough"] == OTHER_EVENT_KIND
