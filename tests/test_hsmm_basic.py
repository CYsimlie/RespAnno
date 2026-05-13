"""Tests for respanno.ml.hsmm — HSMM Viterbi decoder and priors.

All tests exercise the extracted module directly (no QApplication needed).
"""

import numpy as np
import pytest

from respanno.ml.hsmm import (
    estimate_hop_sec,
    estimate_breath_cycle_sec,
    build_hsmm_prior_from_prefix_labels,
    build_hsmm_log_trans,
    hsmm_viterbi,
    state_seq_to_segments,
)


# ---------------------------------------------------------------------------
# estimate_hop_sec
# ---------------------------------------------------------------------------

class TestEstimateHopSec:
    def test_from_sr_hop(self):
        assert estimate_hop_sec(sr=4000, hop_length=256) == pytest.approx(0.064)

    def test_from_times(self):
        times = np.array([0.0, 0.064, 0.128, 0.192, 0.256])
        assert estimate_hop_sec(times=times) == pytest.approx(0.064)

    def test_default(self):
        assert estimate_hop_sec() == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# estimate_breath_cycle_sec
# ---------------------------------------------------------------------------

class TestEstimateBreathCycleSec:
    def test_from_starts(self):
        seg_I = [(0.0, 1.0), (3.0, 4.0), (6.0, 7.0)]
        seg_E = [(1.5, 2.5), (4.5, 5.5)]
        cyc = estimate_breath_cycle_sec(seg_I, seg_E)
        assert cyc > 0

    def test_default_when_empty(self):
        cyc = estimate_breath_cycle_sec([], [], default=4.0)
        assert cyc == 4.0


# ---------------------------------------------------------------------------
# build_hsmm_log_trans
# ---------------------------------------------------------------------------

class TestBuildHSMMLogTrans:
    def test_2state(self):
        A = build_hsmm_log_trans(["Inspiration", "Expiration"])
        assert A.shape == (2, 2)
        assert np.all(np.isfinite(A))

    def test_3state_with_pause(self):
        A = build_hsmm_log_trans(["Inspiration", "Expiration", "Pause"])
        assert A.shape == (3, 3)
        # Pause self-loop should be finite
        assert np.isfinite(A[2, 2])

    def test_3state_insp_to_exp_finite(self):
        A = build_hsmm_log_trans(["Inspiration", "Expiration", "Pause"])
        assert np.isfinite(A[0, 1])  # Insp → Exp
        assert np.isfinite(A[1, 0])  # Exp → Insp


# ---------------------------------------------------------------------------
# hsmm_viterbi
# ---------------------------------------------------------------------------

class TestHSMMViterbi:
    def test_single_state_dominates(self):
        T, S = 40, 2
        log_emit = np.zeros((T, S))
        log_emit[:, 0] = np.log(0.9)
        log_emit[:, 1] = np.log(0.1)
        dmin = [2, 2]
        dmax = [5, 5]
        A = build_hsmm_log_trans(["A", "B"])
        log_pi = np.log(np.array([0.5, 0.5]))
        z = hsmm_viterbi(log_emit, dmin, dmax, A, log_pi)
        assert z.shape == (T,)
        assert np.all(z == 0)

    def test_state_switch_on_flip(self):
        T, S = 60, 2
        log_emit = np.zeros((T, S))
        log_emit[:30, 0] = np.log(0.9)
        log_emit[:30, 1] = np.log(0.1)
        log_emit[30:, 0] = np.log(0.1)
        log_emit[30:, 1] = np.log(0.9)
        dmin = [2, 2]
        dmax = [10, 10]
        A = build_hsmm_log_trans(["A", "B"])
        log_pi = np.log(np.array([0.5, 0.5]))
        z = hsmm_viterbi(log_emit, dmin, dmax, A, log_pi)
        assert z.shape == (T,)
        assert np.mean(z[:20] == 0) > 0.5

    def test_3state(self):
        T, S = 90, 3
        rng = np.random.default_rng(42)
        log_emit = np.log(np.clip(rng.random((T, S)), 0.05, 0.95))
        log_emit[:30, 0] += np.log(2.0)
        log_emit[30:60, 1] += np.log(2.0)
        log_emit[60:, 2] += np.log(2.0)
        dmin = [3, 3, 3]
        dmax = [12, 12, 12]
        A = build_hsmm_log_trans(["Insp", "Exp", "Pause"])
        log_pi = np.log(np.ones(3) / 3)
        z = hsmm_viterbi(log_emit, dmin, dmax, A, log_pi)
        assert z.shape == (T,)
        assert len(np.unique(z)) >= 2

    def test_numerical_stability(self):
        T, S = 200, 2
        rng = np.random.default_rng(999)
        log_emit = np.log(np.clip(rng.random((T, S)), 1e-6, 1.0))
        dmin = [1, 1]
        dmax = [30, 30]
        A = build_hsmm_log_trans(["A", "B"])
        log_pi = np.log(np.array([0.5, 0.5]))
        z = hsmm_viterbi(log_emit, dmin, dmax, A, log_pi)
        assert z.shape == (T,)
        assert np.all(np.isfinite(z))


# ---------------------------------------------------------------------------
# build_hsmm_prior_from_prefix_labels
# ---------------------------------------------------------------------------

class TestHSMMPrior:
    def test_output_has_required_keys(self):
        y = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 1, 1, 0, 0, 0, 0, 2, 2], dtype=int)
        prior = build_hsmm_prior_from_prefix_labels(
            y_prefix=y,
            classes_=[0, 1, 2],
            state_id_to_name={0: "Insp", 1: "Exp", 2: "Pause"},
            hop_sec=0.064,
            cycle_sec=3.0,
        )
        for key in ("classes", "hop_sec", "cycle_sec", "dmin_frames", "dmax_frames"):
            assert key in prior
        assert len(prior["dmin_frames"]) == 3
        assert len(prior["dmax_frames"]) == 3

    def test_dmin_positive(self):
        y = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2], dtype=int)
        prior = build_hsmm_prior_from_prefix_labels(
            y_prefix=y,
            classes_=[0, 1, 2],
            state_id_to_name={0: "Insp", 1: "Exp", 2: "Pause"},
            hop_sec=0.064,
            cycle_sec=3.0,
        )
        for d in prior["dmin_frames"]:
            assert d >= 1
        for d in prior["dmax_frames"]:
            assert d >= prior["dmin_frames"][0]  # reasonable

    def test_single_class_fallback(self):
        y = np.array([0, 0, 0, 0], dtype=int)
        prior = build_hsmm_prior_from_prefix_labels(
            y_prefix=y,
            classes_=[0],
            state_id_to_name={0: "Insp"},
            hop_sec=0.064,
            cycle_sec=3.0,
        )
        assert len(prior["dmin_frames"]) == 1


# ---------------------------------------------------------------------------
# state_seq_to_segments
# ---------------------------------------------------------------------------

class TestStateSeqToSegments:
    def test_basic(self):
        times = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        idx_unr = np.array([0, 1, 2, 3, 4, 5])
        z = np.array([0, 0, 1, 1, 0, 0])
        segs = state_seq_to_segments(times, idx_unr, z, target_state_id=1, min_dur_sec=0.05)
        assert len(segs) == 1
        assert segs[0] == (0.2, 0.3)

    def test_min_dur_filter(self):
        times = np.linspace(0, 1, 20)
        idx_unr = np.arange(20)
        z = np.array([0] * 5 + [1] + [0] * 14)  # only 1 frame as state 1
        segs = state_seq_to_segments(times, idx_unr, z, target_state_id=1, min_dur_sec=0.5)
        assert len(segs) == 0

    def test_empty_unreviewed(self):
        segs = state_seq_to_segments(np.array([]), np.array([]), np.array([]), 0, 0.05)
        assert segs == []
