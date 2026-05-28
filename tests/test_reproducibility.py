"""Tests for deterministic reproducibility across the pipeline.

SoftwareX requirement: same input + same seed → identical output.
"""

import numpy as np
import pytest

from respanno.audio.preprocessing import apply_butter_filter, validate_preprocessing_config
from respanno.dsp.features import compute_short_time_features, build_feature_matrix
from respanno.dsp.fft import compute_fft
from respanno.dsp.spectrogram import compute_stft_db
from respanno.ml.hsmm import (
    build_hsmm_log_trans,
    build_hsmm_prior_from_prefix_labels,
    hsmm_viterbi,
)
from respanno.ml.frame_labels import build_frame_labels

from tests.fixtures.synthetic_signals import (
    generate_tone,
    generate_respiratory_cycle,
)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


class TestPreprocessingDeterminism:
    def test_butter_filter_deterministic(self):
        audio, sr, _ = generate_tone(freq=200.0, sr=4000, duration=2.0, seed=42)
        out1 = apply_butter_filter(audio, sr, "bandpass", 50.0, 500.0, 4)
        out2 = apply_butter_filter(audio, sr, "bandpass", 50.0, 500.0, 4)
        assert np.allclose(out1, out2)

    def test_config_validation_deterministic(self):
        cfg1 = validate_preprocessing_config({"resample_target_sr": 8000})
        cfg2 = validate_preprocessing_config({"resample_target_sr": 8000})
        assert cfg1 == cfg2


# ---------------------------------------------------------------------------
# FFT / Spectrogram
# ---------------------------------------------------------------------------


class TestFFTDeterminism:
    def test_fft_deterministic(self):
        audio, sr, _ = generate_tone(freq=200.0, sr=4000, duration=2.0, seed=42)
        f1, m1 = compute_fft(audio, sr)
        f2, m2 = compute_fft(audio, sr)
        assert np.allclose(f1, f2)
        assert np.allclose(m1, m2)

    def test_stft_deterministic(self):
        audio, sr, _ = generate_tone(freq=200.0, sr=4000, duration=2.0, seed=42)
        S1, f1 = compute_stft_db(audio, sr, n_fft=512, hop_length=128)
        S2, f2 = compute_stft_db(audio, sr, n_fft=512, hop_length=128)
        assert np.allclose(S1, S2, equal_nan=True)
        assert np.allclose(f1, f2)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


class TestFeatureDeterminism:
    def test_short_time_features_deterministic(self):
        audio, sr, _ = generate_tone(freq=200.0, sr=4000, duration=2.0, seed=42)
        t1, fd1 = compute_short_time_features(audio, sr, hop_length=256)
        t2, fd2 = compute_short_time_features(audio, sr, hop_length=256)
        assert np.allclose(t1, t2)
        for k in fd1:
            assert np.allclose(fd1[k], fd2[k], equal_nan=True)

    def test_feature_matrix_deterministic(self):
        audio, sr, _ = generate_tone(freq=200.0, sr=4000, duration=2.0, seed=42)
        t1, fd1 = compute_short_time_features(audio, sr, hop_length=256)
        t2, fd2 = compute_short_time_features(audio, sr, hop_length=256)
        X1, _, names1 = build_feature_matrix(t1, fd1)
        X2, _, names2 = build_feature_matrix(t2, fd2)
        assert np.allclose(X1, X2, equal_nan=True)
        assert names1 == names2


# ---------------------------------------------------------------------------
# Frame labels
# ---------------------------------------------------------------------------


class TestFrameLabelDeterminism:
    def test_build_frame_labels_deterministic(self):
        audio, sr, anns = generate_respiratory_cycle(duration=8.0, seed=42)
        times, _ = compute_short_time_features(audio, sr, hop_length=256)
        y1 = build_frame_labels(anns, times, "Wheeze", neg_margin=0.05)
        y2 = build_frame_labels(anns, times, "Wheeze", neg_margin=0.05)
        if y1 is None:
            assert y2 is None
        else:
            assert np.array_equal(y1, y2)


# ---------------------------------------------------------------------------
# HSMM
# ---------------------------------------------------------------------------


class TestHSMMDeterminism:
    def test_log_trans_deterministic(self):
        names = ["Inspiration", "Expiration", "Pause"]
        lt1 = build_hsmm_log_trans(names)
        lt2 = build_hsmm_log_trans(names)
        assert np.allclose(lt1, lt2)

    def test_hsmm_viterbi_deterministic(self):
        names = ["Inspiration", "Expiration"]
        dmin = np.array([3, 3], dtype=int)
        dmax = np.array([30, 30], dtype=int)
        log_trans = build_hsmm_log_trans(names)
        log_pi = np.log(np.array([0.5, 0.5]))
        rng = np.random.default_rng(42)
        log_emit = np.log(np.abs(rng.standard_normal((50, 2))) + 1e-6)

        z1 = hsmm_viterbi(log_emit, dmin, dmax, log_trans, log_pi)
        z2 = hsmm_viterbi(log_emit, dmin, dmax, log_trans, log_pi)
        assert np.array_equal(z1, z2)

    def test_hsmm_prior_deterministic(self):
        y_prefix = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0], dtype=np.int16)
        classes = [0, 1, 2]
        name_map = {0: "Inspiration", 1: "Expiration", 2: "Pause"}
        p1 = build_hsmm_prior_from_prefix_labels(y_prefix, classes, name_map, 0.064, 4.0)
        p2 = build_hsmm_prior_from_prefix_labels(y_prefix, classes, name_map, 0.064, 4.0)
        assert p1["dmin_frames"] == p2["dmin_frames"]
        assert p1["dmax_frames"] == p2["dmax_frames"]


# ---------------------------------------------------------------------------
# Synthetic signal determinism
# ---------------------------------------------------------------------------


class TestSyntheticSignalDeterminism:
    def test_generator_same_seed_same_output(self):
        a1, sr1, ann1 = generate_respiratory_cycle(seed=42)
        a2, sr2, ann2 = generate_respiratory_cycle(seed=42)
        assert np.allclose(a1, a2)
        assert sr1 == sr2
        assert ann1 == ann2

    def test_generator_different_seed_different_output(self):
        a1, _, _ = generate_respiratory_cycle(seed=42)
        a2, _, _ = generate_respiratory_cycle(seed=99)
        assert not np.allclose(a1, a2)
