"""Tests for respanno.dsp.features — short-time feature computation.

All tests exercise the extracted module directly (no QApplication needed).
"""

import numpy as np
import pytest

from respanno.dsp.features import (
    ALL_FEATURE_NAMES,
    TIME_DOMAIN_FEATURE_NAMES,
    SPECTRAL_FEATURE_NAMES,
    COR_FEATURE_NAMES,
    frame_signal,
    compute_time_domain_features,
    compute_spectral_features,
    compute_short_time_features,
    normalize_feature_for_display,
    build_feature_matrix,
)


@pytest.fixture(scope="module")
def sig_4000():
    sr = 4000
    t = np.linspace(0, 2, sr * 2, endpoint=False)
    sig = 0.5 * np.sin(2 * np.pi * 100 * t) + 0.3 * np.sin(2 * np.pi * 800 * t)
    sig[3000:3500] += 0.4 * np.sin(2 * np.pi * 300 * t[3000:3500])
    return sig.astype(np.float32), sr


class TestFeatureInventory:
    def test_total_count(self):
        assert len(ALL_FEATURE_NAMES) == 56

    def test_time_domain_count(self):
        assert len(TIME_DOMAIN_FEATURE_NAMES) == 7

    def test_spectral_count(self):
        assert len(SPECTRAL_FEATURE_NAMES) == 30

    def test_cor_count(self):
        assert len(COR_FEATURE_NAMES) == 19


class TestFrameSignal:
    def test_output_shape(self, sig_4000):
        sig, sr = sig_4000
        frames, times, T = frame_signal(sig, sr, n_fft=512, hop_length=256)
        assert frames.ndim == 2
        assert frames.shape[1] == T
        assert len(times) == T

    def test_times_are_monotonic(self, sig_4000):
        sig, sr = sig_4000
        _, times, _ = frame_signal(sig, sr, 512, 256)
        assert np.all(np.diff(times) > 0)


class TestComputeShortTimeFeatures:
    def test_returns_dict_with_all_keys(self, sig_4000):
        sig, sr = sig_4000
        times, feat = compute_short_time_features(sig, sr)
        for name in ALL_FEATURE_NAMES:
            assert name in feat, f"Missing feature: {name}"

    def test_times_length_matches(self, sig_4000):
        sig, sr = sig_4000
        times, feat = compute_short_time_features(sig, sr)
        for name, arr in feat.items():
            assert len(arr) == len(times), f"{name}: len(arr)={len(arr)} != len(times)={len(times)}"

    def test_no_nan_inf(self, sig_4000):
        sig, sr = sig_4000
        _, feat = compute_short_time_features(sig, sr)
        for name, arr in feat.items():
            assert np.all(np.isfinite(arr)), f"{name} has NaN or inf"

    def test_empty_signal(self):
        times, feat = compute_short_time_features(np.array([], dtype=np.float32), 4000)
        assert len(times) == 0
        assert feat == {}

    def test_feature_matrix_shape(self, sig_4000):
        sig, sr = sig_4000
        times, feat = compute_short_time_features(sig, sr)
        X_full, t_out, names = build_feature_matrix(times, feat)
        assert X_full.shape[0] == len(times)
        assert X_full.shape[1] == 2 * len(ALL_FEATURE_NAMES)  # raw + smoothed
        assert len(names) == X_full.shape[1]


class TestNormalize:
    def test_range_zero_one(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_norm = normalize_feature_for_display(y)
        assert np.min(y_norm) == pytest.approx(0.0)
        assert np.max(y_norm) == pytest.approx(1.0)

    def test_constant_input(self):
        y = np.array([7.0, 7.0, 7.0])
        y_norm = normalize_feature_for_display(y)
        assert np.all(y_norm == 0.0)


class TestTimeDomainFeatures:
    def test_energy_non_negative(self, sig_4000):
        sig, sr = sig_4000
        feat = compute_time_domain_features(sig, sr, 512, 256)
        assert np.all(feat["短时能量"] >= 0)

    def test_zcr_in_range(self, sig_4000):
        sig, sr = sig_4000
        feat = compute_time_domain_features(sig, sr, 512, 256)
        zcr = feat["过零率"]
        assert np.all(zcr >= 0) and np.all(zcr <= 1)
