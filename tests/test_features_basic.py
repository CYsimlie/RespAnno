"""Tests for respanno.dsp.features — short-time feature computation.

All tests exercise the extracted module directly (no QApplication needed).
"""
import numpy as np
import pytest
from respanno.dsp.features import ALL_FEATURE_NAMES, TIME_DOMAIN_FEATURE_NAMES, SPECTRAL_FEATURE_NAMES, COR_FEATURE_NAMES, frame_signal, compute_time_domain_features, compute_spectral_features, compute_short_time_features, normalize_feature_for_display, build_feature_matrix

@pytest.fixture(scope='module')
def sig_4000():
    sr = 4000
    t = np.linspace(0, 2, sr * 2, endpoint=False)
    sig = 0.5 * np.sin(2 * np.pi * 100 * t) + 0.3 * np.sin(2 * np.pi * 800 * t)
    sig[3000:3500] += 0.4 * np.sin(2 * np.pi * 300 * t[3000:3500])
    return (sig.astype(np.float32), sr)

class TestFeatureInventory:

    def test_total_count(self):
        """Verify total feature count is 56 (7 time-domain + 30 spectral + 19 autocorrelation)."""
        assert len(ALL_FEATURE_NAMES) == 56

    def test_time_domain_count(self):
        """Verify time-domain feature count is 7."""
        assert len(TIME_DOMAIN_FEATURE_NAMES) == 7

    def test_spectral_count(self):
        """Verify spectral feature count is 30."""
        assert len(SPECTRAL_FEATURE_NAMES) == 30

    def test_cor_count(self):
        """Verify autocorrelation feature count is 19."""
        assert len(COR_FEATURE_NAMES) == 19

class TestFrameSignal:

    def test_output_shape(self, sig_4000):
        """Verify output array dimensions and shape."""
        (sig, sr) = sig_4000
        (frames, times, T) = frame_signal(sig, sr, n_fft=512, hop_length=256)
        assert frames.ndim == 2
        assert frames.shape[1] == T
        assert len(times) == T

    def test_times_are_monotonic(self, sig_4000):
        """Verify times are monotonically increasing."""
        (sig, sr) = sig_4000
        (_, times, _) = frame_signal(sig, sr, 512, 256)
        assert np.all(np.diff(times) > 0)

class TestComputeShortTimeFeatures:

    def test_returns_dict_with_all_keys(self, sig_4000):
        """Verify all feature names are present in output dict."""
        (sig, sr) = sig_4000
        (times, feat) = compute_short_time_features(sig, sr)
        for name in ALL_FEATURE_NAMES:
            assert name in feat, f'Missing feature: {name}'

    def test_times_length_matches(self, sig_4000):
        """Verify each feature array has the same length as times."""
        (sig, sr) = sig_4000
        (times, feat) = compute_short_time_features(sig, sr)
        for (name, arr) in feat.items():
            assert len(arr) == len(times), f'{name}: len(arr)={len(arr)} != len(times)={len(times)}'

    def test_no_nan_inf(self, sig_4000):
        """Verify all feature values are finite (no NaN or inf)."""
        (sig, sr) = sig_4000
        (_, feat) = compute_short_time_features(sig, sr)
        for (name, arr) in feat.items():
            assert np.all(np.isfinite(arr)), f'{name} has NaN or inf'

    def test_empty_signal(self):
        """Verify behaviour on empty input."""
        (times, feat) = compute_short_time_features(np.array([], dtype=np.float32), 4000)
        assert len(times) == 0
        assert feat == {}

    def test_feature_matrix_shape(self, sig_4000):
        """Verify feature matrix dimensions and shape."""
        (sig, sr) = sig_4000
        (times, feat) = compute_short_time_features(sig, sr)
        (X_full, t_out, names) = build_feature_matrix(times, feat)
        assert X_full.shape[0] == len(times)
        assert X_full.shape[1] == 2 * len(ALL_FEATURE_NAMES)
        assert len(names) == X_full.shape[1]

class TestNormalize:

    def test_range_zero_one(self):
        """Verify min of normalised signal equals 0 and max equals 1."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_norm = normalize_feature_for_display(y)
        assert np.min(y_norm) == pytest.approx(0.0)
        assert np.max(y_norm) == pytest.approx(1.0)

    def test_constant_input(self):
        """Verify constant input normalises to all zeros."""
        y = np.array([7.0, 7.0, 7.0])
        y_norm = normalize_feature_for_display(y)
        assert np.all(y_norm == 0.0)

class TestTimeDomainFeatures:

    def test_energy_non_negative(self, sig_4000):
        """Verify all energy values are non-negative."""
        (sig, sr) = sig_4000
        feat = compute_time_domain_features(sig, sr, 512, 256)
        assert np.all(feat['短时能量'] >= 0)

    def test_zcr_in_range(self, sig_4000):
        """Verify zero-crossing rate is in [0, 1]."""
        (sig, sr) = sig_4000
        feat = compute_time_domain_features(sig, sr, 512, 256)
        zcr = feat['过零率']
        assert np.all(zcr >= 0) and np.all(zcr <= 1)

class TestGoldenValues:
    """Physical ground-truth: verify short-time feature values against known sine tones."""

    def test_pure_tone_spectral_centroid(self):
        """Verify 500 Hz pure tone: spectral centroid ~500 Hz."""
        import numpy as np
        from respanno.dsp.features import compute_short_time_features

        sr = 4000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        sig = np.sin(2 * np.pi * 500 * t).astype(np.float32)

        times, feat_dict = compute_short_time_features(
            sig, sr, n_fft=256, hop_length=64, f_max=2000
        )

        centroid = feat_dict['谱质心']
        # Golden value: spectral centroid of a pure tone = the tone frequency
        # Allow ±10% tolerance for FFT bin discretization
        mean_centroid = float(np.mean(centroid))
        assert 450 < mean_centroid < 550, (
            f'Spectral centroid of 500 Hz tone should be ~500 Hz, got {mean_centroid:.1f} Hz'
        )

    def test_pure_tone_rms_energy(self):
        """Verify 0.5 amplitude pure tone: RMS energy ~n_fft * amp^2 / 2."""
        import numpy as np
        from respanno.dsp.features import compute_short_time_features

        sr = 4000
        duration = 1.0
        amplitude = 0.5
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        sig = (amplitude * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

        times, feat_dict = compute_short_time_features(
            sig, sr, n_fft=256, hop_length=64, f_max=2000
        )

        energy = feat_dict['短时能量']
        n_fft = 256
        expected_energy = n_fft * (amplitude ** 2 / 2)  # = 32.0
        mean_energy = float(np.mean(energy))

        # Allow +/-15% tolerance (window + boundary effects)
        assert 27.0 < mean_energy < 37.0, (
            f'Energy for amplitude {amplitude} should be ~{expected_energy}, got {mean_energy:.1f}'
        )

    def test_pure_tone_zcr(self):
        """400 Hz pure tone: ZCR should be ~2*f/sr = 0.20."""
        import numpy as np
        from respanno.dsp.features import compute_short_time_features

        sr = 4000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        sig = np.sin(2 * np.pi * 400 * t).astype(np.float32)

        times, feat_dict = compute_short_time_features(
            sig, sr, n_fft=256, hop_length=64, f_max=2000
        )

        zcr = feat_dict['过零率']
        expected = 2 * 400 / sr  # = 0.20
        mean_zcr = float(np.mean(zcr))

        # Allow +/-20% tolerance
        assert 0.16 < mean_zcr < 0.24, (
            f'ZCR of 400 Hz tone should be ~{expected:.3f}，实际 {mean_zcr:.3f}'
        )

    def test_silence_rms_near_zero(self):
        """Verify silence: RMS energy near zero."""
        import numpy as np
        from respanno.dsp.features import compute_short_time_features

        sr = 4000
        duration = 1.0
        sig = np.zeros(int(sr * duration), dtype=np.float32)

        times, feat_dict = compute_short_time_features(
            sig, sr, n_fft=256, hop_length=64, f_max=2000
        )

        energy = feat_dict['短时能量']
        assert float(np.max(energy)) < 1e-6, (
            f'Silence short-time energy should be near 0, max was {float(np.max(energy)):.2e}'
        )
