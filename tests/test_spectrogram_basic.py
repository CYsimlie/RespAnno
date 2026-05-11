"""Tests for STFT spectrogram computation and feature extraction.

Status: TEST SCAFFOLDING — STFT / feature logic is embedded in
AudioViewer.update_spectrogram() and AudioViewer.compute_short_time_features().
Both require a loaded audio file on an AudioViewer instance (needs QApplication).

These tests verify:
1. librosa STFT output shapes and values on synthetic signals.
2. Individual feature calculation correctness (using pure functions).
3. Feature name inventory completeness.

TODO (Phase 3): After extracting `compute_short_time_features` and
`update_spectrogram` into `respanno/dsp/`, rewrite tests to use those
directly without QApplication.
"""

import numpy as np
import pytest
import librosa


# ---------------------------------------------------------------------------
# STFT basic tests (librosa, no GUI needed)
# ---------------------------------------------------------------------------

class TestSTFTBasic:
    """Verify librosa STFT output shapes for a synthetic signal."""

    def test_stft_output_shape(self, synthetic_audio):
        sig, sr = synthetic_audio
        n_fft = 512
        hop_length = 256

        D = librosa.stft(sig, n_fft=n_fft, hop_length=hop_length,
                         center=True, pad_mode="reflect")

        # Expected freq bins: n_fft//2 + 1
        assert D.shape[0] == n_fft // 2 + 1

        # Expected time frames: floor((len(sig) + n_fft) / hop_length)
        # (approximately, with center=True)
        assert D.shape[1] > 0

    def test_stft_freq_axis(self, synthetic_audio):
        sig, sr = synthetic_audio
        n_fft = 512
        D = librosa.stft(sig, n_fft=n_fft, hop_length=256)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        assert len(freqs) == D.shape[0]
        assert freqs[0] == 0.0
        assert freqs[-1] == sr / 2

    def test_stft_db_conversion(self, synthetic_audio):
        sig, sr = synthetic_audio
        D = librosa.stft(sig, n_fft=512, hop_length=256)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        # DB values should be <= 0 (since ref=np.max)
        assert np.all(S_db <= 1e-9)

    def test_stft_f_max_subset(self, synthetic_audio):
        sig, sr = synthetic_audio
        n_fft = 512
        D = librosa.stft(sig, n_fft=n_fft, hop_length=256)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        f_max = 2000.0
        idx = freqs <= f_max
        assert np.any(idx)
        S_sub = np.abs(D)[idx, :]
        assert S_sub.shape[0] > 0
        assert S_sub.shape[1] == D.shape[1]


# ---------------------------------------------------------------------------
# Feature calculation tests (pure functions, no GUI needed)
# ---------------------------------------------------------------------------

class TestFeatureCalculations:
    """Verify individual feature formulas on known inputs."""

    def test_short_time_energy(self):
        """短时能量 = sum(x^2) per frame."""
        # 3 frames, 2 samples each: shape (3, 2)
        frames = np.array([[1.0, 0.5],
                           [2.0, 1.0],
                           [1.0, 0.5]])
        energy = np.sum(frames ** 2, axis=1)  # sum over samples in each frame
        expected = np.array([1.25, 5.0, 1.25])  # frame0:1+0.25, frame1:4+1, frame2:1+0.25
        assert np.allclose(energy, expected)

    def test_zero_crossing_rate(self):
        """过零率 should count sign changes."""
        x = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32)
        # Pad to length > frame_length for librosa
        x_long = np.tile(x, 128)
        zcr = librosa.feature.zero_crossing_rate(x_long, frame_length=256, hop_length=128)[0]
        assert zcr.size > 0
        assert np.all(zcr >= 0) and np.all(zcr <= 1)

    def test_spectral_centroid(self):
        """谱质心 should be within freq range."""
        sig, sr = 0.5 * np.sin(2 * np.pi * 300 * np.linspace(0, 1, 4000)), 4000
        D = librosa.stft(sig, n_fft=512, hop_length=256)
        S = np.abs(D)
        cent = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
        # Centroid of a pure 300 Hz sine should be near 300 Hz
        assert np.all(np.abs(cent - 300) < 100)

    def test_spectral_flatness_range(self):
        """谱平坦度 should be in [0, 1]."""
        sig, sr = 0.5 * np.sin(2 * np.pi * 300 * np.linspace(0, 1, 4000)), 4000
        D = librosa.stft(sig, n_fft=512, hop_length=256)
        flat = librosa.feature.spectral_flatness(S=np.abs(D))[0]
        assert np.all(flat >= 0) and np.all(flat <= 1)


class TestFeatureInventory:
    """Verify the set of feature names matches the legacy code definition."""

    def test_feature_names_completeness(self):
        """All features listed in the legacy UI must be accounted for."""
        time_features = {
            "短时能量", "短时均值", "方差", "峰度", "偏度",
            "过零率", "teager能量算子",
        }
        spec_basic = {
            "谱均值", "谱标准差", "谱中位数", "谱能量", "谱RMS", "谱幅和",
            "谱质心", "谱带宽", "谱偏度", "谱峰度", "谱滚降", "谱平坦度",
            "谱熵", "谱通量",
            "最大谱峰值", "谱峰数量",
            "低频能量占比", "中频能量占比", "高频能量占比",
            "谱四分位距", "谱MAD", "谱差分零交叉率", "谱平滑度",
            "主峰/次峰比", "谱复杂度",
            "主峰能量占比", "前三峰能量占比", "90%能量覆盖频点数",
            "主峰-3dB带宽", "主峰Q因子",
        }
        cor_features = {
            "cor_dist_ratio_mean", "cor_mean_slope", "cor_max_slope",
            "cor_std_slope", "cor_max_peak", "cor_second_peak",
            "cor_peak_count", "cor_peak_density", "cor_area", "cor_std",
            "cor_cv", "cor_skewness", "cor_kurtosis",
            "cor_local_max_slope_mean", "cor_local_max_slope_min",
            "cor_local_std_mean", "cor_local_std_max",
            "cor_local_pk2pk_mean", "cor_local_pk2pk_max",
        }

        all_features = time_features | spec_basic | cor_features
        # Legacy code lists 56 features (7 time + 30 spec + 19 cor)
        assert len(all_features) == 56, f"Expected 56, got {len(all_features)}"
        assert "短时能量" in all_features
        assert "谱质心" in all_features
        assert "cor_dist_ratio_mean" in all_features


# ---------------------------------------------------------------------------
# TODO: Tests that require extraction
# ---------------------------------------------------------------------------

class TestSpectrogramFromAudioViewer:
    """
    TODO: After extracting update_spectrogram to respanno/dsp/stft.py:

    1. test_update_spectrogram_output_shape:
       - Verify S_db shape matches (n_freq_bins <= f_max, n_frames)

    2. test_decimate_spec_for_display:
       - Verify output is smaller than input when exceeding thresholds

    3. test_colorize_spec_output_range:
       - Verify RGB values are in [0, 255]

    4. test_features_matrix_shape:
       - Verify stft_features shape is (T, 2D) after ensure_frame_features

    5. test_features_reproducible:
       - Same audio -> same feature matrix (deterministic)
    """

    def test_todo_placeholder(self):
        pytest.skip("TODO: extract compute_short_time_features to respanno/dsp/")
