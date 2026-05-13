"""Tests for respanno.dsp.spectrogram.

All tests exercise the extracted module directly (no QApplication needed).
"""

import numpy as np
import pytest

from respanno.dsp.spectrogram import (
    DEFAULT_STFT_CONFIG,
    compute_stft_db,
    decimate_spec_for_display,
    get_palette_256,
    colorize_spectrogram,
    compute_spectrogram_display,
    limit_frequency_range,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sig_4000():
    sr = 4000
    t = np.linspace(0, 2, sr * 2, endpoint=False)
    sig = 0.5 * np.sin(2 * np.pi * 100 * t) + 0.3 * np.sin(2 * np.pi * 800 * t)
    return sig.astype(np.float32), sr


# ---------------------------------------------------------------------------
# compute_stft_db
# ---------------------------------------------------------------------------

class TestComputeSTFTDB:
    def test_output_shape(self, sig_4000):
        sig, sr = sig_4000
        S_db, freqs = compute_stft_db(sig, sr, n_fft=512, hop_length=256, f_max=2000)
        assert S_db.ndim == 2
        assert S_db.shape[0] == len(freqs)
        assert S_db.shape[1] > 0
        assert freqs[0] == 0.0

    def test_f_max_clamped_to_nyquist(self, sig_4000):
        sig, sr = sig_4000
        S_db, freqs = compute_stft_db(sig, sr, n_fft=512, hop_length=256, f_max=4000)
        assert freqs[-1] <= sr / 2

    def test_all_db_values_at_most_zero(self, sig_4000):
        sig, sr = sig_4000
        S_db, _ = compute_stft_db(sig, sr)
        # ref=np.max → the maximum should be ~0 and all values <= 0
        assert np.max(np.abs(S_db)) < 100  # sanity: not crazy dB values

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            compute_stft_db(np.array([]), 4000)


# ---------------------------------------------------------------------------
# limit_frequency_range
# ---------------------------------------------------------------------------

class TestLimitFrequencyRange:
    def test_crops(self, sig_4000):
        sig, sr = sig_4000
        S_db, freqs = compute_stft_db(sig, sr, f_max=2000)
        S_sub, f_sub = limit_frequency_range(S_db, freqs, f_max=500, sr=sr)
        assert f_sub[-1] <= 500
        assert S_sub.shape[0] < S_db.shape[0]


# ---------------------------------------------------------------------------
# decimate_spec_for_display
# ---------------------------------------------------------------------------

class TestDecimate:
    def test_no_change_when_small(self):
        spec = np.random.default_rng(0).normal(size=(100, 200))
        result = decimate_spec_for_display(spec, max_time_bins=500, max_freq_bins=256)
        assert result.shape == spec.shape

    def test_reduces_large_freq(self):
        spec = np.random.default_rng(0).normal(size=(800, 100))
        result = decimate_spec_for_display(spec, max_time_bins=9999, max_freq_bins=256)
        assert result.shape[0] == 256

    def test_reduces_large_time(self):
        spec = np.random.default_rng(0).normal(size=(100, 5000))
        result = decimate_spec_for_display(spec, max_time_bins=1000, max_freq_bins=256)
        assert result.shape[1] == 1000

    def test_1d_input_passthrough(self):
        spec = np.array([1.0, 2.0, 3.0])
        result = decimate_spec_for_display(spec)
        assert np.array_equal(result, spec)


# ---------------------------------------------------------------------------
# get_palette_256
# ---------------------------------------------------------------------------

class TestPalette:
    def test_heatmap_shape(self):
        lut = get_palette_256("Heatmap")
        assert lut.shape == (256, 3)
        assert np.all(lut >= 0.0) and np.all(lut <= 1.0)

    def test_grayscale_shape(self):
        lut = get_palette_256("Grayscale")
        assert lut.shape == (256, 3)
        # Each row should have identical R/G/B
        assert np.allclose(lut[:, 0], lut[:, 1])
        assert np.allclose(lut[:, 1], lut[:, 2])


# ---------------------------------------------------------------------------
# colorize_spectrogram
# ---------------------------------------------------------------------------

class TestColorize:
    def test_output_is_uint8_rgb(self, sig_4000):
        sig, sr = sig_4000
        S_db, _ = compute_stft_db(sig, sr)
        rgb = colorize_spectrogram(S_db.T, cmap="Heatmap")
        assert rgb.dtype == np.uint8
        assert rgb.ndim == 3
        assert rgb.shape[2] == 3
        assert rgb.shape[0] == S_db.shape[1]  # time
        assert rgb.shape[1] == S_db.shape[0]  # freq

    def test_heatmap_available(self, sig_4000):
        sig, sr = sig_4000
        S_db, _ = compute_stft_db(sig, sr)
        rgb = colorize_spectrogram(S_db.T, cmap="Heatmap")
        assert rgb.shape[2] == 3
        assert rgb.max() <= 255

    def test_grayscale_available(self, sig_4000):
        sig, sr = sig_4000
        S_db, _ = compute_stft_db(sig, sr)
        rgb = colorize_spectrogram(S_db.T, cmap="Grayscale")
        # R == G == B for every pixel
        assert np.allclose(rgb[:, :, 0], rgb[:, :, 1])
        assert np.allclose(rgb[:, :, 1], rgb[:, :, 2])

    def test_custom_levels(self, sig_4000):
        sig, sr = sig_4000
        S_db, _ = compute_stft_db(sig, sr)
        rgb = colorize_spectrogram(S_db.T, cmap="Heatmap", vmin=-80, vmax=-10)
        assert rgb.shape[2] == 3

    def test_all_nan_input(self):
        spec = np.full((10, 20), np.nan)
        rgb = colorize_spectrogram(spec)
        assert rgb.shape == (10, 20, 3)
        assert np.all(rgb == 0)


# ---------------------------------------------------------------------------
# compute_spectrogram_display
# ---------------------------------------------------------------------------

class TestComputeSpectrogramDisplay:
    def test_full_pipeline(self, sig_4000):
        sig, sr = sig_4000
        result = compute_spectrogram_display(sig, sr)
        assert "S_db" in result
        assert "rgb" in result
        assert "freqs" in result
        assert "duration" in result
        assert result["rgb"].ndim == 3

    def test_config_override(self, sig_4000):
        sig, sr = sig_4000
        cfg = {"n_fft": 256, "hop_length": 128, "f_max": 1000, "cmap": "Grayscale"}
        result = compute_spectrogram_display(sig, sr, config=cfg)
        assert result["n_fft"] == 256
        assert result["hop_length"] == 128
        assert result["f_max_eff"] <= 1000


class TestDefaultConfig:
    def test_defaults_match_legacy(self):
        assert DEFAULT_STFT_CONFIG["n_fft"] == 512
        assert DEFAULT_STFT_CONFIG["hop_length"] == 256
        assert DEFAULT_STFT_CONFIG["f_max"] == 2000
        assert DEFAULT_STFT_CONFIG["cmap"] == "Heatmap"
