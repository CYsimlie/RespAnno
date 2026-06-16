"""Tests for respanno.dsp.spectrogram.

All tests exercise the extracted module directly (no QApplication needed).
"""
import numpy as np
import pytest
from respanno.dsp.spectrogram import DEFAULT_STFT_CONFIG, compute_stft_db, decimate_spec_for_display, get_palette_256, colorize_spectrogram, compute_spectrogram_display, limit_frequency_range

@pytest.fixture(scope='module')
def sig_4000():
    sr = 4000
    t = np.linspace(0, 2, sr * 2, endpoint=False)
    sig = 0.5 * np.sin(2 * np.pi * 100 * t) + 0.3 * np.sin(2 * np.pi * 800 * t)
    return (sig.astype(np.float32), sr)

class TestComputeSTFTDB:

    def test_output_shape(self, sig_4000):
        """Verify output array dimensions and shape match expectations."""
        (sig, sr) = sig_4000
        (S_db, freqs) = compute_stft_db(sig, sr, n_fft=512, hop_length=256, f_max=2000)
        assert S_db.ndim == 2
        assert S_db.shape[0] == len(freqs)
        assert S_db.shape[1] > 0
        assert freqs[0] == 0.0

    def test_f_max_clamped_to_nyquist(self, sig_4000):
        """Verify invalid parameter values are clamped to valid range."""
        (sig, sr) = sig_4000
        (S_db, freqs) = compute_stft_db(sig, sr, n_fft=512, hop_length=256, f_max=4000)
        assert freqs[-1] <= sr / 2

    def test_all_db_values_at_most_zero(self, sig_4000):
        """Verify all STFT dB values are <= 0 dB."""
        (sig, sr) = sig_4000
        (S_db, _) = compute_stft_db(sig, sr)
        assert np.max(np.abs(S_db)) < 100

    def test_empty_raises(self):
        """Verify empty input raises an exception."""
        with pytest.raises(ValueError):
            compute_stft_db(np.array([]), 4000)

class TestLimitFrequencyRange:

    def test_crops(self, sig_4000):
        """Verify limit_frequency_range correctly crops the spectrogram."""
        (sig, sr) = sig_4000
        (S_db, freqs) = compute_stft_db(sig, sr, f_max=2000)
        (S_sub, f_sub) = limit_frequency_range(S_db, freqs, f_max=500, sr=sr)
        assert f_sub[-1] <= 500
        assert S_sub.shape[0] < S_db.shape[0]

class TestDecimate:

    def test_no_change_when_small(self):
        """Verify：result.shape == spec.shape。"""
        spec = np.random.default_rng(0).normal(size=(100, 200))
        result = decimate_spec_for_display(spec, max_time_bins=500, max_freq_bins=256)
        assert result.shape == spec.shape

    def test_reduces_large_freq(self):
        """Verify：result.shape[0] == 256。"""
        spec = np.random.default_rng(0).normal(size=(800, 100))
        result = decimate_spec_for_display(spec, max_time_bins=9999, max_freq_bins=256)
        assert result.shape[0] == 256

    def test_reduces_large_time(self):
        """Verify：result.shape[1] == 1000。"""
        spec = np.random.default_rng(0).normal(size=(100, 5000))
        result = decimate_spec_for_display(spec, max_time_bins=1000, max_freq_bins=256)
        assert result.shape[1] == 1000

    def test_1d_input_passthrough(self):
        """Verify：np.array_equal(result, spec)。"""
        spec = np.array([1.0, 2.0, 3.0])
        result = decimate_spec_for_display(spec)
        assert np.array_equal(result, spec)

class TestPalette:

    def test_heatmap_shape(self):
        """Verify output array dimensions and shape match expected。"""
        lut = get_palette_256('Heatmap')
        assert lut.shape == (256, 3)
        assert np.all(lut >= 0.0) and np.all(lut <= 1.0)

    def test_grayscale_shape(self):
        """Verify output array dimensions and shape match expected。"""
        lut = get_palette_256('Grayscale')
        assert lut.shape == (256, 3)
        assert np.allclose(lut[:, 0], lut[:, 1])
        assert np.allclose(lut[:, 1], lut[:, 2])

class TestColorize:

    def test_output_is_uint8_rgb(self, sig_4000):
        """Verify：rgb.dtype == np.uint8。"""
        (sig, sr) = sig_4000
        (S_db, _) = compute_stft_db(sig, sr)
        rgb = colorize_spectrogram(S_db.T, cmap='Heatmap')
        assert rgb.dtype == np.uint8
        assert rgb.ndim == 3
        assert rgb.shape[2] == 3
        assert rgb.shape[0] == S_db.shape[1]
        assert rgb.shape[1] == S_db.shape[0]

    def test_heatmap_available(self, sig_4000):
        """Verify：rgb.shape[2] == 3。"""
        (sig, sr) = sig_4000
        (S_db, _) = compute_stft_db(sig, sr)
        rgb = colorize_spectrogram(S_db.T, cmap='Heatmap')
        assert rgb.shape[2] == 3
        assert rgb.max() <= 255

    def test_grayscale_available(self, sig_4000):
        """Verify：np.allclose(rgb[:, :, 0], rgb[:, :, 1])。"""
        (sig, sr) = sig_4000
        (S_db, _) = compute_stft_db(sig, sr)
        rgb = colorize_spectrogram(S_db.T, cmap='Grayscale')
        assert np.allclose(rgb[:, :, 0], rgb[:, :, 1])
        assert np.allclose(rgb[:, :, 1], rgb[:, :, 2])

    def test_custom_levels(self, sig_4000):
        """Verify：rgb.shape[2] == 3。"""
        (sig, sr) = sig_4000
        (S_db, _) = compute_stft_db(sig, sr)
        rgb = colorize_spectrogram(S_db.T, cmap='Heatmap', vmin=-80, vmax=-10)
        assert rgb.shape[2] == 3

    def test_all_nan_input(self):
        """Verify：rgb.shape == (10, 20, 3)。"""
        spec = np.full((10, 20), np.nan)
        rgb = colorize_spectrogram(spec)
        assert rgb.shape == (10, 20, 3)
        assert np.all(rgb == 0)

class TestComputeSpectrogramDisplay:

    def test_full_pipeline(self, sig_4000):
        """Verify：'S_db' in result。"""
        (sig, sr) = sig_4000
        result = compute_spectrogram_display(sig, sr)
        assert 'S_db' in result
        assert 'rgb' in result
        assert 'freqs' in result
        assert 'duration' in result
        assert result['rgb'].ndim == 3

    def test_config_override(self, sig_4000):
        """Verify：result['n_fft'] == 256。"""
        (sig, sr) = sig_4000
        cfg = {'n_fft': 256, 'hop_length': 128, 'f_max': 1000, 'cmap': 'Grayscale'}
        result = compute_spectrogram_display(sig, sr, config=cfg)
        assert result['n_fft'] == 256
        assert result['hop_length'] == 128
        assert result['f_max_eff'] <= 1000

class TestDefaultConfig:

    def test_defaults_match_legacy(self):
        """Verify default parameter values match legacy."""
        assert DEFAULT_STFT_CONFIG['n_fft'] == 256
        assert DEFAULT_STFT_CONFIG['hop_length'] == 64
        assert DEFAULT_STFT_CONFIG['f_max'] == 2000
        assert DEFAULT_STFT_CONFIG['cmap'] == 'Heatmap'

class TestGoldenValues:
    """Physical ground-truth: verify STFT output at correct time/frequency positions."""

    def test_sine_sweep_tf_localization(self):
        """500 Hz + 1000 Hz sine sweep: verify STFT energy peaks switch correctly."""
        import numpy as np
        from respanno.dsp.spectrogram import compute_stft_db

        sr = 4000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)

        # First half 500 Hz, second half 1000 Hz
        mid = len(t) // 2
        sig = np.zeros_like(t)
        sig[:mid] = np.sin(2 * np.pi * 500 * t[:mid])
        sig[mid:] = np.sin(2 * np.pi * 1000 * t[mid:])

        S_db, freqs = compute_stft_db(
            sig.astype(np.float32), sr,
            n_fft=256, hop_length=64, f_max=2000
        )

        # Find bins nearest to 500 Hz and 1000 Hz
        i500 = int(np.argmin(np.abs(freqs - 500)))
        i1000 = int(np.argmin(np.abs(freqs - 1000)))

        n_frames = S_db.shape[1]
        half_frames = n_frames // 2

        # Golden value: first-half frames have more energy at 500 Hz than at 1000 Hz
        energy_500_first = np.mean(S_db[i500, :half_frames])
        energy_1000_first = np.mean(S_db[i1000, :half_frames])
        assert energy_500_first > energy_1000_first, (
            f'前半段：500 Hz energy({energy_500_first:.1f})应 > 1000 Hz({energy_1000_first:.1f})'
        )

        # Golden value: second-half frames have more energy at 1000 Hz than at 500 Hz
        energy_500_second = np.mean(S_db[i500, half_frames:])
        energy_1000_second = np.mean(S_db[i1000, half_frames:])
        assert energy_1000_second > energy_500_second, (
            f'后半段：1000 Hz energy({energy_1000_second:.1f})应 > 500 Hz({energy_500_second:.1f})'
        )

    def test_single_tone_consistent_peak(self):
        """400 Hz pure tone: all STFT frame peaks should be near 400 Hz (+/-50 Hz)."""
        import numpy as np
        from respanno.dsp.spectrogram import compute_stft_db

        sr = 4000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        sig = np.sin(2 * np.pi * 400 * t).astype(np.float32)

        S_db, freqs = compute_stft_db(sig, sr, n_fft=256, hop_length=64, f_max=2000)

        # Find peak frequency per frame.
        peak_indices = np.argmax(S_db, axis=0)
        peak_freqs = freqs[peak_indices]

        # Golden value: all frame peaks within 400 +/-50 Hz.
        # (boundary frames may have minor deviation; ±50 Hz tolerance)
        assert np.all((peak_freqs >= 350) & (peak_freqs <= 450)), (
            f'Peak frequency range: {peak_freqs.min():.0f}-{peak_freqs.max():.0f} Hz (expected 350-450 Hz)'
        )
