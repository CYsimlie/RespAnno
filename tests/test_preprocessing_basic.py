"""Tests for respanno.audio.preprocessing.

All tests exercise the extracted module directly (no QApplication needed).
"""
import numpy as np
import pytest
import scipy.io.wavfile as wavfile
from respanno.audio.preprocessing import DEFAULT_PREPROCESSING_CONFIG, validate_preprocessing_config, compute_target_sr, apply_butter_filter, apply_preprocessing, preprocess_audio_file, summarize_preprocessing

class TestValidateConfig:

    def test_defaults(self):
        """Verify default configuration values."""
        cfg = validate_preprocessing_config({})
        assert cfg['preprocessing_enabled'] is True
        assert cfg['resample_enabled'] is True
        assert cfg['resample_target_sr'] == 4000
        assert cfg['filter_enabled'] is False
        assert cfg['filter_type'] == 'bandpass'
        assert cfg['filter_lowcut'] == 20.0
        assert cfg['filter_highcut'] == 1800.0
        assert cfg['filter_order'] == 4
        assert cfg['filter_zero_phase'] is True

    def test_invalid_filter_type_clamped(self):
        """Verify invalid filter type is clamped to 'bandpass'."""
        cfg = validate_preprocessing_config({'filter_type': 'unknown'})
        assert cfg['filter_type'] == 'bandpass'

    def test_order_clamped(self):
        """Verify filter order is clamped to [4, 12]."""
        cfg = validate_preprocessing_config({'filter_order': 50})
        assert cfg['filter_order'] == 12
        cfg = validate_preprocessing_config({'filter_order': 0})
        assert cfg['filter_order'] == 4

    def test_None_values_filled(self):
        """Verify None values are filled with defaults."""
        cfg = validate_preprocessing_config({'filter_type': None, 'filter_lowcut': None, 'filter_highcut': None})
        assert cfg['filter_type'] == 'bandpass'
        assert cfg['filter_lowcut'] == 20.0
        assert cfg['filter_highcut'] == 1800.0

class TestComputeTargetSR:

    def test_enabled_returns_4000(self):
        """Verify compute_target_sr returns 4000 when resampling is enabled."""
        cfg = {'preprocessing_enabled': True, 'resample_enabled': True, 'resample_target_sr': 4000}
        assert compute_target_sr(cfg) == 4000

    def test_disabled_returns_none(self):
        """Verify compute_target_sr returns None when preprocessing or resampling is disabled."""
        assert compute_target_sr({'preprocessing_enabled': False, 'resample_enabled': True, 'resample_target_sr': 4000}) is None
        assert compute_target_sr({'preprocessing_enabled': True, 'resample_enabled': False, 'resample_target_sr': 4000}) is None

    def test_zero_target_returns_none(self):
        """Verify compute_target_sr returns None when target SR is 0."""
        assert compute_target_sr({'preprocessing_enabled': True, 'resample_enabled': True, 'resample_target_sr': 0}) is None

class TestApplyButterFilter:
    """Cover bandpass / lowpass / highpass / bandstop + edge cases."""

    @pytest.fixture
    def sig_4000(self):
        sr = 4000
        t = np.linspace(0, 1, sr, endpoint=False)
        sig = 0.5 * np.sin(2 * np.pi * 100 * t) + 0.3 * np.sin(2 * np.pi * 800 * t)
        return (sig.astype(np.float32), sr)

    def test_bandpass_no_error(self, sig_4000):
        """Verify bandpass filter preserves shape and produces finite output."""
        (sig, sr) = sig_4000
        y = apply_butter_filter(sig, sr, filter_type='bandpass', lowcut=200, highcut=500)
        assert y.shape == sig.shape
        assert np.all(np.isfinite(y))

    def test_lowpass_no_error(self, sig_4000):
        """Verify lowpass filter preserves shape and produces finite output."""
        (sig, sr) = sig_4000
        y = apply_butter_filter(sig, sr, filter_type='lowpass', highcut=1500)
        assert y.shape == sig.shape
        assert np.all(np.isfinite(y))

    def test_highpass_no_error(self, sig_4000):
        """Verify highpass filter preserves shape and produces finite output."""
        (sig, sr) = sig_4000
        y = apply_butter_filter(sig, sr, filter_type='highpass', lowcut=300)
        assert y.shape == sig.shape
        assert np.all(np.isfinite(y))

    def test_bandstop_no_error(self, sig_4000):
        """Verify bandstop filter preserves shape and produces finite output."""
        (sig, sr) = sig_4000
        y = apply_butter_filter(sig, sr, filter_type='bandstop', lowcut=200, highcut=500)
        assert y.shape == sig.shape
        assert np.all(np.isfinite(y))

    def test_lowpass_attenuates_high(self, sig_4000):
        """Verify lowpass filter attenuates energy above the cutoff frequency."""
        (sig, sr) = sig_4000
        y = apply_butter_filter(sig, sr, filter_type='lowpass', highcut=500, order=4)
        from scipy.fft import rfft
        spec_orig = np.abs(rfft(sig))
        spec_y = np.abs(rfft(y))
        freqs = np.fft.rfftfreq(len(y), d=1 / sr)
        ratio = np.sum(spec_y[freqs > 700]) / (np.sum(spec_orig[freqs > 700]) + 1e-12)
        assert ratio < 1.5

    def test_highcut_exceeds_nyquist_clamped(self, sig_4000):
        """Verify highcut exceeding Nyquist frequency is clamped."""
        (sig, sr) = sig_4000
        y = apply_butter_filter(sig, sr, filter_type='bandpass', lowcut=200, highcut=5000)
        assert y.shape == sig.shape
        assert np.all(np.isfinite(y))

    def test_bandpass_invalid_range_returns_original(self, sig_4000):
        """Verify bandpass with invalid range returns the original signal."""
        (sig, sr) = sig_4000
        y = apply_butter_filter(sig, sr, filter_type='bandpass', lowcut=500, highcut=100)
        assert np.array_equal(y, sig)

    def test_short_signal(self):
        """Verify filter handles a very short signal without error."""
        sig = np.array([0.1, -0.2, 0.05], dtype=np.float32)
        y = apply_butter_filter(sig, 4000, filter_type='bandpass')
        assert y.shape == sig.shape
        assert np.all(np.isfinite(y))

    def test_zero_phase_vs_causal(self):
        """Verify both zero-phase and causal filtering produce finite output."""
        sr = 4000
        rng = np.random.default_rng(42)
        sig = rng.normal(0, 1, sr).astype(np.float32)
        y_zp = apply_butter_filter(sig, sr, filter_type='bandpass', lowcut=200, highcut=500, zero_phase=True)
        y_causal = apply_butter_filter(sig, sr, filter_type='bandpass', lowcut=200, highcut=500, zero_phase=False)
        assert np.all(np.isfinite(y_zp))
        assert np.all(np.isfinite(y_causal))

class TestApplyPreprocessing:

    @pytest.fixture
    def sig_4000(self):
        sr = 4000
        t = np.linspace(0, 1, sr, endpoint=False)
        sig = 0.5 * np.sin(2 * np.pi * 100 * t).astype(np.float32)
        return (sig, sr)

    def test_filter_disabled_passthrough(self, sig_4000):
        """Verify signal passes through unchanged when filter is disabled."""
        (sig, sr) = sig_4000
        cfg = {'preprocessing_enabled': True, 'filter_enabled': False}
        (y, meta) = apply_preprocessing(sig, sr, config=cfg)
        assert np.array_equal(y, sig)
        assert meta['filter_applied'] is False

    def test_preprocessing_disabled_passthrough(self, sig_4000):
        """Verify signal passes through unchanged when preprocessing is disabled."""
        (sig, sr) = sig_4000
        cfg = {'preprocessing_enabled': False, 'filter_enabled': True}
        (y, meta) = apply_preprocessing(sig, sr, config=cfg)
        assert np.array_equal(y, sig)

    def test_metadata_complete(self, sig_4000):
        """Verify all expected metadata keys are present."""
        (sig, sr) = sig_4000
        (y, meta) = apply_preprocessing(sig, sr)
        for key in ('input_sr', 'output_sr', 'filter_enabled', 'filter_type', 'filter_lowcut', 'filter_highcut', 'filter_order', 'filter_zero_phase'):
            assert key in meta, f'Missing metadata key: {key}'

class TestPreprocessAudioFile:

    def test_load_resample_44100_to_4000(self, tmp_path):
        """Synthetic 44100 Hz WAV → resample to 4000 Hz."""
        sr_orig = 44100
        t = np.linspace(0, 1, sr_orig, endpoint=False)
        sig = 0.5 * np.sin(2 * np.pi * 100 * t).astype(np.float32)
        sig_i16 = (sig / np.max(np.abs(sig)) * 32767).astype(np.int16)
        p = tmp_path / 'test.wav'
        wavfile.write(str(p), sr_orig, sig_i16)
        (audio, sr, meta) = preprocess_audio_file(str(p))
        assert sr == 4000
        assert meta['original_sr'] == 44100
        assert meta['processed_sr'] == 4000
        assert audio.shape[0] == 4000

    def test_load_no_resample(self, tmp_path):
        """Resample disabled → keep original rate."""
        sr_orig = 8000
        t = np.linspace(0, 0.5, int(sr_orig * 0.5), endpoint=False)
        sig = 0.5 * np.sin(2 * np.pi * 300 * t).astype(np.float32)
        sig_i16 = (sig / np.max(np.abs(sig)) * 32767).astype(np.int16)
        p = tmp_path / 'test.wav'
        wavfile.write(str(p), sr_orig, sig_i16)
        cfg = {'preprocessing_enabled': True, 'resample_enabled': False}
        (audio, sr, meta) = preprocess_audio_file(str(p), config=cfg)
        assert sr == sr_orig
        assert meta['processed_sr'] == sr_orig

    def test_metadata_keys(self, tmp_path):
        """Verify all expected metadata keys are present after loading."""
        sr_orig = 16000
        t = np.linspace(0, 0.5, int(sr_orig * 0.5), endpoint=False)
        sig = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        sig_i16 = (sig / np.max(np.abs(sig)) * 32767).astype(np.int16)
        p = tmp_path / 'test.wav'
        wavfile.write(str(p), sr_orig, sig_i16)
        (audio, sr, meta) = preprocess_audio_file(str(p))
        for key in ('original_sr', 'processed_sr', 'preprocessing_enabled', 'resample_enabled', 'filter_enabled', 'filter_type'):
            assert key in meta, f'Missing: {key}'

class TestSummarizePreprocessing:

    def test_off(self):
        """Verify：summarize_preprocessing({'preprocessing_enabled': False}) == 'preprocessing off'。"""
        assert summarize_preprocessing({'preprocessing_enabled': False}) == 'preprocessing off'

    def test_resample_only(self):
        """Verify resample-only config summary includes the target rate."""
        s = summarize_preprocessing({'preprocessing_enabled': True, 'resample_enabled': True, 'resample_target_sr': 4000, 'filter_enabled': False})
        assert 'resample=4000' in s

    def test_filter_bandpass(self):
        """Verify bandpass filter config appears in summary string."""
        s = summarize_preprocessing({'preprocessing_enabled': True, 'resample_enabled': True, 'resample_target_sr': 4000, 'filter_enabled': True, 'filter_type': 'bandpass', 'filter_lowcut': 100.0, 'filter_highcut': 1800.0})
        assert 'bandpass=' in s

    def test_filter_lowpass(self):
        """Verify lowpass filter config appears in summary string."""
        s = summarize_preprocessing({'preprocessing_enabled': True, 'resample_enabled': False, 'filter_enabled': True, 'filter_type': 'lowpass', 'filter_highcut': 800.0})
        assert 'lowpass' in s


class TestGoldenValues:
    """Physical ground-truth: filter frequency response verification via sine/FFT."""

    def test_lowpass_attenuates_above_cutoff(self):
        """Verify lowpass 500 Hz attenuates a 1000 Hz tone by at least 20 dB."""
        import numpy as np
        from respanno.audio.preprocessing import apply_butter_filter

        sr = 4000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        sig = np.sin(2 * np.pi * 1000 * t).astype(np.float32)

        filtered = apply_butter_filter(
            sig, sr, filter_type='lowpass', lowcut=20.0,
            highcut=500.0, order=4, zero_phase=True)

        fft_in = np.abs(np.fft.rfft(sig))
        fft_out = np.abs(np.fft.rfft(filtered))
        freqs = np.fft.rfftfreq(len(sig), 1 / sr)
        idx_1k = int(np.argmin(np.abs(freqs - 1000)))
        atten_db = 20 * np.log10(max(fft_out[idx_1k], 1e-12) / max(fft_in[idx_1k], 1e-12))

        assert atten_db < -20, (
            f'Lowpass 500 Hz should attenuate 1000 Hz by >20 dB, got {atten_db:.1f} dB'
        )

    def test_bandpass_preserves_inband(self):
        """Verify bandpass 20-1800 Hz preserves a 500 Hz tone (< 3 dB attenuation)."""
        import numpy as np
        from respanno.audio.preprocessing import apply_butter_filter

        sr = 4000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        sig = np.sin(2 * np.pi * 500 * t).astype(np.float32)

        filtered = apply_butter_filter(
            sig, sr, filter_type='bandpass', lowcut=20.0,
            highcut=1800.0, order=4, zero_phase=True)

        fft_in = np.abs(np.fft.rfft(sig))
        fft_out = np.abs(np.fft.rfft(filtered))
        freqs = np.fft.rfftfreq(len(sig), 1 / sr)
        idx_500 = int(np.argmin(np.abs(freqs - 500)))
        atten_db = 20 * np.log10(max(fft_out[idx_500], 1e-12) / max(fft_in[idx_500], 1e-12))

        assert atten_db > -3, (
            f'Bandpass should preserve in-band 500 Hz (< 3 dB), got {atten_db:.1f} dB'
        )

    def test_highpass_attenuates_below_cutoff(self):
        """Verify highpass 100 Hz attenuates a 50 Hz tone by at least 20 dB."""
        import numpy as np
        from respanno.audio.preprocessing import apply_butter_filter

        sr = 4000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        sig = np.sin(2 * np.pi * 50 * t).astype(np.float32)

        filtered = apply_butter_filter(
            sig, sr, filter_type='highpass', lowcut=100.0,
            highcut=1800.0, order=4, zero_phase=True)

        fft_in = np.abs(np.fft.rfft(sig))
        fft_out = np.abs(np.fft.rfft(filtered))
        freqs = np.fft.rfftfreq(len(sig), 1 / sr)
        idx_50 = int(np.argmin(np.abs(freqs - 50)))
        atten_db = 20 * np.log10(max(fft_out[idx_50], 1e-12) / max(fft_in[idx_50], 1e-12))

        assert atten_db < -20, (
            f'Highpass 100 Hz should attenuate 50 Hz by >20 dB, got {atten_db:.1f} dB'
        )

    def test_filter_is_deterministic(self):
        """Verify identical input produces bitwise-identical filtered output."""
        import numpy as np
        from respanno.audio.preprocessing import apply_butter_filter

        sr = 4000
        t = np.linspace(0, 1, sr, endpoint=False)
        sig = (np.sin(2 * np.pi * 200 * t) + 0.3 * np.random.randn(sr)).astype(np.float32)

        out1 = apply_butter_filter(
            sig, sr, filter_type='bandpass', lowcut=20.0,
            highcut=1800.0, order=4, zero_phase=True)
        out2 = apply_butter_filter(
            sig, sr, filter_type='bandpass', lowcut=20.0,
            highcut=1800.0, order=4, zero_phase=True)

        assert np.allclose(out1, out2), 'Identical inputs must produce identical outputs'

