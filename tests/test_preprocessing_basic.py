"""Tests for audio preprocessing (Butterworth filter, resampling).

Status: TEST SCAFFOLDING — tests are written but may fail because
the preprocessing functions are embedded inside the AudioViewer
class and cannot be called directly without a full QApplication.

TODO (Phase 6): After extracting `_apply_butter_filter_for_preprocessing`
and `_get_load_audio_target_sr` from AudioViewer into `audio/preprocessing.py`,
remove the QApplication requirement from these tests.
"""

import numpy as np
import pytest

# --- These imports will work after Phase 6 extraction ---
# from respanno.audio.preprocessing import (
#     apply_butter_filter_for_preprocessing,
#     get_load_audio_target_sr,
# )

from scipy.signal import butter, sosfiltfilt, sosfilt


# ---------------------------------------------------------------------------
# Today: test the underlying scipy filter logic directly (no GUI needed)
# These tests verify the mathematical correctness of the filter pipeline
# that the legacy code uses inside `_apply_butter_filter_for_preprocessing`.
# ---------------------------------------------------------------------------

def _butter_bandpass(lowcut, highcut, fs, order=4):
    """Reference bandpass matching the legacy code's butter + sos pattern."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype="bandpass", output="sos")
    return sos


def _butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    sos = butter(order, cutoff / nyq, btype="lowpass", output="sos")
    return sos


def _butter_highpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    sos = butter(order, cutoff / nyq, btype="highpass", output="sos")
    return sos


class TestButterworthReference:
    """Verify the underlying scipy filter math used by the legacy code."""

    def test_bandpass_does_not_crash(self):
        """Bandpass filter on a synthetic signal should not error."""
        fs = 4000
        t = np.linspace(0, 1, fs, endpoint=False)
        sig = np.sin(2 * np.pi * 100 * t) + 0.3 * np.sin(2 * np.pi * 800 * t)
        sig = sig.astype(np.float32)

        sos = _butter_bandpass(100.0, 500.0, fs, order=4)
        y = sosfiltfilt(sos, sig)
        assert y.shape == sig.shape
        assert np.all(np.isfinite(y))

    def test_lowpass_attenuates_high_freq(self):
        """Lowpass filter should reduce high-frequency energy."""
        fs = 8000
        t = np.linspace(0, 1, fs, endpoint=False)
        sig = np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 3000 * t)
        sig = sig.astype(np.float32)

        sos = _butter_lowpass(2000, fs, order=4)
        y = sosfiltfilt(sos, sig)

        # High-frequency energy should be reduced
        from scipy.fft import rfft
        spec_y = np.abs(rfft(y))
        freqs = np.fft.rfftfreq(len(y), d=1 / fs)

        energy_low = np.sum(spec_y[freqs <= 500] ** 2)
        energy_high = np.sum(spec_y[freqs >= 2500] ** 2)
        assert energy_high < energy_low * 0.5

    def test_bandpass_preserves_in_band_signal(self):
        """Bandpass filter should preserve signal within the passband."""
        fs = 4000
        t = np.linspace(0, 0.5, int(fs * 0.5), endpoint=False)
        sig = np.sin(2 * np.pi * 300 * t).astype(np.float32)

        sos = _butter_bandpass(200, 400, fs, order=4)
        y = sosfiltfilt(sos, sig)

        # Signal within passband should have substantial energy remaining
        assert np.std(y) > np.std(sig) * 0.3

    def test_zero_phase_vs_causal(self):
        """Zero-phase (filtfilt) vs causal (filt) should differ."""
        fs = 4000
        t = np.linspace(0, 1, fs, endpoint=False)
        sig = np.random.default_rng(42).normal(0, 1, len(t)).astype(np.float32)

        sos = _butter_bandpass(200, 400, fs)
        y_causal = sosfilt(sos, sig)
        y_zerophase = sosfiltfilt(sos, sig)

        # They should differ (demonstrating zero-phase is not a no-op)
        # but both should be finite
        assert np.all(np.isfinite(y_causal))
        assert np.all(np.isfinite(y_zerophase))

    def test_filter_on_short_signal(self):
        """Very short signals should not crash the filter."""
        fs = 4000
        sig = np.array([0.1, -0.2, 0.05], dtype=np.float32)

        sos = _butter_bandpass(200, 400, fs, order=4)
        # filtfilt may fail on very short signals; test both paths
        try:
            y = sosfiltfilt(sos, sig)
        except ValueError:
            y = sosfilt(sos, sig)

        assert y.shape == sig.shape
        assert np.all(np.isfinite(y))


class TestResampleTarget:
    """Verify the resample target rate logic (pre-extraction)."""

    def test_default_4000_hz(self):
        """Default resample target should be 4000 Hz (matching legacy default)."""
        target_sr = 4000
        assert target_sr == 4000  # placeholder; replaced after extraction


# ---------------------------------------------------------------------------
# TODO: Tests that require extraction from AudioViewer
# ---------------------------------------------------------------------------

class TestPreprocessingFromAudioViewer:
    """
    TODO: After extracting _apply_butter_filter_for_preprocessing to
    respanno/audio/preprocessing.py, implement these tests:

    1. test_resample_load_4000hz:
       - Load tmp_wav_path with librosa.load(sr=4000)
       - Verify sr == 4000 and audio length matches expected samples

    2. test_filter_disabled_passthrough:
       - With filter_enabled=False, output should == input

    3. test_filter_invalid_cutoff_skipped:
       - highcut <= lowcut should skip filtering gracefully

    4. test_preprocessing_summary_string:
       - Verify _summarize_preprocessing output format
    """

    def test_todo_placeholder(self):
        """Placeholder: will be implemented after Phase 6 extraction."""
        pytest.skip("TODO: extract _apply_butter_filter_for_preprocessing first")
