"""Tests for respanno.dsp.fft — FFT magnitude computation."""

import numpy as np
import pytest

from respanno.dsp.fft import compute_fft
from tests.fixtures.synthetic_signals import (
    generate_tone,
    generate_silence,
    generate_short_signal,
    generate_dc_offset,
)


# ---------------------------------------------------------------------------
# Normal operation
# ---------------------------------------------------------------------------


class TestNormalFFT:
    def test_tone_peak_at_correct_frequency(self):
        """200 Hz tone → magnitude peak near 200 Hz bin."""
        audio, sr, _ = generate_tone(freq=200.0, sr=4000, duration=2.0)
        freqs, mag = compute_fft(audio, sr)
        peak_idx = np.argmax(mag)
        assert freqs[peak_idx] == pytest.approx(200.0, rel=0.05)

    def test_output_shapes_match(self):
        """freqs and mag have same length."""
        audio, sr, _ = generate_tone()
        freqs, mag = compute_fft(audio, sr)
        assert freqs.shape == mag.shape
        assert len(freqs) > 0

    def test_tone_magnitude_positive(self):
        """All magnitude values are non-negative."""
        audio, sr, _ = generate_tone()
        _, mag = compute_fft(audio, sr)
        assert np.all(mag >= 0)


# ---------------------------------------------------------------------------
# max_points decimation
# ---------------------------------------------------------------------------


class TestMaxPoints:
    def test_reduces_to_max_points(self):
        """Output length ≤ max_points when specified."""
        audio, sr, _ = generate_tone(freq=200.0, sr=4000, duration=2.0)
        _, mag = compute_fft(audio, sr, max_points=256)
        assert len(mag) <= 256

    def test_no_reduction_when_small_enough(self):
        """No decimation when spectrum already smaller than max_points."""
        audio, sr, _ = generate_tone(freq=200.0, sr=4000, duration=0.1)
        _, mag_full = compute_fft(audio, sr)
        _, mag_dec = compute_fft(audio, sr, max_points=100000)
        assert len(mag_dec) == len(mag_full)


# ---------------------------------------------------------------------------
# Boundary / error cases
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_empty_audio_returns_empty(self):
        """Empty array → empty freq/mag arrays."""
        freqs, mag = compute_fft(np.array([], dtype=np.float32), 4000)
        assert len(freqs) == 0
        assert len(mag) == 0

    def test_zero_sr_returns_empty(self):
        """sr=0 → safe empty return."""
        audio, _, _ = generate_tone(duration=0.5)
        freqs, mag = compute_fft(audio, 0)
        assert len(freqs) == 0

    def test_none_sr_returns_empty(self):
        """sr=None → safe empty return."""
        audio, _, _ = generate_tone(duration=0.5)
        freqs, mag = compute_fft(audio, None)
        assert len(freqs) == 0

    def test_silence_flat_spectrum(self):
        """Silence → near-zero magnitude across all bins."""
        audio, sr, _ = generate_silence(duration=1.0)
        _, mag = compute_fft(audio, sr)
        assert np.max(mag) < 1e-6

    def test_dc_offset_energy_at_bin_zero(self):
        """DC offset → energy concentrated at DC bin."""
        audio, sr, _ = generate_dc_offset(duration=1.0, offset=2.0)
        freqs, mag = compute_fft(audio, sr)
        assert freqs[0] == pytest.approx(0.0)
        assert mag[0] > np.mean(mag[1:]) * 10

    def test_short_signal_no_crash(self):
        """Very short signal (< 1 frame) does not crash."""
        audio, sr, _ = generate_short_signal(duration=0.005)
        freqs, mag = compute_fft(audio, sr)
        assert isinstance(freqs, np.ndarray)
        assert isinstance(mag, np.ndarray)

    def test_single_sample_no_crash(self):
        """Single sample does not crash."""
        audio = np.array([0.1], dtype=np.float32)
        freqs, mag = compute_fft(audio, 4000)
        assert isinstance(freqs, np.ndarray)
