"""Performance baseline tests.

Measures throughput and latency of key compute stages to establish
baseline expectations. These are NOT hard pass/fail tests; they set
reference values that flag unexpected regressions.
"""

import time

import numpy as np
import pytest

from respanno.audio.preprocessing import apply_butter_filter
from respanno.dsp.spectrogram import compute_stft_db, compute_spectrogram_display
from respanno.dsp.features import compute_short_time_features
from respanno.dsp.fft import compute_fft

from tests.fixtures.synthetic_signals import generate_tone


# ---------------------------------------------------------------------------
# Tolerance: these are generous upper bounds for a modern CPU.
# The actual values on a typical dev machine should be well below these.
# ---------------------------------------------------------------------------

MAX_PREPROCESS_SEC = 2.0       # 2s audio processing
MAX_STFT_SEC = 1.0             # STFT computation
MAX_FEATURES_SEC = 3.0         # 56-feature extraction
MAX_FFT_SEC = 0.5              # FFT computation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tone_2s():
    audio, sr, _ = generate_tone(freq=200.0, sr=4000, duration=2.0, seed=42)
    return audio.astype(np.float32), int(sr)


@pytest.fixture(scope="module")
def tone_10s():
    audio, sr, _ = generate_tone(freq=200.0, sr=4000, duration=10.0, seed=42)
    return audio.astype(np.float32), int(sr)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPreprocessingThroughput:
    def test_butter_filter_latency(self, tone_2s):
        audio, sr = tone_2s
        t0 = time.perf_counter()
        result = apply_butter_filter(audio, sr, "bandpass", 50, 500, 4)
        elapsed = time.perf_counter() - t0
        assert len(result) == len(audio)
        assert elapsed < MAX_PREPROCESS_SEC, f"filter took {elapsed:.3f}s"


class TestSTFTPerformance:
    def test_stft_latency_2s(self, tone_2s):
        audio, sr = tone_2s
        t0 = time.perf_counter()
        spec_db, freqs = compute_stft_db(audio, sr, n_fft=512, hop_length=128)
        elapsed = time.perf_counter() - t0
        assert spec_db.shape[1] > 0
        assert elapsed < MAX_STFT_SEC, f"STFT took {elapsed:.3f}s"

    def test_stft_latency_10s(self, tone_10s):
        audio, sr = tone_10s
        t0 = time.perf_counter()
        spec_db, freqs = compute_stft_db(audio, sr, n_fft=512, hop_length=128)
        elapsed = time.perf_counter() - t0
        assert spec_db.shape[1] > 0
        assert elapsed < MAX_STFT_SEC * 5, f"10s STFT took {elapsed:.3f}s"

    def test_spectrogram_display_pipeline(self, tone_2s):
        audio, sr = tone_2s
        t0 = time.perf_counter()
        result = compute_spectrogram_display(audio, sr)
        elapsed = time.perf_counter() - t0
        assert result["rgb"].shape[2] == 3  # RGB
        assert elapsed < MAX_STFT_SEC * 2, f"display pipeline took {elapsed:.3f}s"


class TestFeaturePerformance:
    def test_56_features_latency_2s(self, tone_2s):
        audio, sr = tone_2s
        t0 = time.perf_counter()
        times, feat_dict = compute_short_time_features(audio, sr, hop_length=256)
        elapsed = time.perf_counter() - t0
        assert len(times) > 0
        assert elapsed < MAX_FEATURES_SEC, f"features took {elapsed:.3f}s"

    def test_56_features_latency_10s(self, tone_10s):
        audio, sr = tone_10s
        t0 = time.perf_counter()
        times, feat_dict = compute_short_time_features(audio, sr, hop_length=256)
        elapsed = time.perf_counter() - t0
        assert len(times) > 0
        assert elapsed < MAX_FEATURES_SEC * 5, f"10s features took {elapsed:.3f}s"


class TestFFTPerformance:
    def test_fft_latency(self, tone_2s):
        audio, sr = tone_2s
        t0 = time.perf_counter()
        freqs, mag = compute_fft(audio, sr)
        elapsed = time.perf_counter() - t0
        assert len(freqs) > 0
        assert elapsed < MAX_FFT_SEC, f"FFT took {elapsed:.3f}s"


class TestMemoryFootprint:
    def test_feature_matrix_memory(self, tone_2s):
        """Feature matrix should fit comfortably in memory."""
        audio, sr = tone_2s
        times, feat_dict = compute_short_time_features(audio, sr, hop_length=256)
        # 2s @ 4000 Hz, hop 256 → ~32 frames × 112 features × 8 bytes ≈ 29 KB
        from respanno.dsp.features import build_feature_matrix
        X, _, _ = build_feature_matrix(times, feat_dict)
        mem_mb = X.nbytes / (1024 * 1024)
        assert mem_mb < 1.0, f"feature matrix {mem_mb:.2f} MB exceeds 1 MB"

    def test_stft_memory(self, tone_10s):
        """10s STFT memory footprint should be modest."""
        audio, sr = tone_10s
        spec_db, freqs = compute_stft_db(audio, sr, n_fft=512, hop_length=128)
        # 10s @ 4000 Hz, hop 128 → ~313 frames × 257 bins × 4 bytes ≈ 322 KB
        mem_mb = spec_db.nbytes / (1024 * 1024)
        assert mem_mb < 5.0, f"STFT {mem_mb:.2f} MB exceeds 5 MB"
