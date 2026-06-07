"""Performance baseline tests.

Measures throughput and latency of key compute stages to establish
baseline expectations. These are REPORT-ONLY (never fail); they log
timings and flag regressions as warnings rather than hard failures.

Set environment variable PERF_STRICT=1 to make thresholds assert.
"""
import os
import time
import warnings
import numpy as np
import pytest
from respanno.audio.preprocessing import apply_butter_filter
from respanno.dsp.spectrogram import compute_stft_db, compute_spectrogram_display
from respanno.dsp.features import compute_short_time_features
from respanno.dsp.fft import compute_fft
from tests.fixtures.synthetic_signals import generate_tone

# Reference thresholds (loose — tuned for CI runners)
MAX_PREPROCESS_SEC = 2.0
MAX_STFT_SEC = 1.0
MAX_FEATURES_SEC = 3.0
MAX_FFT_SEC = 0.5

_STRICT = os.environ.get("PERF_STRICT", "") == "1"


def _check(condition, message):
    """Assert if PERF_STRICT=1, otherwise warn and always pass."""
    if condition:
        return
    if _STRICT:
        pytest.fail(message)
    else:
        warnings.warn(UserWarning(message))


@pytest.fixture(scope='module')
def tone_2s():
    (audio, sr, _) = generate_tone(freq=200.0, sr=4000, duration=2.0, seed=42)
    return (audio.astype(np.float32), int(sr))


@pytest.fixture(scope='module')
def tone_10s():
    (audio, sr, _) = generate_tone(freq=200.0, sr=4000, duration=10.0, seed=42)
    return (audio.astype(np.float32), int(sr))


class TestPreprocessingThroughput:

    def test_butter_filter_latency(self, tone_2s):
        """Report butter filter latency (reference < 2.0 s)."""
        (audio, sr) = tone_2s
        t0 = time.perf_counter()
        result = apply_butter_filter(audio, sr, 'bandpass', 50, 500, 4)
        elapsed = time.perf_counter() - t0
        assert len(result) == len(audio)
        _check(elapsed < MAX_PREPROCESS_SEC, f'filter took {elapsed:.3f}s (ref < {MAX_PREPROCESS_SEC}s)')


class TestSTFTPerformance:

    def test_stft_latency_2s(self, tone_2s):
        """Report STFT latency for 2s signal (reference < 1.0 s)."""
        (audio, sr) = tone_2s
        t0 = time.perf_counter()
        (spec_db, freqs) = compute_stft_db(audio, sr, n_fft=512, hop_length=128)
        elapsed = time.perf_counter() - t0
        assert spec_db.shape[1] > 0
        _check(elapsed < MAX_STFT_SEC, f'2s STFT took {elapsed:.3f}s (ref < {MAX_STFT_SEC}s)')

    def test_stft_latency_10s(self, tone_10s):
        """Report STFT latency for 10s signal (reference < 5.0 s)."""
        (audio, sr) = tone_10s
        t0 = time.perf_counter()
        (spec_db, freqs) = compute_stft_db(audio, sr, n_fft=512, hop_length=128)
        elapsed = time.perf_counter() - t0
        assert spec_db.shape[1] > 0
        _check(elapsed < MAX_STFT_SEC * 5, f'10s STFT took {elapsed:.3f}s (ref < {MAX_STFT_SEC * 5}s)')

    def test_spectrogram_display_pipeline(self, tone_2s):
        """Report full spectrogram display pipeline latency (reference < 2.0 s)."""
        (audio, sr) = tone_2s
        t0 = time.perf_counter()
        result = compute_spectrogram_display(audio, sr)
        elapsed = time.perf_counter() - t0
        assert result['rgb'].shape[2] == 3
        _check(elapsed < MAX_STFT_SEC * 2, f'display pipeline took {elapsed:.3f}s (ref < {MAX_STFT_SEC * 2}s)')


class TestFeaturePerformance:

    def test_56_features_latency_2s(self, tone_2s):
        """Report 56-feature extraction latency for 2s signal (reference < 3.0 s)."""
        (audio, sr) = tone_2s
        t0 = time.perf_counter()
        (times, feat_dict) = compute_short_time_features(audio, sr, hop_length=256)
        elapsed = time.perf_counter() - t0
        assert len(times) > 0
        _check(elapsed < MAX_FEATURES_SEC, f'2s features took {elapsed:.3f}s (ref < {MAX_FEATURES_SEC}s)')

    def test_56_features_latency_10s(self, tone_10s):
        """Report 56-feature extraction latency for 10s signal (reference < 15.0 s)."""
        (audio, sr) = tone_10s
        t0 = time.perf_counter()
        (times, feat_dict) = compute_short_time_features(audio, sr, hop_length=256)
        elapsed = time.perf_counter() - t0
        assert len(times) > 0
        _check(elapsed < MAX_FEATURES_SEC * 5, f'10s features took {elapsed:.3f}s (ref < {MAX_FEATURES_SEC * 5}s)')


class TestFFTPerformance:

    def test_fft_latency(self, tone_2s):
        """Report FFT latency (reference < 0.5 s)."""
        (audio, sr) = tone_2s
        t0 = time.perf_counter()
        (freqs, mag) = compute_fft(audio, sr)
        elapsed = time.perf_counter() - t0
        assert len(freqs) > 0
        _check(elapsed < MAX_FFT_SEC, f'FFT took {elapsed:.3f}s (ref < {MAX_FFT_SEC}s)')


class TestMemoryFootprint:

    def test_feature_matrix_memory(self, tone_2s):
        """Report feature matrix memory footprint (reference < 1 MB)."""
        (audio, sr) = tone_2s
        (times, feat_dict) = compute_short_time_features(audio, sr, hop_length=256)
        from respanno.dsp.features import build_feature_matrix
        (X_full, _, _) = build_feature_matrix(times, feat_dict)
        mem_mb = X_full.nbytes / (1024 * 1024)
        _check(mem_mb < 1.0, f'feature matrix {mem_mb:.2f} MB (ref < 1 MB)')

    def test_stft_memory(self, tone_10s):
        """Report STFT memory footprint (reference < 5 MB)."""
        (audio, sr) = tone_10s
        (spec_db, freqs) = compute_stft_db(audio, sr, n_fft=512, hop_length=128)
        mem_mb = spec_db.nbytes / (1024 * 1024)
        _check(mem_mb < 5.0, f'STFT {mem_mb:.2f} MB (ref < 5 MB)')
