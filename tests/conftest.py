"""Shared fixtures and utilities for respanno tests."""

import os
import sys
import tempfile
import numpy as np
import pytest
import scipy.io.wavfile as wavfile


@pytest.fixture(scope="session")
def qapp():
    """Session-scoped QApplication (offscreen — safe for headless CI)."""
    import os as _os
    _os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication(["pytest"])
    return app


@pytest.fixture(scope="session")
def legacy_root():
    """Absolute path to the legacy directory."""
    root = os.path.join(os.path.dirname(__file__), "..", "legacy")
    return os.path.abspath(root)


@pytest.fixture(scope="session")
def legacy_path(legacy_root):
    """Absolute path to the legacy main program."""
    return os.path.join(legacy_root, "1.0.0.py")


@pytest.fixture(scope="function")
def tmp_wav_path():
    """Create a temporary WAV file with a simple synthetic signal.

    Returns the path string; the file is cleaned up after the test.

    Signal: 5 s, 4000 Hz sample rate, mix of 100 Hz + 400 Hz sine tones
    with a short 800 Hz burst in the middle.
    """
    sr = 4000
    duration = 5.0
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)

    # Mix two tones
    sig = 0.5 * np.sin(2.0 * np.pi * 100.0 * t) + 0.3 * np.sin(2.0 * np.pi * 400.0 * t)

    # Short 800 Hz burst at 2.0–2.3 s
    burst_mask = (t >= 2.0) & (t <= 2.3)
    sig[burst_mask] += 0.4 * np.sin(2.0 * np.pi * 800.0 * t[burst_mask])

    # Normalize to [-1, 1] for int16
    sig /= np.max(np.abs(sig)) + 1e-9
    sig_i16 = (sig * 32767).astype(np.int16)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wavfile.write(f.name, sr, sig_i16)
        path = f.name

    yield path

    try:
        os.unlink(path)
    except Exception:
        pass


@pytest.fixture(scope="function")
def synthetic_audio():
    """Return (audio, sr) as numpy arrays (no file I/O).

    Same signal as tmp_wav_path.
    """
    sr = 4000
    duration = 5.0
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
    sig = 0.5 * np.sin(2.0 * np.pi * 100.0 * t) + 0.3 * np.sin(2.0 * np.pi * 400.0 * t)
    burst_mask = (t >= 2.0) & (t <= 2.3)
    sig[burst_mask] += 0.4 * np.sin(2.0 * np.pi * 800.0 * t[burst_mask])
    sig = sig.astype(np.float32)
    sig /= np.max(np.abs(sig)) + 1e-9
    return sig, sr


@pytest.fixture(scope="function")
def sample_annotations():
    """Return a list of annotation tuples (start, end, label, source)."""
    return [
        (0.5, 1.2, "wheeze", "manual"),
        (2.0, 2.8, "Crackles", "manual"),
        (3.5, 4.0, "Expiration", "manual"),
    ]


@pytest.fixture(scope="function")
def sample_annotations_csv(tmp_path, sample_annotations):
    """Write sample annotations to a CSV file and return its path."""
    import csv
    p = tmp_path / "test_events.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["start", "end", "label", "source"])
        for s, e, lab, src in sample_annotations:
            w.writerow([f"{s:.4f}", f"{e:.4f}", lab, src])
    return str(p)
