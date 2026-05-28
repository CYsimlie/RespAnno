"""Mock AudioViewer for testing classifier.py and phase_model.py.

Provides all attributes and methods expected by the ML training/inference
functions, backed by synthetic data so tests run without a real WAV file
or QApplication.
"""

from __future__ import annotations

import numpy as np

from respanno.ml.frame_labels import (
    build_frame_labels,
    get_manual_segments,
    get_reviewed_prefix,
)


class MockViewer:
    """Minimal stand-in for the AudioViewer GUI class.

    Parameters
    ----------
    stft_features : np.ndarray, shape (n_frames, n_features)
        Pre-computed feature matrix.
    stft_frame_times : np.ndarray, shape (n_frames,)
        Time (seconds) of each frame centre.
    stft_feature_names : list[str]
        Feature name per column.
    annotations : list[tuple]
        Manual annotations as (start, end, label, [source]).
    sr : int
        Sample rate (Hz).
    hop_length : int
        STFT hop length in samples.
    """

    def __init__(
        self,
        stft_features=None,
        stft_frame_times=None,
        stft_feature_names=None,
        annotations=None,
        sr=4000,
        hop_length=256,
    ):
        self.stft_features = (
            np.asarray(stft_features, dtype=np.float64)
            if stft_features is not None
            else None
        )
        self.stft_frame_times = (
            np.asarray(stft_frame_times, dtype=float)
            if stft_frame_times is not None
            else None
        )
        self.stft_feature_names = list(stft_feature_names or [])
        self.annotations = list(annotations or [])
        self.sr = sr
        self.hop_length = hop_length
        self.ml_models = {}
        self.imported = []  # (start, end, text, source)

    # ── methods called by classifier / phase_model ───────────────────────

    def ensure_frame_features(self):
        """No-op: features are pre-computed on construction."""

    def build_frame_labels_for_tag(self, label, neg_margin=0.05):
        return build_frame_labels(
            self.annotations,
            self.stft_frame_times,
            label,
            neg_margin=neg_margin,
        )

    def get_manual_segments_for_label(self, label):
        return get_manual_segments(self.annotations, label)

    def get_reviewed_prefix(self):
        return get_reviewed_prefix(self.annotations)

    def finalize_annotation(self, start, end, text, source="manual"):
        self.imported.append((float(start), float(end), str(text), str(source)))

    # ── status bar (for auto_import / ML messages) ───────────────────────

    def statusBar(self):
        return self

    def showMessage(self, msg, timeout=0):
        pass


# ---------------------------------------------------------------------------
# Helper: build a realistic synthetic feature matrix
# ---------------------------------------------------------------------------


def make_synthetic_features(n_frames=120, n_features=56, seed=42):
    """Return (features, times, names) with controlled class structure.

    The first 60 frames are "reviewed prefix" territory.  Features 0–3 carry
    strong group separation so a classifier can learn a decision boundary;
    the remaining features are low-variance noise.
    """
    rng = np.random.default_rng(seed)
    X = 0.02 * rng.standard_normal((n_frames, n_features))

    # inject discriminative pattern in first 4 features:
    #   frames 10–30 and 50–55 → "positive class" signature
    pos_mask = np.zeros(n_frames, dtype=bool)
    pos_mask[10:30] = True
    pos_mask[50:55] = True
    X[pos_mask, 0] += 0.8
    X[pos_mask, 1] -= 0.6
    X[pos_mask, 2] += 0.4
    X[pos_mask, 3] -= 0.3

    # frames 0–9, 31–49, 56–59 → "negative class" (different pattern)
    neg_mask = np.zeros(n_frames, dtype=bool)
    neg_mask[0:10] = True
    neg_mask[31:50] = True
    neg_mask[56:60] = True
    X[neg_mask, 0] -= 0.5
    X[neg_mask, 1] += 0.4

    times = np.arange(0, n_frames) * 0.064  # 64 ms hop
    names = [f"feat_{i}" for i in range(n_features)]

    return X.astype(np.float64), times.astype(float), names
