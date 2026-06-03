"""Pure STFT / spectrogram helpers (no PyQt / pyqtgraph dependency).

Extracted from AudioViewer.update_spectrogram, _decimate_spec_for_display,
_get_palette_256, and _colorize_spec_with_window in legacy/1.6.6.py.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

try:
    import librosa
except ImportError:
    librosa = None  # type: ignore[assignment]


DEFAULT_STFT_CONFIG = {
    "n_fft": 256,
    "hop_length": 64,
    "f_max": 2000,
    "cmap": "Heatmap",
    "max_time_bins": 2500,
    "max_freq_bins": 512,
}


# ---------------------------------------------------------------------------
# 1. compute_stft_db
# ---------------------------------------------------------------------------

def compute_stft_db(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 256,
    hop_length: int = 64,
    f_max: float = 2000.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the STFT dB spectrogram, cropped to frequencies <= f_max.

    Returns
    -------
    S_db : np.ndarray   shape (f_bins, t_frames)
    freqs : np.ndarray  shape (f_bins,)
    """
    if audio is None or sr is None or sr <= 0 or len(audio) == 0:
        raise ValueError("audio is None or empty")

    f_max_eff = min(float(f_max), float(sr) / 2.0)

    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length,
                    center=True, pad_mode="reflect")
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    idx = freqs <= f_max_eff
    if np.any(idx):
        S_db = S_db[idx, :]
        freqs = freqs[idx]

    return S_db, freqs


# ---------------------------------------------------------------------------
# 2. limit_frequency_range
# ---------------------------------------------------------------------------

def limit_frequency_range(
    S_db: np.ndarray,
    freqs: np.ndarray,
    f_max: float,
    sr: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Crop a spectrogram to frequencies <= f_max (clamped at Nyquist)."""
    if sr is not None and sr > 0:
        f_max = min(float(f_max), float(sr) / 2.0)
    idx = freqs <= f_max
    if not np.any(idx):
        return S_db[:1, :], freqs[:1]
    return S_db[idx, :], freqs[idx]


# ---------------------------------------------------------------------------
# 3. decimate_spec_for_display
# ---------------------------------------------------------------------------

def decimate_spec_for_display(
    spec: np.ndarray,
    max_time_bins: int = 2500,
    max_freq_bins: int = 512,
) -> np.ndarray:
    """Downsample a (freq × time) spectrogram for display only.

    The original audio data is **not** modified.
    """
    spec = np.asarray(spec)
    if spec.ndim != 2:
        return spec
    f_bins, t_bins = spec.shape

    if max_freq_bins > 0 and f_bins > max_freq_bins:
        idx_f = np.linspace(0, f_bins - 1, max_freq_bins).astype(int)
        spec = spec[idx_f, :]
    if max_time_bins > 0 and spec.shape[1] > max_time_bins:
        idx_t = np.linspace(0, spec.shape[1] - 1, max_time_bins).astype(int)
        spec = spec[:, idx_t]
    return spec


# ---------------------------------------------------------------------------
# 4. get_palette_256
# ---------------------------------------------------------------------------

def _interp_palette(points: list) -> np.ndarray:
    pos = np.array([p[0] for p in points], float)
    col = np.array([p[1] for p in points], float)
    xs = np.linspace(0, 1, 256)
    out = np.empty((256, 3), float)
    for k in range(3):
        out[:, k] = np.interp(xs, pos, col[:, k])
    return np.clip(out, 0.0, 1.0)


def get_palette_256(cmap: str = "Heatmap") -> np.ndarray:
    """Return a (256, 3) float32 palette for the named colour map.

    Supported: ``"Heatmap"`` (viridis-like), ``"Grayscale"``.
    """
    if cmap == "Heatmap":
        pts = [
            (0.00, (68 / 255, 1 / 255, 84 / 255)),
            (0.25, (59 / 255, 82 / 255, 139 / 255)),
            (0.50, (33 / 255, 145 / 255, 140 / 255)),
            (0.75, (94 / 255, 201 / 255, 98 / 255)),
            (1.00, (253 / 255, 231 / 255, 37 / 255)),
        ]
        return _interp_palette(pts)
    # Grayscale
    g = np.linspace(0, 1, 256)
    return np.stack([g, g, g], axis=1)


# ---------------------------------------------------------------------------
# 5. colorize_spectrogram
# ---------------------------------------------------------------------------

def colorize_spectrogram(
    spec: np.ndarray,
    cmap: str = "Heatmap",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> np.ndarray:
    """Map a 2-D spectrogram (time × freq) to uint8 RGB (H, W, 3).

    Parameters
    ----------
    spec : np.ndarray  shape (time, freq)
    cmap : str  "Heatmap" or "Grayscale"
    vmin, vmax : float or None
        Display window.  ``None`` auto-computes from 1%–99% percentiles.

    Returns
    -------
    rgb : np.ndarray  shape (H, W, 3), dtype uint8
    """
    Z = np.asarray(spec, float)
    finite = np.isfinite(Z)

    if not np.any(finite):
        return np.zeros((Z.shape[0], Z.shape[1], 3), dtype=np.uint8)

    if vmin is None or vmax is None or not (vmax > vmin):
        vmin = float(np.percentile(Z[finite], 1))
        vmax = float(np.percentile(Z[finite], 99))
        if not (vmax > vmin):
            vmax = vmin + 1.0

    Zn = np.clip((Z - vmin) / (vmax - vmin), 0.0, 1.0)
    lut = (get_palette_256(cmap) * 255.0).astype(np.uint8)
    idx = np.clip((Zn * 255.0 + 0.5).astype(np.int16), 0, 255)

    return lut[idx]  # (H, W, 3)


# ---------------------------------------------------------------------------
# 6. compute_spectrogram_display
# ---------------------------------------------------------------------------

def compute_spectrogram_display(
    audio: np.ndarray,
    sr: int,
    config: Optional[dict] = None,
) -> dict:
    """End-to-end: compute STFT dB, optionally decimate, and colourise.

    Returns a dict with keys:
    - S_db (full-resolution 2-D array, freq × time)
    - S_disp (decimated 2-D array, time × freq)
    - rgb (uint8 (H, W, 3))
    - freqs (1-D array)
    - f_max_eff (float)
    """
    cfg = {**DEFAULT_STFT_CONFIG, **(config or {})}
    n_fft = int(cfg.get("n_fft", 256))
    hop_length = int(cfg.get("hop_length", 64))
    f_max = float(cfg.get("f_max", 2000))
    cmap = str(cfg.get("cmap", "Heatmap"))
    max_tb = int(cfg.get("max_time_bins", 2500))
    max_fb = int(cfg.get("max_freq_bins", 512))
    vmin = cfg.get("vmin", None)
    vmax = cfg.get("vmax", None)

    S_db, freqs = compute_stft_db(audio, sr, n_fft=n_fft, hop_length=hop_length, f_max=f_max)
    f_max_eff = float(freqs[-1]) if len(freqs) else float(min(f_max, sr / 2.0))
    duration = len(audio) / float(sr) if sr > 0 else 0.0

    S_disp = decimate_spec_for_display(S_db, max_time_bins=max_tb, max_freq_bins=max_fb)
    rgb = colorize_spectrogram(S_disp.T, cmap=cmap, vmin=vmin, vmax=vmax)

    return {
        "S_db": S_db,
        "S_disp": S_disp,
        "rgb": rgb,
        "freqs": freqs,
        "f_max_eff": f_max_eff,
        "duration": duration,
        "n_fft": n_fft,
        "hop_length": hop_length,
    }


# ---------------------------------------------------------------------------
# frame times helper
# ---------------------------------------------------------------------------

def compute_stft_frame_times(audio_length: int, sr: int, hop_length: int) -> np.ndarray:
    """Return the time-stamps of STFT frame centres (seconds)."""
    # Same shape as librosa.stft(..., center=True, pad_mode='reflect')
    n_fft = 256  # dummy — frame count is hop-dependent with center=True
    D = librosa.stft(np.zeros(audio_length), n_fft=n_fft, hop_length=hop_length,
                    center=True, pad_mode="reflect")
    T = D.shape[1]
    return (np.arange(T) * hop_length) / sr
