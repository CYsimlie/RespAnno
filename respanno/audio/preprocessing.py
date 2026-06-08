"""Pure audio preprocessing helpers (no PyQt dependency).

Extracted from AudioViewer._apply_butter_filter_for_preprocessing,
AudioViewer._get_load_audio_target_sr, and AudioViewer._summarize_preprocessing
in legacy/1.0.0.py.

Behaviour MUST match the legacy implementation byte-for-byte where
numpy/scipy/librosa determinism allows.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, Optional, Tuple, Union

try:
    from scipy.signal import butter, sosfilt, sosfiltfilt
except ImportError:
    butter = None  # type: ignore[assignment]
    sosfilt = None  # type: ignore[assignment]
    sosfiltfilt = None  # type: ignore[assignment]

try:
    import librosa
except ImportError:
    librosa = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Default preprocessing config (matches legacy AudioViewer defaults)
# ---------------------------------------------------------------------------

DEFAULT_PREPROCESSING_CONFIG: Dict[str, Any] = {
    "preprocessing_enabled": True,
    "resample_enabled": True,
    "resample_target_sr": 4000,
    "filter_enabled": False,
    "filter_type": "bandpass",
    "filter_lowcut": 20.0,
    "filter_highcut": 1800.0,
    "filter_order": 4,
    "filter_zero_phase": True,
}


# ---------------------------------------------------------------------------
# 1. load_audio_file
# ---------------------------------------------------------------------------

def load_audio_file(
    path: str,
    target_sr: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """Load a WAV file using librosa, optionally resampling.

    Parameters
    ----------
    path : str
        Absolute path to the audio file.
    target_sr : int or None
        Target sampling rate for resampling.  ``None`` preserves the
        original sample rate.

    Returns
    -------
    (audio, sr) : (np.ndarray, int)
        The loaded float32 audio array and its sampling rate.
    """
    if librosa is None:
        raise ImportError("librosa is required for audio loading")
    audio, sr = librosa.load(path, sr=target_sr)
    if sr is None or sr <= 0 or audio is None or len(audio) == 0:
        raise ValueError("Empty or invalid audio data")
    return audio, sr


def get_original_sr(path: str) -> int:
    """Return the native sample rate of *path* without loading audio data."""
    if librosa is None:
        raise ImportError("librosa is required for audio loading")
    return int(librosa.get_samplerate(path))


# ---------------------------------------------------------------------------
# 2. validate_preprocessing_config
# ---------------------------------------------------------------------------

def validate_preprocessing_config(
    config: Dict[str, Any],
    sr: Optional[int] = None,
) -> Dict[str, Any]:
    """Validate and normalise a preprocessing config dict.

    Silently clips out-of-range values rather than raising, matching the
    legacy code's defensive style.
    """
    cfg: Dict[str, Any] = {}
    cfg["preprocessing_enabled"] = bool(config.get("preprocessing_enabled", True))
    cfg["resample_enabled"] = bool(config.get("resample_enabled", True))
    try:
        cfg["resample_target_sr"] = int(config.get("resample_target_sr", 4000))
    except (ValueError, TypeError):
        cfg["resample_target_sr"] = 4000

    cfg["filter_enabled"] = bool(config.get("filter_enabled", False))
    cfg["filter_type"] = str(config.get("filter_type", "bandpass") or "bandpass").lower()
    if cfg["filter_type"] not in {"bandpass", "lowpass", "highpass", "bandstop"}:
        cfg["filter_type"] = "bandpass"

    try:
        cfg["filter_lowcut"] = float(config.get("filter_lowcut", 20.0) or 20.0)
    except (ValueError, TypeError):
        cfg["filter_lowcut"] = 20.0
    try:
        cfg["filter_highcut"] = float(config.get("filter_highcut", 1800.0) or 1800.0)
    except (ValueError, TypeError):
        cfg["filter_highcut"] = 1800.0

    try:
        cfg["filter_order"] = int(config.get("filter_order", 4) or 4)
    except (ValueError, TypeError):
        cfg["filter_order"] = 4
    cfg["filter_order"] = max(1, min(12, cfg["filter_order"]))

    cfg["filter_zero_phase"] = bool(config.get("filter_zero_phase", True))
    return cfg


def compute_target_sr(config: Dict[str, Any]) -> Optional[int]:
    """Return the effective target sample rate, or None if resampling is off."""
    if not bool(config.get("preprocessing_enabled", True)):
        return None
    if not bool(config.get("resample_enabled", True)):
        return None
    try:
        sr = int(config.get("resample_target_sr", 0))
    except (ValueError, TypeError):
        return None
    return sr if sr > 0 else None


# ---------------------------------------------------------------------------
# 3. apply_butter_filter
# ---------------------------------------------------------------------------

def apply_butter_filter(
    audio: np.ndarray,
    sr: Union[int, float],
    filter_type: str = "bandpass",
    lowcut: float = 20.0,
    highcut: float = 1800.0,
    order: int = 4,
    zero_phase: bool = True,
) -> np.ndarray:
    """Apply a Butterworth filter to *audio*.

    This is an exact replica of
    ``AudioViewer._apply_butter_filter_for_preprocessing``, extracted
    without the ``self.*`` attribute lookups.

    The function is conservative: invalid cutoff values are clipped or
    cause the original audio to be returned unchanged.
    """
    if audio is None or sr is None:
        return audio
    if butter is None:
        return audio

    x = np.asarray(audio, dtype=np.float32)
    fs = float(sr)
    nyq = fs / 2.0
    if nyq <= 0 or x.size < 4:
        return audio

    ftype = str(filter_type).lower()
    if ftype not in {"bandpass", "lowpass", "highpass", "bandstop"}:
        ftype = "bandpass"

    low = float(lowcut)
    high = float(highcut)
    ord_val = max(1, min(12, int(order)))

    # Safety margin below Nyquist to avoid unstable filters (matches legacy).
    hi_max = max(1e-6, 0.98 * nyq)

    if ftype == "lowpass":
        high = min(max(high, 1e-6), hi_max)
        wn = high / nyq
        sos = butter(ord_val, wn, btype="lowpass", output="sos")
    elif ftype == "highpass":
        low = min(max(low, 1e-6), hi_max)
        wn = low / nyq
        sos = butter(ord_val, wn, btype="highpass", output="sos")
    else:
        low = min(max(low, 1e-6), hi_max)
        high = min(max(high, 1e-6), hi_max)
        if high <= low:
            return audio  # skip filtering (matches legacy)
        wn = [low / nyq, high / nyq]
        sos = butter(ord_val, wn, btype=ftype, output="sos")

    if zero_phase:
        try:
            y = sosfiltfilt(sos, x).astype(np.float32, copy=False)
        except Exception:
            y = sosfilt(sos, x).astype(np.float32, copy=False)
    else:
        y = sosfilt(sos, x).astype(np.float32, copy=False)
    return y


# ---------------------------------------------------------------------------
# 4. apply_preprocessing
# ---------------------------------------------------------------------------

def apply_preprocessing(
    audio: np.ndarray,
    sr: int,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply filtering (resampling is handled by load_audio_file).

    Returns (processed_audio, metadata).
    """
    cfg = validate_preprocessing_config(config or {}, sr=sr)

    metadata: Dict[str, Any] = {
        "input_sr": int(sr),
        "output_sr": int(sr),
        "preprocessing_enabled": cfg["preprocessing_enabled"],
        "resample_enabled": False,  # resampling done during load
        "target_sr": cfg["resample_target_sr"],
        "filter_enabled": cfg["filter_enabled"],
        "filter_type": cfg["filter_type"],
        "filter_lowcut": cfg["filter_lowcut"],
        "filter_highcut": cfg["filter_highcut"],
        "filter_order": cfg["filter_order"],
        "filter_zero_phase": cfg["filter_zero_phase"],
        "filter_applied": False,
    }

    if (not cfg["preprocessing_enabled"]) or (not cfg["filter_enabled"]):
        return np.asarray(audio, dtype=np.float32), metadata

    try:
        processed = apply_butter_filter(
            audio,
            sr=sr,
            filter_type=cfg["filter_type"],
            lowcut=cfg["filter_lowcut"],
            highcut=cfg["filter_highcut"],
            order=cfg["filter_order"],
            zero_phase=cfg["filter_zero_phase"],
        )
    except Exception:
        return np.asarray(audio, dtype=np.float32), metadata

    metadata["filter_applied"] = True
    return np.asarray(processed, dtype=np.float32), metadata


# ---------------------------------------------------------------------------
# 5. preprocess_audio_file (end-to-end convenience)
# ---------------------------------------------------------------------------

def preprocess_audio_file(
    path: str,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """Load and preprocess an audio file in one call.

    Returns (audio, sr, metadata).
    """
    cfg = validate_preprocessing_config(config or {})
    original_sr = get_original_sr(path)

    target_sr = compute_target_sr(cfg)
    audio, sr = load_audio_file(path, target_sr=target_sr)

    processed, metadata = apply_preprocessing(audio, sr, config=cfg)
    metadata["original_sr"] = int(original_sr)
    metadata["processed_sr"] = int(sr)
    metadata["resample_enabled"] = bool(cfg["resample_enabled"])

    return processed, sr, metadata


# ---------------------------------------------------------------------------
# summary helper
# ---------------------------------------------------------------------------

def summarize_preprocessing(config: Dict[str, Any]) -> str:
    """Human-readable summary string (matching legacy _summarize_preprocessing)."""
    cfg = validate_preprocessing_config(config)
    if not cfg["preprocessing_enabled"]:
        return "preprocessing off"
    parts: list = []
    if cfg["resample_enabled"]:
        parts.append(f"resample={cfg['resample_target_sr']} Hz")
    if cfg["filter_enabled"]:
        ftype = cfg["filter_type"]
        if ftype in {"bandpass", "bandstop"}:
            parts.append(f"{ftype}={cfg['filter_lowcut']:.1f}-{cfg['filter_highcut']:.1f} Hz")
        elif ftype == "lowpass":
            parts.append(f"lowpass<={cfg['filter_highcut']:.1f} Hz")
        elif ftype == "highpass":
            parts.append(f"highpass>={cfg['filter_lowcut']:.1f} Hz")
    return "; ".join(parts) if parts else "preprocessing on"
