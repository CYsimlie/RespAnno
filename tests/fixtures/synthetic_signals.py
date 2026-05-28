"""Synthetic respiratory sound signals with known ground-truth annotations.

Design principles
------------------
1. **Physiological plausibility** – each component maps to a real breath-sound
   source (airway turbulence, wall oscillation, airway opening, etc.).
2. **Known ground truth** – every generator returns ``(audio, sr, annotations)``
   where *annotations* are the exact intervals used during generation.
3. **Graded complexity** – Level 0 (pure tone) through Level 3 (full cycle).
4. **Tunable parameters** – all physiological knobs exposed as keyword arguments.
5. **Deterministic reproducibility** – fixed ``seed`` → identical output, every
   random draw via ``np.random.default_rng(seed)``.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Level 0 — pure tone (FFT / spectrogram sanity checks)
# ---------------------------------------------------------------------------


def generate_tone(freq=200.0, sr=4000, duration=2.0, amplitude=0.5, seed=42):
    """Single-frequency sine wave. Minimal test fixture."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, int(sr * duration)) / sr
    noise = 0.001 * rng.standard_normal(len(t))
    audio = (amplitude * np.sin(2 * np.pi * freq * t) + noise).astype(np.float32)
    return audio, sr, []


# ---------------------------------------------------------------------------
# Level 1 — carrier envelope + one adventitious sound
# ---------------------------------------------------------------------------


def generate_wheeze_episode(
    sr=4000, duration=3.0, wheeze_start=1.0, wheeze_dur=0.5,
    wheeze_freq=400.0, snr_db=20.0, seed=42,
):
    """Normal breath with a single wheeze segment.

    Returns (audio, sr, annotations) where annotations = [(wheeze_start, end, "Wheeze", "manual")].
    """
    rng = np.random.default_rng(seed)
    n = int(sr * duration)
    t = np.arange(0, n) / sr

    # --- normal breath: filtered pink-ish noise with respiratory envelope ---
    # pink noise approximated via cumulative sum of white noise
    white = rng.standard_normal(n)
    pink = np.cumsum(white)
    pink -= np.mean(pink)
    pink /= np.std(pink) + 1e-12

    # gentle Hanning-shaped breath envelope
    env = np.sin(np.pi * t / duration) ** 0.6
    breath = 0.15 * env * pink

    # --- wheeze: narrow-band tone with onset/offset ramps ---
    w_start = int(sr * wheeze_start)
    w_end = int(sr * (wheeze_start + wheeze_dur))
    w_n = w_end - w_start
    w_t = t[w_start:w_end] - t[w_start]

    # harmonic series for more realistic timbre
    wheeze_tone = (
        0.6 * np.sin(2 * np.pi * wheeze_freq * w_t)
        + 0.25 * np.sin(2 * np.pi * wheeze_freq * 2 * w_t)
        + 0.10 * np.sin(2 * np.pi * wheeze_freq * 3 * w_t)
        + 0.05 * np.sin(2 * np.pi * wheeze_freq * 4 * w_t)
    )
    # onset/offset ramp (10 ms)
    ramp_n = int(sr * 0.01)
    ramp = np.ones(w_n)
    if w_n > 2 * ramp_n:
        ramp[:ramp_n] = np.linspace(0, 1, ramp_n)
        ramp[-ramp_n:] = np.linspace(1, 0, ramp_n)
    wheeze_tone *= ramp

    # place into full signal
    wheeze = np.zeros(n)
    wheeze[w_start:w_end] = wheeze_tone

    # --- SNR scaling ---
    sig_rms = np.sqrt(np.mean(breath ** 2))
    noise_rms = sig_rms / (10 ** (snr_db / 20))
    noise = noise_rms * rng.standard_normal(n)

    audio = (breath + wheeze + noise).astype(np.float32)

    annotations = [
        (float(wheeze_start), float(wheeze_start + wheeze_dur), "Wheeze", "manual"),
    ]
    return audio, sr, annotations


def generate_crackle_episode(
    sr=4000, duration=3.0, crackle_count=5, crackle_width=0.015,
    snr_db=20.0, seed=42,
):
    """Normal breath with scattered fine crackles.

    Crackles are modelled as exponentially-decaying sine bursts (simulating
    the explosive opening of small airways).
    """
    rng = np.random.default_rng(seed)
    n = int(sr * duration)
    t = np.arange(0, n) / sr

    # normal breath (same as wheeze episode)
    white = rng.standard_normal(n)
    pink = np.cumsum(white)
    pink -= np.mean(pink)
    pink /= np.std(pink) + 1e-12
    env = np.sin(np.pi * t / duration) ** 0.6
    breath = 0.15 * env * pink

    # crackles: randomly placed in [0.3*duration, 0.9*duration]
    crackle_audio = np.zeros(n)
    annotations = []
    margin = int(sr * 0.3)
    end_margin = int(sr * 0.9)
    positions = rng.integers(margin, end_margin, size=crackle_count)

    cw = max(1, int(sr * crackle_width))
    c_t = np.arange(0, cw) / sr
    decay = np.exp(-c_t / (crackle_width / 3))

    for pos in positions:
        freq = rng.uniform(600, 1800)  # fine crackle frequency range
        burst = np.sin(2 * np.pi * freq * c_t) * decay
        burst *= rng.uniform(0.06, 0.12)  # amplitude variation
        end_idx = min(pos + cw, n)
        burst_len = end_idx - pos
        crackle_audio[pos:end_idx] += burst[:burst_len]
        annotations.append(
            (float(pos / sr), float(end_idx / sr), "Crackles", "manual")
        )

    # SNR
    sig_rms = np.sqrt(np.mean(breath ** 2))
    noise_rms = sig_rms / (10 ** (snr_db / 20))
    noise = noise_rms * rng.standard_normal(n)

    audio = (breath + crackle_audio + noise).astype(np.float32)
    return audio, sr, annotations


# ---------------------------------------------------------------------------
# Level 2 — multi-component episode (normal breath + two adventitious sounds)
# ---------------------------------------------------------------------------


def generate_mixed_episode(
    sr=4000, duration=6.0, snr_db=20.0, seed=42,
):
    """Normal breath with wheeze and crackles at known intervals."""
    rng = np.random.default_rng(seed)
    n = int(sr * duration)
    t = np.arange(0, n) / sr

    # normal breath
    white = rng.standard_normal(n)
    pink = np.cumsum(white)
    pink -= np.mean(pink)
    pink /= np.std(pink) + 1e-12
    env = np.sin(np.pi * t / duration) ** 0.6
    breath = 0.12 * env * pink

    # --- wheeze segment (2.0–2.6 s) ---
    w_start = int(sr * 2.0)
    w_end = int(sr * 2.6)
    w_n = w_end - w_start
    w_t = t[w_start:w_end] - t[w_start]
    w_freq = 350.0
    wheeze_tone = (
        0.6 * np.sin(2 * np.pi * w_freq * w_t)
        + 0.3 * np.sin(2 * np.pi * w_freq * 2 * w_t)
        + 0.1 * np.sin(2 * np.pi * w_freq * 3 * w_t)
    )
    ramp_w = int(sr * 0.01)
    ram = np.ones(w_n)
    if w_n > 2 * ramp_w:
        ram[:ramp_w] = np.linspace(0, 1, ramp_w)
        ram[-ramp_w:] = np.linspace(1, 0, ramp_w)
    wheeze_tone *= ram
    wheeze = np.zeros(n)
    wheeze[w_start:w_end] = wheeze_tone

    # --- crackle segments (4 positions) ---
    cw = max(1, int(sr * 0.015))
    c_t = np.arange(0, cw) / sr
    decay = np.exp(-c_t / 0.005)
    crackle_positions = [
        (1.2, 1000.0),
        (1.6, 1400.0),
        (3.5, 900.0),
        (4.8, 1600.0),
    ]
    crackle_audio = np.zeros(n)
    annotations = [
        (2.0, 2.6, "Wheeze", "manual"),
    ]
    for cc_pos, cc_freq in crackle_positions:
        ci = int(sr * cc_pos)
        burst = np.sin(2 * np.pi * cc_freq * c_t) * decay * 0.1
        end_idx = min(ci + cw, n)
        burst_len = end_idx - ci
        crackle_audio[ci:end_idx] += burst[:burst_len]
        annotations.append(
            (float(cc_pos), float(end_idx / sr), "Crackles", "manual")
        )

    # SNR
    sig_rms = np.sqrt(np.mean(breath ** 2))
    noise_rms = sig_rms / (10 ** (snr_db / 20))
    noise = noise_rms * rng.standard_normal(n)

    audio = (breath + wheeze + crackle_audio + noise).astype(np.float32)
    return audio, sr, annotations


# ---------------------------------------------------------------------------
# Level 3 — full respiratory cycle (Inspiration / Expiration / Pause)
# ---------------------------------------------------------------------------


def _breath_envelope(t, t_start, t_peak, t_end):
    """Asymmetric breath-phase envelope: rise → peak → decay."""
    env = np.zeros_like(t)
    n = len(t)
    i_start = int(np.searchsorted(t, t_start))
    i_peak = int(np.searchsorted(t, t_peak))
    i_end = int(np.searchsorted(t, t_end))
    if i_end <= i_start:
        return env
    # rise
    rise_n = max(1, i_peak - i_start)
    env[i_start:i_peak] = np.sin(np.linspace(0, np.pi / 2, rise_n))
    # fall
    fall_n = max(1, i_end - i_peak)
    env[i_peak:i_end] = np.sin(np.linspace(np.pi / 2, np.pi, fall_n))
    return env


def generate_respiratory_cycle(
    sr=4000,
    duration=12.0,
    breath_cycle=4.0,
    ie_ratio=1.2,          # Insp / Exp duration ratio
    pause_frac=0.15,       # fraction of cycle spent in pause
    wheeze_freq=400.0,
    wheeze_dur=0.4,
    crackle_count=6,
    crackle_width=0.015,
    snr_db=20.0,
    seed=42,
):
    """Full respiratory-cycle simulation with ground-truth phase annotations.

    Each cycle: Inspiration → Expiration → Pause.
    A wheeze is placed during the second Expiration; crackles during the
    first Inspiration.
    """
    rng = np.random.default_rng(seed)
    n = int(sr * duration)
    t = np.arange(0, n) / sr

    # breath noise carrier
    white = rng.standard_normal(n)
    pink = np.cumsum(white)
    pink -= np.mean(pink)
    pink /= np.std(pink) + 1e-12

    # low-pass to approximate tracheal breath spectrum (100–600 Hz)
    from scipy.signal import butter, filtfilt
    nyq = sr / 2
    b, a = butter(4, min(600 / nyq, 0.95), btype="low")
    breath_carrier = filtfilt(b, a, pink)
    breath_carrier /= np.std(breath_carrier) + 1e-12

    # build phase envelopes
    phase_env = np.zeros(n)
    annotations = []
    t0 = 0.0
    while t0 < duration:
        insp_dur = (breath_cycle * (1 - pause_frac)) * (ie_ratio / (1 + ie_ratio))
        exp_dur = (breath_cycle * (1 - pause_frac)) - insp_dur
        pause_dur = breath_cycle * pause_frac

        t_peak = t0 + insp_dur / 2
        t_end_insp = t0 + insp_dur
        t_end_exp = t_end_insp + exp_dur
        t_end_pause = min(t_end_exp + pause_dur, duration)

        # Inspiration phase
        phase_env += _breath_envelope(t, t0, t_peak, t_end_insp)
        annotations.append((float(t0), float(t_end_insp), "Inspiration", "manual"))

        # Expiration phase
        phase_env += _breath_envelope(t, t_end_insp, t_end_insp + exp_dur / 3, t_end_exp)
        annotations.append((float(t_end_insp), float(t_end_exp), "Expiration", "manual"))

        # Pause phase (quiet)
        if t_end_pause > t_end_exp:
            annotations.append((float(t_end_exp), float(t_end_pause), "Pause", "manual"))

        t0 = t_end_pause

    breath = 0.2 * breath_carrier * phase_env

    # --- wheeze: placed in the second expiration ---
    exp_annotations = [a for a in annotations if a[2] == "Expiration"]
    wheeze = np.zeros(n)
    if len(exp_annotations) >= 2:
        ws = exp_annotations[1][0] + 0.1
        we = min(ws + wheeze_dur, exp_annotations[1][1] - 0.05)
        wi = int(sr * ws)
        wj = int(sr * we)
        w_n = wj - wi
        if w_n > 0:
            w_t = t[wi:wj] - t[wi]
            w_tone = (
                0.6 * np.sin(2 * np.pi * wheeze_freq * w_t)
                + 0.3 * np.sin(2 * np.pi * wheeze_freq * 2 * w_t)
                + 0.1 * np.sin(2 * np.pi * wheeze_freq * 3 * w_t)
            )
            ramp_w = int(sr * 0.01)
            ram = np.ones(w_n)
            if w_n > 2 * ramp_w:
                ram[:ramp_w] = np.linspace(0, 1, ramp_w)
                ram[-ramp_w:] = np.linspace(1, 0, ramp_w)
            w_tone *= ram
            wheeze[wi:wj] = w_tone
            annotations.append((float(ws), float(we), "Wheeze", "manual"))

    # --- crackles: placed in the first inspiration ---
    insp_annotations = [a for a in annotations if a[2] == "Inspiration"]
    crackle_audio = np.zeros(n)
    cw = max(1, int(sr * crackle_width))
    c_t = np.arange(0, cw) / sr
    decay = np.exp(-c_t / (crackle_width / 3))
    if insp_annotations:
        is0, ie0 = insp_annotations[0][0], insp_annotations[0][1]
        margin = 0.1
        for _ in range(crackle_count):
            cp = rng.uniform(is0 + margin, ie0 - margin - 0.05)
            cf = rng.uniform(600, 1800)
            ci = int(sr * cp)
            burst = np.sin(2 * np.pi * cf * c_t) * decay * 0.12
            ej = min(ci + cw, n)
            bl = ej - ci
            crackle_audio[ci:ej] += burst[:bl]
            annotations.append(
                (float(cp), float(ej / sr), "Crackles", "manual")
            )

    # SNR
    sig_rms = np.sqrt(np.mean(breath ** 2))
    noise_rms = sig_rms / (10 ** (snr_db / 20))
    noise = noise_rms * rng.standard_normal(n)

    audio = (breath + wheeze + crackle_audio + noise).astype(np.float32)
    return audio, sr, annotations


# ---------------------------------------------------------------------------
# Boundary / stress signals
# ---------------------------------------------------------------------------


def generate_silence(sr=4000, duration=1.0, seed=42):
    """Pure silence (zero array)."""
    rng = np.random.default_rng(seed)
    n = int(sr * duration)
    audio = np.zeros(n, dtype=np.float32)
    return audio, sr, []


def generate_short_signal(sr=4000, duration=0.005, seed=42):
    """Signal shorter than one frame — boundary test for FFT/STFT."""
    rng = np.random.default_rng(seed)
    n = max(1, int(sr * duration))
    t = np.arange(0, n) / sr
    audio = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    return audio, sr, []


def generate_dc_offset(sr=4000, duration=1.0, offset=2.0, seed=42):
    """Constant DC signal — tests filter/STFT behaviour on non-zero-mean input."""
    rng = np.random.default_rng(seed)
    n = int(sr * duration)
    audio = (np.full(n, offset, dtype=np.float32)
             + 0.001 * rng.standard_normal(n).astype(np.float32))
    return audio, sr, []


# ---------------------------------------------------------------------------
# Convenience: batch all generators
# ---------------------------------------------------------------------------

ALL_GENERATORS = {
    "tone": generate_tone,
    "wheeze_episode": generate_wheeze_episode,
    "crackle_episode": generate_crackle_episode,
    "mixed_episode": generate_mixed_episode,
    "respiratory_cycle": generate_respiratory_cycle,
    "silence": generate_silence,
    "short_signal": generate_short_signal,
    "dc_offset": generate_dc_offset,
}
