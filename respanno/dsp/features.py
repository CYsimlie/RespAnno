"""Pure short-time feature computation (no PyQt / pyqtgraph dependency).

Extracted from AudioViewer.compute_short_time_features and
AudioViewer.ensure_frame_features in legacy/1.0.0.py.

All feature definitions are preserved exactly — no formula changes.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple, Union

try:
    import librosa
except ImportError:
    librosa = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Feature name inventory (matches legacy code exactly)
# ---------------------------------------------------------------------------

TIME_DOMAIN_FEATURE_NAMES: List[str] = [
    "短时能量",
    "短时均值",
    "方差",
    "峰度",
    "偏度",
    "过零率",
    "teager能量算子",
]

SPECTRAL_FEATURE_NAMES: List[str] = [
    "谱均值", "谱标准差", "谱中位数", "谱能量", "谱RMS", "谱幅和",
    "谱质心", "谱带宽", "谱偏度", "谱峰度", "谱滚降", "谱平坦度",
    "谱熵", "谱通量",
    "最大谱峰值", "谱峰数量",
    "低频能量占比", "中频能量占比", "高频能量占比",
    "谱四分位距", "谱MAD", "谱差分零交叉率", "谱平滑度",
    "主峰/次峰比", "谱复杂度",
    "主峰能量占比", "前三峰能量占比", "90%能量覆盖频点数",
    "主峰-3dB带宽", "主峰Q因子",
]

COR_FEATURE_NAMES: List[str] = [
    "cor_dist_ratio_mean", "cor_mean_slope", "cor_max_slope",
    "cor_std_slope", "cor_max_peak", "cor_second_peak",
    "cor_peak_count", "cor_peak_density",
    "cor_area", "cor_std", "cor_cv", "cor_skewness", "cor_kurtosis",
    "cor_local_max_slope_mean", "cor_local_max_slope_min",
    "cor_local_std_mean", "cor_local_std_max",
    "cor_local_pk2pk_mean", "cor_local_pk2pk_max",
]

ALL_FEATURE_NAMES: List[str] = (
    TIME_DOMAIN_FEATURE_NAMES + SPECTRAL_FEATURE_NAMES + COR_FEATURE_NAMES
)


# ---------------------------------------------------------------------------
# 1. frame_signal
# ---------------------------------------------------------------------------

def frame_signal(
    audio: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Reflection-padded frame decomposition matching librosa STFT semantics.

    Returns (frames, times, T) where *frames* has shape (n_fft, T).
    """
    pad = n_fft // 2
    if pad > 0:
        x_pad = np.pad(audio, (pad, pad), mode="reflect") if len(audio) >= 2 else np.pad(audio, (pad, pad), mode="edge")
    else:
        x_pad = audio.copy()

    frames = librosa.util.frame(x_pad, frame_length=n_fft, hop_length=hop_length)
    T = frames.shape[1]
    times = (np.arange(T) * hop_length) / sr
    return frames, times, T


# ---------------------------------------------------------------------------
# 2. compute_time_domain_features
# ---------------------------------------------------------------------------

def compute_time_domain_features(
    audio: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
) -> Dict[str, np.ndarray]:
    """Compute the 7 time-domain frame features."""
    frames, times, T = frame_signal(audio, sr, n_fft, hop_length)

    feat: Dict[str, np.ndarray] = {}
    feat["短时能量"] = np.sum(frames ** 2, axis=0)
    mu = np.mean(frames, axis=0)
    feat["短时均值"] = mu
    var = np.var(frames, axis=0) + 1e-12
    feat["方差"] = var
    std = np.sqrt(var)
    centered3 = np.mean((frames - mu) ** 3, axis=0)
    centered4 = np.mean((frames - mu) ** 4, axis=0)
    feat["偏度"] = centered3 / (std ** 3 + 1e-12)
    feat["峰度"] = centered4 / (std ** 4 + 1e-12)

    # zero-crossing rate
    pad = n_fft // 2
    if pad > 0:
        x_pad = np.pad(audio, (pad, pad), mode="reflect") if len(audio) >= 2 else np.pad(audio, (pad, pad), mode="edge")
    else:
        x_pad = audio.copy()
    try:
        zcr = librosa.feature.zero_crossing_rate(x_pad, frame_length=n_fft, hop_length=hop_length, center=False)[0]
    except TypeError:
        zcr = librosa.feature.zero_crossing_rate(x_pad, frame_length=n_fft, hop_length=hop_length)[0]
    if zcr.shape[0] != T:
        zcr = zcr[:T] if zcr.shape[0] > T else np.pad(zcr, (0, T - zcr.shape[0]), mode="edge")
    feat["过零率"] = zcr

    # Teager energy operator
    xn = x_pad
    teager = np.zeros_like(xn)
    teager[1:-1] = xn[1:-1] ** 2 - xn[:-2] * xn[2:]
    te_frames = librosa.util.frame(teager, frame_length=n_fft, hop_length=hop_length)
    feat["teager能量算子"] = np.mean(np.abs(te_frames[:, :T]), axis=0)

    return feat


# ---------------------------------------------------------------------------
# 3. compute_spectral_features
# ---------------------------------------------------------------------------

def _cor_curve_features_19(y: np.ndarray, tvec: np.ndarray) -> Tuple:
    """Extract 19 features from a frequency-shift autocorrelation curve.

    Exact replica of the legacy _curve_features_19 inner function.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    tvec = np.asarray(tvec, dtype=float).reshape(-1)
    n0 = y.size
    if n0 < 2 or tvec.size < 2:
        return (0.0,) * 19

    # 1) global trend
    dist = np.sqrt(tvec ** 2 + y ** 2)
    md = float(np.max(dist))
    dist_ratio_mean = float(np.mean(dist / md)) if md > 0 else 0.0

    denom = float(tvec[-1] - tvec[0])
    mean_slope = float((y[-1] - y[0]) / denom) if abs(denom) > 1e-12 else 0.0

    dt = np.diff(tvec)
    dy = np.diff(y)
    if np.all(np.abs(dt) > 1e-12):
        slopes = dy / dt
        max_slope = float(np.min(slopes))
        std_slope = float(np.std(slopes))
    else:
        max_slope = 0.0
        std_slope = 0.0

    # 2) global peak features
    if n0 >= 3:
        pk_mask = (y[1:-1] > y[:-2]) & (y[1:-1] > y[2:])
        pk_vals = y[1:-1][pk_mask]
    else:
        pk_vals = np.array([], dtype=float)
    if pk_vals.size > 0:
        spk = np.sort(pk_vals)[::-1]
        max_peak = float(spk[0])
        second_peak = float(spk[1]) if spk.size >= 2 else float(spk[0])
        peak_count = float(pk_vals.size)
    else:
        max_peak = float(y[0])
        second_peak = float(y[0])
        peak_count = 0.0
    peak_density = float(peak_count / n0)

    # 3) waveform statistics
    area = float(np.trapezoid(y, tvec) if hasattr(np, 'trapezoid') else np.trapz(y, tvec))
    stdv = float(np.std(y))
    meanv = float(np.mean(y))
    cv = float(stdv / (meanv + 1e-12))
    mu0 = meanv
    c3 = float(np.mean((y - mu0) ** 3))
    c4 = float(np.mean((y - mu0) ** 4))
    skewness = float(c3 / (stdv ** 3 + 1e-12))
    kurtosis = float(c4 / (stdv ** 4 + 1e-12))

    # 4) local window features
    win_size = 10
    step = 5
    win_rows = []
    start = 0
    while start < n0:
        end = min(n0, start + win_size)
        yw = y[start:end]
        tw = tvec[start:end]
        if yw.size > 1:
            dtw = np.diff(tw)
            dyw = np.diff(yw)
            if np.all(np.abs(dtw) > 1e-12):
                max_slope_w = float(np.min(dyw / dtw))
            else:
                max_slope_w = 0.0
        else:
            max_slope_w = 0.0
        std_w = float(np.std(yw))
        pk2pk_w = float(np.max(yw) - np.min(yw))
        win_rows.append((max_slope_w, std_w, pk2pk_w))
        start += step
    W = np.array(win_rows, dtype=float) if win_rows else np.zeros((1, 3), dtype=float)
    local_max_slope_mean = float(np.mean(W[:, 0]))
    local_max_slope_min = float(np.min(W[:, 0]))
    local_std_mean = float(np.mean(W[:, 1]))
    local_std_max = float(np.max(W[:, 1]))
    local_pk2pk_mean = float(np.mean(W[:, 2]))
    local_pk2pk_max = float(np.max(W[:, 2]))

    return (
        dist_ratio_mean, mean_slope, max_slope, std_slope,
        max_peak, second_peak, peak_count, peak_density,
        area, stdv, cv, skewness, kurtosis,
        local_max_slope_mean, local_max_slope_min,
        local_std_mean, local_std_max,
        local_pk2pk_mean, local_pk2pk_max,
    )


def compute_spectral_features(
    audio: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
    f_max: float,
) -> Dict[str, np.ndarray]:
    """Compute spectral and COR frame features (30 + 19 = 49 features).

    All feature definitions match the legacy ``compute_short_time_features``.
    """
    from respanno.dsp.spectrogram import compute_stft_db

    S_db, freqs_all = compute_stft_db(audio, sr, n_fft=n_fft, hop_length=hop_length)
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, center=True, pad_mode="reflect")
    T = D.shape[1]

    S = np.abs(D)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    idx_fmax = freqs <= float(f_max)
    S = S[idx_fmax, :]
    freqs = freqs[idx_fmax]

    # Sub-band 200+ Hz
    f_low = 200.0
    idx_sub = (freqs >= f_low) & (freqs <= float(f_max))
    S_sub = S[idx_sub, :]
    freqs_sub = freqs[idx_sub]

    feat: Dict[str, np.ndarray] = {}

    # Power spectrum
    P = S ** 2
    P_sum = np.sum(P, axis=0, keepdims=True) + 1e-12
    Pn = P / P_sum

    # 8. spectral centroid
    cent = np.sum(freqs[:, None] * Pn, axis=0)
    # 9. spectral bandwidth
    var_f = np.sum(((freqs[:, None] - cent[None, :]) ** 2) * Pn, axis=0)
    bw = np.sqrt(var_f)
    # 10/11. spectral skewness / kurtosis
    m3 = np.sum(((freqs[:, None] - cent[None, :]) ** 3) * Pn, axis=0)
    m4 = np.sum(((freqs[:, None] - cent[None, :]) ** 4) * Pn, axis=0)
    skew_s = m3 / (var_f ** 1.5 + 1e-12)
    kurt_s = m4 / (var_f ** 2 + 1e-12)

    feat["谱质心"] = cent
    feat["谱带宽"] = bw
    feat["谱偏度"] = skew_s
    feat["谱峰度"] = kurt_s

    # 12. spectral rolloff (85%)
    roll = librosa.feature.spectral_rolloff(S=S_sub, freq=freqs_sub, roll_percent=0.85)[0]
    feat["谱滚降"] = roll

    # 13. spectral flatness
    flat = librosa.feature.spectral_flatness(S=S_sub)[0]
    feat["谱平坦度"] = flat

    # 14. spectral entropy
    ent = -np.sum(Pn * (np.log(Pn + 1e-12)), axis=0) / np.log(Pn.shape[0])
    feat["谱熵"] = ent

    # 15. spectral flux
    Sn = S / (np.linalg.norm(S, axis=0, keepdims=True) + 1e-12)
    dSn = np.diff(Sn, axis=1)
    flux = np.sqrt(np.sum(np.maximum(dSn, 0.0) ** 2, axis=0))
    flux = np.r_[0.0, flux[:T - 1]] if flux.size < T else flux[:T]
    feat["谱通量"] = flux

    # --- sub-band statistics ---
    spec_mean = np.mean(S_sub, axis=0)
    spec_std = np.std(S_sub, axis=0)
    spec_median = np.median(S_sub, axis=0)
    spec_energy = np.sum(S_sub ** 2, axis=0)
    spec_rms = np.sqrt(np.mean(S_sub ** 2, axis=0))
    spec_sum = np.sum(S_sub, axis=0)

    feat.update({"谱均值": spec_mean, "谱标准差": spec_std, "谱中位数": spec_median,
                 "谱能量": spec_energy, "谱RMS": spec_rms, "谱幅和": spec_sum})

    # --- peak features ---
    if S_sub.shape[0] >= 3:
        pk_mask = (S_sub[1:-1, :] > S_sub[:-2, :]) & (S_sub[1:-1, :] > S_sub[2:, :])
        peak_count = np.sum(pk_mask, axis=0).astype(float)
        pk_vals = np.where(pk_mask, S_sub[1:-1, :], 0.0)
        if pk_vals.shape[0] >= 2:
            top2 = np.partition(pk_vals, -2, axis=0)[-2:, :]
            pk1 = np.max(top2, axis=0)
            pk2 = np.min(top2, axis=0)
        else:
            pk1 = np.max(pk_vals, axis=0) if pk_vals.size else np.zeros(T)
            pk2 = np.zeros_like(pk1)
        max_peak = pk1
        peak_ratio = np.where(peak_count >= 2, pk1 / (pk2 + 1e-12), 0.0)
    else:
        peak_count = np.zeros(T)
        max_peak = np.zeros(T)
        peak_ratio = np.zeros(T)
    feat["最大谱峰值"] = max_peak
    feat["谱峰数量"] = peak_count
    feat["主峰/次峰比"] = peak_ratio

    # --- sub-band energy ratios ---
    E_total = np.sum(P, axis=0) + 1e-12
    low_msk = freqs <= 400.0
    mid_msk = (freqs > 400.0) & (freqs <= 800.0)
    high_msk = freqs > 800.0
    feat["低频能量占比"] = np.sum(P[low_msk, :], axis=0) / E_total if np.any(low_msk) else np.zeros(T)
    feat["中频能量占比"] = np.sum(P[mid_msk, :], axis=0) / E_total if np.any(mid_msk) else np.zeros(T)
    feat["高频能量占比"] = np.sum(P[high_msk, :], axis=0) / E_total if np.any(high_msk) else np.zeros(T)

    # --- IQR, MAD, ZC, smoothness, complexity ---
    q75 = np.percentile(S_sub, 75, axis=0)
    q25 = np.percentile(S_sub, 25, axis=0)
    spec_iqr = q75 - q25
    med0 = np.median(S_sub, axis=0, keepdims=True)
    spec_mad = np.median(np.abs(S_sub - med0), axis=0)

    dS = np.diff(S_sub, axis=0)
    if dS.shape[0] >= 2:
        spec_zc = np.mean((dS[:-1, :] * dS[1:, :]) < 0, axis=0)
    else:
        spec_zc = np.zeros(T)
    spec_smooth = np.mean(np.abs(dS), axis=0) if dS.size else np.zeros(T)
    spec_complex = np.sqrt(np.sum(dS ** 2, axis=0)) / (np.sum(S_sub, axis=0) + 1e-12) if dS.size else np.zeros(T)

    feat.update({"谱四分位距": spec_iqr, "谱MAD": spec_mad,
                 "谱差分零交叉率": spec_zc, "谱平滑度": spec_smooth, "谱复杂度": spec_complex})

    # --- Top-K energy and -3 dB bandwidth ---
    P_sub = S_sub ** 2
    E_sub = np.sum(P_sub, axis=0) + 1e-12
    top1_energy = np.max(P_sub, axis=0)
    feat["主峰能量占比"] = top1_energy / E_sub
    if P_sub.shape[0] >= 3:
        top3_sum = np.sum(np.partition(P_sub, -3, axis=0)[-3:, :], axis=0)
    else:
        top3_sum = np.sum(P_sub, axis=0)
    feat["前三峰能量占比"] = top3_sum / E_sub

    n_bins_90 = np.zeros(T)
    for k in range(T):
        p = P_sub[:, k]
        tot = float(np.sum(p))
        if tot <= 0:
            n_bins_90[k] = 0.0
            continue
        sp = np.sort(p)[::-1]
        cs = np.cumsum(sp)
        n_bins_90[k] = float(np.searchsorted(cs, 0.90 * tot) + 1)
    feat["90%能量覆盖频点数"] = n_bins_90

    dominant_bw = np.zeros(T)
    dominant_Q = np.zeros(T)
    if freqs_sub.size >= 2:
        df_sub = float(np.median(np.diff(freqs_sub)))
    else:
        df_sub = 0.0
    for k in range(T):
        s = S_sub[:, k]
        if s.size < 3:
            continue
        i0 = int(np.argmax(s))
        peak_val = float(s[i0])
        if peak_val <= 0:
            continue
        thr = peak_val / np.sqrt(2.0)
        left = i0
        while left > 0 and s[left] >= thr:
            left -= 1
        left_in = left + 1 if (left < i0 and s[left] < thr) else left
        right = i0
        while right < s.size - 1 and s[right] >= thr:
            right += 1
        right_in = right - 1 if (right > i0 and s[right] < thr) else right
        bw0 = float(freqs_sub[right_in] - freqs_sub[left_in])
        if bw0 <= 0.0 and df_sub > 0.0:
            bw0 = df_sub
        dominant_bw[k] = max(0.0, bw0)
        f0 = float(freqs_sub[i0])
        dominant_Q[k] = (f0 / bw0) if bw0 > 0 else 0.0
    feat["主峰-3dB带宽"] = dominant_bw
    feat["主峰Q因子"] = dominant_Q

    # --- COR (frequency-shift autocorrelation) features ---
    cor_low_hz = 100.0
    cor_high_hz = min(1200.0, float(f_max))
    cor_mask = (freqs_all >= cor_low_hz) & (freqs_all <= cor_high_hz)
    S_cor = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, center=True, pad_mode="reflect"))
    S_cor = S_cor[cor_mask, :]
    freqs_cor = freqs_all[cor_mask]

    # init all COR outputs to zero
    cor_names = [
        "cor_dist_ratio_mean", "cor_mean_slope", "cor_max_slope", "cor_std_slope",
        "cor_max_peak", "cor_second_peak", "cor_peak_count", "cor_peak_density",
        "cor_area", "cor_std", "cor_cv", "cor_skewness", "cor_kurtosis",
        "cor_local_max_slope_mean", "cor_local_max_slope_min",
        "cor_local_std_mean", "cor_local_std_max",
        "cor_local_pk2pk_mean", "cor_local_pk2pk_max",
    ]
    for cn in cor_names:
        feat[cn] = np.zeros(T)

    if S_cor.shape[0] >= 3 and S_cor.shape[1] == T:
        if freqs_cor.size >= 2:
            df_cor = float(np.median(np.diff(freqs_cor)))
        else:
            df_cor = 0.0
        if df_cor > 0.0:
            max_shift_hz = float(cor_high_hz - cor_low_hz)
            N_f = int(freqs_cor.size)
            N_shift = int(round(max_shift_hz / df_cor))
            N_shift = max(1, min(N_shift, N_f - 1))
            tau_axis = np.arange(N_shift + 1, dtype=float) * df_cor
            keep_n = max(1, int(round(N_f * 0.5)))

            for k in range(T):
                s0 = S_cor[:, k]
                if s0.size != N_f:
                    continue
                if keep_n >= N_f:
                    s = s0.astype(float)
                else:
                    idx_keep = np.argpartition(s0, -keep_n)[-keep_n:]
                    s = np.zeros_like(s0, dtype=float)
                    s[idx_keep] = s0[idx_keep]
                r = np.zeros(N_shift + 1, dtype=float)
                r[0] = float(np.dot(s, s))
                for tau_i in range(1, N_shift + 1):
                    r[tau_i] = float(np.dot(s[tau_i:], s[:-tau_i]))
                r = r - float(np.min(r))
                mx = float(np.max(r))
                if mx > 0:
                    r = r / mx
                vals = _cor_curve_features_19(r, tau_axis)
                for j, cn in enumerate(cor_names):
                    feat[cn][k] = vals[j]

    return feat


# ---------------------------------------------------------------------------
# 4. compute_short_time_features
# ---------------------------------------------------------------------------

def compute_short_time_features(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 256,
    hop_length: int = 64,
    f_max: float = 2000.0,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Compute all short-time features (time + spectral + COR).

    Returns
    -------
    times : (T,) float    frame centre times
    feat_dict : {name: (T,) float}   all features
    """
    if audio is None or sr is None or sr <= 0 or len(audio) == 0:
        return np.array([], dtype=float), {}

    # Time axis from STFT frame count
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, center=True, pad_mode="reflect")
    T = D.shape[1]
    times = (np.arange(T) * hop_length) / sr

    # Compute
    f_time = compute_time_domain_features(audio, sr, n_fft, hop_length)
    f_spec = compute_spectral_features(audio, sr, n_fft, hop_length, f_max)

    feat = {**f_time, **f_spec}
    return times, feat


# ---------------------------------------------------------------------------
# 5. normalize_feature_for_display
# ---------------------------------------------------------------------------

def normalize_feature_for_display(values: np.ndarray) -> np.ndarray:
    """0–1 normalisation per-feature for overlay display."""
    y = np.asarray(values, dtype=float)
    ymin = float(np.min(y))
    ymax = float(np.max(y))
    if abs(ymax - ymin) < 1e-12:
        return np.zeros_like(y)
    return (y - ymin) / (ymax - ymin)


# ---------------------------------------------------------------------------
# 6. build_feature_matrix (matching ensure_frame_features)
# ---------------------------------------------------------------------------

def build_feature_matrix(
    times: np.ndarray,
    feat_dict: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Stack features into a (T, 2D) matrix with smoothed copies.

    X_sm[k] = 0.5*X[k] + 0.25*X[k-1] + 0.25*X[k+1]
    """
    names = list(feat_dict.keys())
    feat_list = [np.asarray(feat_dict[n], dtype=float).reshape(-1) for n in names]
    X = np.vstack(feat_list).T

    if X.shape[0] != times.shape[0]:
        T = min(X.shape[0], times.shape[0])
        X = X[:T, :]
        times = times[:T]

    X_sm = 0.5 * X.copy()
    if X.shape[0] > 1:
        X_sm[1:] += 0.25 * X[:-1]
        X_sm[:-1] += 0.25 * X[1:]
    X_full = np.concatenate([X, X_sm], axis=1)
    full_names = names + [n + "_sm" for n in names]

    return X_full, times, full_names
