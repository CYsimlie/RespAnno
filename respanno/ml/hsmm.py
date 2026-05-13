"""Pure HSMM (Hidden Semi-Markov Model) utilities.

Extracted from MLService._hsmm_viterbi, _build_hsmm_log_trans,
_build_hsmm_prior_from_prefix_labels, _estimate_hop_sec,
_estimate_breath_cycle_sec, and _state_seq_to_segments
in legacy/1.6.6.py.

Zero dependency on PyQt / sklearn / LightGBM — only numpy.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# 1. estimate_hop_sec
# ---------------------------------------------------------------------------

def estimate_hop_sec(
    times: Optional[np.ndarray] = None,
    sr: Optional[int] = None,
    hop_length: Optional[int] = None,
) -> float:
    """Estimate the frame hop in seconds.

    Priority: sr + hop_length > median time-array diff > 0.05 default.
    """
    if sr is not None and hop_length is not None:
        try:
            sr_f = float(sr)
            hop_f = float(hop_length)
            if sr_f > 0 and hop_f > 0:
                return hop_f / sr_f
        except (ValueError, TypeError):
            pass

    if times is not None:
        t = np.asarray(times, dtype=float)
        dt = np.diff(t)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if dt.size:
            return float(np.median(dt))

    return 0.05


# ---------------------------------------------------------------------------
# 2. estimate_breath_cycle_sec
# ---------------------------------------------------------------------------

def estimate_breath_cycle_sec(
    seg_I: Sequence[Tuple[float, float]],
    seg_E: Sequence[Tuple[float, float]],
    default: float = 3.0,
) -> float:
    """Estimate breath-cycle duration (seconds) from manual phase annotations.

    Uses the median inter-start gap of the same phase (Insp or Exp).
    """
    starts: List[float] = []
    for segs in (seg_I, seg_E):
        ss = sorted([float(s) for s, e in segs if float(e) > float(s)])
        if len(ss) >= 2:
            dif = [ss[i + 1] - ss[i] for i in range(len(ss) - 1)]
            dif = [d for d in dif if d > 0.1]
            if dif:
                starts.extend(dif)
    if starts:
        return float(np.median(starts))
    return float(default)


# ---------------------------------------------------------------------------
# 3. build_hsmm_prior_from_prefix_labels
# ---------------------------------------------------------------------------

def build_hsmm_prior_from_prefix_labels(
    y_prefix: np.ndarray,
    classes_: Sequence[int],
    state_id_to_name: Dict[int, str],
    hop_sec: float,
    cycle_sec: float,
    dmax_cap_sec: float = 15.0,
) -> Dict[str, Any]:
    """Build HSMM duration priors (dmin/dmax in frames) from labelled frames.

    * classes_  — sklearn classifier class IDs (order must match).
    * state_id_to_name — maps class ID → human name (Inspiration, Expiration, Pause).

    Returns a dict with keys:
    - classes, hop_sec, cycle_sec, dmin_frames, dmax_frames
    """
    classes: List[int] = [int(c) for c in np.asarray(classes_).tolist()]
    y = np.asarray(y_prefix, dtype=int)

    # Collect run-lengths per state
    runs: Dict[int, List[int]] = {c: [] for c in classes}
    if y.size:
        cur = int(y[0])
        ln = 1
        for v in y[1:]:
            v = int(v)
            if v == cur:
                ln += 1
            else:
                if cur in runs:
                    runs[cur].append(int(ln))
                cur = v
                ln = 1
        if cur in runs:
            runs[cur].append(int(ln))

    cycle = float(cycle_sec) if (cycle_sec and cycle_sec > 0) else 3.0

    # Default seconds per named state
    def_sec: Dict[str, Tuple[float, float]] = {
        "Inspiration": (max(0.10, 0.08 * cycle), max(0.40, 1.20 * cycle)),
        "Expiration": (max(0.10, 0.08 * cycle), max(0.40, 1.20 * cycle)),
        "Pause": (0.05, max(0.30, 0.80 * cycle)),
    }

    def sec_to_frames(sec: float) -> int:
        return max(1, int(round(float(sec) / max(hop_sec, 1e-6))))

    dmax_cap = sec_to_frames(dmax_cap_sec)
    dmin: List[int] = []
    dmax: List[int] = []

    for c in classes:
        name = str(state_id_to_name.get(int(c), str(c)))
        arr = np.asarray(runs.get(int(c), []), dtype=float)
        if arr.size >= 5:
            mn = int(max(1, round(np.percentile(arr, 5))))
            mx = int(max(mn, round(np.percentile(arr, 95))))
        elif arr.size >= 1:
            mn = int(max(1, np.min(arr)))
            mx = int(max(mn, np.max(arr) * 2))
        else:
            a, b = def_sec.get(name, (0.10, 1.20 * cycle))
            mn = sec_to_frames(a)
            mx = sec_to_frames(b)

        mn = int(max(1, mn))
        mx = int(min(max(mn, mx), dmax_cap))
        dmin.append(mn)
        dmax.append(mx)

    return {
        "classes": classes,
        "hop_sec": float(hop_sec),
        "cycle_sec": float(cycle),
        "dmin_frames": [int(x) for x in dmin],
        "dmax_frames": [int(x) for x in dmax],
    }


# ---------------------------------------------------------------------------
# 4. build_hsmm_log_trans
# ---------------------------------------------------------------------------

def build_hsmm_log_trans(state_names: Sequence[str]) -> np.ndarray:
    """Build the HSMM log-transition matrix given human-readable state names.

    * 2 states: mutual transitions + self-loops.
    * 3 states (with ``"pause"``): Insp↔Exp can also go via Pause;
      Pause can self-loop.

    Returns (S, S) float array (log probabilities).
    """
    names = [str(x) for x in state_names]
    S = len(names)
    logA = np.full((S, S), -np.inf, dtype=float)

    def _norm_row(i: int, js: List[int]) -> None:
        if not js:
            return
        p = 1.0 / float(len(js))
        for j in js:
            logA[i, j] = np.log(p)

    if S == 2:
        for i in range(2):
            _norm_row(i, [0, 1])
        return logA

    # S == 3 (with Pause)
    idx_pause: Optional[int] = None
    for i, n in enumerate(names):
        if n.lower() == "pause":
            idx_pause = i
            break

    if idx_pause is None:
        for i in range(S):
            _norm_row(i, list(range(S)))
        return logA

    idx_insp: Optional[int] = None
    idx_exp: Optional[int] = None
    for i, n in enumerate(names):
        ln = n.lower()
        if ln == "inspiration":
            idx_insp = i
        elif ln == "expiration":
            idx_exp = i

    if idx_insp is None or idx_exp is None:
        for i in range(S):
            _norm_row(i, list(range(S)))
        return logA

    _norm_row(idx_insp, [idx_exp, idx_pause])
    _norm_row(idx_exp, [idx_insp, idx_pause])
    _norm_row(idx_pause, [idx_insp, idx_exp, idx_pause])
    return logA


# ---------------------------------------------------------------------------
# 5. hsmm_viterbi
# ---------------------------------------------------------------------------

def hsmm_viterbi(
    log_emit: np.ndarray,
    dmin: Sequence[int],
    dmax: Sequence[int],
    log_trans: np.ndarray,
    log_pi: np.ndarray,
) -> np.ndarray:
    """HSMM Viterbi decoder with explicit duration modelling.

    Parameters
    ----------
    log_emit : (T, S) float  log emission probabilities
    dmin, dmax : (S,) int   minimum / maximum duration in frames
    log_trans : (S, S) float  log transition matrix
    log_pi : (S,) float       log initial distribution

    Returns
    -------
    z : (T,) int   viterbi state index per frame
    """
    log_emit = np.asarray(log_emit, dtype=float)
    T, S = log_emit.shape
    dmin = np.asarray(dmin, dtype=int).reshape(-1)
    dmax = np.asarray(dmax, dtype=int).reshape(-1)
    log_trans = np.asarray(log_trans, dtype=float)
    log_pi = np.asarray(log_pi, dtype=float).reshape(-1)

    # cumulative sum for segment likelihood
    cum = np.zeros((T + 1, S), dtype=float)
    cum[1:, :] = np.cumsum(log_emit, axis=0)

    def seg_sum(t_end: int, t_start: int, s: int) -> float:
        return float(cum[t_end, s] - cum[t_start, s])

    neg_inf = -1e300
    dp = np.full((T + 1, S), neg_inf, dtype=float)
    bp_state = np.full((T + 1, S), -1, dtype=int)
    bp_dur = np.full((T + 1, S), 0, dtype=int)

    for t in range(1, T + 1):
        for s in range(S):
            best = neg_inf
            best_p = -1
            best_d = 0
            d_lo = int(max(1, dmin[s] if s < len(dmin) else 1))
            d_hi = int(min(dmax[s] if s < len(dmax) else t, t))
            if d_hi < d_lo:
                d_lo, d_hi = 1, t

            log_dur = -np.log(float(d_hi - d_lo + 1))

            for d in range(d_lo, d_hi + 1):
                start = t - d
                sll = seg_sum(t, start, s)

                if start == 0:
                    score = float(log_pi[s]) + log_dur + sll
                    prev = -1
                else:
                    prev_scores = dp[start, :] + log_trans[:, s]
                    prev = int(np.argmax(prev_scores))
                    score = float(prev_scores[prev]) + log_dur + sll

                if score > best:
                    best = score
                    best_p = prev
                    best_d = d

            dp[t, s] = best
            bp_state[t, s] = best_p
            bp_dur[t, s] = best_d

    # backtrack
    z = np.zeros(T, dtype=int)
    s = int(np.argmax(dp[T, :]))
    t = T
    while t > 0:
        d = int(bp_dur[t, s])
        start = t - d
        z[start:t] = s
        s_prev = int(bp_state[t, s])
        t = start
        if t <= 0 or s_prev < 0:
            break
        s = s_prev

    return z


# ---------------------------------------------------------------------------
# 6. state_seq_to_segments
# ---------------------------------------------------------------------------

def state_seq_to_segments(
    times: np.ndarray,
    idx_unr: np.ndarray,
    z_state_ids: np.ndarray,
    target_state_id: int,
    min_dur_sec: float,
) -> List[Tuple[float, float]]:
    """Convert a state sequence over unreviewed frames into (start, end) segments.

    * times         — time-stamps for ALL frames, shape (T_all,)
    * idx_unr       — indices of unreviewed frames within times, shape (T,)
    * z_state_ids   — state-id assignments for the unreviewed frames, shape (T,)
    * target_state_id — the state we are extracting segments for
    * min_dur_sec   — minimum segment duration in seconds
    """
    times = np.asarray(times, dtype=float)
    idx_unr = np.asarray(idx_unr, dtype=int)
    z = np.asarray(z_state_ids, dtype=int)
    if idx_unr.size == 0 or z.size == 0:
        return []
    if idx_unr.size != z.size:
        raise ValueError("idx_unr and z_state_ids must have the same length")

    segs: List[Tuple[float, float]] = []
    in_run = False
    start_i = 0
    for i, sid in enumerate(z):
        if sid == int(target_state_id) and not in_run:
            in_run = True
            start_i = i
        elif sid != int(target_state_id) and in_run:
            frame_idxs = idx_unr[start_i:i]
            s = float(times[frame_idxs[0]])
            e = float(times[frame_idxs[-1]])
            if e - s >= float(min_dur_sec):
                segs.append((s, e))
            in_run = False

    if in_run:
        frame_idxs = idx_unr[start_i:len(z)]
        s = float(times[frame_idxs[0]])
        e = float(times[frame_idxs[-1]])
        if e - s >= float(min_dur_sec):
            segs.append((s, e))

    return segs
