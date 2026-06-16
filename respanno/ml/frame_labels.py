"""Frame-level label builders for ML training data preparation.

Pure numpy — no PyQt or pyqtgraph dependency.
"""

import numpy as np


def _iter_reviewed_annotations(annotations):
    """Iterate over (start, end, label) tuples from reviewed annotations.

    Reviewed sources include: manual, auto_accepted, auto_edited, merged,
    merged_thresh_ctx. Three-element tuples are treated as manual.
    """

    def _norm_src(x):
        try:
            return str(x).strip().lower()
        except Exception:
            return ""

    reviewed_sources = {
        "manual",
        "auto_accepted",
        "auto_edited",
        "merged",
        "merged_thresh_ctx",
    }
    for item in annotations:
        if item is None:
            continue
        try:
            if len(item) == 3:
                s, e, t = item
                src = "manual"
            elif len(item) >= 4:
                s, e, t, src = item[:4]
            else:
                continue
        except Exception:
            continue

        src_n = _norm_src(src)
        if src_n in reviewed_sources:
            yield float(s), float(e), str(t)


def get_manual_segments(annotations, label):
    """Return [(s, e), ...] for reviewed annotations matching `label`."""
    segs = []
    _lab_low = str(label).strip().lower()
    for s, e, t in _iter_reviewed_annotations(annotations):
        if str(t).strip().lower() == _lab_low:
            segs.append((s, e))
    return segs


def get_reviewed_prefix(annotations):
    """Return max end time among reviewed annotations."""
    T = 0.0
    for s, e, _ in _iter_reviewed_annotations(annotations):
        T = max(T, e)
    return float(T)


def build_frame_labels(
    annotations,
    frame_times,
    label,
    neg_segments=None,
    neg_margin=0.05,
):
    """Build frame-level label vector y for a single label.

    Returns
    -------
    y : np.ndarray or None
        Shape (T,) with values: 1=positive, 0=safe negative, -1=ignore.
        Returns None if no valid frames are available for training.
    """
    if frame_times is None or len(frame_times) == 0:
        return None

    times = np.asarray(frame_times, dtype=float)

    T_used = get_reviewed_prefix(annotations)
    if T_used is None or T_used <= 0:
        return None

    y = np.full(times.shape, -1, dtype=np.int8)

    segs_pos = get_manual_segments(annotations, label)

    # 1) positive samples: entire segment
    for (s, e) in segs_pos:
        if e <= s:
            continue
        idx = np.where((times >= s) & (times <= e))[0]
        y[idx] = 1

    # 2) extended positive mask to exclude "too close" negatives
    mask_pos_ext = np.zeros_like(times, dtype=bool)
    for (s, e) in segs_pos:
        if e <= s:
            continue
        ext_start = max(0.0, s - neg_margin)
        ext_end = e + neg_margin
        if ext_end <= ext_start:
            continue
        idx = np.where((times >= ext_start) & (times <= ext_end))[0]
        mask_pos_ext[idx] = True

    mask_prefix = times <= T_used

    # 3) safe negatives: in prefix, not in extended-positive, not positive
    idx_neg = np.where(mask_prefix & (~mask_pos_ext) & (y != 1))[0]
    y[idx_neg] = 0

    # 3.5) hard negative segments (from deletions/corrections)
    # Hard negatives can lie beyond the reviewed prefix (e.g. deleted ML
    # false-positives in unreviewed regions).  They are placed without
    # mask_prefix, and T_used is extended accordingly so the function
    # does not return None when all negatives are beyond the prefix.
    if neg_segments:
        _lab_low = str(label).strip().lower()
        neg_list = []
        for _nk, _nv in neg_segments.items():
            if str(_nk).strip().lower() == _lab_low:
                neg_list = _nv
                break
        if neg_list:
            for it in neg_list:
                try:
                    s, e = float(it[0]), float(it[1])
                except Exception:
                    continue
                if e <= s:
                    continue
                idx = np.where(
                    (times >= s) & (times <= e) & (y != 1)
                )[0]
                y[idx] = 0
                T_used = max(T_used, e)

    if not np.any(y == 1) and not np.any(y == 0):
        return None

    return y
