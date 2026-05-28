"""Phase model training and inference using HSMM post-processing.

Provides two standalone functions that operate on an AudioViewer instance:

- train_phase_model(viewer, label, ...) → bool
- apply_phase_model(viewer, label, ...) → bool

Both import HSMM utilities from respanno.ml.hsmm and the annotation-clear
helper from respanno.ml.label_taxonomy.
"""

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

from PyQt5.QtWidgets import QMessageBox

from respanno.ml.hsmm import (
    estimate_hop_sec,
    estimate_breath_cycle_sec,
    build_hsmm_prior_from_prefix_labels,
    build_hsmm_log_trans,
    hsmm_viterbi,
    state_seq_to_segments,
)
from respanno.ml.label_taxonomy import clear_ml_annotations


def train_phase_model(
    viewer,
    label,
    min_pos_frames=30,
    neg_pos_ratio=5,
    random_state=None,
):
    """Train a phase model (Inspiration/Expiration + optional Pause).

    Builds multi-class frame labels from manual phase annotations, trains a
    LightGBM classifier with optional MI feature selection, estimates HSMM
    duration priors, and stores the model in viewer.ml_models.
    """
    # 1) prepare features and time axis
    viewer.ensure_frame_features()
    X_all = getattr(viewer, "stft_features", None)
    times = getattr(viewer, "stft_frame_times", None)
    if X_all is None or times is None or len(times) == 0:
        QMessageBox.information(viewer, "Machine Learning", "No available short-time features; cannot train the phase model.")
        return False

    times = np.asarray(times, dtype=float)

    # 2) only use reviewed prefix
    T_used = viewer.get_reviewed_prefix()
    if T_used is None or T_used <= 0:
        QMessageBox.information(viewer, "Machine Learning", "No manual annotations yet; cannot train the phase model.")
        return False

    idx_prefix = np.where(times <= float(T_used))[0]
    if idx_prefix.size == 0:
        QMessageBox.information(viewer, "Machine Learning", "No available frames in the reviewed prefix; cannot train the phase model.")
        return False

    # 3) collect manual phase annotations
    LAB_I = "Inspiration"
    LAB_E = "Expiration"
    seg_I = viewer.get_manual_segments_for_label(LAB_I)
    seg_E = viewer.get_manual_segments_for_label(LAB_E)

    has_I = len(seg_I) > 0
    has_E = len(seg_E) > 0
    if not has_I and not has_E:
        QMessageBox.information(viewer, "Machine Learning", "No manual Inspiration/Expiration annotations in the reviewed prefix; cannot train the phase model.")
        return False

    # 4) build multi-class frame labels
    y_state = np.full(len(times), -1, dtype=np.int16)

    for (s, e) in seg_I:
        if e <= s:
            continue
        idx = np.where((times >= float(s)) & (times <= float(e)))[0]
        y_state[idx] = 0
    for (s, e) in seg_E:
        if e <= s:
            continue
        idx = np.where((times >= float(s)) & (times <= float(e)))[0]
        y_state[idx] = 1

    # blank region handling
    prefix_mask = np.zeros(len(times), dtype=bool)
    prefix_mask[idx_prefix] = True
    idx_blank = np.where(prefix_mask & (y_state < 0))[0]

    if has_I and has_E:
        y_state[idx_blank] = 2
        state_id_to_name = {0: LAB_I, 1: LAB_E, 2: "Pause"}
    else:
        if has_I and not has_E:
            y_state[idx_blank] = 1
        elif has_E and not has_I:
            y_state[idx_blank] = 0
        state_id_to_name = {0: LAB_I, 1: LAB_E}

    y_prefix = y_state[idx_prefix]
    uniq = np.unique(y_prefix)
    if uniq.size < 2:
        QMessageBox.information(viewer, "Machine Learning", "The phase training data has too few valid classes (at least two are required). Please add the other phase or leave blank intervals.")
        return False

    # 5) train LightGBM classifier
    FS_ENABLE = True
    FS_KBEST = 25
    X_prefix = X_all[idx_prefix, :]

    D_all = int(X_prefix.shape[1])
    use_fs = bool(FS_ENABLE and D_all > 6)
    k_best = int(min(FS_KBEST, max(2, D_all)))

    steps = [("scaler", StandardScaler())]
    if use_fs:
        steps.append(("select", SelectKBest(score_func=mutual_info_classif, k=k_best)))

    if LGBMClassifier is None:
        QMessageBox.warning(viewer, "Machine Learning",
                            "lightgbm is not installed; cannot train a LightGBM model. Please run: pip install lightgbm")
        return False

    classes_uniq = np.unique(y_prefix)
    n_classes = int(len(classes_uniq))
    counts = {int(c): int(np.sum(y_prefix == c)) for c in classes_uniq}

    if n_classes <= 2:
        n0 = counts.get(0, 0)
        n1 = counts.get(1, 0)
        scale_pos_weight = float(n0) / float(max(1, n1))

        clf = LGBMClassifier(
            objective="binary",
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            random_state=int(random_state) if random_state is not None else 0,
            n_jobs=1
        )
    else:
        total = float(len(y_prefix))
        class_weight = {
            int(c): float(total) / (float(n_classes) * float(max(1, counts.get(int(c), 1))))
            for c in classes_uniq
        }

        clf = LGBMClassifier(
            objective="multiclass",
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_lambda=1.0,
            class_weight=class_weight,
            random_state=int(random_state) if random_state is not None else 0,
            n_jobs=1
        )

    steps.append(("clf", clf))
    pipe = Pipeline(steps)
    try:
        pipe.fit(X_prefix, y_prefix)
    except Exception as e:
        QMessageBox.warning(viewer, "Machine Learning", f"Phase model training failed: {e}")
        return False

    # 6) estimate HSMM priors
    sr = hop_length = None
    try:
        sv = getattr(viewer, "sr", None)
        hv = getattr(viewer, "hop_length", None)
        if sv is not None and hv is not None and float(sv) > 0 and float(hv) > 0:
            sr = float(sv)
            hop_length = float(hv)
    except Exception:
        pass
    hop_sec = estimate_hop_sec(times=times, sr=sr, hop_length=hop_length)
    cycle_sec = estimate_breath_cycle_sec(seg_I, seg_E)
    hsmm_prior = build_hsmm_prior_from_prefix_labels(
        y_prefix=y_prefix,
        classes_=pipe.named_steps["clf"].classes_,
        state_id_to_name=state_id_to_name,
        hop_sec=hop_sec,
        cycle_sec=cycle_sec,
    )

    # 6.1) inject HSMM initial distribution pi
    try:
        PI_TAIL_SEC = 1.0
        PI_MIX_LAMBDA = 0.85
        proba_prefix = pipe.predict_proba(X_prefix)
        hop_eff = float(hop_sec) if (hop_sec and hop_sec > 1e-6) else 0.05
        n_tail = int(max(1, round(float(PI_TAIL_SEC) / hop_eff)))
        n_tail = int(min(n_tail, proba_prefix.shape[0]))
        pi_hat = np.mean(proba_prefix[-n_tail:, :], axis=0)
        S_pi = int(pi_hat.size)
        pi_uniform = np.full((S_pi,), 1.0 / float(max(1, S_pi)), dtype=float)
        pi_init = (1.0 - float(PI_MIX_LAMBDA)) * pi_uniform + float(PI_MIX_LAMBDA) * pi_hat
        pi_init = np.clip(pi_init, 1e-12, None)
        pi_init = pi_init / float(np.sum(pi_init) + 1e-12)
        hsmm_prior["pi_init"] = [float(x) for x in pi_init.tolist()]
        hsmm_prior["pi_init_classes"] = [int(c) for c in hsmm_prior.get("classes", [])]
        hsmm_prior["pi_init_tail_sec"] = float(PI_TAIL_SEC)
        hsmm_prior["pi_init_lambda"] = float(PI_MIX_LAMBDA)
    except Exception:
        pass

    # 7) feature selection info
    full_names = list(getattr(viewer, "stft_feature_names", []))
    selected_idx = list(range(D_all))
    if use_fs and "select" in pipe.named_steps:
        try:
            selected_idx = pipe.named_steps["select"].get_support(indices=True).tolist()
        except Exception:
            selected_idx = list(range(D_all))

    model_info = {
        "model_kind": "phase",
        "clf": pipe,
        "classes": [int(x) for x in pipe.named_steps["clf"].classes_.tolist()],
        "state_id_to_name": dict((int(k), str(v)) for k, v in state_id_to_name.items()),
        "hsmm_prior": hsmm_prior,
        "feature_names": list(full_names),
        "selected_feature_indices": [int(i) for i in selected_idx],
        "feature_select_method": "mutual_info_kbest" if use_fs else "none",
        "feature_select_k": int(k_best) if use_fs else int(D_all),
        "train_prefix_sec": float(T_used),
    }

    # 8) shared mount to both Inspiration / Expiration
    if not hasattr(viewer, "ml_models"):
        viewer.ml_models = {}
    viewer.ml_models[LAB_I] = model_info
    viewer.ml_models[LAB_E] = model_info

    # 9) report
    counts = {int(c): int(np.sum(y_prefix == c)) for c in np.unique(y_prefix)}
    scheme = "Three-state (with Pause)" if ("Pause" in model_info["state_id_to_name"].values()) else "Two-state"
    msg = (
        f"Phase model trained ({scheme}):\n"
        f"Prefix length: {float(T_used):.2f}s, hop={hop_sec:.4f}s, estimated cycle={cycle_sec:.2f}s\n"
        f"Class frame counts: " + ", ".join([f"{model_info['state_id_to_name'][k]}={v}" for k, v in sorted(counts.items())]) + "\n"
        f"Feature selection: {'MI-TopK' if use_fs else 'None'} (kept {len(selected_idx)}/{D_all})\n"
        f"HSMM duration (frames): " + ", ".join([f"{model_info['state_id_to_name'][sid]}[{hsmm_prior['dmin_frames'][i]}..{hsmm_prior['dmax_frames'][i]}]" for i, sid in enumerate(hsmm_prior['classes'])])
    )
    QMessageBox.information(viewer, "Machine Learning", msg)
    return True


def apply_phase_model(viewer, label, min_dur_sec=0.05):
    """Apply trained phase model on unreviewed region using HSMM Viterbi decoding.

    Loads the model from viewer.ml_models, runs the classifier on unreviewed
    frames, decodes with HSMM Viterbi, and writes resulting segments as
    machine annotations.
    """
    LAB_I = "Inspiration"
    LAB_E = "Expiration"
    lab = str(label).strip().lower()
    if lab == "inspiration":
        target_label = LAB_I
    elif lab == "expiration":
        target_label = LAB_E
    else:
        target_label = str(label)

    # 1) check model
    if (not hasattr(viewer, "ml_models")) or (LAB_I not in viewer.ml_models):
        QMessageBox.information(viewer, "Auto Annotation", "The phase model has not been trained. Please train the Inspiration/Expiration phase model first.")
        return False

    model_info = viewer.ml_models.get(target_label, viewer.ml_models.get(LAB_I, None))
    if not model_info or model_info.get("model_kind") != "phase":
        QMessageBox.information(viewer, "Auto Annotation", "The phase model is missing or has a mismatched type. Please retrain the phase model.")
        return False

    clear_ml_annotations(viewer, target_label)

    # 2) prepare features and time
    viewer.ensure_frame_features()
    times = getattr(viewer, "stft_frame_times", None)
    X = getattr(viewer, "stft_features", None)
    if times is None or X is None or len(times) == 0:
        QMessageBox.information(viewer, "Auto Annotation", "No available short-time features; cannot auto-label.")
        return False

    times = np.asarray(times, dtype=float)

    # 3) unreviewed region
    T_used = viewer.get_reviewed_prefix()
    if T_used is None or T_used <= 0:
        QMessageBox.information(viewer, "Auto Annotation", "No manual annotations yet; cannot determine the unreviewed region.")
        return False

    idx_unr = np.where(times > float(T_used))[0]
    if idx_unr.size == 0:
        QMessageBox.information(viewer, "Auto Annotation", "The current record has already been fully reviewed; there are no unreviewed frames.")
        return False

    X_unr = X[idx_unr, :]
    clf = model_info["clf"]
    try:
        proba = clf.predict_proba(X_unr)
    except Exception as e:
        QMessageBox.warning(viewer, "Auto Annotation", f"Phase model prediction failed: {e}")
        return False

    classes = [int(c) for c in model_info.get("classes", [])]
    if not classes:
        try:
            classes = [int(c) for c in clf.named_steps["clf"].classes_.tolist()]
        except Exception:
            classes = list(range(proba.shape[1]))

    try:
        clf_classes = [int(c) for c in clf.named_steps["clf"].classes_.tolist()]
    except Exception:
        clf_classes = classes

    if len(clf_classes) != proba.shape[1]:
        QMessageBox.warning(viewer, "Auto Annotation", "The phase model has an abnormal class dimension. Retraining is recommended.")
        return False

    # build log-emission
    eps = 1e-12
    log_emit = np.log(np.clip(proba, eps, 1.0))

    # HSMM priors
    prior = model_info.get("hsmm_prior", {})
    dmin = prior.get("dmin_frames", None)
    dmax = prior.get("dmax_frames", None)
    if dmin is None or dmax is None:
        QMessageBox.warning(viewer, "Auto Annotation", "The phase model lacks HSMM priors (duration). Please retrain the phase model.")
        return False

    # transition matrix (log)
    state_id_to_name = model_info.get("state_id_to_name", {})
    state_names = [state_id_to_name.get(int(cid), str(cid)) for cid in clf_classes]
    log_trans = build_hsmm_log_trans(state_names)

    # initial distribution pi
    log_pi = np.full((len(state_names),), -np.log(len(state_names) + 1e-12), dtype=float)
    try:
        pi_init = prior.get("pi_init", None)
        pi_classes = prior.get("pi_init_classes", prior.get("classes", None))
        if (pi_init is not None) and (pi_classes is not None) and (len(pi_init) == len(pi_classes)) and len(pi_init) > 0:
            cls2pi = {int(c): float(p) for c, p in zip(pi_classes, pi_init)}
            S0 = float(len(state_names))
            pi_vec = np.array([cls2pi.get(int(cid), 1.0 / max(1.0, S0)) for cid in clf_classes], dtype=float)
            pi_vec = np.clip(pi_vec, 1e-12, None)
            pi_vec = pi_vec / float(np.sum(pi_vec) + 1e-12)
            log_pi = np.log(pi_vec)
    except Exception:
        pass

    # 4) HSMM decoding
    try:
        z_hat = hsmm_viterbi(log_emit, np.asarray(dmin, dtype=int), np.asarray(dmax, dtype=int), log_trans, log_pi)
    except Exception as e:
        QMessageBox.warning(viewer, "Auto Annotation", f"HSMM decoding failed: {e}")
        return False

    # 5) generate target_label continuous intervals
    name_to_stateid = {str(v): int(k) for k, v in state_id_to_name.items()}
    if target_label not in name_to_stateid:
        QMessageBox.information(viewer, "Auto Annotation", f"The current phase model does not contain state {target_label}; cannot output intervals for this label.")
        return False

    target_state_id = name_to_stateid[target_label]
    idx_to_state_id = [int(cid) for cid in clf_classes]
    z_state_ids = np.array([idx_to_state_id[int(k)] for k in z_hat], dtype=int)

    new_segments = state_seq_to_segments(times, idx_unr, z_state_ids, target_state_id, float(min_dur_sec))
    if not new_segments:
        QMessageBox.information(viewer, "Auto Annotation", "No candidate phase intervals were detected in the unreviewed region.")
        return False

    # 6) dedup against manual annotations
    manual_segs = viewer.get_manual_segments_for_label(target_label)

    def overlap_ratio(seg, base):
        s1, e1 = seg
        s2, e2 = base
        inter = min(e1, e2) - max(s1, s2)
        if inter <= 0:
            return 0.0
        return inter / max(e1 - s1, 1e-6)

    final_segments = []
    for (s, e) in new_segments:
        skip = False
        for (ms, me) in manual_segs:
            if overlap_ratio((s, e), (ms, me)) >= 0.5:
                skip = True
                break
        if not skip:
            final_segments.append((s, e))

    if not final_segments:
        QMessageBox.information(viewer, "Auto Annotation", "Candidate intervals highly overlap with existing manual annotations; no machine annotations were added.")
        return False

    # 7) write annotations
    if not hasattr(viewer, "annotations"):
        viewer.annotations = []

    for (s, e) in final_segments:
        viewer.finalize_annotation(s, e, text=target_label, source="ml")

    QMessageBox.information(viewer, "Auto Annotation", f"Phase '{target_label}' added {len(final_segments)} machine-annotation segments in the unreviewed region.")
    return True
