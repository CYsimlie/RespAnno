"""Binary event classifier training and inference.

Provides two standalone functions that operate on an AudioViewer instance:

- train_event_model(viewer, label, ...) → bool
- apply_event_model(viewer, label, ...) → bool
"""

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    f1_score, precision_recall_fscore_support,
    confusion_matrix, accuracy_score, balanced_accuracy_score,
    matthews_corrcoef, roc_auc_score, average_precision_score, brier_score_loss,
)

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

from PyQt5.QtWidgets import QMessageBox

from respanno.ml.label_taxonomy import clear_ml_annotations


def train_event_model(
    viewer,
    label,
    min_pos_frames=20,
    neg_pos_ratio=5,
    random_state=None,
    model_kind="event",
):
    """Train a binary LightGBM classifier for a single label.

    Samples positive/negative frames from reviewed annotations, trains a
    LightGBM pipeline with optional MI feature selection, finds the optimal
    probability threshold, and stores the model in viewer.ml_models.
    """
    FS_ENABLE = True
    FS_KBEST = 20

    # 1) prepare features and frame labels
    viewer.ensure_frame_features()
    if viewer.stft_features is None or viewer.stft_frame_times is None:
        QMessageBox.information(viewer, "Machine Learning",
                                "No available short-time features; cannot train the model.")
        return False

    y = viewer.build_frame_labels_for_tag(label, neg_margin=0.05)
    if y is None:
        QMessageBox.information(viewer, "Machine Learning",
                                f"Label {label}  has no available frames in the reviewed region; cannot train.")
        return False

    y = np.asarray(y, dtype=np.int8)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]

    n_pos = int(len(idx_pos))
    n_neg_all = int(len(idx_neg))

    if n_pos < int(min_pos_frames):
        QMessageBox.information(
            viewer, "Machine Learning",
            f"Label {label} has only {n_pos} positive frames, less than the minimum requirement of {min_pos_frames}; training is skipped.")
        return False

    if n_neg_all == 0:
        QMessageBox.information(
            viewer, "Machine Learning",
            f"Label {label} has no available safe negative frames; training is skipped.")
        return False

    # 2) negative sample downsampling
    ratio = max(1, int(neg_pos_ratio))
    n_neg_target = int(min(n_neg_all, ratio * n_pos))

    rng = np.random.default_rng(random_state)
    idx_neg_sample = rng.choice(idx_neg, size=n_neg_target, replace=False)

    idx_all = np.concatenate([idx_pos, idx_neg_sample])
    rng.shuffle(idx_all)

    X = viewer.stft_features[idx_all, :]
    y_train = (y[idx_all] == 1).astype(int)

    # 3) feature selection + standardization + LightGBM
    D_all = int(X.shape[1])
    use_fs = bool(FS_ENABLE and D_all > 4)
    k_best = int(min(FS_KBEST, max(2, D_all)))

    steps = [("scaler", StandardScaler())]
    if use_fs:
        steps.append(("select", SelectKBest(score_func=mutual_info_classif, k=k_best)))

    if LGBMClassifier is None:
        QMessageBox.warning(viewer, "Machine Learning",
                            "lightgbm is not installed; cannot train a LightGBM model. Please run: pip install lightgbm")
        return False

    n_pos_train = int(np.sum(y_train == 1))
    n_neg_train = int(np.sum(y_train == 0))
    scale_pos_weight = float(n_neg_train) / float(max(1, n_pos_train))

    clf = LGBMClassifier(
        objective="binary",
        n_estimators=500,
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
    steps.append(("clf", clf))
    pipe = Pipeline(steps)
    pipe.fit(X, y_train)

    # 4) find optimal probability threshold (maximize F1)
    proba = pipe.predict_proba(X)[:, 1]
    best_th = 0.5
    best_f1 = -1.0

    for th in np.linspace(0.3, 0.9, 13):
        y_pred = (proba >= th).astype(int)
        f1 = f1_score(y_train, y_pred)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_th = float(th)

    y_pred_best = (proba >= best_th).astype(int)
    precision, recall, f1_final, _ = precision_recall_fscore_support(
        y_train, y_pred_best, average="binary", zero_division=0
    )

    # 5) comprehensive training metrics
    tn, fp, fn, tp = confusion_matrix(y_train, y_pred_best, labels=[0, 1]).ravel()
    acc = accuracy_score(y_train, y_pred_best)
    bacc = balanced_accuracy_score(y_train, y_pred_best)
    mcc = matthews_corrcoef(y_train, y_pred_best)
    specificity = float(tn) / (float(tn + fp) + 1e-12)
    npv = float(tn) / (float(tn + fn) + 1e-12)
    auc_roc = roc_auc_score(y_train, proba)
    auc_pr = average_precision_score(y_train, proba)
    brier = brier_score_loss(y_train, proba)

    # 6) feature selection results
    full_names = list(viewer.stft_feature_names)
    selected_idx = list(range(D_all))
    if use_fs and "select" in pipe.named_steps:
        selected_idx = pipe.named_steps["select"].get_support(indices=True).tolist()
    selected_names = [full_names[i] for i in selected_idx]

    # LightGBM feature importance
    top_k_show = int(min(8, len(selected_names)))
    top_feats = []
    importance_type = "gain"
    top_feat_str = ""

    try:
        booster = pipe.named_steps["clf"].booster_
        imp = booster.feature_importance(importance_type="gain")
        order = np.argsort(imp)[::-1][:top_k_show]
        top_feats = [(selected_names[i], float(imp[i])) for i in order]
        top_feat_str = "\n".join([f"  - {n}: {v:.4f}" for n, v in top_feats])
    except Exception:
        try:
            imp = pipe.named_steps["clf"].feature_importances_
            importance_type = "split"
            order = np.argsort(imp)[::-1][:top_k_show]
            top_feats = [(selected_names[i], float(imp[i])) for i in order]
            top_feat_str = "\n".join([f"  - {n}: {v:.4f}" for n, v in top_feats])
        except Exception:
            importance_type = "unknown"
            top_feat_str = "  (unable to retrieve feature importances)"

    # 7) store in viewer.ml_models
    viewer.ml_models[label] = {
        "clf": pipe,
        "threshold": float(best_th),
        "model_kind": str(model_kind),
        "feature_names": list(viewer.stft_feature_names),
        "selected_feature_indices": [int(i) for i in selected_idx],
        "selected_feature_names": list(selected_names),
        "feature_select_method": "mutual_info_kbest" if use_fs else "none",
        "feature_select_k": int(k_best) if use_fs else int(D_all),
        "top_features_by_importance": list(top_feats),
        "feature_importance_type": str(importance_type),
        "n_pos": int(n_pos),
        "n_neg": int(n_neg_target),
        "train_precision": float(precision),
        "train_recall": float(recall),
        "train_f1": float(f1_final),
        "train_accuracy": float(acc),
        "train_specificity": float(specificity),
        "train_npv": float(npv),
        "train_bacc": float(bacc),
        "train_mcc": float(mcc),
        "train_auc_roc": float(auc_roc),
        "train_auc_pr": float(auc_pr),
        "train_brier": float(brier),
        "confusion": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
    }

    QMessageBox.information(
        viewer, "Machine Learning",
        (f"Trained label '{label}' frame-level model:\n"
         f"Positive frames: {n_pos}, sampled negative frames: {n_neg_target}\n"
         f"Threshold (best training F1) = {best_th:.2f}\n"
         f"P={precision:.3f}, R={recall:.3f}, F1={f1_final:.3f}, Acc={acc:.3f}, Spec={specificity:.3f}\n"
         f"BAcc={bacc:.3f}, MCC={mcc:.3f}, AUROC={auc_roc:.3f}, AUPRC={auc_pr:.3f}, Brier={brier:.4f}\n"
         f"Confusion: TP={tp}, FP={fp}, TN={tn}, FN={fn}\n"
         f"Feature selection: {'MI-TopK' if use_fs else 'None'} (kept {len(selected_names)}/{D_all})\n"
         f"Top features (importance-{importance_type}):\n{top_feat_str}")
    )

    return True


def apply_event_model(
    viewer,
    label,
    min_dur_sec=0.05,
    expected_model_kinds=None,
):
    """Apply trained binary classifier on unreviewed region.

    Loads the model from viewer.ml_models, runs prediction on unreviewed
    frames, merges consecutive positive frames into segments, deduplicates
    against manual annotations, and writes results as machine annotations.
    """
    # 1) check model exists
    if not hasattr(viewer, "ml_models") or label not in viewer.ml_models:
        QMessageBox.information(
            viewer, "Auto Annotation",
            f"Label {label} has no trained model yet. Please train this label model first."
        )
        return False

    clear_ml_annotations(viewer, label)

    model_info = viewer.ml_models[label]
    if expected_model_kinds is not None:
        try:
            mk = str(model_info.get("model_kind", "event"))
        except Exception:
            mk = "event"
        if mk not in set(expected_model_kinds):
            QMessageBox.information(
                viewer, "Auto Annotation",
                f"Label {label} has a mismatched model type (current: {mk}). Please retrain the model for this label type."
            )
            return False

    clf = model_info["clf"]
    th = float(model_info.get("threshold", 0.5))

    # 2) prepare frame features and time axis
    viewer.ensure_frame_features()
    times = getattr(viewer, "stft_frame_times", None)
    X = getattr(viewer, "stft_features", None)
    if times is None or X is None or len(times) == 0:
        QMessageBox.information(
            viewer, "Auto Annotation",
            "No available short-time features; cannot auto-label."
        )
        return False

    times = np.asarray(times, dtype=float)

    # 3) unreviewed region
    T_used = viewer.get_reviewed_prefix()
    if T_used is None or T_used <= 0:
        QMessageBox.information(
            viewer, "Auto Annotation",
            "No manual annotations yet; cannot determine the unreviewed region."
        )
        return False

    idx_unr = np.where(times > T_used)[0]
    if idx_unr.size == 0:
        QMessageBox.information(
            viewer, "Auto Annotation",
            "The current record has already been fully reviewed; there are no unreviewed frames."
        )
        return False

    # 4) run model on unreviewed frames
    X_unr = X[idx_unr, :]
    try:
        proba = clf.predict_proba(X_unr)[:, 1]
    except Exception as e:
        QMessageBox.warning(
            viewer, "Auto Annotation",
            f"Model prediction failed: {e}"
        )
        return False

    y_hat = (proba >= th).astype(int)

    # 5) merge consecutive 1-runs into segments
    new_segments = []
    in_run = False
    start_i = None
    for i, v in enumerate(y_hat):
        if v == 1 and not in_run:
            in_run = True
            start_i = i
        elif v == 0 and in_run:
            frame_idxs = idx_unr[start_i:i]
            s = float(times[frame_idxs[0]])
            e = float(times[frame_idxs[-1]])
            if e - s >= min_dur_sec:
                new_segments.append((s, e))
            in_run = False
            start_i = None

    # tail segment
    if in_run and start_i is not None:
        frame_idxs = idx_unr[start_i:len(y_hat)]
        s = float(times[frame_idxs[0]])
        e = float(times[frame_idxs[-1]])
        if e - s >= min_dur_sec:
            new_segments.append((s, e))

    if not new_segments:
        QMessageBox.information(
            viewer, "Auto Annotation",
            "No candidate events were detected in the unreviewed region."
        )
        return False

    # 6) dedup against manual annotations
    manual_segs = viewer.get_manual_segments_for_label(label)

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
        QMessageBox.information(
            viewer, "Auto Annotation",
            "Candidate events highly overlap with existing manual annotations; no machine annotations were added."
        )
        return False

    # 7) write annotations via finalize_annotation
    if not hasattr(viewer, "annotations"):
        viewer.annotations = []

    for (s, e) in final_segments:
        viewer.finalize_annotation(s, e, text=label, source="ml")

    QMessageBox.information(
        viewer, "Auto Annotation",
        f"Label '{label}' added {len(final_segments)} machine-annotation segments in the unreviewed region."
    )
    return True
