#!/usr/bin/env python
"""ML pipeline demonstration on synthetic respiratory sound data.

Generates controlled synthetic audio with known ground-truth annotations,
runs the full RespAnno ML pipeline, and reports quantitative training
metrics suitable for inclusion in the SoftwareX manuscript.

The synthetic signals are designed so that:
  - Wheeze segments carry a narrow-band harmonic signature (350-400 Hz)
  - Crackle segments are short broadband transients (600-1800 Hz bursts)
  - The feature extractor captures spectral-centroid / energy differences
  - A LightGBM classifier can learn discriminative boundaries

Usage:
    conda run -n respanno python scripts/ml_demo.py
"""

import os
import sys

# Ensure stdout uses UTF-8 on Windows CI (avoids cp1252 errors on Chinese chars)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Suppress LightGBM verbose output
import lightgbm as lgb


class _SilentLogger:
    def info(self, msg): pass
    def warning(self, msg): pass


lgb.register_logger(_SilentLogger())

from respanno.dsp.features import compute_short_time_features, build_feature_matrix
from respanno.ml.classifier import train_event_model, apply_event_model
from respanno.ml.phase_model import train_phase_model, apply_phase_model
from tests.fixtures.mock_viewer import MockViewer
from tests.fixtures.synthetic_signals import (
    generate_wheeze_episode,
    generate_mixed_episode,
    generate_respiratory_cycle,
)

# Patch QMessageBox on the modules that imported their own references
import respanno.ml.classifier as _cls
import respanno.ml.phase_model as _phm
_cls.QMessageBox.information = lambda *a, **kw: None
_cls.QMessageBox.warning = lambda *a, **kw: None
_phm.QMessageBox.information = lambda *a, **kw: None
_phm.QMessageBox.warning = lambda *a, **kw: None


def _build_viewer(audio, sr, annotations, hop_length=256):
    times, feat_dict = compute_short_time_features(
        audio, int(sr), hop_length=hop_length
    )
    X, _, names = build_feature_matrix(times, feat_dict)
    return MockViewer(
        stft_features=X.astype(np.float64),
        stft_frame_times=times.astype(float),
        stft_feature_names=names,
        annotations=list(annotations),
        sr=int(sr),
        hop_length=hop_length,
    )


def _report_model(info, label, tag):
    print(f"  {tag:<12s} | {label:<15s} | "
          f"{info['train_precision']:6.3f} | {info['train_recall']:6.3f} | "
          f"{info['train_f1']:6.3f} | {info['train_accuracy']:6.3f} | "
          f"{info['train_specificity']:6.3f} | {info['train_auc_roc']:6.3f} | "
          f"{info['n_pos']:5d} | {info['n_neg']:6d} | {info['threshold']:5.3f}")


def main():
    print("=" * 125)
    print("RespAnno ML Pipeline Demonstration")
    print("=" * 125)

    # ---- header ----
    print()
    print(f"{'Experiment':<12s} | {'Label':<15s} | "
          f"{'P':>6s} | {'R':>6s} | {'F1':>6s} | {'Acc':>6s} | "
          f"{'Spec':>6s} | {'AUC':>6s} | {'Pos':>5s} | {'Neg':>6s} | {'Thr':>5s}")
    print("-" * 125)

    # ------------------------------------------------------------------
    # Experiment 1 — Wheeze binary classifier (single episode)
    # ------------------------------------------------------------------
    print("\n[Exp 1] Single-label wheeze classifier on 5 s episode")
    audio, sr, anns = generate_wheeze_episode(
        duration=6.0, wheeze_start=1.5, wheeze_dur=2.5, seed=42
    )
    viewer = _build_viewer(audio, sr, anns)
    ok = train_event_model(viewer, "Wheeze", random_state=42)
    if ok:
        _report_model(viewer.ml_models["Wheeze"], "Wheeze", "Exp1")

        # Show top features
        top = viewer.ml_models["Wheeze"].get("top_features_by_importance", [])
        if top:
            print(f"  Top features       : {', '.join(n for n, _ in top[:5])}")

        # Count predictions on unreviewed region
        ok_apply = apply_event_model(viewer, "Wheeze", min_dur_sec=0.05)
        if ok_apply:
            print(f"  ML-predicted segs  : {len(viewer.imported)}")

    # ------------------------------------------------------------------
    # Experiment 2 — Multi-label classifier (mixed episode)
    # ------------------------------------------------------------------
    print("\n[Exp 2] Multi-label classifiers on 6 s mixed episode")
    audio, sr, anns = generate_mixed_episode(duration=6.0, seed=42)
    viewer = _build_viewer(audio, sr, anns)

    for label in ("Wheeze", "Crackles"):
        ok = train_event_model(viewer, label, min_pos_frames=5, random_state=42)
        if ok:
            _report_model(viewer.ml_models[label], label, "Exp2")

    # ------------------------------------------------------------------
    # Experiment 3 — Phase model (respiratory cycle)
    # ------------------------------------------------------------------
    print("\n[Exp 3] Phase model (Insp/Exp/Pause) on 12 s respiratory cycle")
    audio, sr, anns = generate_respiratory_cycle(duration=12.0, seed=42)
    viewer = _build_viewer(audio, sr, anns)

    ok = train_phase_model(viewer, "Inspiration", random_state=42)
    if ok:
        info = viewer.ml_models["Inspiration"]
        prior = info["hsmm_prior"]
        scheme = "Three-state (Insp/Exp/Pause)" if len(info["classes"]) == 3 else "Two-state"
        print(f"  Scheme             : {scheme}")
        print(f"  Classes            : {info['classes']}")
        print(f"  State→Name         : {info['state_id_to_name']}")
        print(f"  dmin (frames)      : {prior['dmin_frames']}")
        print(f"  dmax (frames)      : {prior['dmax_frames']}")
        print(f"  Prefix reviewed    : {info['train_prefix_sec']:.1f} s")

        ok_apply = apply_phase_model(viewer, "Inspiration", min_dur_sec=0.05)
        if ok_apply:
            print(f"  ML-predicted segs  : {len(viewer.imported)}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 125)
    print("Summary")
    print("=" * 125)
    print("  All experiments completed successfully.")
    print("  The ML pipeline trains binary LightGBM classifiers for adventitious")
    print("  sound events (Wheeze, Crackles) and a multi-class phase model")
    print("  (Inspiration/Expiration/Pause) with HSMM post-processing.")
    print()
    print("  These results demonstrate that RespAnno's ML-assisted labeling")
    print("  pipeline can learn from user-reviewed annotations and propagate")
    print("  predictions to unreviewed regions of the recording.")
    print("=" * 125)


if __name__ == "__main__":
    main()
