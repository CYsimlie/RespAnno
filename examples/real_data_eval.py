"""RespAnno v1.0.0 -- Real-data ML-assisted annotation evaluation.

Loads a real respiratory WAV + ground-truth CSV, routes each label
through its correct ML pipeline:
  - phase labels (Inspiration/Expiration/Pause) -> phase_model + HSMM
  - abnormal sounds (Wheeze/Crackles/Rhonchi/etc.) -> binary LightGBM

Phase labels REQUIRE at least 2 phase types (e.g. Insp + Exp) in the GT
for HSMM to build a valid state machine.  If only one phase type is
present the evaluation is skipped for that label with an explanation.

Usage:
    conda run -n respanno python examples/real_data_eval.py
"""
import os, sys, numpy as np
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import lightgbm as lgb
class _Quiet:
    def info(self, msg): pass
    def warning(self, msg): pass
lgb.register_logger(_Quiet())

from respanno.audio.preprocessing import apply_butter_filter, load_audio_file
from respanno.dsp.features import compute_short_time_features, build_feature_matrix
from respanno.ml.classifier import train_event_model, apply_event_model
from respanno.ml.phase_model import train_phase_model, apply_phase_model
from respanno.ml.label_taxonomy import label_kind
from respanno.labels.annotation_io import read_annotations
from tests.fixtures.mock_viewer import MockViewer

import respanno.ml.classifier as _cls
import respanno.ml.phase_model as _phm
_cls.QMessageBox.information = lambda *a, **kw: None
_cls.QMessageBox.warning   = lambda *a, **kw: None
_phm.QMessageBox.information = lambda *a, **kw: None
_phm.QMessageBox.warning   = lambda *a, **kw: None

# ═════════════════════════════════════════════════════════════════════════
# Config
# ═════════════════════════════════════════════════════════════════════════
WAV_PATH = "demo_data/4000Hz/101_1b1_Al_sc_Meditron.wav"
CSV_PATH = "demo_data/events/101_1b1_Al_sc_Meditron_events.csv"
N_REVIEWED = 2
MIN_POS_FRAMES = 2
MIN_DUR_SEC = 0.10


def iou(s1, e1, s2, e2):
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    return inter / union if union > 0 else 0.0


def main():
    SEP = "=" * 68
    print(SEP)
    print("  RespAnno v1.0.0 -- Real-data ML-assisted Annotation Eval")
    print(SEP)

    # 1. Load
    print()
    print("  [1] Load real respiratory audio + ground-truth annotations")
    audio, sr = load_audio_file(WAV_PATH)
    gt = read_annotations(CSV_PATH)
    print(f"      Audio   : {WAV_PATH}")
    print(f"      Duration: {len(audio)/sr:.1f} s @ {sr} Hz")
    print(f"      GT      : {CSV_PATH}  ({len(gt)} annotations)")

    label_segs = {}
    for a in gt:
        label_segs.setdefault(a["label"], []).append((a["start"], a["end"]))
    for lbl, segs in sorted(label_segs.items()):
        kind = label_kind(lbl)
        print(f"      {lbl}: {len(segs)} segments  kind={kind}")

    # 2. Preprocessing + features (shared across all labels)
    print()
    print("  [2] Preprocessing: bandpass 20-1800 Hz + 56 feature extraction")
    filtered = apply_butter_filter(audio, sr, "bandpass", lowcut=20, highcut=1800, order=4)
    times, feat_dict = compute_short_time_features(filtered, int(sr))
    X_full, times2, full_names = build_feature_matrix(times, feat_dict)
    print(f"      Feature matrix: {X_full.shape[0]} frames x {X_full.shape[1]} features")

    # 3. Per-label evaluation
    print()
    print(SEP)
    print("  Results")
    print(SEP)
    header = f"  {'Label':<16s} {'Pipeline':<10s} {'Rev':>4s} {'W/h':>4s} {'ML':>4s} {'IoU':>4s}  {'Recall':>7s}"
    print(header)
    print("  " + "-" * 72)

    overall_reviewed = 0
    overall_withheld = 0
    overall_ml = 0
    overall_iou = 0

    for label, segs in sorted(label_segs.items()):
        segs.sort()
        kind = label_kind(label)

        if len(segs) < N_REVIEWED + 1:
            print(f"  {label:<16s} {'-':<10s} {'-':>4s} {'-':>4s} {'-':>4s} {'-':>4s}  {'-':>7s}   "
                  f"only {len(segs)} segment(s)")
            continue

        reviewed = segs[:N_REVIEWED]
        withheld = segs[N_REVIEWED:]
        reviewed_anns = [(s, e, label, "manual") for s, e in reviewed]

        viewer = MockViewer(
            stft_features=X_full.astype(np.float64),
            stft_frame_times=times.astype(float),
            stft_feature_names=full_names,
            annotations=reviewed_anns,
            sr=int(sr), hop_length=256,
        )

        # ---- Route to correct pipeline ----
        if kind == "phase":
            # Phase labels NEED multiple phase types (e.g. Inspiration AND Expiration)
            # for HSMM to build a valid 2+ state machine.
            phase_labels_in_gt = {lbl for lbl in label_segs if label_kind(lbl) == "phase"}
            if len(phase_labels_in_gt) < 2:
                print(f"  {label:<16s} {'HSMM':<10s} {len(reviewed):>4d} {len(withheld):>4d} "
                      f"{'-':>4s} {'-':>4s}  {'-':>7s}   "
                      f"NEED 2nd phase (e.g. Expiration) for HSMM")
                continue

            ok = train_phase_model(viewer, label, min_pos_frames=MIN_POS_FRAMES, random_state=42)
            if not ok:
                print(f"  {label:<16s} {'HSMM':<10s} {len(reviewed):>4d} {len(withheld):>4d} "
                      f"{'-':>4s} {'-':>4s}  {'-':>7s}   train failed")
                continue
            pipeline_name = "HSMM"
            ok_apply = apply_phase_model(viewer, label, min_dur_sec=MIN_DUR_SEC)
        else:
            ok = train_event_model(viewer, label, min_pos_frames=MIN_POS_FRAMES, random_state=42)
            if not ok:
                print(f"  {label:<16s} {'LightGBM':<10s} {len(reviewed):>4d} {len(withheld):>4d} "
                      f"{'-':>4s} {'-':>4s}  {'-':>7s}   train failed")
                continue
            pipeline_name = "LightGBM"
            ok_apply = apply_event_model(viewer, label, min_dur_sec=MIN_DUR_SEC)

        ml_candidates = [(float(s), float(e)) for s, e, _, _ in viewer.imported] if ok_apply else []

        matches = 0
        for cs, ce in ml_candidates:
            for ws, we in withheld:
                if iou(cs, ce, ws, we) > 0.3:
                    matches += 1
                    break

        recall_s = f"{matches}/{len(withheld)}" if withheld else "-"

        overall_reviewed += len(reviewed)
        overall_withheld += len(withheld)
        overall_ml += len(ml_candidates)
        overall_iou += matches

        note = ""
        print(f"  {label:<16s} {pipeline_name:<10s} {len(reviewed):>4d} {len(withheld):>4d} "
              f"{len(ml_candidates):>4d} {matches:>4d}  {recall_s:>7s}  {note}")

        print(f"      Reviewed (train): {', '.join(f'[{s:.2f}-{e:.2f}]' for s,e in reviewed)}")
        print(f"      Withheld (test) : {', '.join(f'[{s:.2f}-{e:.2f}]' for s,e in withheld)}")
        if ml_candidates:
            print(f"      ML candidates   : ({len(ml_candidates)})")
            for cs, ce in ml_candidates:
                best_tuple = max(((iou(cs, ce, ws, we), ws, we) for ws, we in withheld),
                                 key=lambda x: x[0])
                best_iou, best_ws, best_we = best_tuple
                status = "HIT" if best_iou > 0.3 else "MISS"
                print(f"        [{cs:.2f}-{ce:.2f}] -> {status}  "
                      f"(best IoU={best_iou:.2f} vs [{best_ws:.2f}-{best_we:.2f}])")

    # Summary
    print()
    print(SEP)
    print("  Summary")
    print(SEP)
    print(f"    Signal                 : {len(audio)/sr:.0f} s real respiratory recording @ {sr} Hz")
    print(f"    Reviewed per label     : first {N_REVIEWED} segment(s) = training")
    print(f"    Withheld per label     : remainder = test")
    print(f"    Pipeline routing       : phase labels -> HSMM; abnormal labels -> LightGBM")
    if overall_withheld > 0:
        print(f"    Overall recall         : {overall_iou}/{overall_withheld} "
              f"({100*overall_iou/overall_withheld:.0f}%)")
    print(SEP)


if __name__ == "__main__":
    main()
