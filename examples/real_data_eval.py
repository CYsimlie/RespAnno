"""Real-data ML-assisted annotation evaluation.

Loads a paired (WAV, ground-truth CSV), trains one model per label using
the first N_reviewed ground-truth segments, then auto-labels the unreviewed
remainder and reports per-label recall/IoU.

Pipeline routing follows label taxonomy:
  - Phase labels (Inspiration/Expiration/Pause): HSMM
  - Abnormal sound labels (Wheeze/Crackles/Rhonchi/Stridor/etc.): LightGBM

Usage:
    cd examples/
    python real_data_eval.py <wav_path> <csv_path> [--n_reviewed 2]
"""
import os, sys, argparse, numpy as np
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Quiet LightGBM output
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

# Suppress QMessageBox from ML modules
import respanno.ml.classifier as _cls
import respanno.ml.phase_model as _phm
_cls.QMessageBox.information = lambda *a, **kw: None
_cls.QMessageBox.warning   = lambda *a, **kw: None
_phm.QMessageBox.information = lambda *a, **kw: None
_phm.QMessageBox.warning   = lambda *a, **kw: None


def iou(s1, e1, s2, e2):
    """Intersection-over-Union between two intervals."""
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    return inter / union if union > 0 else 0.0


def evaluate_one_file(wav_path, csv_path, n_reviewed=2, min_dur_sec=0.10):
    """Run end-to-end ML evaluation on a single WAV + ground-truth pair."""
    SEP = "=" * 72

    # ── 1. Load ──────────────────────────────────────────────────────────
    audio, sr_raw = load_audio_file(wav_path)
    original_sr = getattr(load_audio_file, '__wrapped__', lambda x: x)
    try:
        sr = sr_raw
    except Exception:
        sr = sr_raw
    gt = read_annotations(csv_path)

    print(f"\n  Audio     : {wav_path}")
    print(f"  Duration  : {len(audio)/sr:.1f} s @ {sr} Hz")
    print(f"  GT anns   : {len(gt)} ({csv_path})")

    label_segs = {}
    for a in gt:
        label_segs.setdefault(a["label"], []).append((a["start"], a["end"]))
    for lbl, segs in sorted(label_segs.items()):
        print(f"    {lbl}: {len(segs)} segments  ({label_kind(lbl)})")

    # ── 2. Preprocessing + features ──────────────────────────────────────
    filtered = apply_butter_filter(audio, sr, "bandpass", lowcut=20, highcut=1800, order=4)
    times, feat_dict = compute_short_time_features(filtered, int(sr))
    X_full, times2, full_names = build_feature_matrix(times, feat_dict)

    # ── 3. Per-label evaluation ──────────────────────────────────────────
    print(f"\n  {'Label':<16s} {'Pipeline':<10s} {'Rev':>4s} {'W/h':>4s} {'ML':>4s} {'IoU>0.3':>8s}")
    print("  " + "-" * 56)

    total_reviewed, total_withheld, total_ml, total_hits = 0, 0, 0, 0

    for label, segs in sorted(label_segs.items()):
        segs.sort()
        kind = label_kind(label)

        if len(segs) < n_reviewed + 1:
            print(f"  {label:<16s} {'-':<10s} {'-':>4s} {'-':>4s} {'-':>4s} {'-':>8s}   "
                  f"(only {len(segs)} segments, need >{n_reviewed})")
            continue

        reviewed = segs[:n_reviewed]
        withheld = segs[n_reviewed:]
        reviewed_anns = [(s, e, label, "manual") for s, e in reviewed]

        viewer = MockViewer(
            stft_features=X_full.astype(np.float64),
            stft_frame_times=times.astype(float),
            stft_feature_names=full_names,
            annotations=reviewed_anns,
            sr=int(sr), hop_length=256,
        )

        # Route to ML pipeline.
        if kind == "phase":
            phase_labels_in_gt = {lbl for lbl in label_segs if label_kind(lbl) == "phase"}
            if len(phase_labels_in_gt) < 2:
                print(f"  {label:<16s} {'HSMM':<10s} {len(reviewed):>4d} {len(withheld):>4d} "
                      f"{'-':>4s} {'-':>8s}   (need 2+ phase types for HSMM)")
                continue
            ok = train_phase_model(viewer, label, min_pos_frames=2, random_state=42)
            pipeline = "HSMM"
            ok_apply = apply_phase_model(viewer, label, min_dur_sec=min_dur_sec)
        else:
            ok = train_event_model(viewer, label, min_pos_frames=2, random_state=42)
            pipeline = "LightGBM"
            ok_apply = apply_event_model(viewer, label, min_dur_sec=min_dur_sec)

        if not ok:
            print(f"  {label:<16s} {pipeline:<10s} {len(reviewed):>4d} {len(withheld):>4d} "
                  f"{'-':>4s} {'-':>8s}   (training failed)")
            continue

        ml_segs = [(float(s), float(e)) for s, e, _, _ in viewer.imported] if ok_apply else []

        hits = 0
        for cs, ce in ml_segs:
            for ws, we in withheld:
                if iou(cs, ce, ws, we) > 0.3:
                    hits += 1
                    break

        recall = f"{hits}/{len(withheld)}" if withheld else "-"
        print(f"  {label:<16s} {pipeline:<10s} {len(reviewed):>4d} {len(withheld):>4d} "
              f"{len(ml_segs):>4d} {recall:>8s}")

        total_reviewed += len(reviewed)
        total_withheld += len(withheld)
        total_ml += len(ml_segs)
        total_hits += hits

    # ── 4. Summary ───────────────────────────────────────────────────────
    print()
    print(SEP)
    print(f"  Summary: {total_reviewed} reviewed, {total_withheld} withheld, "
          f"{total_ml} ML predictions")
    if total_withheld > 0:
        print(f"  Overall recall: {total_hits}/{total_withheld} "
              f"({100*total_hits/total_withheld:.0f}%)")
    print(SEP)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RespAnno ML annotation on real lung sound recordings.")
    parser.add_argument("wav", nargs="?", default="",
                        help="Path to WAV file (4000 Hz or will be resampled)")
    parser.add_argument("csv", nargs="?", default="",
                        help="Path to ground-truth CSV (start,end,label[,source] columns)")
    parser.add_argument("--n_reviewed", type=int, default=2,
                        help="Number of ground-truth segments to use for training per label")
    args = parser.parse_args()

    if args.wav and args.csv:
        evaluate_one_file(args.wav, args.csv, n_reviewed=args.n_reviewed)
    else:
        # Run on bundled demo data.
        demo_wav = os.path.join(ROOT, "demo_data", "4000Hz", "101_1b1_Al_sc_Meditron.wav")
        demo_csv = os.path.join(ROOT, "demo_data", "events", "101_1b1_Al_sc_Meditron_events.csv")
        if not os.path.exists(demo_wav) or not os.path.exists(demo_csv):
            print("ERROR: demo data not found. Provide WAV and CSV paths:")
            print("  python real_data_eval.py <wav> <csv> [--n_reviewed 2]")
            return
        print("Using bundled demo data. Usage: python real_data_eval.py <wav> <csv>")
        evaluate_one_file(demo_wav, demo_csv, n_reviewed=args.n_reviewed)


if __name__ == "__main__":
    main()
