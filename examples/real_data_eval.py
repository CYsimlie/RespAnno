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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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


def evaluate_one_file(wav_path, csv_path, n_reviewed=2, min_dur_sec=0.10, plot=False):
    """Run end-to-end ML evaluation on a single WAV + ground-truth pair.

    Returns: (audio, sr, filtered, label_segs, results, S_db, spec_t, freqs, dur)
    where results = [(label, pipeline, reviewed, withheld, ml_segs, hits), ...]
    """
    SEP = "=" * 72

    # ── 1. Load ──────────────────────────────────────────────────────────
    audio, sr_raw = load_audio_file(wav_path)
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

    # Spectrogram for plotting
    from respanno.dsp.spectrogram import compute_stft_db
    S_db, freqs = compute_stft_db(filtered, int(sr), n_fft=256, hop_length=64, f_max=2000)
    spec_t = np.arange(S_db.shape[1]) * 64 / sr
    dur = len(audio) / sr

    # ── 3. Per-label evaluation ──────────────────────────────────────────
    print(f"\n  {'Label':<16s} {'Pipeline':<10s} {'Rev':>4s} {'W/h':>4s} {'ML':>4s} {'IoU>0.3':>8s}")
    print("  " + "-" * 56)

    total_reviewed, total_withheld, total_ml, total_hits = 0, 0, 0, 0
    results = []

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
            sr=int(sr), hop_length=64,
        )

        if kind == "phase":
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
            results.append((label, pipeline, reviewed, withheld, [], 0))
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
        results.append((label, pipeline, reviewed, withheld, ml_segs, hits))

    # ── 4. Summary ───────────────────────────────────────────────────────
    print()
    print(SEP)
    print(f"  Summary: {total_reviewed} reviewed, {total_withheld} withheld, "
          f"{total_ml} ML predictions")
    if total_withheld > 0:
        print(f"  Overall recall: {total_hits}/{total_withheld} "
              f"({100*total_hits/total_withheld:.0f}%)")
    print(SEP)

    # ── 5. Plot ──────────────────────────────────────────────────────────
    if plot and results:
        _plot_results(wav_path, audio, sr, filtered, label_segs, results,
                      S_db, spec_t, freqs, dur, n_reviewed)

    return audio, sr, filtered, label_segs, results, S_db, spec_t, freqs, dur


def _plot_results(wav_path, audio, sr, filtered, label_segs, results,
                  S_db, spec_t, freqs, dur, n_reviewed):
    """Generate waveform + spectrogram plot with annotation bands."""
    base_name = os.path.splitext(os.path.basename(wav_path))[0]
    out_path = os.path.join(os.path.dirname(wav_path) or ".", f"{base_name}_eval.png")

    labels = [r[0] for r in results]
    n_labels = len(labels)
    n_rows = 1 + n_labels
    fig_height = 4.5 * n_rows

    fig, axes = plt.subplots(n_rows, 1, figsize=(18, fig_height), sharex=True)
    if n_rows == 1:
        axes = [axes]

    # Shared axis handle
    ax_w = axes[0]
    ax_w.plot(np.arange(len(audio)) / sr, audio, color="#2c3e50", linewidth=0.25)

    # Reviewed prefix boundary line
    max_reviewed = max(max(e for _, e in r[2]) for r in results) if results else 0.0
    ax_w.axvline(max_reviewed, color="black", lw=1.5, ls="--", alpha=0.5)

    for i, (label, pipeline, reviewed, withheld, ml_segs, hits) in enumerate(results):
        ax = axes[1 + i] if (i + 1) < len(axes) else axes[0]
        if i > 0 or n_labels > 1:
            # Duplicate waveform for per-label rows
            ax.plot(np.arange(len(audio)) / sr, audio, color="#2c3e50", linewidth=0.20)
        if i > 0:
            ax.set_title(f"{label}  ({pipeline}: {hits}/{len(withheld)} hits)",
                         fontsize=11, fontweight="bold", loc="left", color="#333333")

        # Green: reviewed segments (training)
        for ws, we in reviewed or []:
            ax.axvspan(ws, we, alpha=0.28, color="#27ae60")
        # Blue: withheld GT (test)
        n_withheld = len(withheld) if withheld else 0
        for ws, we in withheld or []:
            ax.axvspan(ws, we, alpha=0.15, color="#2980b9")
        # Red hatched: ML predicted
        for ws, we in ml_segs or []:
            ax.axvspan(ws, we, alpha=0.22, color="#e74c3c", hatch="////")

        if i == 0:
            ax.set_ylabel("Amplitude", fontsize=10)
            handles = [
                mpatches.Patch(color="#27ae60", alpha=0.28, label="Reviewed (training)"),
                mpatches.Patch(color="#2980b9", alpha=0.15, label="GT withheld (test)"),
                mpatches.Patch(color="#e74c3c", alpha=0.22, hatch="////", label="ML predicted"),
            ]
            ax.legend(handles=handles, loc="upper right", fontsize=9, ncol=3, framealpha=0.9)
            ax.text(max_reviewed / 2, 0.92, "REVIEWED", ha="center", fontsize=10, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#27ae60", alpha=0.22))
            ax.text(max_reviewed + (dur - max_reviewed) / 2, 0.92, "UNREVIEWED", ha="center",
                    fontsize=10, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#e74c3c", alpha=0.18))
        ax.set_ylim(-1.2, 1.2)

    axes[-1].set_xlabel("Time (s)", fontsize=11)

    # Summary title
    total_hits = sum(r[5] for r in results)
    total_withheld = sum(len(r[3]) for r in results)
    fig.suptitle(
        f"RespAnno ML Evaluation — {os.path.basename(wav_path)}\n"
        f"{n_reviewed} reviewed segments per label | "
        f"Overall recall: {total_hits}/{total_withheld} "
        f"({100*total_hits/total_withheld:.0f}%)" if total_withheld > 0 else "",
        fontsize=13, fontweight="bold", y=0.998,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Plot saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RespAnno ML annotation on real lung sound recordings.")
    parser.add_argument("wav", nargs="?", default="",
                        help="Path to WAV file (4000 Hz or will be resampled)")
    parser.add_argument("csv", nargs="?", default="",
                        help="Path to ground-truth CSV (start,end,label[,source] columns)")
    parser.add_argument("--n_reviewed", type=int, default=2,
                        help="Number of ground-truth segments to use for training per label")
    parser.add_argument("--plot", action="store_true",
                        help="Generate PNG visualization of results")
    args = parser.parse_args()

    if args.wav and args.csv:
        evaluate_one_file(args.wav, args.csv, n_reviewed=args.n_reviewed, plot=args.plot)
    else:
        # Run on all bundled demo data.
        demo_dir = os.path.join(ROOT, "demo_data")
        pairs = [
            (os.path.join(demo_dir, "4000Hz", "101_1b1_Al_sc_Meditron.wav"),
             os.path.join(demo_dir, "events", "101_1b1_Al_sc_Meditron_events.csv")),
            (os.path.join(demo_dir, "4000Hz", "103_2b2_Ar_mc_LittC2SE.wav"),
             os.path.join(demo_dir, "events", "103_2b2_Ar_mc_LittC2SE_events.csv")),
            (os.path.join(demo_dir, "4000Hz", "130_1p3_Ll_mc_AKGC417L.wav"),
             os.path.join(demo_dir, "events", "130_1p3_Ll_mc_AKGC417L_events.csv")),
        ]
        for wav, csv in pairs:
            if not os.path.exists(wav) or not os.path.exists(csv):
                print(f"SKIP: missing {'WAV' if not os.path.exists(wav) else 'CSV'} for {os.path.basename(wav)}")
                continue
            evaluate_one_file(wav, csv, n_reviewed=args.n_reviewed, plot=args.plot)


if __name__ == "__main__":
    main()
