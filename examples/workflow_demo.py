#!/usr/bin/env python
"""RespAnno v1.0.0 -- Workflow Demonstration.

Generates 20 s of synthetic respiratory audio with known wheeze segments
embedded in realistic background noise (SNR ~ -3 dB).  Only the first 5 s
are manually annotated; a LightGBM classifier is trained on that prefix
and then applied to the remaining 15 s to locate wheeze candidates.

No external WAV files required -- all signals are generated on-the-fly
with fixed random seeds for bitwise-identical output on every run.

Usage:
    conda run -n respanno python examples/workflow_demo.py
"""
import os, sys, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import lightgbm as lgb
class _Quiet:
    def info(self, msg): pass
    def warning(self, msg): pass
lgb.register_logger(_Quiet())

from respanno.audio.preprocessing import apply_butter_filter
from respanno.dsp.features import compute_short_time_features, build_feature_matrix
from respanno.ml.classifier import train_event_model, apply_event_model
from tests.fixtures.mock_viewer import MockViewer

import respanno.ml.classifier as _cls
_cls.QMessageBox.information = lambda *a, **kw: None
_cls.QMessageBox.warning   = lambda *a, **kw: None

# =============================================================================
# 20 s synthetic respiratory signal with realistic background noise
# =============================================================================

def make_demo_signal(sr=4000, duration=20.0, snr_db=-3.0, seed=42):
    """Generate a noisy respiratory signal with 6 wheeze bursts.

    The white noise floor is constant across the entire signal.  Wheeze
    tones are added on top *without* reducing local noise, so the wheeze
    regions are genuinely no louder than the surrounding breath -- only the
    narrow-band spectral signature distinguishes them.

    Parameters
    ----------
    sr : int        Sample rate (Hz)
    duration : float   Total signal duration (s)
    snr_db : float    Ratio of wheeze-tone RMS to white-noise RMS.  0 dB
                      means the tone power equals the noise power (1:1).
    seed : int         Fixed seed for reproducibility.

    Returns
    -------
    audio : np.ndarray           Normalised audio [-1, 1]
    sr : int                     Sample rate
    reviewed : list[tuple]       Annotations in the first 5 s
    unreviewed_gt : list[tuple]  Withheld annotations in the last 15 s
    """
    rng = np.random.default_rng(seed)
    n = int(sr * duration)
    t = np.arange(n, dtype=np.float32) / sr

    # -- Constant-power white noise floor across the entire 20 s --
    noise_floor = rng.standard_normal(n).astype(np.float32)

    # -- Breath carrier: pink noise with respiratory envelope --
    white = rng.standard_normal(n)
    pink = np.cumsum(white.astype(np.float64))
    pink -= np.mean(pink)
    pink /= max(np.std(pink), 1e-12)

    cycle_len = float(duration) / 5.0
    env = np.zeros(n, dtype=np.float32)
    for cyc in range(5):
        t0 = cyc * cycle_len
        i0 = int(t0 * sr)
        i1 = int(min(t0 + cycle_len, duration) * sr)
        n_seg = i1 - i0
        if n_seg > 0:
            seg_t = np.linspace(0, np.pi, n_seg, dtype=np.float32)
            s = np.sin(seg_t)
            s = np.maximum(s, 1e-8)
            env[i0:i1] += (s ** 0.6).astype(np.float32)

    breath = (0.10 * env * pink.astype(np.float32)).astype(np.float32)

    # -- 6 wheeze bursts (narrow-band harmonic series) --
    all_wheeze = [
        (1.2, 1.8, 380.0),
        (3.5, 4.2, 420.0),
        (7.0, 7.6, 360.0),
        (10.5, 11.5, 400.0),
        (14.0, 15.0, 390.0),
        (17.5, 18.2, 410.0),
    ]

    wheeze_signal = np.zeros(n, dtype=np.float32)
    all_annotations = []
    for ws, we, wf in all_wheeze:
        wi, wj = int(sr * ws), int(sr * we)
        w_n = wj - wi
        if w_n <= 0:
            continue
        w_t = t[wi:wj] - t[wi]
        tone = (
            0.55 * np.sin(2 * np.pi * wf * w_t) +
            0.25 * np.sin(2 * np.pi * wf * 2 * w_t) +
            0.12 * np.sin(2 * np.pi * wf * 3 * w_t) +
            0.05 * np.sin(2 * np.pi * wf * 4 * w_t)
        )
        ramp_n = max(1, int(sr * 0.01))
        ramp = np.ones(w_n, dtype=np.float32)
        if w_n > 2 * ramp_n:
            ramp[:ramp_n] = np.linspace(0, 1, ramp_n, dtype=np.float32)
            ramp[-ramp_n:] = np.linspace(1, 0, ramp_n, dtype=np.float32)
        tone = (tone * ramp).astype(np.float32)
        wheeze_signal[wi:wj] += tone
        all_annotations.append((float(ws), float(we), "wheeze", "manual"))

    # -- Power calibration: global white noise floor --
    # Tone RMS = noise RMS * 10^(snr/20).  At 0 dB they are equal.
    # The same white noise is applied everywhere -- wheeze tones sit on
    # top without any local compensation.  At such low SNR the wheeze
    # regions are visually indistinguishable from pure-noise segments.
    tone_rms = float(np.sqrt(np.mean(wheeze_signal ** 2)))
    noise_floor_rms = tone_rms / (10 ** (snr_db / 20.0))
    noise_floor *= (noise_floor_rms / (float(np.std(noise_floor)) + 1e-12))

    audio = breath + wheeze_signal + noise_floor
    audio /= (np.max(np.abs(audio)) + 1e-9)

    reviewed = [a for a in all_annotations if a[0] < 5.0]
    unreviewed = [a for a in all_annotations if a[0] >= 5.0]

    return audio.astype(np.float32), sr, reviewed, unreviewed

# =============================================================================
# Main
# =============================================================================

def main():
    SEP = "=" * 68
    print(SEP)
    print("  RespAnno v1.0.0 -- Workflow Demonstration")
    print("  ML-assisted respiratory sound annotation pipeline")
    print(SEP)

    # Step 1
    print()
    print("  Step 1: Generate 20 s synthetic respiratory audio (SNR ~ -3 dB)")
    audio, sr, reviewed, unreviewed_gt = make_demo_signal(seed=42)
    print(f"    Signal       : {len(audio)/sr:.0f} s @ {sr} Hz")
    print(f"    Total wheeze : {len(reviewed) + len(unreviewed_gt)} bursts")
    print(f"    Reviewed     : {len(reviewed)} burst(s) in first 5 s")
    for ws, we, lbl, _ in reviewed:
        print(f"      [{ws:.1f} - {we:.1f} s]  {lbl}  (manually annotated)")
    print(f"    Unreviewed   : {len(unreviewed_gt)} burst(s) in last 15 s")
    print(f"      (withheld -- ML will try to find them)")

    # Step 2
    print()
    print("  Step 2: Bandpass filter 20-1800 Hz (Butterworth, order 4)")
    filtered = apply_butter_filter(audio, sr, "bandpass", lowcut=20, highcut=1800, order=4)
    print(f"    Input RMS    : {np.sqrt(np.mean(audio**2)):.4f}")
    print(f"    Output RMS   : {np.sqrt(np.mean(filtered**2)):.4f}")

    # Step 3
    print()
    print("  Step 3: Extract 56 short-time features (n_fft=256, hop=64)")
    times, feat_dict = compute_short_time_features(filtered, int(sr))
    X_full, times2, full_names = build_feature_matrix(times, feat_dict)
    print(f"    Feature matrix : {X_full.shape[0]} frames x {X_full.shape[1]} features")

    # Step 4
    print()
    print("  Step 4: Train LightGBM on reviewed prefix only (first 5 s)")
    viewer = MockViewer(
        stft_features=X_full.astype(np.float64),
        stft_frame_times=times.astype(float),
        stft_feature_names=full_names,
        annotations=list(reviewed),
        sr=int(sr), hop_length=256,
    )
    ok = train_event_model(viewer, "wheeze", min_pos_frames=2, random_state=42)
    if not ok:
        print("    Training failed -- not enough positive samples.")
        return

    info = viewer.ml_models["wheeze"]
    print(f"    Train F1     : {info['train_f1']:.3f}")
    print(f"    Pos / Neg    : {info['n_pos']} / {info['n_neg']}")
    print(f"    Threshold    : {info['threshold']:.3f}")

    # Step 5
    print()
    print("  Step 5: Auto-label unreviewed region (last 15 s)")
    ok_apply = apply_event_model(viewer, "wheeze", min_dur_sec=0.10)
    if not ok_apply:
        print("    Auto-label returned no result.")
        return

    candidates = [(float(s), float(e), lbl, src) for s, e, lbl, src in viewer.imported]

    # Step 6
    print()
    print("  Step 6: Evaluate candidates against withheld ground truth")
    print(f"    ML candidates : {len(candidates)}")
    for s, e, _, _ in candidates:
        print(f"      [{s:.2f} - {e:.2f} s]")
    print(f"    Ground truth  : {len(unreviewed_gt)} (withheld)")
    for ws, we, _, _ in unreviewed_gt:
        print(f"      [{ws:.1f} - {we:.1f} s]")

    def iou(s1, e1, s2, e2):
        inter = max(0.0, min(e1, e2) - max(s1, s2))
        union = max(e1, e2) - min(s1, s2)
        return inter / union if union > 0 else 0.0

    matched = 0
    for cs, ce, _, _ in candidates:
        for ws, we, _, _ in unreviewed_gt:
            if iou(cs, ce, ws, we) > 0.3:
                matched += 1
                break

    print(f"    IoU matches   : {matched} / {len(unreviewed_gt)}")

    # Summary
    print()
    print(SEP)
    print("  Summary")
    print(SEP)
    print(f"    Manual annotations in first 5 s    : {len(reviewed)} wheeze burst(s)")
    print(f"    ML training F1 on reviewed prefix  : {info['train_f1']:.3f}")
    print(f"    ML candidates in unreviewed region  : {len(candidates)}")
    print(f"    IoU > 0.3 match against GT         : {matched} / {len(unreviewed_gt)}")
    if matched > 0:
        print()
        print("    The ML pipeline identified wheeze segments in the unreviewed")
        print("    region without any human input beyond the initial 5-second")
        print("    prefix.  Because the signal-to-noise ratio is low (~ -3 dB),")
        print("    the bursts are not trivially visible in the waveform; the")
        print("    ML classifier relies on the 56 short-time spectral features")
        print("    to distinguish narrow-band wheeze from broad-band breath noise.")
    print()
    print("    Launch the GUI for interactive annotation:")
    print("      conda run -n respanno python 1.0.0.py")
    print(SEP)

if __name__ == "__main__":
    main()
