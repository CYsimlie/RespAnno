#!/usr/bin/env python
"""RespAnno v1.0.0 -- Demo signal visualization.

Waveform + spectrogram with aligned annotation bands:
  green  = reviewed (manual, first 5 s)
  blue   = ground truth (withheld, last 15 s)
  red    = ML predicted candidates
  purple = IoU-match connectors

Output: examples/demo_signal_visualization.png
"""
import os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.patches as mpatches

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests"))

from examples.workflow_demo import make_demo_signal
from respanno.audio.preprocessing import apply_butter_filter
from respanno.dsp.spectrogram import compute_stft_db
from respanno.dsp.features import compute_short_time_features, build_feature_matrix
from respanno.ml.classifier import train_event_model, apply_event_model
from tests.fixtures.mock_viewer import MockViewer

import lightgbm as lgb
class _Q:
    def info(self, m): pass
    def warning(self, m): pass
lgb.register_logger(_Q())
import respanno.ml.classifier as _c
_c.QMessageBox.information = lambda *a, **kw: None
_c.QMessageBox.warning   = lambda *a, **kw: None

# =============================================================================
# 1. Generate + run pipeline
# =============================================================================
audio, sr, reviewed, unreviewed_gt = make_demo_signal(seed=42)
filtered = apply_butter_filter(audio, sr, "bandpass", lowcut=20, highcut=1800)
times_f, feat_dict = compute_short_time_features(filtered, int(sr))
X_full, _, full_names = build_feature_matrix(times_f, feat_dict)

viewer = MockViewer(
    stft_features=X_full.astype(np.float64),
    stft_frame_times=times_f.astype(float),
    stft_feature_names=full_names,
    annotations=list(reviewed), sr=int(sr), hop_length=256,
)
train_event_model(viewer, "wheeze", min_pos_frames=2, random_state=42)
apply_event_model(viewer, "wheeze", min_dur_sec=0.10)

rev   = [(a[0], a[1]) for a in reviewed]
gt_un = [(a[0], a[1]) for a in unreviewed_gt]
ml    = [(float(s), float(e)) for s, e, _, _ in viewer.imported]

# Spectrogram
S_db, freqs = compute_stft_db(filtered, int(sr), n_fft=256, hop_length=64, f_max=2000)
spec_t = np.arange(S_db.shape[1]) * 64 / sr
dur = len(audio) / sr

# =============================================================================
# 2. Plot
# =============================================================================
fig = plt.figure(figsize=(22, 10))

# ---- waveform ----
ax_w = fig.add_axes([0.06, 0.58, 0.91, 0.36])
ax_w.plot(np.arange(len(audio))/sr, audio, color="#2c3e50", linewidth=0.30)
for ws, we in rev:   ax_w.axvspan(ws, we, alpha=0.22, color="#27ae60")
for ws, we in gt_un: ax_w.axvspan(ws, we, alpha=0.12, color="#2980b9")
for ws, we in ml:    ax_w.axvspan(ws, we, alpha=0.22, color="#e74c3c", hatch="////")
ax_w.axvline(5.0, color="black", lw=1.8, ls="--", alpha=0.55)
ax_w.text(2.5, 0.93, "REVIEWED", ha="center", fontsize=11, fontweight="bold",
          bbox=dict(boxstyle="round,pad=0.3", fc="#27ae60", alpha=0.22))
ax_w.text(12.5, 0.93, "UNREVIEWED  (ML auto-labeled)", ha="center", fontsize=11,
          fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="#e74c3c", alpha=0.18))
ax_w.set_ylabel("Amplitude", fontsize=11)
ax_w.set_ylim(-1.15, 1.15)
ax_w.set_xlim(0, dur)
ax_w.set_xticklabels([])
ax_w.tick_params(axis="both", labelsize=9)

h1 = mpatches.Patch(color="#27ae60", alpha=0.22, label=f"Reviewed ({len(rev)} bursts)")
h2 = mpatches.Patch(color="#2980b9", alpha=0.12, label=f"GT withheld ({len(gt_un)} bursts)")
h3 = mpatches.Patch(color="#e74c3c", alpha=0.22, hatch="////",
                    label=f"ML predicted ({len(ml)} candidates)")
ax_w.legend(handles=[h1, h2, h3], loc="lower right", fontsize=9, ncol=3, framealpha=0.9)

# ---- spectrogram ----
ax_s = fig.add_axes([0.06, 0.08, 0.91, 0.44])
# Use tighter dB range to show noise structure
vmin_disp = -70.0
S_clip = np.clip(S_db, vmin_disp, 0)
im = ax_s.pcolormesh(spec_t, freqs, S_clip, shading="auto", cmap="inferno",
                     vmin=vmin_disp, vmax=0, rasterized=True)
ax_s.set_ylabel("Frequency (Hz)", fontsize=11)
ax_s.set_xlabel("Time (s)", fontsize=11)
ax_s.set_ylim(0, 2000)
ax_s.set_xlim(0, dur)
ax_s.tick_params(axis="both", labelsize=9)

cb_ax = fig.add_axes([0.975, 0.08, 0.012, 0.44])
fig.colorbar(im, cax=cb_ax).set_label("dB", fontsize=9)

for ws, we in rev:   ax_s.axvspan(ws, we, alpha=0.22, color="#27ae60", zorder=5)
for ws, we in gt_un: ax_s.axvspan(ws, we, alpha=0.12, color="#2980b9", zorder=4)
for ws, we in ml:    ax_s.axvspan(ws, we, alpha=0.22, color="#e74c3c", hatch="////", zorder=6)
ax_s.axvline(5.0, color="white", lw=1.8, ls="--", alpha=0.55)

# ---- IoU match connectors ----
def iou(s1, e1, s2, e2):
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    return inter / union if union > 0 else 0.0

matched = 0
for cs, ce in ml:
    for ws, we in gt_un:
        if iou(cs, ce, ws, we) > 0.3:
            matched += 1
            ax_s.plot([(ws+we)/2, (cs+ce)/2], [280, 200], color="#9b59b6",
                     lw=1.8, alpha=0.85, zorder=10)
            ax_s.plot((ws+we)/2, 280, marker="v", color="#2980b9", markersize=7, zorder=10)
            ax_s.plot((cs+ce)/2, 200, marker="^", color="#e74c3c", markersize=7, zorder=10)
            break

h4 = mpatches.Patch(color="#27ae60", alpha=0.22, label="Reviewed")
h5 = mpatches.Patch(color="#2980b9", alpha=0.12, label="GT withheld")
h6 = mpatches.Patch(color="#e74c3c", alpha=0.22, hatch="////", label="ML predicted")
h7 = mpatches.Patch(color="#9b59b6", alpha=0.85, label=f"IoU match ({matched}/{len(gt_un)})")
ax_s.legend(handles=[h4, h5, h6, h7], loc="upper right", fontsize=9, ncol=4, framealpha=0.9)

# ---- title ----
train_f1 = viewer.ml_models["wheeze"]["train_f1"]
fig.suptitle(
    f"RespAnno v1.0.0 -- ML-Assisted Annotation Demo\n"
    f"20 s synthetic respiratory signal ({sr} Hz, SNR ~ -3 dB)  |  "
    f"Reviewed: {len(rev)} wheeze in first 5 s  |  "
    f"ML found: {matched}/{len(gt_un)} in unreviewed  |  "
    f"Train F1: {train_f1:.3f}",
    fontsize=12, fontweight="bold", y=0.995,
)

out = os.path.join(os.path.dirname(__file__), "demo_signal_visualization.png")
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
print(f"  Reviewed    : {len(rev)} bursts")
print(f"  GT withheld : {len(gt_un)} bursts")
print(f"  ML predicted: {len(ml)} candidates")
print(f"  IoU matched : {matched}/{len(gt_un)}")
print(f"  Train F1    : {train_f1:.3f}")
plt.close(fig)
