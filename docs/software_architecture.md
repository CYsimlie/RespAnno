# Software Architecture

## Overview

RespAnno is a respiratory sound annotation tool built with PyQt5 + pyqtgraph +
librosa + scipy + LightGBM.  The codebase has been modularised from a single
6600-line legacy file into a maintainable package structure.

## Current Architecture (v1.0.0)

```
1.0.0.py                       # PyQt5 GUI entry point (~2446 lines)
  |
  +-- respanno/                # Computational back-end (~3900 lines)
  |   ├── main.py              # CLI launcher
  |   ├── audio/
  |   │   └── preprocessing.py # Butterworth filtering, resampling, config
  |   ├── dsp/
  |   │   ├── spectrogram.py   # STFT computation, display colourisation
  |   │   ├── fft.py           # FFT magnitude spectrum
  |   │   └── features.py      # 56 short-time features
  |   ├── ml/
  |   │   ├── service.py       # ML pipeline dispatcher
  |   │   ├── classifier.py    # LightGBM binary event classifier
  |   │   ├── phase_model.py   # HSMM-based respiratory phase model
  |   │   ├── hsmm.py          # HSMM Viterbi decoder, duration priors
  |   │   ├── label_taxonomy.py# Label-to-pipeline routing
  |   │   ├── frame_labels.py  # Frame-level training label builder
  |   │   └── negatives.py     # Hard-negative sample manager
  |   ├── labels/
  |   │   ├── annotation_io.py # CSV/TXT/JSON read & write
  |   │   └── events_importer.py# Auto-import matching _events files
  |   └── gui/                 # Reusable PyQt5 components
  |       ├── dialogs/         # SettingsDialog, LoopPlayer, AnnotationLabelDialog
  |       ├── spans/           # BoxSpan, SpanLabelItem
  |       ├── views/           # AnnotViewBox, WaveViewBox
  |       └── widgets/         # ColorBarWidget, ClickableSlider, ColorCheckDelegate
  |
  +-- legacy/1.0.0.py          # Frozen original single-file program (unmodified)
  +-- tests/                   # 535 tests, 24 files, 535 pass, 0 skip
```

## Data Flow

```
WAV file
  │
  ▼
load_audio (librosa) ──► preprocessing (resample + butter) ──► audio array + sr
  │                                                                   │
  ▼                                                                   ▼
STFT (spectrogram.py) ◄────────────────────────────── draw_waveform (GUI)
  │
  ├──► colorize → RGB image → QImage (GUI)
  │
  ▼
features.py ──► (T, 56) matrix + smoothed copies (T, 112)
  │
  ▼
ML training (LightGBM) ──► predict_proba
  │
  ├──► frame-level 0/1 → segments (event model)
  │
  └──► HSMM Viterbi → phase segments (phase model)

annotations ──► CSV/TXT/JSON export (annotation_io.py)
```

## Annotation Data Model

Standard internal representation (used throughout `respanno.labels`):

```python
{
    "start":  float,   # seconds
    "end":    float,   # seconds (must be > start)
    "label":  str,     # e.g. "wheeze", "Crackles", "Inspiration"
    "source": str,     # "manual" | "ml" | "auto_accepted" | "auto_edited" | "merged"
}
```

Legacy compatibility: tuples `(start, end, label)` or `(start, end, label, source)`
are accepted by `normalize_annotation()` and converted to dicts.

## Key Design Decisions

1. **PyQt-free modules** — All extracted modules (`labels`, `audio`, `dsp`, `ml`)
   have zero dependency on PyQt / pyqtgraph.  Only `respanno/main.py` touches the
   legacy GUI code.

2. **Behaviour preservation** — Algorithm logic is copied byte-for-byte from the
   legacy code.  No defaults, thresholds, or formulas have been changed.

3. **Test-first extraction** — Every module extraction is preceded by writing
   tests against pure-function replicas, so behaviour drift is caught immediately.

## States & Transitions

- **Phase 1–5 (COMPLETE)**: Pure logic modules extracted and tested.
- **Phase 6 (COMPLETE)**: Extracted modules connected to the GUI entry
  point (`1.0.0.py`), replacing inline code with module imports.
