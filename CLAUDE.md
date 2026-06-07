# RespAnno — Project Knowledge for Claude Code

## Project Overview

RespAnno is an interactive respiratory sound annotation tool with ML-assisted labeling, targeting SoftwareX journal submission.

- **Version:** v1.0.0
- **License:** MIT
- **Entry point:** `1.6.6.py` (PyQt5 GUI, ~2446 lines, single class AudioViewer)
- **Backend:** `respanno/` package (~5600 lines, 34 .py files, zero GUI dependency)
- **Tests:** 535 tests across 26 test files (534 pass, 1 skip)
- **Test runner:** `conda run -n respanno python -m pytest tests -q`
- **Compile check:** `conda run -n respanno python -m py_compile <file>`

## Development Environment

| Component | Version | Notes |
|-----------|---------|-------|
| Python | **3.10.20** | requires-python >= 3.9 |
| numpy | 1.24.4 | |
| scipy | 1.10.1 | |
| librosa | 0.11.0 | |
| scikit-learn | 1.3.2 | |
| lightgbm | 4.6.0 | |
| PyQt5 | 5.15.11 | GUI, not importable in headless CI |
| pyqtgraph | 0.13.3 | |
| sounddevice | 0.5.2 | Requires PortAudio (absent on headless Linux, lazy-imported) |
| pytest | 9.0.3 | |

**Conda environment:** `respanno` (defined in `environment.yml`)

## Architecture

```
1.6.6.py (2446 lines, AudioViewer class only)
  └─► respanno/
        ├── audio/preprocessing.py   — WAV loading, resampling, Butterworth filter
        ├── dsp/spectrogram.py       — STFT computation & display colorization (n_fft=256, hop=64)
        ├── dsp/fft.py               — FFT magnitude spectrum
        ├── dsp/features.py          — 56 short-time features
        ├── ml/service.py            — ML pipeline dispatcher (train/apply routing)
        ├── ml/classifier.py         — LightGBM binary event classifier
        ├── ml/phase_model.py        — HSMM-based respiratory phase model
        ├── ml/hsmm.py               — HSMM Viterbi decoder & duration priors
        ├── ml/label_taxonomy.py     — Label-to-pipeline routing (phase/abnormal/other)
        ├── ml/frame_labels.py       — Frame-level training label builder
        ├── ml/negatives.py          — Hard-negative sample manager
        ├── labels/annotation_io.py  — CSV/TXT/JSON annotation I/O
        ├── labels/events_importer.py— Auto-import of matching _events files
        ├── gui/dialogs/             — SettingsDialog, LoopPlayer, AnnotationLabelDialog
        ├── gui/spans/               — BoxSpan, SpanLabelItem
        ├── gui/views/               — AnnotViewBox, WaveViewBox
        └── gui/widgets/             — ColorBarWidget, ClickableSlider, ColorCheckDelegate
```

## Key Design Decisions

1. **MLService** uses lazy imports (`from respanno.ml.classifier import ...` inside methods) to avoid direct GUI file imports of ML modules.
2. **NegSampleManager** manages hard negative samples collected from user deletions. `neg_manager.to_dict()` feeds into `build_frame_labels()`.
3. **STFT defaults:** n_fft=256, hop_length=64, f_max=2000 (changed from legacy 512/256).
4. **sounddevice** is lazy-imported via `_get_sd()` for headless CI compatibility.
5. **Chinese feature names** in features.py (e.g., `谱质心`, `短时能量`, `过零率`), not English.

## Refactoring Constraints (NEVER VIOLATE)

- Never modify `legacy/1.6.6.py` (frozen original snapshot)
- Never start GUI on Linux headless (no display)
- Never run: `python 1.6.6.py`, `python legacy/1.6.6.py`, `python -m respanno.main`
- Use CRLF line endings for 1.6.6.py (it's a Windows file)
- Use Python scripts for large replacements (>100 lines) to avoid Edit tool issues

## Refactoring Workflow (8 steps)

1. Read module API + find target methods in 1.6.6.py
2. Replace method bodies with delegation calls
3. Update tests if needed
4. Run: `conda run -n respanno python -m py_compile 1.6.6.py`
5. Run: `conda run -n respanno python -m py_compile legacy/1.6.6.py`
6. Run: `conda run -n respanno python -m pytest tests -q`
7. Run: `git diff -- legacy/1.6.6.py` (must be empty)
8. Commit with descriptive message

## Keyboard Shortcuts (complete list)

| Shortcut | Action |
|----------|--------|
| Ctrl+O | Import WAV |
| Ctrl+E | Export annotations |
| Ctrl+I | Import annotations |
| Ctrl+P | Settings |
| Ctrl+Z | Undo |
| Space | Play/Pause |
| Left/Right | Seek ±1s |
| Up/Down | Prev/Next file |
| Delete/Backspace | Delete selected annotation |
| Ctrl+A | Accept ML annotation |
| Ctrl+T | Train model for current label |
| Ctrl+M | Auto-label for current label |
| Enter/Esc | Commit/Cancel span edit |
| F1 | About |
| Ctrl+Q | Exit |

## Test Suite (454 tests, 23 files)

### Golden Value Tests (physical ground-truth, all verified)
- Spectrogram: 500/1000 Hz sine sweep time-frequency localization
- Features: 500 Hz tone → spectral centroid 499.8 Hz (theory ~500)
- Features: amp=0.5 sine → short-time energy 32.00 (theory = n_fft × amp²/2 = 32.0)
- Features: 400 Hz tone → ZCR 0.1982 (theory 2f/sr = 0.200)
- Preprocessing: 500 Hz lowpass → -57.7 dB at 1 kHz
- Preprocessing: 20-1800 Hz bandpass → -0.00 dB at 500 Hz (passband preserved)
- Preprocessing: 100 Hz highpass → -48.2 dB at 50 Hz

### Test Quality Metrics
- 329/329 test functions have docstrings (100%)
- 5 modules rated ⭐⭐⭐⭐⭐: module_imports, label_taxonomy, negatives, fft_basic, reproducibility
- 3 modules rated ⭐⭐⭐☆☆: gui_static_integration (AST-only), performance_baseline (env-sensitive), icbhi_compatibility (naming only)

### Known Test Gaps (functional coverage)
1. MLService dispatcher routing — zero tests for the central pipeline router
2. Annotation CRUD semantics — undo/redo cycle with negative samples
3. Multi-lane layout (_pick_lane) — 3-lane collision avoidance algorithm
4. Hard negative feedback effectiveness — delete→retrain→improved predictions
5. All GUI modules — require PyQt5, manual testing only

## SoftwareX Submission Status

### Done
- ✅ README with Code Metadata + Software Metadata tables
- ✅ CITATION.cff, CONTRIBUTING.md, CODE_OF_CONDUCT.md
- ✅ MIT LICENSE
- ✅ CHANGELOG.md updated
- ✅ PROJECT_OUTLINE.md (paper writing prompt)
- ✅ TEST_RELIABILITY_REPORT.md (scientific validity analysis)
- ✅ GitHub Actions CI config (.github/workflows/test.yml)

### Pending (user action required)
- ⏳ Author full name in LICENSE, pyproject.toml, CITATION.cff (currently "C.Y.Pan")
- ⏳ Real email address (currently placeholder)
- ⏳ Screenshots in screenshots/ directory
- ⏳ Examples in examples/ directory
- ⏳ Zenodo DOI (archive before submission)
- ⏳ Cross-platform testing (Windows/macOS)
- ⏳ Comparison experiments with Audacity/ELAN for paper
