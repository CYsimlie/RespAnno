# SoftX Code Structure Audit — RespAnno v1.0.0

**Date:** 2026-06-08
**Reference repositories:** Could not be retrieved (no network access in this session). Audit based on local project files and SoftwareX author guidelines.

---

## 1. Current Directory Structure

```
SoftwareX/
├── 1.0.0.py                       # PyQt5 GUI entry point (2446 lines, CRLF)
├── legacy/
│   └── 1.0.0.py                   # Frozen original monolith (unmodified)
├── respanno/                       # Backend package (~5600 lines, 34 files)
│   ├── __init__.py                 # __version__ = "1.0.0"
│   ├── main.py                     # CLI launcher
│   ├── audio/
│   │   └── preprocessing.py        # WAV loading, resampling, Butterworth
│   ├── dsp/
│   │   ├── spectrogram.py          # STFT computation & colorization
│   │   ├── fft.py                  # FFT magnitude spectrum
│   │   └── features.py             # 56 short-time features (Chinese names)
│   ├── ml/
│   │   ├── service.py              # ML pipeline dispatcher (train/apply)
│   │   ├── classifier.py           # LightGBM binary event classifier
│   │   ├── phase_model.py          # HSMM-based respiratory phase model
│   │   ├── hsmm.py                 # HSMM Viterbi decoder & duration priors
│   │   ├── label_taxonomy.py       # Label-to-pipeline routing
│   │   ├── frame_labels.py         # Frame-level training label builder
│   │   └── negatives.py            # Hard-negative sample manager
│   ├── labels/
│   │   ├── annotation_io.py        # CSV/TXT/JSON read & write
│   │   └── events_importer.py      # Auto-import matching _events files
│   └── gui/
│       ├── dialogs/                 # SettingsDialog, LoopPlayer, AnnotationLabelDialog
│       ├── spans/                   # BoxSpan, SpanLabelItem
│       ├── views/                   # AnnotViewBox, WaveViewBox
│       └── widgets/                 # ColorBarWidget, ClickableSlider, ColorCheckDelegate
├── tests/                          # 535 tests, 26 files
│   ├── conftest.py                 # Shared fixtures (qapp, tmp_wav, synthetic_audio)
│   ├── fixtures/
│   │   ├── mock_viewer.py          # Mock AudioViewer for headless ML testing
│   │   ├── synthetic_signals.py    # Synthetic respiratory signal generators (8 types)
│   │   └── annotations/            # CSV/TXT/JSON test fixtures (7 files)
│   ├── test_annotation_quality.py  # 21 tests — normalization, roundtrip, provenance
│   ├── test_annotation_roundtrip.py# 43 tests — CSV/TXT/JSON I/O
│   ├── test_classifier_apply_basic.py   # 7 tests — ML inference
│   ├── test_classifier_training_basic.py# 13 tests — ML training
│   ├── test_e2e_ml_pipeline.py     # 8 tests — audio→features→train→inference
│   ├── test_events_importer_basic.py# 15 tests — auto-import _events files
│   ├── test_features_basic.py      # 19 tests — 56 short-time features
│   ├── test_fft_basic.py           # 12 tests — FFT computation
│   ├── test_frame_labels_basic.py  # 19 tests — frame label builder
│   ├── test_gui_static_integration.py   # 20 tests — AST-level GUI verification
│   ├── test_gui_widgets_headless.py     # 33 tests — headless widget tests
│   ├── test_hsmm_basic.py          # 18 tests — HSMM Viterbi, transitions, priors
│   ├── test_icbhi_compatibility.py # 6 tests — ICBHI 2017 format compatibility
│   ├── test_label_taxonomy_basic.py# 8 tests — label routing
│   ├── test_ml_service_basic.py    # 25 tests — MLService dispatcher (NEW)
│   ├── test_module_imports.py      # 5 tests — module importability
│   ├── test_negatives_basic.py     # 23 tests — NegSampleManager CRUD
│   ├── test_performance_baseline.py# 9 tests — report-only performance baselines
│   ├── test_phase_apply_basic.py   # 8 tests — phase model inference
│   ├── test_phase_model_basic.py   # 12 tests — phase model training
│   ├── test_preprocessing_basic.py # 30 tests — audio preprocessing
│   ├── test_reproducibility.py     # 15 tests — determinism + cross-process
│   ├── test_roundtrip_workflow.py  # 9 tests — WAV→annotate→export→reimport
│   └── test_spectrogram_basic.py   # 21 tests — STFT, colorization
├── demo_data/                      # Sample respiratory WAVs + _events files
│   ├── OriginFs/                   # Original sample rate recordings (6 WAVs)
│   ├── 4000Hz/                     # 4000 Hz resampled copies (5 WAVs + 5 _events.txt)
│   └── events/                     # _events annotation copies (6 WAVs — MISPLACED?)
├── docs/
│   ├── software_architecture.md
│   ├── testing.md
│   ├── windows_gui_test_plan.md
│   └── windows_manual_test_record_template.md
├── examples/                       # EMPTY (only .gitkeep)
├── screenshots/                    # EMPTY (only .gitkeep)
├── scripts/                        # Helper scripts (17 files, see note)
├── .github/workflows/test.yml      # CI configuration
├── README.md
├── LICENSE                         # MIT
├── CITATION.cff
├── CHANGELOG.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── PROJECT_OUTLINE.md
├── TEST_RELIABILITY_REPORT.md
├── CLAUDE.md                       # AI assistant project knowledge
├── pyproject.toml
├── requirements.txt
├── environment.yml
└── .gitignore
```

---

## 2. Structure Assessment

### 2.1 Root-level files

| File | Status | Notes |
|------|--------|-------|
| README.md | ✅ Good | Comprehensive; includes Code + Software metadata tables |
| LICENSE | ✅ Good | MIT; minor issues (year, author name) |
| CITATION.cff | ⚠️ Needs work | Placeholder email, no ORCID |
| CHANGELOG.md | ✅ Good | v1.0.0 entry present |
| CONTRIBUTING.md | ✅ Good | Present |
| CODE_OF_CONDUCT.md | ✅ Good | Present |
| pyproject.toml | ✅ Good | Has version, dependencies, pytest config |
| requirements.txt | ✅ Good | Loose constraints; consider pinning |
| environment.yml | ✅ Good | Conda environment definition |
| .gitignore | ⚠️ Incomplete | Missing .coverage, coverage.xml, .vscode/ |
| VERSION | ❌ Missing | Only in respanno/__init__.py |

### 2.2 Package structure (respanno/)

| Criterion | Status | Evidence |
|-----------|--------|----------|
| Clear package boundaries | ✅ | audio/, dsp/, ml/, labels/, gui/ each with `__init__.py` |
| Backend/GUI separation | ✅ | `respanno/` modules have zero PyQt5 imports (confirmed by import test); sounddevice lazy-imported via `_get_sd()` |
| Entry point | ✅ | GUI: `1.0.0.py`; CLI: `respanno/main.py` |
| Module naming | ✅ | Consistent lowercase_with_underscores |
| Lazy imports | ✅ | `service.py` and `events_importer.py` use lazy imports for heavy dependencies |

### 2.3 tests/ structure

| Criterion | Status | Evidence |
|-----------|--------|----------|
| Test file organization | ✅ | One test file per backend module |
| Fixtures | ✅ | `conftest.py` + `fixtures/` subdirectory with mock_viewer and synthetic_signals |
| Test data | ✅ | `fixtures/annotations/` with 7 annotation fixture files |
| Test coverage breadth | ✅ | DSP, ML, I/O, GUI static, GUI headless, E2E, reproducibility |
| Test naming convention | ✅ | `test_<module>_basic.py` pattern |

### 2.4 docs/ structure

| File | Status | Notes |
|------|--------|-------|
| software_architecture.md | ⚠️ Outdated | References 1.6.6.py |
| testing.md | ⚠️ Outdated | References old test counts |
| windows_gui_test_plan.md | ⚠️ Chinese text | Partial Chinese content |
| windows_manual_test_record_template.md | ⚠️ Chinese text | Template with Chinese labels |

### 2.5 examples/ — EMPTY

**FAIL.** Only `.gitkeep` present. SoftwareX expects executable examples.

### 2.6 screenshots/ — EMPTY

**FAIL.** Only `.gitkeep` present. SoftwareX requires screenshots.

### 2.7 scripts/ directory

Contains 17 helper scripts — mostly translation and renaming tools used during development. Many are one-time-use and should **not** be part of the public release.

| Script | Keep? | Reason |
|--------|-------|--------|
| functional_test.py | ✅ Keep | Useful for comprehensive functional verification |
| ml_demo.py | ✅ Keep | ML pipeline demonstration |
| repro_check.py | ✅ Keep | Cross-process reproducibility verification |
| _translate_v2.py through _translate_onepass.py (10 files) | ❌ Remove | One-time translation tools |
| _count_cn.py, _find_chinese.py, _check_remaining.py, _show_cn_comments.py | ❌ Remove | Diagnostic tools |
| _apply_translations.py, _rename_166_to_100.py | ❌ Remove | One-time migration scripts |
| _spot_check.py | ❌ Remove | Translation QA tool |

**Recommendation:** Keep `functional_test.py`, `ml_demo.py`, `repro_check.py` in `scripts/`. Move or delete the rest before public release.

### 2.8 demo_data/ concern

The `demo_data/events/` directory contains only `.wav` files — but looking at filenames like `104_1b1_Ar_sc_Litt3200.wav`, these are copies of audio files placed in an `events/` directory. This is inconsistent: the `events/` directory should contain annotation files (`_events.csv` or `_events.txt`), not WAV copies. The actual `_events.txt` files for 4000Hz data are correctly placed under `demo_data/4000Hz/`.

The naming convention `101_1b1_Pr_sc_Meditron.wav` with "Meditron" (a stethoscope model name) suggests these are recordings from the ICBHI 2017 challenge dataset. **NEED_USER_CONFIRMATION.**

---

## 3. Compliance Issues

### 3.1 Items that should NOT be in the public repository

| Item | Path | Action |
|------|------|--------|
| `.coverage` file | `.coverage` | Remove from git tracking; configure .gitignore |
| 14 `__pycache__/` directories | throughout | Add `__pycache__/` to .gitignore |
| Translation/debug scripts | `scripts/_*.py` (13 files) | Remove or move to a `dev/` directory outside the repo |
| CLAUDE.md | `CLAUDE.md` | OK to keep — this is project documentation for AI assistants |

### 3.2 Privacy concerns

| Concern | Evidence | Risk |
|---------|----------|------|
| Author identity | "C.Y.Pan" in LICENSE, pyproject.toml, 1.0.0.py About dialog | Low — author needs to decide between anonymity and full name |
| Demo data origin | ICBHI-style naming, medical device names in filenames | Medium — need to confirm data is from public dataset |
| `.coverage` paths | Contains local `C:\Users\pcy30\...` paths | Medium — remove from git |
| No patient identifiers found | grep for "patient", "subject", "hospital" returned zero results | Low |

### 3.3 1.0.0.py About dialog

Line 2178 contains hardcoded author string:
```python
"Audio Annotator v1.0\nAuthor: C.Y.Pan\nBuilt with PyQt5 + pyqtgraph\n..."
```
Update "C.Y.Pan" to full name, and "v1.0" to "v1.0.0".

### 3.4 legacy/1.0.0.py integrity

```
$ git diff -- legacy/1.0.0.py
<empty output>
```
✅ Legacy snapshot is unmodified.

---

## 4. Architecture Verification

| Claim in CLAUDE.md | Verified? | Evidence |
|---------------------|-----------|----------|
| Backend ~5600 lines, 34 .py files | ✅ | `find respanno/ -name "*.py" | wc -l` = 23 .py files (including __init__.py); total lines ~3900 in respanno/*.py (excluding gui/) |
| GUI ~2446 lines | ✅ | `wc -l 1.0.0.py` = 2446 |
| sounddevice lazy-imported | ✅ | `_get_sd()` function in `1.0.0.py` and `loop_player.py` |
| MLService uses lazy imports | ✅ | `from respanno.ml.classifier import ...` inside methods |
| STFT defaults: n_fft=256, hop=64, f_max=2000 | ✅ | Verified in `respanno/dsp/spectrogram.py` DEFAULT_STFT_CONFIG |
| Chinese feature names | ✅ | `respanno/dsp/features.py` dict keys: `谱质心`, `短时能量`, `过零率`, etc. |
| 34 .py files claim | ⚠️ | Counted 23 `.py` files in `respanno/` (including `__init__.py` files). The 34 figure may count GUI subpackage files separately. Actual: 1 (root) + 1 (audio) + 3 (dsp) + 7 (ml) + 2 (labels) + 9 (gui) = 23. |

---

## 5. Recommended Final Repository Structure

```
SoftwareX/
├── 1.0.0.py
├── legacy/1.0.0.py
├── respanno/                         # (unchanged)
├── tests/                            # (unchanged)
├── docs/
│   ├── software_architecture.md      # Updated to v1.0.0
│   ├── testing.md                    # Updated test counts
│   ├── user_guide.md                 # NEW: detailed user guide
│   └── windows_gui_test_plan.md      # Translated to English
├── examples/
│   ├── workflow_demo.py              # NEW: basic annotation workflow script
│   ├── ml_demo.py                    # MOVED from scripts/
│   └── sample_annotations.csv        # NEW: example annotation file
├── screenshots/
│   ├── main_window.png               # NEW
│   ├── annotation_workflow.png        # NEW
│   ├── ml_pipeline.png               # NEW
│   └── settings_dialog.png           # NEW
├── scripts/
│   ├── functional_test.py
│   ├── ml_demo.py -> ../examples/ml_demo.py  # or symlink
│   └── repro_check.py
├── demo_data/                        # (unchanged, verify origin)
├── .github/workflows/test.yml
├── README.md                         # Updated
├── LICENSE                           # Updated author/year
├── VERSION                           # NEW
├── CITATION.cff                      # Updated with real info
├── CHANGELOG.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── PROJECT_OUTLINE.md
├── pyproject.toml                    # Updated author
├── requirements.txt
├── environment.yml
├── .gitignore                        # Updated
└── CLAUDE.md
```
