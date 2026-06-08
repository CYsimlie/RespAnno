# Testing

## Running Tests

```bash
conda run -n respanno python -m pytest tests -q
```

535 collected, 534 pass, 1 skip (LoopPlayer requires sounddevice/PortAudio).

## Test Organization

```
tests/
├── conftest.py                     # Shared fixtures (qapp, synthetic signals, temp files)
├── fixtures/
│   ├── mock_viewer.py              # Mock AudioViewer for headless ML testing
│   ├── synthetic_signals.py        # 8 synthetic respiratory signal generators
│   └── annotations/                # CSV/TXT/JSON test fixtures
├── test_annotation_roundtrip.py     # 43 tests — CSV/TXT/JSON I/O roundtrip
├── test_gui_widgets_headless.py     # 33 tests — headless PyQt5 widget checks
├── test_preprocessing_basic.py      # 30 tests — Butterworth filter, golden values
├── test_ml_service_basic.py         # 25 tests — MLService dispatcher routing
├── test_negatives_basic.py          # 23 tests — NegSampleManager CRUD
├── test_annotation_quality.py       # 21 tests — source provenance, dedup
├── test_spectrogram_basic.py        # 21 tests — STFT, colourisation, golden values
├── test_gui_static_integration.py   # 20 tests — AST-level GUI import verification
├── test_frame_labels_basic.py       # 19 tests — frame label builder
├── test_features_basic.py           # 19 tests — 56 short-time features, golden values
├── test_hsmm_basic.py               # 18 tests — Viterbi, transitions, priors
├── test_events_importer_basic.py    # 15 tests — _events file auto-import
├── test_reproducibility.py          # 15 tests — determinism + cross-process hashing
├── test_classifier_training_basic.py# 13 tests — LightGBM binary classifier
├── test_fft_basic.py                # 12 tests — FFT magnitude computation
├── test_phase_model_basic.py        # 12 tests — HSMM phase model training
├── test_performance_baseline.py     #  9 tests — report-only latency/memory baselines
├── test_roundtrip_workflow.py       #  9 tests — WAV -> annotate -> export -> re-import
├── test_e2e_ml_pipeline.py          #  8 tests — audio -> features -> train -> predict
├── test_phase_apply_basic.py        #  8 tests — HSMM Viterbi decoding
├── test_classifier_apply_basic.py   #  7 tests — ML inference, dedup, min-dur filtering
├── test_label_taxonomy_basic.py     #  8 tests — label routing (phase/abnormal/other)
├── test_icbhi_compatibility.py      #  6 tests — ICBHI 2017 format conventions
└── test_module_imports.py           #  5 tests — module importability
```

## Disclaimer

These tests demonstrate functional correctness, file-format robustness, and
reproducibility of the software infrastructure. They should **not** be
interpreted as clinical performance validation or as a substitute for
independent detection model evaluation. The ML-assisted annotation pipeline
produces **candidate suggestions** that require human review; it does not
generate clinical diagnoses.
