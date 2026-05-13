# Testing

## Running Tests

```bash
conda run -n respanno python -m pytest tests -q
```

## Test Organization

```
tests/
├── conftest.py                      # Shared fixtures (synthetic signals, temp files)
├── test_annotation_roundtrip.py     # 43 tests — label I/O (CSV, TXT, JSON)
├── test_preprocessing_basic.py      # 26 tests — Butterworth filter, resample config
├── test_spectrogram_basic.py        # 19 tests — STFT, palette, colourisation
├── test_hsmm_basic.py               # 18 tests — Viterbi, transitions, priors
└── test_features_basic.py           # 15 tests — 56 short-time features
```

## What Each Test Suite Covers

### test_annotation_roundtrip.py
- normalize_annotation (tuple → dict)
- parse_annotation_row (delimited row → dict, column mapping)
- CSV read (comma, source column, header skip, invalid row skip)
- TXT read (tab, semicolon, space, custom delimiter)
- JSON read (array-of-dicts, nested keys, fallback keys, case-insensitive)
- Write-read roundtrip (CSV & JSON)
- Archived annotation filtering
- Default config values

### test_preprocessing_basic.py
- Config validation (defaults, clamping, None-filling)
- Target sample-rate computation (enabled/disabled/zero gating)
- Butterworth bandpass / lowpass / highpass / bandstop
- Nyquist safety clipping
- High <= low cutoff handling
- Zero-phase vs causal filtering
- Full preprocess_audio_file pipeline (44100→4000 Hz resample)
- Metadata completeness
- Summary string generation

### test_spectrogram_basic.py
- STFT output shape and frequency axis
- f_max Nyquist clamping
- display decimation (freq & time reduction)
- palette generation (256×3, Heatmap & Grayscale)
- colourisation (uint8 RGB output, custom levels, all-NaN input)
- End-to-end compute_spectrogram_display pipeline

### test_hsmm_basic.py
- hop_sec estimation (from sr+hop, from times, default)
- breath-cycle estimation (from phase starts, default)
- 2-state & 3-state log transition matrix construction
- Viterbi: dominant-state detection, state-switch behaviour, 3-state, numerical stability
- Duration priors (required keys, positive dmin, single-class fallback)
- State sequence → (start, end) segment conversion
- min_dur_sec filtering

### test_features_basic.py
- Feature inventory (7 time + 30 spectral + 19 COR = 56)
- frame_signal output shape
- All 56 feature keys present
- Feature array lengths match time axis
- No NaN / inf
- Feature matrix shape (T, 2D) with smoothed copies
- 0-1 normalisation (range check, constant-input)

## Adding New Tests

When adding a new module, follow this pattern:

1. Write pure-function replicas of the legacy logic in the test file.
2. Verify tests pass against the replicas.
3. Extract the logic into a `respanno/` module.
4. Rewrite tests to import from the module.
5. Verify tests still pass.
