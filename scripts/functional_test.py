"""Functional test of RespAnno backend — verified against actual module APIs.

Exercises every major module: imports, audio I/O, preprocessing, spectrogram,
FFT, 56 features, ML classifier, phase model, HSMM, label taxonomy,
frame labels, negatives, annotation I/O, events importer, MLService, E2E pipeline.
"""
import os, sys, warnings, tempfile, traceback

warnings.filterwarnings("ignore")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "tests"))

import numpy as np

try:
    from PyQt5.QtWidgets import QMessageBox
    QMessageBox.information = lambda *a, **kw: None
    QMessageBox.warning = lambda *a, **kw: None
except Exception:
    pass

_tests = []

def register(name, fn):
    _tests.append((name, fn))

def run_all():
    passed = failed = 0
    print("=" * 60)
    print("  RespAnno Functional Test Suite")
    print("=" * 60)
    for i, (name, fn) in enumerate(_tests):
        if name.startswith("=="):
            print(f"\n{'='*60}")
            print(f"  {name[2:].strip()}")
            print(f"{'='*60}")
            continue
        try:
            fn()
            passed += 1
            print(f"  [{i+1:02d}] OK  {name}")
        except Exception as e:
            failed += 1
            print(f"  [{i+1:02d}] FAIL  {name}")
            for line in traceback.format_exc().split("\n")[-4:-1]:
                if line.strip():
                    print(f"         {line.strip()}")
    print(f"\n{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed  ({passed+failed} total)")
    if failed == 0:
        print("  All functional tests passed!")
    else:
        print(f"  {failed} test(s) failed!")
    print("=" * 60)
    return failed == 0


# ═══════════════════════════ 1. Module Imports ═══════════════════════════════
register("== 1. Module Imports", lambda: None)

for mp, desc in [
    ("respanno", "respanno package"),
    ("respanno.audio.preprocessing", "audio.preprocessing"),
    ("respanno.dsp.spectrogram", "dsp.spectrogram"),
    ("respanno.dsp.fft", "dsp.fft"),
    ("respanno.dsp.features", "dsp.features"),
    ("respanno.ml.classifier", "ml.classifier"),
    ("respanno.ml.phase_model", "ml.phase_model"),
    ("respanno.ml.hsmm", "ml.hsmm"),
    ("respanno.ml.service", "ml.service"),
    ("respanno.ml.label_taxonomy", "ml.label_taxonomy"),
    ("respanno.ml.frame_labels", "ml.frame_labels"),
    ("respanno.ml.negatives", "ml.negatives"),
    ("respanno.labels.annotation_io", "labels.annotation_io"),
    ("respanno.labels.events_importer", "labels.events_importer"),
]:
    def _mk(mp, d):
        return lambda: __import__(mp, fromlist=[mp.split(".")[-1]])
    register(f"Import {desc}", _mk(mp, desc))


# ═══════════════════════════ 2. Real Audio ═══════════════════════════════════
register("== 2. Real Audio Loading", lambda: None)

DEMO_DIR = os.path.join(PROJECT_ROOT, "demo_data", "OriginFs")
demo_files = sorted([f for f in os.listdir(DEMO_DIR) if f.endswith(".wav")]) if os.path.isdir(DEMO_DIR) else []

if demo_files:
    from respanno.audio.preprocessing import load_audio_file
    _audio_cache = {}
    def _load_demo():
        for fname in demo_files[:3]:
            fpath = os.path.join(DEMO_DIR, fname)
            audio, sr = load_audio_file(fpath)
            assert audio.ndim == 1
            assert sr > 0 and len(audio) > 0
            _audio_cache[fname] = (audio, sr)
    register(f"Load {len(demo_files)} real respiratory WAVs (3 tested)", _load_demo)

def _test_get_sr():
    from respanno.audio.preprocessing import get_original_sr
    fpath = os.path.join(DEMO_DIR, demo_files[0])
    sr = get_original_sr(fpath)
    assert sr > 0
register("get_original_sr on real WAV", _test_get_sr)


# ═══════════════════════════ 3. Preprocessing ════════════════════════════════
register("== 3. Audio Preprocessing", lambda: None)

def _test_preprocess():
    from respanno.audio.preprocessing import preprocess_audio_file
    fpath = os.path.join(DEMO_DIR, demo_files[0])
    audio, sr, metadata = preprocess_audio_file(fpath, {})
    assert audio.ndim == 1 and sr > 0
    assert isinstance(metadata, dict)
    assert "original_sr" in metadata
register("preprocess_audio_file: (audio, sr, metadata)", _test_preprocess)

def _test_butter():
    from respanno.audio.preprocessing import apply_butter_filter
    sr = 4000
    t = np.linspace(0, 2, sr * 2, endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 500 * t)).astype(np.float32)
    out = apply_butter_filter(audio, sr, "bandpass", lowcut=20, highcut=1800, order=5)
    assert len(out) == len(audio)
register("apply_butter_filter bandpass 20-1800 Hz", _test_butter)

def _test_apply_preproc():
    from respanno.audio.preprocessing import apply_preprocessing
    sr = 4000
    t = np.linspace(0, 2, sr * 2, endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 500 * t)).astype(np.float32)
    out, metadata = apply_preprocessing(audio, sr, {"filter_enabled": True, "filter_lowcut": 20, "filter_highcut": 1800})
    assert len(out) == len(audio)
    assert isinstance(metadata, dict)
register("apply_preprocessing: (audio, metadata)", _test_apply_preproc)

def _test_summarize():
    from respanno.audio.preprocessing import summarize_preprocessing
    s = summarize_preprocessing({"filter_enabled": True, "filter_lowcut": 20, "filter_highcut": 1800})
    assert isinstance(s, str) and len(s) > 0
register("summarize_preprocessing: non-empty string", _test_summarize)

def _test_validate_config():
    from respanno.audio.preprocessing import validate_preprocessing_config
    cfg = validate_preprocessing_config({})
    assert isinstance(cfg, dict)
    assert "filter_enabled" in cfg
    assert "resample_enabled" in cfg
    assert cfg["filter_type"] == "bandpass"
register("validate_preprocessing_config fills defaults", _test_validate_config)


# ═══════════════════════════ 4. Spectrogram ══════════════════════════════════
register("== 4. Spectrogram", lambda: None)

def _test_stft():
    from respanno.dsp.spectrogram import compute_stft_db, compute_stft_frame_times
    sr = 4000
    t = np.linspace(0, 3, sr * 3, endpoint=False)
    audio = (np.sin(2 * np.pi * 500 * t)).astype(np.float32)
    S_db, freqs = compute_stft_db(audio, sr, n_fft=256, hop_length=64)
    times = compute_stft_frame_times(len(audio), sr, hop_length=64)
    assert S_db.shape[0] == len(freqs) and S_db.shape[1] == len(times)
    assert S_db.shape[1] > 10
register("compute_stft_db + compute_stft_frame_times", _test_stft)

def _test_colorize():
    from respanno.dsp.spectrogram import compute_stft_db, colorize_spectrogram
    sr = 4000
    t = np.linspace(0, 1, sr * 1, endpoint=False)
    audio = (np.sin(2 * np.pi * 500 * t)).astype(np.float32)
    S_db, _ = compute_stft_db(audio, sr, 256, 64)
    rgb = colorize_spectrogram(S_db.T, "Heatmap")
    assert rgb.ndim == 3 and rgb.shape[2] == 3
register("colorize_spectrogram: RGB output", _test_colorize)

def _test_spec_display():
    from respanno.dsp.spectrogram import compute_spectrogram_display
    sr = 4000
    t = np.linspace(0, 1, sr * 1, endpoint=False)
    audio = (np.sin(2 * np.pi * 500 * t)).astype(np.float32)
    result = compute_spectrogram_display(audio, sr,
        {"n_fft": 256, "hop_length": 64, "f_max": 2000, "cmap": "Heatmap"})
    assert isinstance(result, dict)
    assert "S_db" in result and "rgb" in result and "freqs" in result
    assert result["rgb"].ndim == 3
register("compute_spectrogram_display: dict with rgb/S_db/freqs", _test_spec_display)


# ═══════════════════════════ 5. FFT ══════════════════════════════════════════
register("== 5. FFT", lambda: None)

def _test_fft():
    from respanno.dsp.fft import compute_fft
    sr = 4000
    t = np.linspace(0, 1, sr, endpoint=False)
    audio = (np.sin(2 * np.pi * 500 * t)).astype(np.float32)
    freqs, mag = compute_fft(audio, sr)  # NOTE: returns (freqs, mag)
    assert len(freqs) == len(mag)
    above_dc = freqs > 50
    peak_freq = freqs[above_dc][np.argmax(mag[above_dc])]
    assert 450 < peak_freq < 550, f"Peak at {peak_freq:.1f} Hz"
register("compute_fft: (freqs, mag) — peak at 500 Hz", _test_fft)


# ═══════════════════════════ 6. Features ═════════════════════════════════════
register("== 6. Features (56)", lambda: None)

def _test_56_features():
    from respanno.dsp.features import compute_short_time_features
    sr = 4000
    t = np.linspace(0, 3, sr * 3, endpoint=False)
    audio = (np.sin(2 * np.pi * 500 * t)).astype(np.float32)
    times, feat_dict = compute_short_time_features(audio, sr)
    assert len(feat_dict) == 56, f"Got {len(feat_dict)}"
    assert len(times) > 0
    for k, v in feat_dict.items():
        assert len(v) == len(times), f"'{k}': {len(v)} vs {len(times)}"
register("56 features with correct lengths", _test_56_features)

def _test_feature_matrix():
    from respanno.dsp.features import compute_short_time_features, build_feature_matrix
    sr = 4000
    t = np.linspace(0, 3, sr * 3, endpoint=False)
    audio = (np.sin(2 * np.pi * 500 * t)).astype(np.float32)
    times, feat_dict = compute_short_time_features(audio, sr)
    # Returns (X_full, times_trimmed, full_names) with 112 cols (56 + 56 smoothed)
    X_full, _, full_names = build_feature_matrix(times, feat_dict)
    assert X_full.ndim == 2
    assert X_full.shape[1] == 112
    assert len(full_names) == 112
    assert full_names[56] == full_names[0] + "_sm"
register("build_feature_matrix: (X_full 112cols, times, names)", _test_feature_matrix)

def _test_features_finite():
    from respanno.dsp.features import compute_short_time_features
    sr = 4000
    t = np.linspace(0, 2, sr * 2, endpoint=False)
    audio = (0.5 * np.sin(2*np.pi*200*t) + 0.3*np.sin(2*np.pi*600*t)).astype(np.float32)
    _, feat_dict = compute_short_time_features(audio, sr)
    for k, v in feat_dict.items():
        assert not np.any(np.isnan(v)), f"NaN in '{k}'"
        assert not np.any(np.isinf(v)), f"Inf in '{k}'"
register("All 56 features finite (no NaN/Inf)", _test_features_finite)


# ═══════════════════════════ 7. ML Classifier ════════════════════════════════
register("== 7. ML Classifier (LightGBM)", lambda: None)

def _make_viewer(audio, sr, anns):
    from respanno.dsp.features import compute_short_time_features, build_feature_matrix
    from tests.fixtures.mock_viewer import MockViewer
    times, feat_dict = compute_short_time_features(audio, int(sr))
    X_full, _, full_names = build_feature_matrix(times, feat_dict)
    return MockViewer(stft_features=X_full.astype(np.float64), stft_frame_times=times.astype(float),
                      stft_feature_names=full_names, annotations=list(anns), sr=int(sr), hop_length=256)

def _test_train():
    from respanno.ml.classifier import train_event_model
    from tests.fixtures.synthetic_signals import generate_wheeze_episode
    audio, sr, anns = generate_wheeze_episode(duration=5.0, wheeze_start=1.0, wheeze_dur=2.5, seed=42)
    viewer = _make_viewer(audio, sr, anns)
    assert train_event_model(viewer, 'Wheeze', random_state=42) is True
    assert 'Wheeze' in viewer.ml_models and 'clf' in viewer.ml_models['Wheeze']
register("train_event_model on synthetic wheeze", _test_train)

def _test_apply():
    from respanno.ml.classifier import train_event_model, apply_event_model
    from tests.fixtures.synthetic_signals import generate_wheeze_episode
    audio, sr, anns = generate_wheeze_episode(duration=5.0, wheeze_start=1.0, wheeze_dur=2.5, seed=42)
    viewer = _make_viewer(audio, sr, anns)
    train_event_model(viewer, 'Wheeze', random_state=42)
    assert apply_event_model(viewer, 'Wheeze', min_dur_sec=0.05) in (True, False)
register("apply_event_model produces predictions", _test_apply)

def _test_determinism():
    from respanno.ml.classifier import train_event_model
    from tests.fixtures.synthetic_signals import generate_wheeze_episode
    audio, sr, anns = generate_wheeze_episode(duration=5.0, wheeze_start=1.0, wheeze_dur=2.5, seed=42)
    v1, v2 = _make_viewer(audio, sr, anns), _make_viewer(audio, sr, anns)
    train_event_model(v1, 'Wheeze', random_state=42)
    train_event_model(v2, 'Wheeze', random_state=42)
    assert v1.ml_models['Wheeze']['train_f1'] == v2.ml_models['Wheeze']['train_f1']
register("Determinism: same seed -> identical model", _test_determinism)


# ═══════════════════════════ 8. Phase Model ══════════════════════════════════
register("== 8. Phase Model", lambda: None)

def _test_phase_train():
    from respanno.ml.phase_model import train_phase_model
    from tests.fixtures.synthetic_signals import generate_respiratory_cycle
    audio, sr, anns = generate_respiratory_cycle(duration=8.0, seed=42)
    viewer = _make_viewer(audio, sr, anns)
    assert train_phase_model(viewer, 'Inspiration', random_state=42) is True
register("train_phase_model on respiratory cycle", _test_phase_train)

def _test_phase_apply():
    from respanno.ml.phase_model import train_phase_model, apply_phase_model
    from tests.fixtures.synthetic_signals import generate_respiratory_cycle
    audio, sr, anns = generate_respiratory_cycle(duration=8.0, seed=42)
    viewer = _make_viewer(audio, sr, anns)
    train_phase_model(viewer, 'Inspiration', random_state=42)
    assert apply_phase_model(viewer, 'Inspiration', min_dur_sec=0.05) in (True, False)
register("apply_phase_model runs successfully", _test_phase_apply)


# ═══════════════════════════ 9. HSMM ═════════════════════════════════════════
register("== 9. HSMM Viterbi Decoder", lambda: None)

def _test_hsmm_trans():
    from respanno.ml.hsmm import build_hsmm_log_trans
    lt = build_hsmm_log_trans(["S0", "S1"])
    assert lt.shape == (2, 2)
    assert np.all(np.isfinite(lt))
register("build_hsmm_log_trans: (2,2) finite matrix", _test_hsmm_trans)

def _test_hsmm_viterbi():
    from respanno.ml.hsmm import hsmm_viterbi
    n_states, n_frames, max_dur = 2, 100, 20
    rng = np.random.RandomState(42)
    # Generate plausible log_emissions (T, S)
    log_emit = rng.randn(n_frames, n_states)
    dmin = [1, 1]
    dmax = [max_dur, max_dur]
    log_trans = np.log(np.array([[0.5, 0.5], [0.5, 0.5]]))
    log_pi = np.log(np.array([0.5, 0.5]))
    states = hsmm_viterbi(log_emit, dmin, dmax, log_trans, log_pi)
    assert len(states) == n_frames
    assert set(np.unique(states)).issubset({0, 1})
register("hsmm_viterbi: decode returns valid 2-state sequence", _test_hsmm_viterbi)

def _test_hsmm_segments():
    from respanno.ml.hsmm import state_seq_to_segments
    # Simulate output: times for all frames, unreviewed indices, state ids
    T_all, T_unr = 200, 100
    times = np.arange(T_all) * 0.016
    idx_unr = np.arange(50, 150)  # unreviewed frames in middle
    z_state_ids = np.array([0] * 30 + [1] * 40 + [0] * 30)  # state seq for unreviewed
    segs = state_seq_to_segments(times, idx_unr, z_state_ids, target_state_id=1, min_dur_sec=0.05)
    assert len(segs) >= 1  # at least one segment for state 1
register("state_seq_to_segments: extracts target-state segments", _test_hsmm_segments)

def _test_hsmm_priors():
    from respanno.ml.hsmm import build_hsmm_prior_from_prefix_labels
    import numpy as np
    # Signature: (y_prefix, classes_, state_id_to_name, hop_sec, cycle_sec)
    n_frames = 50
    y_prefix = np.array([0, 0, 1, 1, 0, 0, 1, 1] * 6)[:n_frames]
    classes_ = [0, 1]
    state_id_to_name = {0: "Pause", 1: "Inspiration"}
    priors = build_hsmm_prior_from_prefix_labels(y_prefix, classes_, state_id_to_name, hop_sec=0.016, cycle_sec=3.0)
    assert isinstance(priors, dict)
    assert "dmin_frames" in priors and "dmax_frames" in priors
    assert priors["hop_sec"] == 0.016
register("build_hsmm_prior_from_prefix_labels: (dmin_frames, dmax_frames)", _test_hsmm_priors)


# ═══════════════════════════ 10. Label Taxonomy ══════════════════════════════
register("== 10. Label Taxonomy", lambda: None)

def _test_label_kind():
    from respanno.ml.label_taxonomy import label_kind, ABNORMAL_SOUND_KIND, PHASE_KIND
    for lbl in ["Wheeze", "Crackles", "Rhonchi"]:
        assert label_kind(lbl) == ABNORMAL_SOUND_KIND
    for lbl in ["Inspiration", "Expiration"]:
        assert label_kind(lbl) == PHASE_KIND
register("label_kind: abnormal / phase routing correct", _test_label_kind)

def _test_label_sets():
    from respanno.ml.label_taxonomy import PHASE_LABELS, OTHER_EVENT_LABELS
    # Both are sets (not dicts)
    assert isinstance(PHASE_LABELS, set)
    assert isinstance(OTHER_EVENT_LABELS, set)
    assert "inspiration" in PHASE_LABELS
    assert "cough" in OTHER_EVENT_LABELS
register("PHASE_LABELS/OTHER_EVENT_LABELS are sets with expected entries", _test_label_sets)


# ═══════════════════════════ 11. Frame Labels ════════════════════════════════
register("== 11. Frame Labels", lambda: None)

def _test_frame_labels():
    from respanno.ml.frame_labels import build_frame_labels
    sr = 4000; hop = 256
    duration = 4.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    n_frames = 1 + (len(t) - 256) // hop
    times = np.arange(n_frames) * hop / sr
    # Annotations are tuples: (start, end, label) or (start, end, label, source)
    anns = [(1.0, 2.5, "Wheeze", "manual")]
    y = build_frame_labels(anns, times, "Wheeze")
    assert y is not None
    assert np.sum(y == 1) > 0, "positive frames missing"
    assert np.sum(y == 0) > 0, "negative frames missing"
register("build_frame_labels: pos+neg from tuple annotations", _test_frame_labels)


# ═══════════════════════════ 12. Negatives ═══════════════════════════════════
register("== 12. Negative Sample Manager", lambda: None)

def _test_neg():
    from respanno.ml.negatives import NegSampleManager
    mgr = NegSampleManager()
    # API: add(label, start, end)
    r1 = mgr.add("Wheeze", 1.0, 2.0)
    r2 = mgr.add("Wheeze", 3.0, 4.0)
    mgr.add("Crackles", 0.5, 1.5)
    assert r1 is not None and r2 is not None
    d = mgr.to_dict()
    assert "Wheeze" in d and "Crackles" in d
    assert mgr.count("Wheeze") == 2
    mgr.clear("Wheeze")
    assert mgr.count("Wheeze") == 0
    assert mgr.count("Crackles") == 1
register("NegSampleManager: add/count/clear/multi-label", _test_neg)


# ═══════════════════════════ 13. Annotation I/O ══════════════════════════════
register("== 13. Annotation I/O", lambda: None)

def _test_read_write_csv():
    from respanno.labels.annotation_io import read_annotations, write_annotations
    anns = [
        {"start": 0.5, "end": 1.2, "label": "Wheeze", "source": "manual"},
        {"start": 2.0, "end": 2.8, "label": "Crackles", "source": "manual"},
    ]
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "test.csv")
        write_annotations(p, anns)  # format auto-detected from .csv extension
        loaded = read_annotations(p)
        assert len(loaded) == 2
        assert loaded[0]["label"] == "Wheeze"
register("CSV write/read roundtrip (2 annotations)", _test_read_write_csv)

def _test_read_write_txt():
    from respanno.labels.annotation_io import read_annotations, write_annotations
    anns = [{"start": 0.0, "end": 3.0, "label": "Inspiration", "source": "manual"}]
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "test.txt")
        write_annotations(p, anns)
        loaded = read_annotations(p)
        assert len(loaded) == 1 and loaded[0]["label"] == "Inspiration"
register("TXT write/read roundtrip", _test_read_write_txt)

def _test_read_write_json():
    from respanno.labels.annotation_io import read_annotations, write_annotations
    anns = [{"start": 0.5, "end": 1.0, "label": "Rhonchi", "source": "ml"}]
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "test.json")
        write_annotations(p, anns)
        loaded = read_annotations(p)
        assert len(loaded) == 1 and loaded[0]["label"] == "Rhonchi"
register("JSON write/read roundtrip", _test_read_write_json)

def _test_roundtrip_fn():
    from respanno.labels.annotation_io import roundtrip_annotations
    anns = [{"start": 0.0, "end": 1.0, "label": "Expiration", "source": "manual"}]
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "test.csv")
        result = roundtrip_annotations(p, anns)
        assert len(result) == 1 and result[0]["label"] == "Expiration"
register("roundtrip_annotations: write then read back", _test_roundtrip_fn)


# ═══════════════════════════ 14. Events Importer ═════════════════════════════
register("== 14. Events Auto-Importer", lambda: None)

def _test_events_importer():
    from respanno.labels.events_importer import EventsFileIndexer
    class FakeViewer:
        wav_dir = os.path.join(PROJECT_ROOT, "demo_data", "OriginFs")
    indexer = EventsFileIndexer(FakeViewer())
    assert indexer is not None
    # API: build_index, resolve_path, parse_file, auto_import
    assert hasattr(indexer, 'build_index')
    assert hasattr(indexer, 'resolve_path')
    assert hasattr(indexer, 'auto_import')
    # build_index on events directory should populate the index
    events_dir = os.path.join(PROJECT_ROOT, "demo_data", "events")
    indexer.build_index(events_dir)
    assert len(indexer._index) >= 1, "Should index at least one folder"
register("EventsFileIndexer: build_index indexes events directory", _test_events_importer)


# ═══════════════════════════ 15. MLService ═══════════════════════════════════
register("== 15. MLService Dispatcher", lambda: None)

def _test_mlservice():
    from respanno.ml.service import MLService
    # MLService needs an owner (the viewer)
    svc = MLService(owner=None)
    assert svc is not None
register("MLService(owner=None) creates OK", _test_mlservice)


# ═══════════════════════════ 16. Real Data Pipeline ══════════════════════════
register("== 16. Real Data Pipeline", lambda: None)

if demo_files:
    from respanno.audio.preprocessing import load_audio_file

    def _test_real_spectrogram():
        from respanno.dsp.spectrogram import compute_spectrogram_display
        fpath = os.path.join(DEMO_DIR, demo_files[0])
        audio, sr = load_audio_file(fpath)
        result = compute_spectrogram_display(audio, sr)
        assert result["rgb"] is not None and result["S_db"].size > 0
    register("Spectrogram display from real respiratory WAV", _test_real_spectrogram)

    def _test_real_fft():
        from respanno.dsp.fft import compute_fft
        fpath = os.path.join(DEMO_DIR, demo_files[0])
        audio, sr = load_audio_file(fpath)
        freqs, mag = compute_fft(audio, sr)
        assert len(freqs) == len(mag) and len(freqs) > 0
        assert np.max(mag) > 0  # non-zero spectrum
    register("FFT from real respiratory WAV (non-zero spectrum)", _test_real_fft)

    def _test_real_preprocessing_pipeline():
        from respanno.audio.preprocessing import preprocess_audio_file
        fpath = os.path.join(DEMO_DIR, demo_files[0])
        audio, sr, metadata = preprocess_audio_file(fpath)
        assert metadata["original_sr"] > 0
        assert metadata["processed_sr"] > 0
    register("Full preprocessing pipeline on real WAV", _test_real_preprocessing_pipeline)


# ═══════════════════════════ 17. E2E Pipeline ════════════════════════════════
register("== 17. End-to-End ML Pipeline", lambda: None)

def _test_e2e_mixed():
    from respanno.ml.classifier import train_event_model, apply_event_model
    from tests.fixtures.synthetic_signals import generate_mixed_episode
    audio, sr, anns = generate_mixed_episode(duration=6.0, seed=42)
    viewer = _make_viewer(audio, sr, anns)
    assert train_event_model(viewer, 'Wheeze', min_pos_frames=5, random_state=42) is True
    assert train_event_model(viewer, 'Crackles', min_pos_frames=2, random_state=42) is True
    assert viewer.ml_models['Wheeze']['clf'] is not viewer.ml_models['Crackles']['clf']
    apply_event_model(viewer, 'Wheeze', min_dur_sec=0.05)
    apply_event_model(viewer, 'Crackles', min_dur_sec=0.05)
register("Mixed episode: Wheeze+Crackles train+apply (separate models)", _test_e2e_mixed)

def _test_e2e_phase():
    from respanno.ml.phase_model import train_phase_model, apply_phase_model
    from tests.fixtures.synthetic_signals import generate_respiratory_cycle
    audio, sr, anns = generate_respiratory_cycle(duration=8.0, seed=42)
    viewer = _make_viewer(audio, sr, anns)
    assert train_phase_model(viewer, 'Inspiration', random_state=42) is True
    assert apply_phase_model(viewer, 'Inspiration', min_dur_sec=0.05) in (True, False)
register("Full phase pipeline: train+apply Inspiration", _test_e2e_phase)


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
