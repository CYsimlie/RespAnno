"""Roundtrip workflow tests: WAV → preprocess → annotate → export → re-import.

Verifies that annotations survive a full write-then-read cycle in all
supported formats, and that source provenance is preserved.
"""
import json
import os
import tempfile
import numpy as np
import pytest
import scipy.io.wavfile as wavfile
from respanno.audio.preprocessing import preprocess_audio_file
from respanno.labels.annotation_io import normalize_annotation, read_annotations, write_annotations
from tests.fixtures.synthetic_signals import generate_tone

def _save_wav(path, audio, sr):
    wavfile.write(path, int(sr), audio.astype(np.float32))

class TestCSVRoundtrip:

    def test_wav_to_annotations_csv_roundtrip(self):
        """Generate WAV → load+preprocess → annotate → export CSV → re-import."""
        (audio, sr, _) = generate_tone(freq=200.0, sr=4000, duration=2.0, seed=42)
        with tempfile.TemporaryDirectory() as d:
            wav_path = os.path.join(d, 'test.wav')
            csv_path = os.path.join(d, 'test_annotations.csv')
            _save_wav(wav_path, audio, sr)
            (processed, loaded_sr, meta) = preprocess_audio_file(wav_path)
            assert loaded_sr == sr
            assert meta['processed_sr'] == sr
            original = [(0.5, 1.2, 'Wheeze', 'manual'), (1.5, 1.8, 'Crackles', 'ml')]
            anns = [normalize_annotation(a) for a in original]
            write_annotations(csv_path, anns)
            loaded = read_annotations(csv_path)
            assert len(loaded) == 2
            for (o, l) in zip(original, loaded):
                assert o[0] == pytest.approx(l['start'])
                assert o[1] == pytest.approx(l['end'])
                assert o[2] == l['label']
                src = o[3] if len(o) >= 4 else 'manual'
                assert src == l['source']

    def test_json_roundtrip(self):
        """Full roundtrip with JSON format."""
        original = [(0.2, 0.8, 'Inspiration', 'manual'), (0.9, 1.5, 'Expiration', 'auto_accepted'), (1.6, 2.0, 'Pause', 'merged')]
        anns = [normalize_annotation(a) for a in original]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'annotations.json')
            write_annotations(path, anns)
            loaded = read_annotations(path)
            assert len(loaded) == 3
            for (o, l) in zip(original, loaded):
                src = o[3] if len(o) >= 4 else 'manual'
                assert src == l['source']

class TestEmptyPipeline:

    def test_no_annotations_roundtrip(self):
        """Empty annotation list roundtrips without error."""
        with tempfile.TemporaryDirectory() as d:
            csv_path = os.path.join(d, 'empty.csv')
            write_annotations(csv_path, [])
            loaded = read_annotations(csv_path)
            assert loaded == []

class TestSourceProvenanceRoundtrip:
    """Every valid source type must survive the roundtrip."""

    def test_all_sources_survive(self):
        """验证标注的 source 溯源信息正确保留。"""
        sources = ['manual', 'ml', 'auto_accepted', 'auto_edited', 'merged']
        original = [(i * 0.5, i * 0.5 + 0.4, f'L{i}', src) for (i, src) in enumerate(sources)]
        anns = [normalize_annotation(a) for a in original]
        with tempfile.TemporaryDirectory() as d:
            for (fmt, ext) in [('csv', '.csv'), ('json', '.json')]:
                path = os.path.join(d, f'test{ext}')
                write_annotations(path, anns)
                loaded = read_annotations(path)
                assert len(loaded) == len(sources)
                for (a, b) in zip(original, loaded):
                    src_a = a[3] if len(a) >= 4 else 'manual'
                    assert b['source'] == src_a, f'format={fmt}: source mismatch'


# ═══════════════════════════════════════════════════════════════════════════
# Extended roundtrip tests (SoftwareX review readiness)
# ═══════════════════════════════════════════════════════════════════════════

class TestRealAnnotationWorkflow:
    """Full pipeline: synthetic WAV with annotations → preprocess → export → re-import → verify values."""

    def _build_wav_with_annotations(self, d, name, audio, sr, annotations):
        """Helper: write WAV, preprocess it, write annotations, return metadata."""
        wav_path = os.path.join(d, name)
        _save_wav(wav_path, audio, sr)
        processed, loaded_sr, meta = preprocess_audio_file(wav_path)
        anns = [normalize_annotation(a) for a in annotations]
        return wav_path, anns, meta

    def test_wheeze_annotation_roundtrip_csv(self):
        """WAV with wheeze → export CSV → re-import → wheeze interval preserved."""
        from tests.fixtures.synthetic_signals import generate_wheeze_episode
        audio, sr, annotations = generate_wheeze_episode(duration=4.0, wheeze_start=1.0, wheeze_dur=1.5, seed=42)
        with tempfile.TemporaryDirectory() as d:
            wav_path, anns, meta = self._build_wav_with_annotations(d, 'test.wav', audio, sr, annotations)
            csv_path = os.path.join(d, 'events.csv')
            write_annotations(csv_path, anns)
            loaded = read_annotations(csv_path)
            assert len(loaded) == 1
            assert loaded[0]['label'] == 'Wheeze'
            assert loaded[0]['start'] == pytest.approx(1.0, abs=0.02)
            assert loaded[0]['end'] == pytest.approx(2.5, abs=0.02)

    def test_respiratory_cycle_roundtrip_json(self):
        """Full respiratory cycle annotations survive JSON roundtrip."""
        from tests.fixtures.synthetic_signals import generate_respiratory_cycle
        dur = 8.0
        audio, sr, annotations = generate_respiratory_cycle(duration=dur, seed=42)
        with tempfile.TemporaryDirectory() as d:
            wav_path, anns, meta = self._build_wav_with_annotations(d, 'cycle.wav', audio, sr, annotations)
            json_path = os.path.join(d, 'cycle.json')
            write_annotations(json_path, anns)
            loaded = read_annotations(json_path)
            # Should have multiple phase annotations + adventitious sounds
            assert len(loaded) >= 6, f"Expected >=6 annotations in respiratory cycle, got {len(loaded)}"
            labels = {a['label'] for a in loaded}
            assert 'Inspiration' in labels
            assert 'Expiration' in labels
            # Verify all start times are within signal bounds
            for a in loaded:
                assert 0 <= a['start'] < a['end'] <= dur, \
                    f"annotation out of bounds: {a}"

    def test_multi_label_roundtrip_txt(self):
        """Mixed episode (wheeze + crackles) survives TXT roundtrip."""
        from tests.fixtures.synthetic_signals import generate_mixed_episode
        audio, sr, annotations = generate_mixed_episode(duration=6.0, seed=42)
        with tempfile.TemporaryDirectory() as d:
            wav_path, anns, meta = self._build_wav_with_annotations(d, 'mixed.wav', audio, sr, annotations)
            txt_path = os.path.join(d, 'mixed.txt')
            write_annotations(txt_path, anns)
            loaded = read_annotations(txt_path)
            assert len(loaded) >= 4
            labels = {a['label'] for a in loaded}
            assert 'Wheeze' in labels
            assert 'Crackles' in labels

    def test_numeric_precision_preserved_csv(self):
        """Timestamps with 3 decimal digits must survive CSV roundtrip exactly."""
        annotations = [
            (0.123, 0.456, 'Event', 'manual'),
            (1.789, 2.345, 'Phase', 'ml'),
            (3.001, 4.999, 'Test', 'auto_accepted'),
        ]
        anns = [normalize_annotation(a) for a in annotations]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'precision.csv')
            write_annotations(path, anns)
            loaded = read_annotations(path)
            assert len(loaded) == 3
            assert loaded[0]['start'] == pytest.approx(0.123, abs=0.001)
            assert loaded[0]['end'] == pytest.approx(0.456, abs=0.001)
            assert loaded[2]['start'] == pytest.approx(3.001, abs=0.001)
            assert loaded[2]['end'] == pytest.approx(4.999, abs=0.001)

    def test_empty_annotations_all_formats(self):
        """Empty annotation list roundtrips through all three formats."""
        with tempfile.TemporaryDirectory() as d:
            for ext in ['csv', 'txt', 'json']:
                path = os.path.join(d, f'empty.{ext}')
                write_annotations(path, [])
                loaded = read_annotations(path)
                assert loaded == [], f"Expected empty list for {ext}, got {loaded}"