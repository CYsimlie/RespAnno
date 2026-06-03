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