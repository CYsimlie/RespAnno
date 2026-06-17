"""ICBHI 2017 Challenge dataset compatibility tests.

Validates that RespAnno can load and process annotation files following
the ICBHI 2017 Challenge naming and format conventions:

- WAV files named like ``101_1b1_Pr_sc_Meditron.wav``
- Annotation files use tab-separated ``start\\tend\\tlabel`` format
- Labels use lowercase (wheeze, crackle, inspiration, expiration)
"""
import os
import tempfile
import pytest
from respanno.labels.annotation_io import read_annotations
ICBHI_LABELS = {'wheeze', 'crackle', 'rhonchi', 'stridor', 'coarse crackle', 'fine crackle', 'inspiration', 'expiration', 'normal', 'artifact'}

def _write(path, content):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

class TestICBHIFormat:
    """ICBHI annotations are tab-separated with 3 columns: start, end, label."""

    def test_load_icbhi_tab_format(self):
        """Verify ICBHI 2017 Challenge dataset format compatibility."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'test_events.txt')
            _write(path, '2.122\t2.879\twheeze\n10.350\t10.807\twheeze\n')
            rows = read_annotations(path)
            assert len(rows) == 2
            assert rows[0]['start'] == pytest.approx(2.122)
            assert rows[0]['end'] == pytest.approx(2.879)
            assert rows[0]['label'] == 'wheeze'
            assert rows[0]['source'] == 'manual'

    def test_load_icbhi_with_all_labels(self):
        """All ICBHI label types should parse without error."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'test_events.txt')
            lines = []
            for (i, lab) in enumerate(sorted(ICBHI_LABELS)):
                lines.append(f'{i * 1.0}\t{i * 1.0 + 0.5}\t{lab}')
            _write(path, '\n'.join(lines))
            rows = read_annotations(path)
            assert len(rows) == len(ICBHI_LABELS)

    def test_icbhi_mixed_case_labels(self):
        """ICBHI labels are conventionally lowercase — verify case preserved."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'test_events.txt')
            _write(path, '1.0\t2.0\tWheeze\n3.0\t4.0\tCrackle\n')
            rows = read_annotations(path)
            assert rows[0]['label'] == 'Wheeze'
            assert rows[1]['label'] == 'Crackle'

class TestICBHIFileNaming:
    """ICBHI WAV files follow a specific naming pattern."""

    def test_icbhi_filename_pattern(self):
        """Verify the ICBHI naming pattern is recognised."""
        import re
        pattern = '^\\d{3}_\\db\\d_[A-Z][a-z]_(sc|mc)_\\w+\\.wav$'
        examples = ['101_1b1_Pr_sc_Meditron.wav', '102_1b1_Ar_sc_Meditron.wav', '103_2b2_Ar_mc_LittC2SE.wav', '104_1b1_Al_sc_Litt3200.wav']
        for name in examples:
            assert re.match(pattern, name), f"'{name}' does not match ICBHI pattern"

    def test_events_suffix_convention(self):
        """ICBHI events files are named `<wav_base>_events.txt`."""
        wav_name = '101_1b1_Pr_sc_Meditron.wav'
        base = os.path.splitext(wav_name)[0]
        events_name = f'{base}_events.txt'
        assert events_name == '101_1b1_Pr_sc_Meditron_events.txt'

    def test_our_demo_data_matches_icbhi_naming(self):
        """Demo data 4000Hz WAV files have matching _events and _example CSV pairs."""
        import glob
        demo_dir = os.path.join(os.path.dirname(__file__), '..', 'demo_data', '4000Hz')
        if not os.path.isdir(demo_dir):
            pytest.skip('demo_data/4000Hz directory not found')
        wavs = sorted(f for f in os.listdir(demo_dir) if f.endswith('.wav'))
        assert len(wavs) >= 3, f'should have at least 3 WAV files, found {len(wavs)}'
        for wav in wavs:
            base = os.path.splitext(wav)[0]
            # Each WAV should have a corresponding _events.csv
            events_file = os.path.join(os.path.dirname(demo_dir), 'events', f'{base}_events.csv')
            assert os.path.isfile(events_file), f'Missing events CSV for {wav}: {events_file}'
            # Each WAV should have a corresponding _example.csv
            example_file = os.path.join(demo_dir, f'{base}_example.csv')
            assert os.path.isfile(example_file), f'Missing example CSV for {wav}: {example_file}'