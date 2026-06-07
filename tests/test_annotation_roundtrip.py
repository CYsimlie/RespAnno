"""Tests for respanno.labels.annotation_io — import/export roundtrip.

All tests now exercise the extracted module directly (no QApplication needed).
"""
import json
import pytest
from respanno.labels.annotation_io import DEFAULT_LABEL_CONFIG, normalize_annotation, parse_annotation_row, read_annotations, read_annotations_csv, read_annotations_txt, read_annotations_json, write_annotations, write_annotations_csv, write_annotations_json, roundtrip_annotations

class TestNormalizeAnnotation:

    def test_3tuple(self):
        """Verify三元组format (start, end, label) 正确规范化为标准 dict，source default manual。"""
        ann = normalize_annotation((0.5, 1.2, 'wheeze'))
        assert ann == {'start': 0.5, 'end': 1.2, 'label': 'wheeze', 'source': 'manual'}

    def test_4tuple(self):
        """Verify四元组format (start, end, label, source) 正确规范化为标准 dict。"""
        ann = normalize_annotation((0.5, 1.2, 'wheeze', 'ml'))
        assert ann == {'start': 0.5, 'end': 1.2, 'label': 'wheeze', 'source': 'ml'}

    def test_dict(self):
        """Verify dict formatannotation直接透传，保留所有字段。"""
        ann = normalize_annotation({'start': 0.5, 'end': 1.2, 'label': 'wheeze'})
        assert ann == {'start': 0.5, 'end': 1.2, 'label': 'wheeze', 'source': 'manual'}

    def test_target_source_override(self):
        """Verifyannotation的 source 溯源info正ensure留。"""
        ann = normalize_annotation((0.5, 1.2, 'wheeze', 'ml'), target_source='auto_accepted')
        assert ann['source'] == 'auto_accepted'

    def test_none(self):
        """Verify空input或 None input时的行为。"""
        assert normalize_annotation(None) is None

    def test_end_equals_start(self):
        """Verify：normalize_annotation((1.0, 1.0, 'x')) is None。"""
        assert normalize_annotation((1.0, 1.0, 'x')) is None

    def test_end_before_start(self):
        """Verify：normalize_annotation((2.0, 1.0, 'x')) is None。"""
        assert normalize_annotation((2.0, 1.0, 'x')) is None

    def test_empty_label(self):
        """Verify空input或 None input时的行为。"""
        assert normalize_annotation((0.5, 1.2, '   ')) is None

class TestParseAnnotationRow:

    def test_default_columns_comma(self):
        """Verifydefaultparameter值符合预期。"""
        result = parse_annotation_row(['0.5000', '1.2000', 'wheeze'])
        assert result == {'start': 0.5, 'end': 1.2, 'label': 'wheeze', 'source': 'manual'}

    def test_default_columns_with_source(self):
        """Verifydefaultparameter值符合预期。"""
        result = parse_annotation_row(['0.5000', '1.2000', 'wheeze', 'ml'])
        assert result == {'start': 0.5, 'end': 1.2, 'label': 'wheeze', 'source': 'ml'}

    def test_custom_column_order(self):
        """Verify：result == {'start': 0.8, 'end': 2.5, 'label': 'Crackles', 'source': 'manual'}。"""
        cfg = {'start_col': 2, 'end_col': 3, 'label_col': 1, 'source_col': 4}
        parts = ['Crackles', '0.8', '2.5', 'manual']
        result = parse_annotation_row(parts, config=cfg)
        assert result == {'start': 0.8, 'end': 2.5, 'label': 'Crackles', 'source': 'manual'}

    def test_source_col_zero_disabled(self):
        """Verifyannotation的 source 溯源info正ensure留。"""
        cfg = {'source_col': 0}
        parts = ['0.5', '1.2', 'wheeze']
        result = parse_annotation_row(parts, config=cfg)
        assert result == {'start': 0.5, 'end': 1.2, 'label': 'wheeze', 'source': 'manual'}

    def test_invalid_numeric_returns_none(self):
        """Verify空input或 None input时的行为。"""
        assert parse_annotation_row(['abc', 'def', 'label']) is None

    def test_empty_label_returns_none(self):
        """Verify空input或 None input时的行为。"""
        assert parse_annotation_row(['0.5', '1.2', '']) is None

    def test_too_few_columns(self):
        """Verify：parse_annotation_row(['0.5']) is None。"""
        assert parse_annotation_row(['0.5']) is None

    def test_end_before_start_skipped(self):
        """Verify：parse_annotation_row(['3.0', '1.0', 'x']) is None。"""
        assert parse_annotation_row(['3.0', '1.0', 'x']) is None

def _write_text(path, lines):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

class TestReadAnnotationsCSV:

    def test_basic_comma(self, tmp_path):
        """Verify：len(anns) == 2。"""
        p = tmp_path / 't.csv'
        _write_text(p, ['start,end,label,source', '0.5000,1.2000,wheeze,manual', '2.0000,2.8000,Crackles,manual'])
        anns = read_annotations_csv(str(p))
        assert len(anns) == 2
        assert anns[0]['label'] == 'wheeze'

    def test_with_source(self, tmp_path):
        """Verifyannotation的 source 溯源info正ensure留。"""
        p = tmp_path / 't.csv'
        _write_text(p, ['start,end,label,source', '0.5,1.2,wheeze,ml'])
        anns = read_annotations_csv(str(p))
        assert anns[0]['source'] == 'ml'

    def test_default_source_manual(self, tmp_path):
        """Verifydefaultparameter值符合预期。"""
        p = tmp_path / 't.csv'
        _write_text(p, ['0.5,1.2,wheeze'])
        anns = read_annotations_csv(str(p))
        assert anns[0]['source'] == 'manual'

    def test_skip_header_with_config(self, tmp_path):
        """Verify：len(anns) == 1。"""
        p = tmp_path / 't.csv'
        _write_text(p, ['skip this line', '0.5,1.2,wheeze'])
        cfg = {'skip_header_lines': 1}
        anns = read_annotations_csv(str(p), config=cfg)
        assert len(anns) == 1
        assert anns[0]['label'] == 'wheeze'

    def test_invalid_row_skipped(self, tmp_path):
        """Verify：len(anns) == 1。"""
        p = tmp_path / 't.csv'
        _write_text(p, ['abc,def,xxx', '0.5,1.2,good'])
        anns = read_annotations_csv(str(p))
        assert len(anns) == 1
        assert anns[0]['label'] == 'good'

    def test_end_before_start_skipped(self, tmp_path):
        """Verify：len(anns) == 1。"""
        p = tmp_path / 't.csv'
        _write_text(p, ['3.0,1.0,bad', '0.5,1.2,good'])
        anns = read_annotations_csv(str(p))
        assert len(anns) == 1
        assert anns[0]['label'] == 'good'

class TestDelimiterVariants:

    def test_tab(self, tmp_path):
        """Verify：len(anns) == 1。"""
        p = tmp_path / 't.txt'
        _write_text(p, ['0.5\t1.2\twheeze'])
        anns = read_annotations_txt(str(p))
        assert len(anns) == 1

    def test_semicolon(self, tmp_path):
        """Verify：len(anns) == 1。"""
        p = tmp_path / 't.txt'
        _write_text(p, ['0.5;1.2;wheeze'])
        anns = read_annotations_txt(str(p))
        assert len(anns) == 1

    def test_space_explicit(self, tmp_path):
        """Verify：len(anns) == 1。"""
        p = tmp_path / 't.txt'
        _write_text(p, ['0.5 1.2 wheeze'])
        cfg = {'delimiter': 'space'}
        anns = read_annotations_txt(str(p), config=cfg)
        assert len(anns) == 1

    def test_custom_delimiter(self, tmp_path):
        """Verify：len(anns) == 1。"""
        p = tmp_path / 't.txt'
        _write_text(p, ['0.5|1.2|wheeze'])
        cfg = {'delimiter': 'custom', 'custom_delimiter': '|'}
        anns = read_annotations_txt(str(p), config=cfg)
        assert len(anns) == 1

    def test_custom_column_order(self, tmp_path):
        """Verify：anns[0] == {'start': 0.8, 'end': 2.5, 'label': 'Crackles', 'source': 'ml'}。"""
        p = tmp_path / 't.csv'
        _write_text(p, ['Crackles,0.8,2.5,ml'])
        cfg = {'start_col': 2, 'end_col': 3, 'label_col': 1, 'source_col': 4}
        anns = read_annotations_csv(str(p), config=cfg)
        assert anns[0] == {'start': 0.8, 'end': 2.5, 'label': 'Crackles', 'source': 'ml'}

class TestReadAnnotationsJSON:

    def test_array_of_dicts(self, tmp_path):
        """Verify：len(anns) == 1。"""
        p = tmp_path / 't.json'
        p.write_text(json.dumps([{'start': 0.5, 'end': 1.2, 'label': 'wheeze', 'source': 'ml'}]), encoding='utf-8')
        anns = read_annotations_json(str(p))
        assert len(anns) == 1
        assert anns[0]['label'] == 'wheeze'

    def test_nested_annotations_key(self, tmp_path):
        """Verify：len(anns) == 1。"""
        p = tmp_path / 't.json'
        p.write_text(json.dumps({'annotations': [{'start': 0.5, 'end': 1.2, 'label': 'Crackles'}]}), encoding='utf-8')
        anns = read_annotations_json(str(p))
        assert len(anns) == 1
        assert anns[0]['label'] == 'Crackles'

    def test_nested_events_key(self, tmp_path):
        """Verify：len(anns) == 1。"""
        p = tmp_path / 't.json'
        p.write_text(json.dumps({'events': [{'start': 0.5, 'end': 1.2, 'label': 'Speech'}]}), encoding='utf-8')
        anns = read_annotations_json(str(p))
        assert len(anns) == 1

    def test_missing_source_defaults_manual(self, tmp_path):
        """Verifydefaultparameter值符合预期。"""
        p = tmp_path / 't.json'
        p.write_text(json.dumps([{'start': 0.5, 'end': 1.2, 'label': 'wheeze'}]), encoding='utf-8')
        anns = read_annotations_json(str(p))
        assert anns[0]['source'] == 'manual'

    def test_fallback_keys(self, tmp_path):
        """Verify：anns[0]['label'] == 'Crackles'。"""
        p = tmp_path / 't.json'
        p.write_text(json.dumps([{'onset': 0.5, 'offset': 1.2, 'type': 'Crackles'}]), encoding='utf-8')
        anns = read_annotations_json(str(p))
        assert anns[0]['label'] == 'Crackles'

    def test_case_insensitive_match(self, tmp_path):
        """Verify source 字段size写不敏感匹配。"""
        p = tmp_path / 't.json'
        p.write_text(json.dumps([{'Start': 0.5, 'End': 1.2, 'Label': 'wheeze'}]), encoding='utf-8')
        anns = read_annotations_json(str(p))
        assert anns[0]['label'] == 'wheeze'

    def test_custom_keys(self, tmp_path):
        """Verify：anns[0]['label'] == 'Stridor'。"""
        p = tmp_path / 't.json'
        cfg = {'json_start_key': 'time_start', 'json_end_key': 'time_end', 'json_label_key': 'class_name'}
        p.write_text(json.dumps([{'time_start': 0.5, 'time_end': 1.2, 'class_name': 'Stridor'}]), encoding='utf-8')
        anns = read_annotations_json(str(p), config=cfg)
        assert anns[0]['label'] == 'Stridor'

    def test_array_rows(self, tmp_path):
        """Verify：len(anns) == 1。"""
        p = tmp_path / 't.json'
        p.write_text(json.dumps([[0.5, 1.2, 'wheeze']]), encoding='utf-8')
        anns = read_annotations_json(str(p))
        assert len(anns) == 1

    def test_end_before_start_skipped(self, tmp_path):
        """Verify：len(anns) == 1。"""
        p = tmp_path / 't.json'
        p.write_text(json.dumps([{'start': 3.0, 'end': 1.0, 'label': 'bad'}, {'start': 0.5, 'end': 1.2, 'label': 'good'}]), encoding='utf-8')
        anns = read_annotations_json(str(p))
        assert len(anns) == 1
        assert anns[0]['label'] == 'good'

class TestReadAnnotationsAuto:

    def test_detects_csv(self, tmp_path):
        """Verify：len(anns) == 1。"""
        p = tmp_path / 't.csv'
        _write_text(p, ['0.5,1.2,wheeze'])
        anns = read_annotations(str(p))
        assert len(anns) == 1

    def test_detects_json(self, tmp_path):
        """Verify：len(anns) == 1。"""
        p = tmp_path / 't.json'
        p.write_text(json.dumps([{'start': 0.5, 'end': 1.2, 'label': 'x'}]), encoding='utf-8')
        anns = read_annotations(str(p))
        assert len(anns) == 1

    def test_force_json_via_config(self, tmp_path):
        """Verify：len(anns) == 1。"""
        p = tmp_path / 't.csv'
        p.write_text(json.dumps([{'start': 0.5, 'end': 1.2, 'label': 'x'}]), encoding='utf-8')
        anns = read_annotations(str(p), config={'file_format': 'json'})
        assert len(anns) == 1

class TestWriteAnnotations:

    def test_csv_roundtrip(self, tmp_path):
        """Verify csv format的写→读往返data完全一致。"""
        anns = [{'start': 0.5, 'end': 1.2, 'label': 'wheeze', 'source': 'manual'}, {'start': 2.0, 'end': 2.8, 'label': 'Crackles', 'source': 'ml'}]
        p = tmp_path / 'out.csv'
        roundtripped = roundtrip_annotations(str(p), anns)
        assert len(roundtripped) == 2
        for (a, b) in zip(anns, roundtripped):
            assert abs(a['start'] - b['start']) < 0.001
            assert abs(a['end'] - b['end']) < 0.001
            assert a['label'] == b['label']
            assert a['source'] == b['source']

    def test_json_roundtrip(self, tmp_path):
        """Verify json format的写→读往返data完全一致。"""
        anns = [{'start': 0.5, 'end': 1.2, 'label': 'wheeze', 'source': 'manual'}]
        p = tmp_path / 'out.json'
        roundtripped = roundtrip_annotations(str(p), anns)
        assert len(roundtripped) == 1
        assert roundtripped[0]['label'] == 'wheeze'

    def test_archived_skipped_on_export(self, tmp_path):
        """Verify：len(read_back) == 1。"""
        anns = [{'start': 0.5, 'end': 1.2, 'label': 'wheeze', 'source': 'manual'}, {'start': 2.0, 'end': 2.8, 'label': 'old', 'source': 'archived'}]
        p = tmp_path / 'out.csv'
        write_annotations_csv(str(p), anns)
        read_back = read_annotations_csv(str(p))
        assert len(read_back) == 1
        assert read_back[0]['label'] == 'wheeze'

class TestDefaultConfig:

    def test_defaults_match_legacy(self):
        """Verifydefaultparameter值符合预期。"""
        assert DEFAULT_LABEL_CONFIG['start_col'] == 1
        assert DEFAULT_LABEL_CONFIG['end_col'] == 2
        assert DEFAULT_LABEL_CONFIG['label_col'] == 3
        assert DEFAULT_LABEL_CONFIG['source_col'] == 4
        assert DEFAULT_LABEL_CONFIG['delimiter'] == 'auto'
        assert DEFAULT_LABEL_CONFIG['skip_header_lines'] == 0