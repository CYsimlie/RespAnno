"""Tests for respanno.ml.label_taxonomy — label routing."""
import pytest
from respanno.ml.label_taxonomy import label_kind, PHASE_KIND, OTHER_EVENT_KIND, ABNORMAL_SOUND_KIND, PHASE_LABELS, OTHER_EVENT_LABELS

class TestPhaseRouting:

    @pytest.mark.parametrize('label', ['Inspiration', 'inspiration', 'INSPIRATION', 'Expiration', 'expiration', 'EXPIRATION', 'Pause', 'pause', 'PAUSE', 'insp', 'exp', 'inhale', 'exhale', 'exspiration'])
    def test_phase_labels(self, label):
        """验证英文呼吸时相标签（Inspiration/Expiration/Pause 等）被路由到 PHASE_KIND。"""
        assert label_kind(label) == PHASE_KIND

    @pytest.mark.parametrize('label', ['吸气', '呼气', '停顿'])
    def test_chinese_phase_labels(self, label):
        """验证英文呼吸时相标签（Inspiration/Expiration/Pause 等）被路由到 PHASE_KIND。"""
        assert label_kind(label) == PHASE_KIND

class TestOtherEventRouting:

    @pytest.mark.parametrize('label', ['Speech', 'speech', 'talk', 'talking', 'voice', 'whisper', 'Cough', 'cough', 'coughing', 'Sneeze', 'sneeze', 'Snore', 'snore', 'Noise', 'noise', 'artifact', 'movement', 'background'])
    def test_other_event_labels(self, label):
        """验证英文其他事件标签（Speech/Cough/Noise 等）被路由到 OTHER_EVENT_KIND。"""
        assert label_kind(label) == OTHER_EVENT_KIND

    @pytest.mark.parametrize('label', ['说话', '讲话', '咳嗽', '咳'])
    def test_chinese_other_event_labels(self, label):
        """验证英文其他事件标签（Speech/Cough/Noise 等）被路由到 OTHER_EVENT_KIND。"""
        assert label_kind(label) == OTHER_EVENT_KIND

class TestAbnormalSoundRouting:

    @pytest.mark.parametrize('label', ['Wheeze', 'wheeze', 'Crackles', 'crackles', 'Rhonchi', 'rhonchi', 'Stridor', 'stridor', 'Pleural Rub', 'pleural rub'])
    def test_abnormal_sound_default(self, label):
        """验证默认参数值符合预期。"""
        assert label_kind(label) == ABNORMAL_SOUND_KIND

    def test_unknown_label_defaults_to_abnormal_sound(self):
        """Any label not in PHASE or OTHER_EVENT sets → ABNORMAL_SOUND_KIND."""
        assert label_kind('custom_label') == ABNORMAL_SOUND_KIND
        assert label_kind('xyz') == ABNORMAL_SOUND_KIND
        assert label_kind('') == ABNORMAL_SOUND_KIND

class TestEdgeCases:

    def test_leading_trailing_whitespace_trimmed(self):
        """验证标签前后空格被正确去除后再路由。"""
        assert label_kind('  Inspiration  ') == PHASE_KIND
        assert label_kind('\tCough\n') == OTHER_EVENT_KIND

    def test_non_string_input(self):
        """Non-string inputs should not crash."""
        label_kind(123)
        label_kind(None)