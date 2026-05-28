"""Tests for respanno.ml.label_taxonomy — label routing."""

import pytest

from respanno.ml.label_taxonomy import (
    label_kind,
    PHASE_KIND,
    OTHER_EVENT_KIND,
    ABNORMAL_SOUND_KIND,
    PHASE_LABELS,
    OTHER_EVENT_LABELS,
)


# ---------------------------------------------------------------------------
# Phase labels
# ---------------------------------------------------------------------------


class TestPhaseRouting:
    @pytest.mark.parametrize("label", [
        "Inspiration", "inspiration", "INSPIRATION",
        "Expiration", "expiration", "EXPIRATION",
        "Pause", "pause", "PAUSE",
        "insp", "exp", "inhale", "exhale", "exspiration",
    ])
    def test_phase_labels(self, label):
        assert label_kind(label) == PHASE_KIND

    @pytest.mark.parametrize("label", ["吸气", "呼气", "停顿"])
    def test_chinese_phase_labels(self, label):
        assert label_kind(label) == PHASE_KIND


# ---------------------------------------------------------------------------
# Other-event labels
# ---------------------------------------------------------------------------


class TestOtherEventRouting:
    @pytest.mark.parametrize("label", [
        "Speech", "speech", "talk", "talking", "voice", "whisper",
        "Cough", "cough", "coughing",
        "Sneeze", "sneeze", "Snore", "snore",
        "Noise", "noise", "artifact", "movement", "background",
    ])
    def test_other_event_labels(self, label):
        assert label_kind(label) == OTHER_EVENT_KIND

    @pytest.mark.parametrize("label", ["说话", "讲话", "咳嗽", "咳"])
    def test_chinese_other_event_labels(self, label):
        assert label_kind(label) == OTHER_EVENT_KIND


# ---------------------------------------------------------------------------
# Default (abnormal sound) labels
# ---------------------------------------------------------------------------


class TestAbnormalSoundRouting:
    @pytest.mark.parametrize("label", [
        "Wheeze", "wheeze",
        "Crackles", "crackles",
        "Rhonchi", "rhonchi",
        "Stridor", "stridor",
        "Pleural Rub", "pleural rub",
    ])
    def test_abnormal_sound_default(self, label):
        assert label_kind(label) == ABNORMAL_SOUND_KIND

    def test_unknown_label_defaults_to_abnormal_sound(self):
        """Any label not in PHASE or OTHER_EVENT sets → ABNORMAL_SOUND_KIND."""
        assert label_kind("custom_label") == ABNORMAL_SOUND_KIND
        assert label_kind("xyz") == ABNORMAL_SOUND_KIND
        assert label_kind("") == ABNORMAL_SOUND_KIND


# ---------------------------------------------------------------------------
# White-space / edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_leading_trailing_whitespace_trimmed(self):
        assert label_kind("  Inspiration  ") == PHASE_KIND
        assert label_kind("\tCough\n") == OTHER_EVENT_KIND

    def test_non_string_input(self):
        """Non-string inputs should not crash."""
        label_kind(123)      # should not raise
        label_kind(None)     # should not raise
