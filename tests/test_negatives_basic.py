"""Tests for respanno.ml.negatives — NegSampleManager."""
import pytest
from respanno.ml.negatives import NegSampleManager

class TestNegSampleManager:

    def test_initial_count_is_zero(self):
        """Verify initial state is empty."""
        mgr = NegSampleManager()
        assert mgr.count('Wheeze') == 0
        assert mgr.count('Crackles') == 0
        assert mgr.count('') == 0

    def test_initial_get_returns_empty_list(self):
        """Verify initial state is empty."""
        mgr = NegSampleManager()
        assert mgr.get('Wheeze') == []

    def test_initial_to_dict_is_empty(self):
        """Verify initial state is empty."""
        mgr = NegSampleManager()
        assert mgr.to_dict() == {}

    def test_add_returns_tuple(self):
        """Verify add returns (s, e, neg_id) tuple."""
        mgr = NegSampleManager()
        item = mgr.add('Wheeze', 1.0, 2.0)
        assert item is not None
        assert len(item) == 3
        (s, e, neg_id) = item
        assert s == 1.0
        assert e == 2.0
        assert isinstance(neg_id, int)

    def test_add_increments_count(self):
        """Verify consecutive adds increment count correctly."""
        mgr = NegSampleManager()
        mgr.add('Wheeze', 1.0, 2.0)
        assert mgr.count('Wheeze') == 1
        mgr.add('Wheeze', 3.0, 4.0)
        assert mgr.count('Wheeze') == 2

    def test_add_unique_ids(self):
        """Verify each add generates a globally unique neg_id."""
        mgr = NegSampleManager()
        id1 = mgr.add('Wheeze', 1.0, 2.0)[2]
        id2 = mgr.add('Wheeze', 3.0, 4.0)[2]
        assert id1 != id2

    def test_add_empty_label_returns_none(self):
        """Verify empty-label add returns None."""
        mgr = NegSampleManager()
        assert mgr.add('', 1.0, 2.0) is None       # empty → None
        assert mgr.add('x', 1.0, 2.0) is not None   # non-empty → ok
        # Whitespace-only strings are truthy in Python, so they are valid labels
        assert mgr.add('  ', 1.0, 2.0) is not None

    def test_add_multiple_labels_independent(self):
        """Verify multiple labels have independent negative samples."""
        mgr = NegSampleManager()
        mgr.add('Wheeze', 1.0, 2.0)
        mgr.add('Crackles', 3.0, 4.0)
        assert mgr.count('Wheeze') == 1
        assert mgr.count('Crackles') == 1

    def test_get_returns_correct_tuples(self):
        """Verify get returns correct (s, e, neg_id) tuple list."""
        mgr = NegSampleManager()
        mgr.add('Wheeze', 1.0, 2.0)
        mgr.add('Wheeze', 3.0, 4.0)
        items = mgr.get('Wheeze')
        assert len(items) == 2
        assert items[0][0] == 1.0
        assert items[0][1] == 2.0
        assert items[1][0] == 3.0
        assert items[1][1] == 4.0

    def test_remove_decrements_count(self):
        """Verify remove decrements count correctly."""
        mgr = NegSampleManager()
        item = mgr.add('Wheeze', 1.0, 2.0)
        neg_id = item[2]
        assert mgr.count('Wheeze') == 1
        mgr.remove('Wheeze', neg_id)
        assert mgr.count('Wheeze') == 0

    def test_remove_nonexistent_does_not_crash(self):
        """Verify behaviour on empty or None input。"""
        mgr = NegSampleManager()
        mgr.remove('Wheeze', 999)

    def test_remove_nonexistent_label_does_not_crash(self):
        """Verify behaviour on empty or None input。"""
        mgr = NegSampleManager()
        mgr.remove('nonexistent', 1)

    def test_remove_only_removes_correct_id(self):
        """Verify remove only deletes the specified neg_id."""
        mgr = NegSampleManager()
        item1 = mgr.add('Wheeze', 1.0, 2.0)
        item2 = mgr.add('Wheeze', 3.0, 4.0)
        mgr.remove('Wheeze', item1[2])
        assert mgr.count('Wheeze') == 1
        remaining = mgr.get('Wheeze')
        assert remaining[0][2] == item2[2]

    def test_count_zero_for_nonexistent_label(self):
        """Verify behaviour on empty or None input。"""
        mgr = NegSampleManager()
        assert mgr.count('nonexistent') == 0

    def test_clear_single_label(self):
        """Verify clear(label) only clears one label."""
        mgr = NegSampleManager()
        mgr.add('Wheeze', 1.0, 2.0)
        mgr.add('Crackles', 3.0, 4.0)
        mgr.clear('Wheeze')
        assert mgr.count('Wheeze') == 0
        assert mgr.count('Crackles') == 1

    def test_clear_nonexistent_label_does_not_crash(self):
        """Verify behaviour on empty or None input。"""
        mgr = NegSampleManager()
        mgr.clear('nonexistent')

    def test_clear_all_removes_everything(self):
        """Verify clear_all removes all samples for all labels."""
        mgr = NegSampleManager()
        mgr.add('Wheeze', 1.0, 2.0)
        mgr.add('Crackles', 3.0, 4.0)
        mgr.add('Rhonchi', 5.0, 6.0)
        mgr.clear_all()
        assert mgr.count('Wheeze') == 0
        assert mgr.count('Crackles') == 0
        assert mgr.count('Rhonchi') == 0

    def test_clear_all_resets_id_counter(self):
        """Verify clear_all resets the neg_id counter."""
        mgr = NegSampleManager()
        item = mgr.add('Wheeze', 1.0, 2.0)
        old_id = item[2]
        mgr.clear_all()
        new_item = mgr.add('Wheeze', 1.0, 2.0)
        assert new_item[2] == 1

    def test_to_dict_is_live_reference(self):
        """Verify to_dict returns a live dict (mutations affect the manager)."""
        mgr = NegSampleManager()
        mgr.add('Wheeze', 1.0, 2.0)
        d = mgr.to_dict()
        assert 'Wheeze' in d
        assert len(d['Wheeze']) == 1
        d['Wheeze'].clear()
        assert mgr.count('Wheeze') == 0

    def test_to_dict_multiple_labels(self):
        """Verify to_dict includes all label keys when multiple exist."""
        mgr = NegSampleManager()
        mgr.add('Wheeze', 1.0, 2.0)
        mgr.add('Crackles', 3.0, 4.0)
        d = mgr.to_dict()
        assert 'Wheeze' in d
        assert 'Crackles' in d

    def test_add_with_negative_zero_times(self):
        """Negative or zero start/end should still be recorded (caller validates)."""
        mgr = NegSampleManager()
        item = mgr.add('Test', -1.0, 0.0)
        assert item is not None
        assert item[0] == -1.0

    def test_add_large_number_of_segments(self):
        """Verify adding 100+ segments counts correctly."""
        mgr = NegSampleManager()
        for i in range(100):
            mgr.add('Wheeze', float(i), float(i + 1))
        assert mgr.count('Wheeze') == 100

    def test_add_with_various_label_types(self):
        """Labels are converted to str."""
        mgr = NegSampleManager()
        assert mgr.add('Wheeze', 1.0, 2.0) is not None
        assert mgr.add('123', 1.0, 2.0) is not None