from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from audit_ternary_v7_forensics import metrics, packed_transition_count


def test_counts_codes_not_different_packed_words():
    before = np.zeros((1, 2), dtype=np.uint32)
    after = before.copy()
    after[0, 0] = np.uint32(1 | (2 << 4))
    assert packed_transition_count(before, after) == (2, 32)
    assert packed_transition_count(after, after) == (0, 32)


def test_empty_arrays_are_not_zero_change_evidence():
    with pytest.raises(ValueError, match="empty"):
        packed_transition_count(np.zeros(0, np.uint32), np.zeros(0, np.uint32))


def test_reserved_code_is_rejected():
    with pytest.raises(ValueError, match="reserved"):
        packed_transition_count(np.array([0], np.uint32), np.array([3], np.uint32))


def test_shape_or_dtype_drift_is_rejected():
    with pytest.raises(ValueError, match="uint32"):
        packed_transition_count(np.zeros(2, np.uint32), np.zeros(1, np.uint32))
    with pytest.raises(ValueError, match="uint32"):
        packed_transition_count(np.zeros(2, np.int32), np.zeros(2, np.int32))


def test_metrics_keep_amplitude_error_when_cosine_is_perfect():
    result = metrics(np.array([1., 2.]), np.array([2., 4.]))
    assert result["cosine"] == pytest.approx(1.)
    assert result["relative_l2"] == pytest.approx(1.)
    assert result["rms_ratio"] == pytest.approx(2.)
