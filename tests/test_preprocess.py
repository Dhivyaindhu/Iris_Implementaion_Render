import numpy as np
import pytest

from src.preprocess import scale_minmax, validate_features


def test_validate_returns_row_vector():
    out = validate_features([5.1, 3.5, 1.4, 0.2])
    assert out.shape == (1, 4)


def test_validate_rejects_wrong_length():
    with pytest.raises(ValueError, match="expected 4 features"):
        validate_features([1, 2, 3])


def test_validate_rejects_nan():
    with pytest.raises(ValueError, match="NaN"):
        validate_features([1, 2, float("nan"), 4])


@pytest.mark.parametrize("bad", [[-1, 2, 3, 4], [1, -2, 3, 4], [1, 2, 3, -0.1]])
def test_validate_rejects_negative(bad):
    with pytest.raises(ValueError, match="non-negative"):
        validate_features(bad)


def test_scale_minmax_range():
    out = scale_minmax([10, 20, 30])
    assert out.min() == 0.0 and out.max() == 1.0


def test_scale_minmax_constant_array():
    assert np.all(scale_minmax([5, 5, 5]) == 0)
