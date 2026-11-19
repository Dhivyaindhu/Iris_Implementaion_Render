"""Small, pure functions: the easiest code to unit test."""
import numpy as np

FEATURE_COUNT = 4  # sepal length, sepal width, petal length, petal width


def validate_features(features):
    """Return features as a float array of shape (1, 4) or raise ValueError."""
    if features is None:
        raise ValueError("features must not be None")
    arr = np.asarray(features, dtype=float).reshape(1, -1)
    if arr.shape[1] != FEATURE_COUNT:
        raise ValueError(f"expected {FEATURE_COUNT} features, got {arr.shape[1]}")
    if np.isnan(arr).any():
        raise ValueError("features must not contain NaN")
    if (arr < 0).any():
        raise ValueError("features must be non-negative")
    return arr


def scale_minmax(values):
    """Scale a 1-D array to the 0-1 range. A constant array becomes all zeros."""
    arr = np.asarray(values, dtype=float)
    span = arr.max() - arr.min()
    if span == 0:
        return np.zeros_like(arr)
    return (arr - arr.min()) / span
