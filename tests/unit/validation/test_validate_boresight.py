import numpy as np
import pytest

try:
    import saltro_py

    validateBoresight = lambda boresight: saltro_py.validateBoresight(boresight)
    MAX_LENGTH_TRAJ = saltro_py.limits.MAX_LENGTH_TRAJ
except ImportError as e:
    pytest.skip(f"saltro Python bindings not available: {e}", allow_module_level=True)


def valid_boresight_history():
    boresight = np.zeros((3, 2), dtype=float)
    boresight[:, 0] = [1.0, 0.0, 0.0]
    boresight[:, 1] = [0.0, 1.0, 0.0]
    return boresight


def test_valid_boresight_history_passes_validation():
    boresight = valid_boresight_history()

    ok, error_msg = validateBoresight(boresight)

    assert ok
    assert error_msg == "" or error_msg is None


def test_valid_single_column_boresight_passes_validation():
    boresight = np.array([[0.0], [0.0], [1.0]], dtype=float)

    ok, error_msg = validateBoresight(boresight)

    assert ok
    assert error_msg == "" or error_msg is None


def test_valid_multiple_columns_with_normalized_vectors_pass():
    boresight = np.zeros((3, 4), dtype=float)
    boresight[:, 0] = [1.0, 0.0, 0.0]
    boresight[:, 1] = [0.0, 1.0, 0.0]
    boresight[:, 2] = [0.0, 0.0, 1.0]
    boresight[:, 3] = [1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]

    ok, error_msg = validateBoresight(boresight)

    assert ok
    assert error_msg == "" or error_msg is None


def test_invalid_row_count_fails():
    boresight = np.zeros((2, 2), dtype=float)

    ok, error_msg = validateBoresight(boresight)

    assert not ok
    assert error_msg == "boresight must have shape (3, N)"


def test_invalid_row_count_4_rows_fails():
    boresight = np.zeros((4, 2), dtype=float)

    ok, error_msg = validateBoresight(boresight)

    assert not ok
    assert error_msg == "boresight must have shape (3, N)"


def test_empty_boresight_fails():
    boresight = np.zeros((3, 0), dtype=float)

    ok, error_msg = validateBoresight(boresight)

    assert not ok
    assert error_msg == "boresight must have at least one column"


def test_boresight_exceeding_max_length_traj_fails():
    boresight = np.zeros((3, MAX_LENGTH_TRAJ + 1), dtype=float)
    boresight[0, :] = 1.0

    ok, error_msg = validateBoresight(boresight)

    assert not ok
    assert error_msg == "boresight exceeds MAX_LENGTH_TRAJ"


def test_nan_in_boresight_is_rejected():
    boresight = valid_boresight_history()
    boresight[2, 1] = np.nan

    ok, error_msg = validateBoresight(boresight)

    assert not ok
    assert error_msg == "boresight column 1 contains NaN or Inf"


def test_infinity_in_boresight_is_rejected():
    boresight = valid_boresight_history()
    boresight[0, 0] = np.inf

    ok, error_msg = validateBoresight(boresight)

    assert not ok
    assert error_msg == "boresight column 0 contains NaN or Inf"


def test_zero_norm_boresight_is_rejected():
    boresight = np.array([[0.0], [0.0], [0.0]], dtype=float)

    ok, error_msg = validateBoresight(boresight)

    assert not ok
    assert error_msg == "boresight column 0 has zero or non-finite norm"


def test_boresight_not_normalized_is_rejected():
    boresight = np.array([[2.0], [0.0], [0.0]], dtype=float)

    ok, error_msg = validateBoresight(boresight)

    assert not ok
    assert "boresight column 0 is not normalized" in error_msg


def test_boresight_norm_at_tolerance_boundary_passes():
    boresight = np.array([[1.001], [0.0], [0.0]], dtype=float)

    ok, error_msg = validateBoresight(boresight)

    assert ok


def test_boresight_norm_beyond_tolerance_fails():
    boresight = np.array([[1.0011], [0.0], [0.0]], dtype=float)

    ok, error_msg = validateBoresight(boresight)

    assert not ok
    assert "boresight column 0 is not normalized" in error_msg


def test_error_pinpoints_failing_column_in_set():
    boresight = np.zeros((3, 3), dtype=float)
    boresight[:, 0] = [1.0, 0.0, 0.0]
    boresight[:, 1] = [0.0, 1.0, 0.0]
    boresight[:, 2] = [0.0, 0.0, 0.0]

    ok, error_msg = validateBoresight(boresight)

    assert not ok
    assert error_msg == "boresight column 2 has zero or non-finite norm"


def test_negative_infinity_is_rejected():
    boresight = valid_boresight_history()
    boresight[1, 0] = -np.inf

    ok, error_msg = validateBoresight(boresight)

    assert not ok
    assert error_msg == "boresight column 0 contains NaN or Inf"


def test_very_small_but_normalized_vector_passes():
    val = 1.0 / np.sqrt(3.0)
    boresight = np.array([[val], [val], [val]], dtype=float)

    ok, error_msg = validateBoresight(boresight)

    assert ok
