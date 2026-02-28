import numpy as np
import pytest

try:
    import saltro_py

    validateQGoal = lambda q_goal: saltro_py.validateQGoal(q_goal)
    MAX_LENGTH_TRAJ = saltro_py.limits.MAX_LENGTH_TRAJ
except ImportError as e:
    pytest.skip(f"saltro Python bindings not available: {e}", allow_module_level=True)


def valid_quaternion_goals():
    q_goal = np.zeros((4, 2), dtype=float)
    q_goal[:, 0] = [1.0, 0.0, 0.0, 0.0]
    q_goal[:, 1] = [0.0, 1.0, 0.0, 0.0]
    return q_goal


def valid_eci_goals():
    q_goal = np.zeros((4, 2), dtype=float)
    q_goal[:, 0] = [np.nan, 1.0, 0.0, 0.0]
    q_goal[:, 1] = [np.nan, 0.0, 0.0, 1.0]
    return q_goal


def test_valid_quaternion_goals_pass_validation():
    q_goal = valid_quaternion_goals()

    ok, error_msg = validateQGoal(q_goal)

    assert ok
    assert error_msg == "" or error_msg is None


def test_valid_eci_goals_pass_validation():
    q_goal = valid_eci_goals()

    ok, error_msg = validateQGoal(q_goal)

    assert ok
    assert error_msg == "" or error_msg is None


def test_valid_mixed_quaternion_and_eci_goals_pass_validation():
    q_goal = np.zeros((4, 4), dtype=float)
    q_goal[:, 0] = [1.0, 0.0, 0.0, 0.0]
    q_goal[:, 1] = [np.nan, 0.0, 1.0, 0.0]
    q_goal[:, 2] = [0.0, 0.0, 0.0, 1.0]
    q_goal[:, 3] = [np.nan, 1.0, 0.0, 0.0]

    ok, error_msg = validateQGoal(q_goal)

    assert ok
    assert error_msg == "" or error_msg is None


def test_invalid_row_count_fails():
    q_goal = np.zeros((3, 2), dtype=float)

    ok, error_msg = validateQGoal(q_goal)

    assert not ok
    assert error_msg == "q_goal must have shape (4, N)"


def test_empty_qgoal_fails():
    q_goal = np.zeros((4, 0), dtype=float)

    ok, error_msg = validateQGoal(q_goal)

    assert not ok
    assert error_msg == "q_goal must have at least one column"


def test_qgoal_exceeding_max_length_traj_fails():
    q_goal = np.zeros((4, MAX_LENGTH_TRAJ + 1), dtype=float)
    q_goal[0, :] = 1.0

    ok, error_msg = validateQGoal(q_goal)

    assert not ok
    assert error_msg == "q_goal exceeds MAX_LENGTH_TRAJ"


def test_nan_in_rows_2_to_4_is_rejected():
    q_goal = valid_quaternion_goals()
    q_goal[2, 1] = np.nan

    ok, error_msg = validateQGoal(q_goal)

    assert not ok
    assert error_msg == "q_goal column 1 has invalid NaN/Inf in rows 2-4"


def test_infinity_in_rows_2_to_4_is_rejected():
    q_goal = valid_eci_goals()
    q_goal[3, 0] = np.inf

    ok, error_msg = validateQGoal(q_goal)

    assert not ok
    assert error_msg == "q_goal column 0 has invalid NaN/Inf in rows 2-4"


def test_zero_direction_norm_is_rejected():
    q_goal = np.array([[np.nan], [0.0], [0.0], [0.0]], dtype=float)

    ok, error_msg = validateQGoal(q_goal)

    assert not ok
    assert error_msg == "q_goal column 0 has zero or non-finite direction norm"


def test_eci_direction_not_normalized_is_rejected():
    q_goal = np.array([[np.nan], [2.0], [0.0], [0.0]], dtype=float)

    ok, error_msg = validateQGoal(q_goal)

    assert not ok
    assert error_msg == "q_goal column 0 ECI direction is not normalized"


def test_invalid_q0_value_is_rejected():
    q_goal = np.array([[np.inf], [1.0], [0.0], [0.0]], dtype=float)

    ok, error_msg = validateQGoal(q_goal)

    assert not ok
    assert error_msg == "q_goal column 0 has invalid q0 value"


def test_quaternion_not_normalized_is_rejected():
    q_goal = np.array([[1.0], [1.0], [0.0], [0.0]], dtype=float)

    ok, error_msg = validateQGoal(q_goal)

    assert not ok
    assert error_msg == "q_goal column 0 quaternion is not normalized"


def test_quaternion_norm_at_tolerance_boundary_passes():
    q_goal = np.array([[1.001], [0.0], [0.0], [0.0]], dtype=float)

    ok, error_msg = validateQGoal(q_goal)

    assert ok


def test_quaternion_norm_beyond_tolerance_fails():
    q_goal = np.array([[1.0011], [0.0], [0.0], [0.0]], dtype=float)

    ok, error_msg = validateQGoal(q_goal)

    assert not ok
    assert error_msg == "q_goal column 0 quaternion is not normalized"


def test_eci_norm_at_tolerance_boundary_passes():
    q_goal = np.array([[np.nan], [1.001], [0.0], [0.0]], dtype=float)

    ok, error_msg = validateQGoal(q_goal)

    assert ok


def test_eci_norm_beyond_tolerance_fails():
    q_goal = np.array([[np.nan], [1.0011], [0.0], [0.0]], dtype=float)

    ok, error_msg = validateQGoal(q_goal)

    assert not ok
    assert error_msg == "q_goal column 0 ECI direction is not normalized"


def test_error_pinpoints_failing_column_in_mixed_set():
    q_goal = np.zeros((4, 3), dtype=float)
    q_goal[:, 0] = [1.0, 0.0, 0.0, 0.0]
    q_goal[:, 1] = [np.nan, 0.0, 1.0, 0.0]
    q_goal[:, 2] = [0.5, 0.5, 0.5, 0.5]
    q_goal[2, 2] = np.nan

    ok, error_msg = validateQGoal(q_goal)

    assert not ok
    assert error_msg == "q_goal column 2 has invalid NaN/Inf in rows 2-4"
