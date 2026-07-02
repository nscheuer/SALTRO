import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "build"))

try:
    import saltro_py
except ImportError as e:
    pytest.skip(f"saltro Python bindings not available: {e}", allow_module_level=True)


def norm_q(q):
    return q / np.linalg.norm(q)


def test_quat_error_of_identical_quaternions_is_identity():
    q = norm_q(np.array([0.7, 0.2, -0.5, 0.4]))
    e = saltro_py.quatError(q, q)
    assert abs(e[0] - 1.0) < 1e-12
    assert np.linalg.norm(e[1:]) < 1e-12


def test_quat_error_composes_back_to_current_quaternion():
    q_goal = norm_q(np.array([0.9, 0.1, 0.2, -0.3]))
    q = norm_q(np.array([0.6, -0.4, 0.5, 0.3]))
    e = saltro_py.quatError(q_goal, q)

    a0, a1, a2, a3 = q_goal
    b0, b1, b2, b3 = e
    q_rec = np.array([
        a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
        a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
        a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
        a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0,
    ])

    err = min(np.linalg.norm(q_rec - q), np.linalg.norm(q_rec + q))
    assert err < 1e-12


def test_quat_error_enforces_nonnegative_scalar_component():
    q_goal = np.array([1.0, 0.0, 0.0, 0.0])
    q = norm_q(np.array([-0.2, 0.8, 0.3, -0.4]))
    e = saltro_py.quatError(q_goal, q)
    assert e[0] >= 0.0


def test_quat_to_mrp_of_identity_is_zero():
    s = saltro_py.quatToMRP(np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.linalg.norm(s) < 1e-14


def test_quat_to_mrp_small_angle_limit():
    theta = 1e-3
    axis = np.array([0.3, -0.6, 0.7], dtype=float)
    axis /= np.linalg.norm(axis)
    q = np.zeros(4)
    q[0] = np.cos(theta / 2.0)
    q[1:] = np.sin(theta / 2.0) * axis
    s = saltro_py.quatToMRP(q)
    expected = (theta / 2.0) * axis
    assert np.linalg.norm(s - expected) < 1e-9


def test_quat_to_mrp_round_trip_closed_form():
    theta = 0.5
    axis = np.array([1.0, 2.0, -1.0], dtype=float)
    axis /= np.linalg.norm(axis)
    q = np.zeros(4)
    q[0] = np.cos(theta / 2.0)
    q[1:] = np.sin(theta / 2.0) * axis

    s = saltro_py.quatToMRP(q)
    s2 = float(s @ s)
    q0_rec = (4.0 - s2) / (4.0 + s2)
    qv_rec = s * (1.0 + q0_rec) / 2.0
    assert abs(q0_rec - q[0]) < 1e-12
    assert np.linalg.norm(qv_rec - q[1:]) < 1e-12


def test_find_gmat_has_correct_dimensions():
    g = saltro_py.findGMat(norm_q(np.array([0.8, 0.2, -0.4, 0.3])), 3)
    assert g.shape == (9, 10)


def test_find_gmat_identity_and_rw_blocks_are_correct():
    g = saltro_py.findGMat(norm_q(np.array([0.5, 0.5, 0.5, 0.5])), 2)
    assert np.linalg.norm(g[0:3, 0:3] - np.eye(3)) < 1e-14
    assert abs(g[6, 7] - 1.0) < 1e-14
    assert abs(g[7, 8] - 1.0) < 1e-14


def test_find_gmat_attitude_block_matches_w_transpose():
    q = norm_q(np.array([0.8, 0.2, -0.4, 0.3]))
    g = saltro_py.findGMat(q, 1)
    wt = g[3:6, 3:7]
    proj = wt @ q
    assert np.linalg.norm(proj) < 1e-12
    assert np.linalg.norm(wt - saltro_py.findWMat(q).T) < 1e-14


def test_find_gmat_wt_block_linearizes_quaternion_kinematics():
    q = norm_q(np.array([0.9, -0.1, 0.2, 0.3]))
    omega = np.array([0.11, -0.07, 0.05])
    g = saltro_py.findGMat(q, 0)
    wt = g[3:6, 3:7]
    w = saltro_py.findWMat(q)
    qdot = 0.5 * (w @ omega)
    recovered = wt @ qdot
    assert np.linalg.norm(recovered - 0.5 * omega) < 1e-12
