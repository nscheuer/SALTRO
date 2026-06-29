"""Python twin of tests/unit/math/test_mrp.cpp."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "build"))

import numpy as np
import saltro_py as s


def _normq(q):
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q)


def _quat_mul(p, r):
    """Scalar-first Hamilton product p (x) r."""
    out = np.empty(4)
    out[0] = p[0] * r[0] - p[1] * r[1] - p[2] * r[2] - p[3] * r[3]
    out[1] = p[0] * r[1] + p[1] * r[0] + p[2] * r[3] - p[3] * r[2]
    out[2] = p[0] * r[2] - p[1] * r[3] + p[2] * r[0] + p[3] * r[1]
    out[3] = p[0] * r[3] + p[1] * r[2] - p[2] * r[1] + p[3] * r[0]
    return out


# ----------------------------------------------------------------------------
# quatError
# ----------------------------------------------------------------------------
def test_quat_error_of_identical_is_identity():
    q = _normq([0.7, 0.2, -0.5, 0.4])
    e = np.array(s.quatError(q, q))
    assert abs(e[0] - 1.0) < 1e-12
    assert np.linalg.norm(e[1:]) < 1e-12


def test_quat_error_composes_back_to_current_quaternion():
    q_goal = _normq([0.9, 0.1, 0.2, -0.3])
    q = _normq([0.6, -0.4, 0.5, 0.3])
    e = np.array(s.quatError(q_goal, q))
    q_rec = _quat_mul(q_goal, e)
    # equal up to global sign
    err = min(np.linalg.norm(q_rec - q), np.linalg.norm(q_rec + q))
    assert err < 1e-12


def test_quat_error_enforces_nonnegative_scalar():
    q_goal = _normq([1.0, 0.0, 0.0, 0.0])
    q = _normq([-0.2, 0.8, 0.3, -0.4])
    e = np.array(s.quatError(q_goal, q))
    assert e[0] >= 0.0


# ----------------------------------------------------------------------------
# quatToMRP
# ----------------------------------------------------------------------------
def test_quat_to_mrp_of_identity_is_zero():
    sm = np.array(s.quatToMRP([1.0, 0.0, 0.0, 0.0]))
    assert np.linalg.norm(sm) < 1e-14


def test_quat_to_mrp_small_angle_limit():
    theta = 1e-3
    axis = _normq([0.3, -0.6, 0.7, 0.0])[:3]
    axis = axis / np.linalg.norm(axis)
    q = np.empty(4)
    q[0] = math.cos(theta / 2.0)
    q[1:] = math.sin(theta / 2.0) * axis
    sm = np.array(s.quatToMRP(q))
    expected = (theta / 2.0) * axis
    assert np.linalg.norm(sm - expected) < 1e-9


def test_quat_to_mrp_round_trips():
    theta = 0.5
    axis = np.array([1.0, 2.0, -1.0])
    axis = axis / np.linalg.norm(axis)
    q = np.empty(4)
    q[0] = math.cos(theta / 2.0)
    q[1:] = math.sin(theta / 2.0) * axis

    sm = np.array(s.quatToMRP(q))
    s2 = float(sm @ sm)
    q0_rec = (4.0 - s2) / (4.0 + s2)
    qv_rec = sm * (1.0 + q0_rec) / 2.0
    assert abs(q0_rec - q[0]) < 1e-12
    assert np.linalg.norm(qv_rec - q[1:]) < 1e-12


# ----------------------------------------------------------------------------
# findGMat
# ----------------------------------------------------------------------------
def test_find_gmat_has_correct_dimensions():
    nRW = 3
    q = _normq([0.8, 0.2, -0.4, 0.3])
    G = np.array(s.findGMat(q, nRW))
    assert G.shape == (6 + nRW, 7 + nRW)


def test_find_gmat_identity_and_rw_blocks():
    nRW = 2
    q = _normq([0.5, 0.5, 0.5, 0.5])
    G = np.array(s.findGMat(q, nRW))
    assert np.linalg.norm(G[0:3, 0:3] - np.eye(3)) < 1e-14
    for i in range(nRW):
        assert abs(G[6 + i, 7 + i] - 1.0) < 1e-14


def test_find_gmat_attitude_block_is_wmat_transpose():
    nRW = 1
    q = _normq([0.8, 0.2, -0.4, 0.3])
    G = np.array(s.findGMat(q, nRW))
    Wt = G[3:6, 3:7]
    # tangent projection W^T q == 0 for a unit quaternion
    assert np.linalg.norm(Wt @ q) < 1e-12
    W = np.array(s.findWMat(q))
    assert np.linalg.norm(Wt - W.T) < 1e-14


def test_find_gmat_wmat_block_linearizes_kinematics():
    q = _normq([0.9, -0.1, 0.2, 0.3])
    omega = np.array([0.11, -0.07, 0.05])
    G = np.array(s.findGMat(q, 0))
    Wt = G[3:6, 3:7]
    W = np.array(s.findWMat(q))
    qdot = 0.5 * W @ omega
    recovered = Wt @ qdot
    assert np.linalg.norm(recovered - 0.5 * omega) < 1e-12
