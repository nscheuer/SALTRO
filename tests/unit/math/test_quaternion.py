"""Python twin of tests/unit/math/test_quaternion.cpp.

Includes the finite-difference checks for the analytic Jacobian/Hessian
helpers (quatNormJacobian, drotmatTvecdq, ddrotmatTvecdqdq). The last group is
the regression coverage for the rotation-Hessian fix: the analytic second
derivative must be symmetric in its two quaternion indices and match the finite
difference of drotmatTvecdq.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "build"))

import numpy as np
import pytest
import saltro_py as s


def _fixed_unit_quat():
    q = np.array([0.8, 0.2, -0.4, 0.3])
    return q / np.linalg.norm(q)


def _fixed_vec():
    return np.array([1.3, -0.7, 2.1])


def _quat_mul(p, r):
    """Scalar-first Hamilton product p (x) r."""
    out = np.empty(4)
    out[0] = p[0] * r[0] - p[1] * r[1] - p[2] * r[2] - p[3] * r[3]
    out[1] = p[0] * r[1] + p[1] * r[0] + p[2] * r[3] - p[3] * r[2]
    out[2] = p[0] * r[2] - p[1] * r[3] + p[2] * r[0] + p[3] * r[1]
    out[3] = p[0] * r[3] + p[1] * r[2] - p[2] * r[1] + p[3] * r[0]
    return out


def _perturb_on_sphere(q, theta):
    """q' = q (x) exp(theta/2): perturb a unit quaternion, staying on S^3."""
    n = np.linalg.norm(theta)
    dq = np.empty(4)
    if n < 1e-300:
        dq[:] = [1.0, 0.0, 0.0, 0.0]
    else:
        dq[0] = math.cos(n / 2.0)
        dq[1:] = math.sin(n / 2.0) * np.asarray(theta) / n
    return _quat_mul(q, dq)


# ----------------------------------------------------------------------------
# normalizeQuat
# ----------------------------------------------------------------------------
def test_normalize_quat_produces_unit_norm():
    q = np.array([2.0, -1.0, 0.5, 3.0])
    qn = np.array(s.normalizeQuat(q))
    assert abs(np.linalg.norm(qn) - 1.0) < 1e-12
    assert np.linalg.norm(qn * np.linalg.norm(q) - q) < 1e-12


def test_normalize_quat_of_unit_is_unchanged():
    q = _fixed_unit_quat()
    qn = np.array(s.normalizeQuat(q))
    assert np.linalg.norm(qn - q) < 1e-12


def test_normalize_quat_throws_on_near_zero():
    q = np.array([1e-15, 0.0, -1e-15, 0.0])
    with pytest.raises(Exception):
        s.normalizeQuat(q)


# ----------------------------------------------------------------------------
# rotationMatrix
# ----------------------------------------------------------------------------
def test_rotation_matrix_orthonormal_det_plus_one():
    R = np.array(s.rotationMatrix(_fixed_unit_quat()))
    assert np.linalg.norm(R @ R.T - np.eye(3)) < 1e-12
    assert abs(np.linalg.det(R) - 1.0) < 1e-12


def test_rotation_matrix_normalizes_input():
    q = _fixed_unit_quat()
    R1 = np.array(s.rotationMatrix(q))
    R2 = np.array(s.rotationMatrix(2.5 * q))
    assert np.linalg.norm(R1 - R2) < 1e-12


def test_rotation_matrix_identity_quaternion():
    R = np.array(s.rotationMatrix([1.0, 0.0, 0.0, 0.0]))
    assert np.linalg.norm(R - np.eye(3)) < 1e-12


def test_rotation_matrix_known_90deg_about_z():
    c = math.cos(math.pi / 4.0)
    sn = math.sin(math.pi / 4.0)
    R = np.array(s.rotationMatrix([c, 0.0, 0.0, sn]))
    expected = np.array([[0.0, -1.0, 0.0],
                         [1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0]])
    assert np.linalg.norm(R - expected) < 1e-12


# ----------------------------------------------------------------------------
# skewSymmetric
# ----------------------------------------------------------------------------
def test_skew_symmetric_implements_cross_product():
    a = np.array([1.0, -2.0, 3.0])
    b = np.array([-0.5, 0.4, 1.7])
    S = np.array(s.skewSymmetric(a))
    assert np.linalg.norm(np.cross(a, b) - S @ b) < 1e-14


def test_skew_symmetric_is_antisymmetric_zero_diagonal():
    a = np.array([1.3, -0.7, 2.1])
    S = np.array(s.skewSymmetric(a))
    assert np.linalg.norm(S + S.T) < 1e-14
    assert abs(S[0, 0]) < 1e-14
    assert abs(S[1, 1]) < 1e-14
    assert abs(S[2, 2]) < 1e-14


# ----------------------------------------------------------------------------
# findWMat / quatNormJacobian (FD vs analytic)
# ----------------------------------------------------------------------------
def test_quat_norm_jacobian_matches_finite_differences():
    q = np.array([0.8, 0.2, -0.4, 0.3])  # raw (unnormalized direction)
    Janalytic = np.array(s.quatNormJacobian(q))  # 4x3

    eps = 1e-6
    Jnum = np.zeros((4, 4))
    for j in range(4):
        qp = q.copy(); qm = q.copy()
        qp[j] += eps; qm[j] -= eps
        fp = qp / np.linalg.norm(qp)
        fm = qm / np.linalg.norm(qm)
        Jnum[:, j] = (fp - fm) / (2.0 * eps)

    # Compare the 4x3 analytic block against the first 3 numerical columns.
    assert np.max(np.abs(Janalytic[:, :3] - Jnum[:, :3])) < 1e-6


def test_find_wmat_satisfies_quaternion_kinematics():
    q = _fixed_unit_quat()
    omega = np.array([0.13, -0.07, 0.21])
    W = np.array(s.findWMat(q))

    q0 = q[0]; qv = q[1:]
    qdot_ref = np.empty(4)
    qdot_ref[0] = -0.5 * float(qv @ omega)
    qdot_ref[1:] = 0.5 * (q0 * omega + np.cross(qv, omega))

    qdot = 0.5 * W @ omega
    assert np.linalg.norm(qdot - qdot_ref) < 1e-12


def test_find_wmat_columns_orthogonal_to_q():
    q = _fixed_unit_quat()
    W = np.array(s.findWMat(q))
    assert np.linalg.norm(W.T @ q) < 1e-12


# ----------------------------------------------------------------------------
# drotmatTvecdq (on-manifold FD of R(q)^T v vs analytic)
# ----------------------------------------------------------------------------
def test_drotmat_tvecdq_matches_on_manifold_finite_differences():
    v = _fixed_vec()
    for q in (np.array([1.0, 0.0, 0.0, 0.0]), _fixed_unit_quat()):
        J = np.array(s.drotmatTvecdq(q, v))  # 4x3
        W = np.array(s.findWMat(q))
        analytic = J.T @ (0.5 * W)  # 3x3

        eps = 1e-7
        numerical = np.zeros((3, 3))
        for k in range(3):
            tp = np.zeros(3); tm = np.zeros(3)
            tp[k] = eps; tm[k] = -eps
            fp = np.array(s.rotationMatrix(_perturb_on_sphere(q, tp))).T @ v
            fm = np.array(s.rotationMatrix(_perturb_on_sphere(q, tm))).T @ v
            numerical[:, k] = (fp - fm) / (2.0 * eps)

        assert np.max(np.abs(analytic - numerical)) < 1e-6


# ----------------------------------------------------------------------------
# ddrotmatTvecdqdq -- regression coverage for the rotation-Hessian fix
# ----------------------------------------------------------------------------
def test_ddrotmat_slices_are_symmetric_in_the_two_q_indices():
    q = _fixed_unit_quat()
    v = _fixed_vec()
    H = [np.array(h) for h in s.ddrotmatTvecdqdq(q, v)]
    for k in range(3):
        assert np.linalg.norm(H[k] - H[k].T) < 1e-12


def test_ddrotmat_matches_finite_differences_of_drotmat():
    q = _fixed_unit_quat()
    v = _fixed_vec()
    Hanalytic = [np.array(h) for h in s.ddrotmatTvecdqdq(q, v)]

    eps = 1e-6
    Hnum = [np.zeros((4, 4)) for _ in range(3)]
    for b in range(4):
        qp = q.copy(); qm = q.copy()
        qp[b] += eps; qm[b] -= eps
        dJ = (np.array(s.drotmatTvecdq(qp, v)) - np.array(s.drotmatTvecdq(qm, v))) / (2.0 * eps)
        for k in range(3):
            for a in range(4):
                Hnum[k][a, b] = dJ[a, k]

    for k in range(3):
        assert np.max(np.abs(Hanalytic[k] - Hnum[k])) < 1e-5


def test_ddrotmat_finite_difference_reference_is_symmetric():
    q = _fixed_unit_quat()
    v = _fixed_vec()
    eps = 1e-6
    Hnum = [np.zeros((4, 4)) for _ in range(3)]
    for b in range(4):
        qp = q.copy(); qm = q.copy()
        qp[b] += eps; qm[b] -= eps
        dJ = (np.array(s.drotmatTvecdq(qp, v)) - np.array(s.drotmatTvecdq(qm, v))) / (2.0 * eps)
        for k in range(3):
            for a in range(4):
                Hnum[k][a, b] = dJ[a, k]
    for k in range(3):
        assert np.linalg.norm(Hnum[k] - Hnum[k].T) < 1e-5
