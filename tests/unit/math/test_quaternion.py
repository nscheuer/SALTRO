import math
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


def fixed_unit_quat():
    q = np.array([0.8, 0.2, -0.4, 0.3], dtype=float)
    return q / np.linalg.norm(q)


def fixed_vec():
    return np.array([1.3, -0.7, 2.1], dtype=float)


def quat_mul(p, r):
    return np.array([
        p[0] * r[0] - p[1] * r[1] - p[2] * r[2] - p[3] * r[3],
        p[0] * r[1] + p[1] * r[0] + p[2] * r[3] - p[3] * r[2],
        p[0] * r[2] - p[1] * r[3] + p[2] * r[0] + p[3] * r[1],
        p[0] * r[3] + p[1] * r[2] - p[2] * r[1] + p[3] * r[0],
    ])


def perturb_on_sphere(q, theta):
    n = np.linalg.norm(theta)
    if n < 1e-300:
        dq = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        dq = np.zeros(4)
        dq[0] = np.cos(n / 2.0)
        dq[1:] = np.sin(n / 2.0) * theta / n
    return quat_mul(q, dq)


def test_normalize_quat_produces_unit_norm():
    q = np.array([2.0, -1.0, 0.5, 3.0])
    qn = saltro_py.normalizeQuat(q)
    assert abs(np.linalg.norm(qn) - 1.0) < 1e-12
    assert abs(np.linalg.norm(qn * np.linalg.norm(q) - q)) < 1e-12


def test_normalize_quat_preserves_unit_quaternion():
    q = fixed_unit_quat()
    qn = saltro_py.normalizeQuat(q)
    assert np.linalg.norm(qn - q) < 1e-12


def test_normalize_quat_throws_on_near_zero_quaternion():
    with pytest.raises(Exception):
        saltro_py.normalizeQuat(np.array([1e-15, 0.0, -1e-15, 0.0]))


def test_rotation_matrix_is_orthonormal_with_det_plus_one():
    r = saltro_py.rotationMatrix(fixed_unit_quat())
    assert np.linalg.norm(r @ r.T - np.eye(3)) < 1e-12
    assert abs(np.linalg.det(r) - 1.0) < 1e-12


def test_rotation_matrix_normalizes_input():
    q = fixed_unit_quat()
    r1 = saltro_py.rotationMatrix(q)
    r2 = saltro_py.rotationMatrix(2.5 * q)
    assert np.linalg.norm(r1 - r2) < 1e-12


def test_rotation_matrix_identity_quaternion_gives_identity():
    r = saltro_py.rotationMatrix(np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.linalg.norm(r - np.eye(3)) < 1e-12


def test_rotation_matrix_known_90deg_about_z():
    c = np.cos(np.pi / 4.0)
    s = np.sin(np.pi / 4.0)
    r = saltro_py.rotationMatrix(np.array([c, 0.0, 0.0, s]))
    expected = np.array([[0.0, -1.0, 0.0],
                         [1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0]])
    assert np.linalg.norm(r - expected) < 1e-12


def test_skew_symmetric_implements_cross_product():
    a = np.array([1.0, -2.0, 3.0])
    b = np.array([-0.5, 0.4, 1.7])
    assert np.linalg.norm(np.cross(a, b) - saltro_py.skewSymmetric(a) @ b) < 1e-14


def test_skew_symmetric_is_antisymmetric_with_zero_diagonal():
    s = saltro_py.skewSymmetric(np.array([1.3, -0.7, 2.1]))
    assert np.linalg.norm(s + s.T) < 1e-14
    assert abs(s[0, 0]) < 1e-14
    assert abs(s[1, 1]) < 1e-14
    assert abs(s[2, 2]) < 1e-14


def test_quat_norm_jacobian_matches_finite_difference_first_three_columns():
    q = np.array([0.8, 0.2, -0.4, 0.3], dtype=float)
    janalytic = saltro_py.quatNormJacobian(q)
    eps = 1e-6
    jnum = np.zeros((4, 4))
    for j in range(4):
        qp = q.copy()
        qm = q.copy()
        qp[j] += eps
        qm[j] -= eps
        fp = qp / np.linalg.norm(qp)
        fm = qm / np.linalg.norm(qm)
        jnum[:, j] = (fp - fm) / (2.0 * eps)
    assert np.max(np.abs(janalytic - jnum[:, :3])) < 1e-6


def test_find_w_mat_satisfies_quaternion_kinematics():
    q = fixed_unit_quat()
    omega = np.array([0.13, -0.07, 0.21])
    w = saltro_py.findWMat(q)

    q0 = q[0]
    qv = q[1:]
    qdot_ref = np.zeros(4)
    qdot_ref[0] = -0.5 * np.dot(qv, omega)
    qdot_ref[1:] = 0.5 * (q0 * omega + np.cross(qv, omega))

    qdot = 0.5 * (w @ omega)
    assert np.linalg.norm(qdot - qdot_ref) < 1e-12


def test_find_w_mat_columns_are_orthogonal_to_q():
    q = fixed_unit_quat()
    w = saltro_py.findWMat(q)
    assert np.linalg.norm(w.T @ q) < 1e-12


def test_drotmat_t_vec_dq_matches_on_manifold_finite_differences():
    qs = [np.array([1.0, 0.0, 0.0, 0.0]), fixed_unit_quat()]
    v = fixed_vec()

    for q in qs:
        j = saltro_py.drotmatTvecdq(q, v)
        w = saltro_py.findWMat(q)
        analytic = j.T @ (0.5 * w)

        eps = 1e-7
        numerical = np.zeros((3, 3))
        for k in range(3):
            tp = np.zeros(3)
            tm = np.zeros(3)
            tp[k] = eps
            tm[k] = -eps
            fp = saltro_py.rotationMatrix(perturb_on_sphere(q, tp)).T @ v
            fm = saltro_py.rotationMatrix(perturb_on_sphere(q, tm)).T @ v
            numerical[:, k] = (fp - fm) / (2.0 * eps)

        assert np.max(np.abs(analytic - numerical)) < 1e-6


def test_ddrotmat_t_vec_dqdq_slices_are_symmetric():
    h = saltro_py.ddrotmatTvecdqdq(fixed_unit_quat(), fixed_vec())
    for block in h:
        assert np.linalg.norm(block - block.T) < 1e-12


def test_ddrotmat_t_vec_dqdq_matches_finite_differences_of_drotmat():
    q = fixed_unit_quat()
    v = fixed_vec()
    h_analytic = saltro_py.ddrotmatTvecdqdq(q, v)

    eps = 1e-6
    h_num = [np.zeros((4, 4)), np.zeros((4, 4)), np.zeros((4, 4))]
    for b in range(4):
        qp = q.copy()
        qm = q.copy()
        qp[b] += eps
        qm[b] -= eps
        jp = saltro_py.drotmatTvecdq(qp, v)
        jm = saltro_py.drotmatTvecdq(qm, v)
        dj = (jp - jm) / (2.0 * eps)
        for k in range(3):
            h_num[k][:, b] = dj[:, k]

    for k in range(3):
        assert np.max(np.abs(h_analytic[k] - h_num[k])) < 1e-5


def test_ddrotmat_t_vec_dqdq_finite_difference_reference_is_symmetric():
    q = fixed_unit_quat()
    v = fixed_vec()
    eps = 1e-6
    h_num = [np.zeros((4, 4)), np.zeros((4, 4)), np.zeros((4, 4))]
    for b in range(4):
        qp = q.copy()
        qm = q.copy()
        qp[b] += eps
        qm[b] -= eps
        dj = (saltro_py.drotmatTvecdq(qp, v) - saltro_py.drotmatTvecdq(qm, v)) / (2.0 * eps)
        for k in range(3):
            h_num[k][:, b] = dj[:, k]

    for k in range(3):
        assert np.linalg.norm(h_num[k] - h_num[k].T) < 1e-5
