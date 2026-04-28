"""FD tests for the satellite cost: gradients, Hessians, vec/quat
   modes, and crossterm shape.

   Note: file kept at this path for git history; ω_ff feed-forward machinery
   was removed (was always zero in production), so the tests previously
   parametrized over `omega_ref` are simplified accordingly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "build"))

import saltro_py as saltro


def _make_satellite():
    ps = saltro.PlannerSettings()
    J = np.diag([0.067, 0.071, 0.069])
    sat = saltro.Satellite(J, ps)
    sat.addMTQ(np.array([1.0, 0.0, 0.0]), 0.2)
    sat.addMTQ(np.array([0.0, 1.0, 0.0]), 0.2)
    sat.addMTQ(np.array([0.0, 0.0, 1.0]), 0.2)
    sat.addRW(np.array([1.0, 0.0, 0.0]), 0.001, 1e-5, 0.0, 0.02)
    sat.addRW(np.array([0.0, 1.0, 0.0]), 0.001, 1e-5, 0.0, 0.02)
    sat.addRW(np.array([0.0, 0.0, 1.0]), 0.001, 1e-5, 0.0, 0.02)
    return sat


def _nominal_state(sat):
    nx = 7 + sat.numRW
    nu = sat.numMTQ + sat.numRW
    x = np.zeros(nx)
    x[0:3] = np.array([0.01, -0.005, 0.008])
    q = np.array([0.92, 0.10, -0.20, 0.32])
    x[3:7] = q / np.linalg.norm(q)
    x[7] = 1e-4
    x[8] = -5e-5
    x[9] = 2e-5
    u = np.zeros(nu)
    u[:] = [0.05, -0.02, 0.04, 1e-5, -5e-6, 2e-6]
    return x, u


_BORESIGHT = np.array([1.0, 0.0, 0.0])
_TARGET_Q = np.array([np.sqrt(2)/2, 0.0, 0.0, np.sqrt(2)/2])  # 90° about z
_TARGET_VEC = np.array([np.nan, np.cos(np.radians(30)), np.sin(np.radians(30)), 0.0])
_B_ECI = np.array([2.5e-5, -1.5e-5, 3.0e-5])


def _cost_cfg(angle=1e3, ang_vel=1e4, ang_vel_err_dir=0.0,
              ang_vel_err_dir_ratio=0.0, ang_vel_roll_ratio=1.0,
              use_hess=True):
    cfg = saltro.CostConfig()
    cfg.angle = angle
    cfg.ang_vel = ang_vel
    cfg.ang_vel_err_dir = ang_vel_err_dir
    cfg.ang_vel_err_dir_ratio = ang_vel_err_dir_ratio
    cfg.ang_vel_roll_ratio = ang_vel_roll_ratio
    cfg.ang_vel_mag = 0.0
    cfg.mtq_control_weight = 0.0
    cfg.rw_control_weight = 0.0
    cfg.rw_AM_weight = 0.0
    cfg.rw_stic_weight = 0.0
    cfg.RWh_max_mult = 1.0
    cfg.RWh_ok_mult = 0.0
    cfg.RWh_stiction_mult = 0.0
    cfg.use_cost_hess = use_hess
    cfg.setTerminalEmphasis(1.0)
    return cfg


# Hamilton-convention helpers.
def _quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _quat_mult(a, b):
    return np.array([
        a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
        a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
        a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
        a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0],
    ])


def _rot_matrix(q):
    q0, qv = q[0], q[1:]
    qx, qy, qz = qv
    skew = np.array([[0,-qz,qy],[qz,0,-qx],[-qy,qx,0]])
    return (q0*q0 - qv.dot(qv)) * np.eye(3) + 2*np.outer(qv, qv) + 2*q0*skew


# ============================================================================
# Cost shape sanity tests.
# ============================================================================

def test_legacy_and_newpath_both_callable():
    """Sanity: cost callable in both crossterm regimes (legacy w_avang and new α-from-β)."""
    sat = _make_satellite()
    x, u = _nominal_state(sat)
    cfg_legacy = _cost_cfg(ang_vel_err_dir=0.7)
    cfg_newpath = _cost_cfg(ang_vel_err_dir_ratio=0.5)
    c_legacy = sat.stageCost(0, 100, x, u, _BORESIGHT, _TARGET_Q, _B_ECI, cfg_legacy)
    c_newpath = sat.stageCost(0, 100, x, u, _BORESIGHT, _TARGET_Q, _B_ECI, cfg_newpath)
    assert c_legacy >= 0
    assert c_newpath >= 0


def test_crossterm_rewards_error_reducing_motion():
    """ratio>0: ω in error-reducing direction lowers cost."""
    sat = _make_satellite()
    x, _ = _nominal_state(sat)
    u = np.zeros(sat.numMTQ + sat.numRW)
    cfg = _cost_cfg(ang_vel_err_dir_ratio=0.5)

    q = x[3:7]
    q_g = _TARGET_Q.copy()
    if q.dot(q_g) < 0:
        q_g = -q_g
    q_e = _quat_mult(_quat_conj(q), q_g)
    err_axis = q_e[1:4]
    err_axis = err_axis / max(np.linalg.norm(err_axis), 1e-12)

    eps = 1e-4
    x_plus = x.copy(); x_plus[0:3] = +eps * err_axis
    x_minus = x.copy(); x_minus[0:3] = -eps * err_axis
    c_plus = sat.stageCost(0, 100, x_plus, u, _BORESIGHT, _TARGET_Q, _B_ECI, cfg)
    c_minus = sat.stageCost(0, 100, x_minus, u, _BORESIGHT, _TARGET_Q, _B_ECI, cfg)
    assert c_plus < c_minus, (c_plus, c_minus)


# ============================================================================
# Gradient FD tests (quaternion mode).
# ============================================================================

def _fd_grad_x(sat, x, u, cfg, target=None, k=0, N=100, eps=1e-6):
    if target is None:
        target = _TARGET_Q
    g = np.zeros_like(x)
    for i in range(x.size):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        cp = sat.stageCost(k, N, xp, u, _BORESIGHT, target, _B_ECI, cfg)
        cm = sat.stageCost(k, N, xm, u, _BORESIGHT, target, _B_ECI, cfg)
        g[i] = (cp - cm) / (2 * eps)
    return g


@pytest.mark.parametrize("ratio,name", [
    (0.0, "default"),
    (0.5, "lyapunov"),
])
def test_grad_fd_new_path(ratio, name):
    sat = _make_satellite()
    x, u = _nominal_state(sat)
    cfg = _cost_cfg(ang_vel_err_dir_ratio=ratio)
    grad_ana, _, _ = sat.stageCostJacobians(0, 100, x, u,
                                            _BORESIGHT, _TARGET_Q, _B_ECI, cfg)
    grad_fd = _fd_grad_x(sat, x, u, cfg)
    q = x[3:7]
    proj = np.eye(4) - np.outer(q, q)
    grad_fd[3:7] = proj @ grad_fd[3:7]
    np.testing.assert_allclose(grad_ana[0:3], grad_fd[0:3], atol=5e-4,
                                err_msg=f"[{name}] ω-grad mismatch")
    np.testing.assert_allclose(grad_ana[3:7], grad_fd[3:7], atol=5e-4,
                                err_msg=f"[{name}] q-grad mismatch")


@pytest.mark.parametrize("ang_vel_err_dir,name", [
    (0.5, "low"),
    (5.0, "mid"),
])
def test_grad_fd_legacy_path(ang_vel_err_dir, name):
    """Quat-mode legacy formula, ang_vel_err_dir > 0."""
    sat = _make_satellite()
    x, u = _nominal_state(sat)
    cfg = _cost_cfg(ang_vel_err_dir=ang_vel_err_dir)
    grad_ana, _, _ = sat.stageCostJacobians(0, 100, x, u,
                                            _BORESIGHT, _TARGET_Q, _B_ECI, cfg)
    grad_fd = _fd_grad_x(sat, x, u, cfg)
    q = x[3:7]
    proj = np.eye(4) - np.outer(q, q)
    grad_fd[3:7] = proj @ grad_fd[3:7]
    np.testing.assert_allclose(grad_ana[0:3], grad_fd[0:3], atol=5e-4,
                                err_msg=f"[{name}] ω-grad mismatch")
    np.testing.assert_allclose(grad_ana[3:7], grad_fd[3:7], atol=5e-4,
                                err_msg=f"[{name}] q-grad mismatch")


# ============================================================================
# Hessian FD tests (quaternion mode).
# ============================================================================

def _fd_hess_x(sat, x, u, cfg, target=None, k=0, N=100, eps=1e-4):
    if target is None:
        target = _TARGET_Q
    n = x.size
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            xpp = x.copy(); xpp[i] += eps; xpp[j] += eps
            xmm = x.copy(); xmm[i] -= eps; xmm[j] -= eps
            xpm = x.copy(); xpm[i] += eps; xpm[j] -= eps
            xmp = x.copy(); xmp[i] -= eps; xmp[j] += eps
            cpp = sat.stageCost(k, N, xpp, u, _BORESIGHT, target, _B_ECI, cfg)
            cmm = sat.stageCost(k, N, xmm, u, _BORESIGHT, target, _B_ECI, cfg)
            cpm = sat.stageCost(k, N, xpm, u, _BORESIGHT, target, _B_ECI, cfg)
            cmp = sat.stageCost(k, N, xmp, u, _BORESIGHT, target, _B_ECI, cfg)
            H[i, j] = (cpp + cmm - cpm - cmp) / (4 * eps * eps)
    return H


@pytest.mark.parametrize("ratio,name", [
    (0.0, "default"),
    (0.5, "lyapunov"),
])
def test_hess_omega_omega(ratio, name):
    sat = _make_satellite()
    x, u = _nominal_state(sat)
    cfg = _cost_cfg(ang_vel_err_dir_ratio=ratio)
    Hxx, _, _ = sat.stageCostHessians(0, 100, x, u,
                                        _BORESIGHT, _TARGET_Q, _B_ECI, cfg)
    expected = cfg.ang_vel * np.eye(3)
    np.testing.assert_allclose(Hxx[0:3, 0:3], expected, atol=1e-6,
                                err_msg=f"[{name}] (ω,ω) block mismatch")


def test_hess_fd_omega_q_block():
    sat = _make_satellite()
    x, u = _nominal_state(sat)
    cfg = _cost_cfg(ang_vel_err_dir_ratio=0.5)
    Hxx, _, _ = sat.stageCostHessians(0, 100, x, u,
                                        _BORESIGHT, _TARGET_Q, _B_ECI, cfg)
    Hxx_fd = _fd_hess_x(sat, x, u, cfg)
    np.testing.assert_allclose(Hxx[0:3, 3:7], Hxx_fd[0:3, 3:7], atol=2e-2,
                                err_msg="(ω,q) block mismatch")
    np.testing.assert_allclose(Hxx[3:7, 0:3], Hxx[0:3, 3:7].T, atol=1e-10,
                                err_msg="(q,ω) ≠ (ω,q)^T")


# ============================================================================
# Vector-mode (boresight pointing): axis-aware W_ω cost reduction.
# ============================================================================

def test_vector_mode_axis_aware_w_default_no_op():
    sat = _make_satellite()
    x, u = _nominal_state(sat)
    cfg_uniform = _cost_cfg(ang_vel_roll_ratio=1.0)
    cfg_axis = _cost_cfg(ang_vel_roll_ratio=1.0)
    c_uniform = sat.stageCost(0, 100, x, u, _BORESIGHT, _TARGET_VEC, _B_ECI, cfg_uniform)
    c_axis = sat.stageCost(0, 100, x, u, _BORESIGHT, _TARGET_VEC, _B_ECI, cfg_axis)
    assert np.isclose(c_uniform, c_axis)


def test_vector_mode_axis_aware_w_lowers_roll_cost():
    sat = _make_satellite()
    x, u = _nominal_state(sat)
    x_roll = x.copy(); x_roll[0:3] = np.array([0.05, 0.0, 0.0])
    cfg_uniform = _cost_cfg(ang_vel_roll_ratio=1.0)
    cfg_axis = _cost_cfg(ang_vel_roll_ratio=0.05)
    c_uniform = sat.stageCost(0, 100, x_roll, u, _BORESIGHT, _TARGET_VEC, _B_ECI, cfg_uniform)
    c_axis = sat.stageCost(0, 100, x_roll, u, _BORESIGHT, _TARGET_VEC, _B_ECI, cfg_axis)
    assert c_axis < c_uniform, (c_axis, c_uniform)


def test_vector_mode_axis_aware_w_offaxis_unchanged():
    sat = _make_satellite()
    x, u = _nominal_state(sat)
    x_perp = x.copy(); x_perp[0:3] = np.array([0.0, 0.05, 0.0])
    cfg_uniform = _cost_cfg(ang_vel_roll_ratio=1.0)
    cfg_axis = _cost_cfg(ang_vel_roll_ratio=0.05)
    c_uniform = sat.stageCost(0, 100, x_perp, u, _BORESIGHT, _TARGET_VEC, _B_ECI, cfg_uniform)
    c_axis = sat.stageCost(0, 100, x_perp, u, _BORESIGHT, _TARGET_VEC, _B_ECI, cfg_axis)
    assert np.isclose(c_uniform, c_axis), (c_uniform, c_axis)


def test_vector_mode_axis_aware_w_quaternion_mode_unaffected():
    sat = _make_satellite()
    x, u = _nominal_state(sat)
    cfg_uniform = _cost_cfg(ang_vel_roll_ratio=1.0)
    cfg_low_roll = _cost_cfg(ang_vel_roll_ratio=0.05)
    c_uniform = sat.stageCost(0, 100, x, u, _BORESIGHT, _TARGET_Q, _B_ECI, cfg_uniform)
    c_low = sat.stageCost(0, 100, x, u, _BORESIGHT, _TARGET_Q, _B_ECI, cfg_low_roll)
    assert np.isclose(c_uniform, c_low)


@pytest.mark.parametrize("roll_ratio", [0.05, 0.5])
def test_vector_mode_grad_fd(roll_ratio):
    sat = _make_satellite()
    x, u = _nominal_state(sat)
    cfg = _cost_cfg(ang_vel_roll_ratio=roll_ratio)
    grad_ana, _, _ = sat.stageCostJacobians(0, 100, x, u, _BORESIGHT, _TARGET_VEC, _B_ECI, cfg)
    grad_fd = _fd_grad_x(sat, x, u, cfg, target=_TARGET_VEC)
    q = x[3:7]
    proj = np.eye(4) - np.outer(q, q)
    grad_fd[3:7] = proj @ grad_fd[3:7]
    np.testing.assert_allclose(grad_ana[0:3], grad_fd[0:3], atol=5e-4,
                                err_msg=f"[roll={roll_ratio}] ω-grad mismatch")


@pytest.mark.parametrize("roll_ratio", [0.05, 0.5])
def test_vector_mode_hess_omega_omega_block(roll_ratio):
    sat = _make_satellite()
    x, u = _nominal_state(sat)
    cfg = _cost_cfg(ang_vel_roll_ratio=roll_ratio)
    Hxx, _, _ = sat.stageCostHessians(0, 100, x, u, _BORESIGHT, _TARGET_VEC, _B_ECI, cfg)
    bs = _BORESIGHT / np.linalg.norm(_BORESIGHT)
    expected = cfg.ang_vel * np.eye(3) - cfg.ang_vel * (1.0 - roll_ratio) * np.outer(bs, bs)
    np.testing.assert_allclose(Hxx[0:3, 0:3], expected, atol=1e-8,
                                err_msg=f"[roll={roll_ratio}] (ω,ω) block mismatch")


def test_vector_mode_crossterm_rewards_error_reducing_motion():
    sat = _make_satellite()
    x, _ = _nominal_state(sat)
    u = np.zeros(sat.numMTQ + sat.numRW)
    cfg = _cost_cfg(ang_vel_roll_ratio=0.5, ang_vel_err_dir_ratio=0.5)

    q = x[3:7]
    R_T = _rot_matrix(q).T
    r_eci = _TARGET_VEC[1:4] / np.linalg.norm(_TARGET_VEC[1:4])
    bs = _BORESIGHT / np.linalg.norm(_BORESIGHT)
    err_reduce_axis = np.cross(bs, R_T @ r_eci)
    err_reduce_axis = err_reduce_axis / max(np.linalg.norm(err_reduce_axis), 1e-12)

    eps = 1e-4
    x_plus = x.copy(); x_plus[0:3] = +eps * err_reduce_axis
    x_minus = x.copy(); x_minus[0:3] = -eps * err_reduce_axis
    c_plus = sat.stageCost(0, 100, x_plus, u, _BORESIGHT, _TARGET_VEC, _B_ECI, cfg)
    c_minus = sat.stageCost(0, 100, x_minus, u, _BORESIGHT, _TARGET_VEC, _B_ECI, cfg)
    assert c_plus < c_minus, (c_plus, c_minus)


@pytest.mark.parametrize("ratio,name", [
    (0.0, "no_cross"),
    (0.5, "lyapunov"),
])
def test_vector_mode_grad_fd_full(ratio, name):
    sat = _make_satellite()
    x, u = _nominal_state(sat)
    cfg = _cost_cfg(ang_vel_roll_ratio=0.5, ang_vel_err_dir_ratio=ratio)
    grad_ana, _, _ = sat.stageCostJacobians(0, 100, x, u,
                                            _BORESIGHT, _TARGET_VEC, _B_ECI, cfg)
    grad_fd = _fd_grad_x(sat, x, u, cfg, target=_TARGET_VEC)
    q = x[3:7]
    proj = np.eye(4) - np.outer(q, q)
    grad_fd[3:7] = proj @ grad_fd[3:7]
    np.testing.assert_allclose(grad_ana[0:3], grad_fd[0:3], atol=5e-4,
                                err_msg=f"[{name}] vec-mode ω-grad mismatch")
    np.testing.assert_allclose(grad_ana[3:7], grad_fd[3:7], atol=5e-4,
                                err_msg=f"[{name}] vec-mode q-grad mismatch")


def test_vector_mode_hess_omega_q_block():
    sat = _make_satellite()
    x, u = _nominal_state(sat)
    cfg = _cost_cfg(ang_vel_roll_ratio=0.5, ang_vel_err_dir_ratio=0.5)
    Hxx, _, _ = sat.stageCostHessians(0, 100, x, u,
                                        _BORESIGHT, _TARGET_VEC, _B_ECI, cfg)
    Hxx_fd = _fd_hess_x(sat, x, u, cfg, target=_TARGET_VEC)
    np.testing.assert_allclose(Hxx[0:3, 3:7], Hxx_fd[0:3, 3:7], atol=2e-2,
                                err_msg="vec-mode (ω,q) block mismatch")
    np.testing.assert_allclose(Hxx[3:7, 0:3], Hxx[0:3, 3:7].T, atol=1e-10,
                                err_msg="vec-mode (q,ω) ≠ (ω,q)^T")


# ============================================================================
# Vector-mode ANGLE COST refactor — 5 ang_cost_func_type values.
# ============================================================================

def _vec_only_cfg(ang_cost_func_type):
    cfg = saltro.CostConfig()
    cfg.angle = 1e2
    cfg.ang_vel = 0.0
    cfg.ang_vel_mag = 0.0
    cfg.ang_vel_err_dir = 0.0
    cfg.ang_vel_err_dir_ratio = 0.0
    cfg.ang_vel_roll_ratio = 1.0
    cfg.mtq_control_weight = 0.0
    cfg.rw_control_weight = 0.0
    cfg.rw_AM_weight = 0.0
    cfg.rw_stic_weight = 0.0
    cfg.RWh_max_mult = 1.0
    cfg.RWh_ok_mult = 0.0
    cfg.RWh_stiction_mult = 0.0
    cfg.use_cost_hess = True
    cfg.ang_cost_func_type = ang_cost_func_type
    cfg.setTerminalEmphasis(1.0)
    return cfg


@pytest.mark.parametrize("act", [0, 1, 2, 3, 4])
def test_vec_ang_cost_grad_fd(act):
    sat = _make_satellite()
    x, u = _nominal_state(sat)
    cfg = _vec_only_cfg(act)
    grad_ana, _, _ = sat.stageCostJacobians(0, 100, x, u,
                                            _BORESIGHT, _TARGET_VEC, _B_ECI, cfg)
    grad_fd = _fd_grad_x(sat, x, u, cfg, target=_TARGET_VEC)
    q = x[3:7]
    proj = np.eye(4) - np.outer(q, q)
    grad_fd[3:7] = proj @ grad_fd[3:7]
    np.testing.assert_allclose(grad_ana[0:3], 0, atol=1e-10,
                                err_msg=f"[act={act}] ω-grad nonzero (should be 0)")
    np.testing.assert_allclose(grad_ana[7:], 0, atol=1e-10,
                                err_msg=f"[act={act}] RW-grad nonzero (should be 0)")
    np.testing.assert_allclose(grad_ana[3:7], grad_fd[3:7], atol=5e-4,
                                err_msg=f"[act={act}] q-grad mismatch")


@pytest.mark.parametrize("act", [0, 1, 2, 3, 4])
def test_vec_ang_cost_hess_qq_fd(act):
    """(q,q) Hessian projected to tangent plane vs FD."""
    sat = _make_satellite()
    x, u = _nominal_state(sat)
    cfg = _vec_only_cfg(act)
    Hxx, _, _ = sat.stageCostHessians(0, 100, x, u,
                                        _BORESIGHT, _TARGET_VEC, _B_ECI, cfg)
    eps = 1e-4
    H_qq_fd = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            xpp = x.copy(); xpp[3+i] += eps; xpp[3+j] += eps
            xmm = x.copy(); xmm[3+i] -= eps; xmm[3+j] -= eps
            xpm = x.copy(); xpm[3+i] += eps; xpm[3+j] -= eps
            xmp = x.copy(); xmp[3+i] -= eps; xmp[3+j] += eps
            cpp = sat.stageCost(0, 100, xpp, u, _BORESIGHT, _TARGET_VEC, _B_ECI, cfg)
            cmm = sat.stageCost(0, 100, xmm, u, _BORESIGHT, _TARGET_VEC, _B_ECI, cfg)
            cpm = sat.stageCost(0, 100, xpm, u, _BORESIGHT, _TARGET_VEC, _B_ECI, cfg)
            cmp = sat.stageCost(0, 100, xmp, u, _BORESIGHT, _TARGET_VEC, _B_ECI, cfg)
            H_qq_fd[i, j] = (cpp + cmm - cpm - cmp) / (4 * eps * eps)
    q = x[3:7]
    P = np.eye(4) - np.outer(q, q)
    Hqq_proj_fd = P @ H_qq_fd @ P
    Hqq_proj_ana = P @ Hxx[3:7, 3:7] @ P
    np.testing.assert_allclose(Hqq_proj_ana, Hqq_proj_fd, atol=5e-2,
                                err_msg=f"[act={act}] (q,q) projected-Hess mismatch")


def test_vec_ang_cost_aligned_zero_at_target():
    sat = _make_satellite()
    nx = 7 + sat.numRW
    nu = sat.numMTQ + sat.numRW
    x = np.zeros(nx)
    th = np.radians(30) / 2
    x[3:7] = np.array([np.cos(th), 0, 0, np.sin(th)])
    u = np.zeros(nu)
    for act in range(5):
        cfg = _vec_only_cfg(act)
        c = sat.stageCost(0, 100, x, u, _BORESIGHT, _TARGET_VEC, _B_ECI, cfg)
        assert abs(c) < 1e-8, f"[act={act}] cost at aligned q should be 0, got {c}"


def test_vec_ang_cost_no_synthetic_q_roll_invariance():
    """Cost is INVARIANT to roll about boresight when at target (2-DOF semantics)."""
    sat = _make_satellite()
    nx = 7 + sat.numRW
    nu = sat.numMTQ + sat.numRW
    x_aligned = np.zeros(nx)
    th = np.radians(30) / 2
    x_aligned[3:7] = np.array([np.cos(th), 0, 0, np.sin(th)])
    u = np.zeros(nu)
    cfg = _vec_only_cfg(3)
    c0 = sat.stageCost(0, 100, x_aligned, u, _BORESIGHT, _TARGET_VEC, _B_ECI, cfg)
    bs_unit = _BORESIGHT / np.linalg.norm(_BORESIGHT)
    roll_th = np.radians(45) / 2
    q_roll_local = np.array([np.cos(roll_th)] + list(np.sin(roll_th) * bs_unit))
    q_aligned = x_aligned[3:7]
    q_combined = _quat_mult(q_aligned, q_roll_local)
    x_rolled = x_aligned.copy()
    x_rolled[3:7] = q_combined / np.linalg.norm(q_combined)
    c_rolled = sat.stageCost(0, 100, x_rolled, u, _BORESIGHT, _TARGET_VEC, _B_ECI, cfg)
    assert abs(c0) < 1e-8
    assert abs(c_rolled) < 1e-8, f"Roll about boresight should leave cost at 0, got {c_rolled}"
