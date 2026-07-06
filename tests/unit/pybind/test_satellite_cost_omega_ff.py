"""Focused cost tests for the legacy/new ω-error crossterm and roll-aware
vector-mode weighting.

The original ω_ff feed-forward path was removed, but this file still covers the
branch-specific behavior that is not well exercised by the broader cost suites:
legacy vs ratio-based crossterms, vector-mode roll-axis de-weighting, the mixed
(ω,q) derivative blocks for those branches, and the 2-DOF pointing semantics.
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
    sat = saltro.Satellite(np.diag([0.067, 0.071, 0.069]), ps)
    for axis in np.eye(3):
        sat.addMTQ(axis, 0.2)
        sat.addRW(axis, 0.001, 1e-5, 0.0, 0.02)
    return sat


def _nominal_state(sat):
    x = np.zeros(7 + sat.numRW)
    x[0:3] = np.array([0.01, -0.005, 0.008])
    q = np.array([0.92, 0.10, -0.20, 0.32])
    x[3:7] = q / np.linalg.norm(q)
    x[7:10] = np.array([1e-4, -5e-5, 2e-5])

    u = np.zeros(sat.numMTQ + sat.numRW)
    u[:] = [0.05, -0.02, 0.04, 1e-5, -5e-6, 2e-6]
    return x, u


_BORESIGHT = np.array([1.0, 0.0, 0.0])
_TARGET_Q = np.array([np.sqrt(2.0) / 2.0, 0.0, 0.0, np.sqrt(2.0) / 2.0])
_TARGET_VEC = np.array([np.nan, np.cos(np.radians(30.0)), np.sin(np.radians(30.0)), 0.0])
_B_ECI = np.array([2.5e-5, -1.5e-5, 3.0e-5])


def _cost_cfg(*, angle=1e3, ang_vel=1e4, ang_vel_err_dir=0.0,
              ang_vel_err_dir_ratio=0.0, ang_vel_roll_ratio=1.0, use_hess=True):
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
    cfg.RWh_knee_frac = 1.0
    cfg.RWh_desat_mult = 0.0
    cfg.RWh_stiction_mult = 0.0
    cfg.use_cost_hess = use_hess
    cfg.setTerminalEmphasis(1.0)
    return cfg


def _quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _quat_mult(a, b):
    return np.array([
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ])


def _rot_matrix(q):
    q0, qx, qy, qz = q
    qv = q[1:]
    skew = np.array([[0.0, -qz, qy], [qz, 0.0, -qx], [-qy, qx, 0.0]])
    return (q0 * q0 - qv.dot(qv)) * np.eye(3) + 2.0 * np.outer(qv, qv) + 2.0 * q0 * skew


def _projected_qblock_fd_grad(sat, x, u, cfg, target, eps=1e-6):
    grad = np.zeros_like(x)
    for i in range(x.size):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        cp = sat.stageCost(0, 100, xp, u, _BORESIGHT, target, _B_ECI, cfg)
        cm = sat.stageCost(0, 100, xm, u, _BORESIGHT, target, _B_ECI, cfg)
        grad[i] = (cp - cm) / (2.0 * eps)
    q = x[3:7]
    proj = np.eye(4) - np.outer(q, q)
    grad[3:7] = proj @ grad[3:7]
    return grad


def _fd_hess_x(sat, x, u, cfg, target, eps=1e-4):
    hess = np.zeros((x.size, x.size))
    for i in range(x.size):
        for j in range(x.size):
            xpp = x.copy()
            xmm = x.copy()
            xpm = x.copy()
            xmp = x.copy()
            xpp[i] += eps
            xpp[j] += eps
            xmm[i] -= eps
            xmm[j] -= eps
            xpm[i] += eps
            xpm[j] -= eps
            xmp[i] -= eps
            xmp[j] += eps
            cpp = sat.stageCost(0, 100, xpp, u, _BORESIGHT, target, _B_ECI, cfg)
            cmm = sat.stageCost(0, 100, xmm, u, _BORESIGHT, target, _B_ECI, cfg)
            cpm = sat.stageCost(0, 100, xpm, u, _BORESIGHT, target, _B_ECI, cfg)
            cmp = sat.stageCost(0, 100, xmp, u, _BORESIGHT, target, _B_ECI, cfg)
            hess[i, j] = (cpp + cmm - cpm - cmp) / (4.0 * eps * eps)
    return hess


def test_quat_crossterm_paths_callable():
    sat = _make_satellite()
    x, u = _nominal_state(sat)
    for cfg in (
        _cost_cfg(ang_vel_err_dir=0.7),
        _cost_cfg(ang_vel_err_dir_ratio=0.5),
    ):
        cost = sat.stageCost(0, 100, x, u, _BORESIGHT, _TARGET_Q, _B_ECI, cfg)
        assert cost >= 0.0


def test_quat_crossterm_rewards_error_reducing_motion():
    sat = _make_satellite()
    x, _ = _nominal_state(sat)
    u = np.zeros(sat.numMTQ + sat.numRW)
    cfg = _cost_cfg(ang_vel_err_dir_ratio=0.5)

    q = x[3:7]
    q_goal = _TARGET_Q.copy()
    if q.dot(q_goal) < 0.0:
        q_goal = -q_goal
    q_err = _quat_mult(_quat_conj(q), q_goal)
    err_axis = q_err[1:4]
    err_axis /= max(np.linalg.norm(err_axis), 1e-12)

    eps = 1e-4
    x_plus = x.copy()
    x_minus = x.copy()
    x_plus[0:3] = eps * err_axis
    x_minus[0:3] = -eps * err_axis

    c_plus = sat.stageCost(0, 100, x_plus, u, _BORESIGHT, _TARGET_Q, _B_ECI, cfg)
    c_minus = sat.stageCost(0, 100, x_minus, u, _BORESIGHT, _TARGET_Q, _B_ECI, cfg)
    assert c_plus < c_minus


@pytest.mark.parametrize(("cfg", "label"), [
    (_cost_cfg(ang_vel_err_dir_ratio=0.0), "new-default"),
    (_cost_cfg(ang_vel_err_dir_ratio=0.5), "new-ratio"),
    (_cost_cfg(ang_vel_err_dir=0.7), "legacy"),
])
def test_quat_crossterm_gradients_match_fd(cfg, label):
    sat = _make_satellite()
    x, u = _nominal_state(sat)
    grad_ana, _, _ = sat.stageCostJacobians(0, 100, x, u, _BORESIGHT, _TARGET_Q, _B_ECI, cfg)
    grad_fd = _projected_qblock_fd_grad(sat, x, u, cfg, _TARGET_Q)
    np.testing.assert_allclose(grad_ana[0:3], grad_fd[0:3], atol=5e-4, err_msg=f"[{label}] ω-grad")
    np.testing.assert_allclose(grad_ana[3:7], grad_fd[3:7], atol=5e-4, err_msg=f"[{label}] q-grad")


def test_quat_crossterm_hessian_blocks_match_fd():
    sat = _make_satellite()
    x, u = _nominal_state(sat)
    cfg = _cost_cfg(ang_vel_err_dir_ratio=0.5)
    hxx, _, _ = sat.stageCostHessians(0, 100, x, u, _BORESIGHT, _TARGET_Q, _B_ECI, cfg)
    hxx_fd = _fd_hess_x(sat, x, u, cfg, _TARGET_Q)
    np.testing.assert_allclose(hxx[0:3, 0:3], cfg.ang_vel * np.eye(3), atol=1e-6)
    np.testing.assert_allclose(hxx[0:3, 3:7], hxx_fd[0:3, 3:7], atol=2e-2)
    np.testing.assert_allclose(hxx[3:7, 0:3], hxx[0:3, 3:7].T, atol=1e-10)


def test_vector_roll_ratio_semantics():
    sat = _make_satellite()
    x, u = _nominal_state(sat)
    cfg_uniform = _cost_cfg(ang_vel_roll_ratio=1.0)
    cfg_low_roll = _cost_cfg(ang_vel_roll_ratio=0.05)

    x_roll = x.copy()
    x_roll[0:3] = np.array([0.05, 0.0, 0.0])
    x_perp = x.copy()
    x_perp[0:3] = np.array([0.0, 0.05, 0.0])

    c_same = sat.stageCost(0, 100, x, u, _BORESIGHT, _TARGET_VEC, _B_ECI, cfg_uniform)
    c_same2 = sat.stageCost(0, 100, x, u, _BORESIGHT, _TARGET_VEC, _B_ECI, _cost_cfg(ang_vel_roll_ratio=1.0))
    c_roll_uniform = sat.stageCost(0, 100, x_roll, u, _BORESIGHT, _TARGET_VEC, _B_ECI, cfg_uniform)
    c_roll_low = sat.stageCost(0, 100, x_roll, u, _BORESIGHT, _TARGET_VEC, _B_ECI, cfg_low_roll)
    c_perp_uniform = sat.stageCost(0, 100, x_perp, u, _BORESIGHT, _TARGET_VEC, _B_ECI, cfg_uniform)
    c_perp_low = sat.stageCost(0, 100, x_perp, u, _BORESIGHT, _TARGET_VEC, _B_ECI, cfg_low_roll)
    c_quat_uniform = sat.stageCost(0, 100, x, u, _BORESIGHT, _TARGET_Q, _B_ECI, cfg_uniform)
    c_quat_low = sat.stageCost(0, 100, x, u, _BORESIGHT, _TARGET_Q, _B_ECI, cfg_low_roll)

    assert np.isclose(c_same, c_same2)
    assert c_roll_low < c_roll_uniform
    assert np.isclose(c_perp_uniform, c_perp_low)
    assert np.isclose(c_quat_uniform, c_quat_low)


def test_vector_crossterm_rewards_error_reducing_motion():
    sat = _make_satellite()
    x, _ = _nominal_state(sat)
    u = np.zeros(sat.numMTQ + sat.numRW)
    cfg = _cost_cfg(ang_vel_roll_ratio=0.5, ang_vel_err_dir_ratio=0.5)

    q = x[3:7]
    r_eci = _TARGET_VEC[1:4] / np.linalg.norm(_TARGET_VEC[1:4])
    err_axis = np.cross(_BORESIGHT, _rot_matrix(q).T @ r_eci)
    err_axis /= max(np.linalg.norm(err_axis), 1e-12)

    eps = 1e-4
    x_plus = x.copy()
    x_minus = x.copy()
    x_plus[0:3] = eps * err_axis
    x_minus[0:3] = -eps * err_axis

    c_plus = sat.stageCost(0, 100, x_plus, u, _BORESIGHT, _TARGET_VEC, _B_ECI, cfg)
    c_minus = sat.stageCost(0, 100, x_minus, u, _BORESIGHT, _TARGET_VEC, _B_ECI, cfg)
    assert c_plus < c_minus


@pytest.mark.parametrize("ratio", [0.0, 0.5])
def test_vector_roll_and_crossterm_gradients_match_fd(ratio):
    sat = _make_satellite()
    x, u = _nominal_state(sat)
    cfg = _cost_cfg(ang_vel_roll_ratio=0.5, ang_vel_err_dir_ratio=ratio)
    grad_ana, _, _ = sat.stageCostJacobians(0, 100, x, u, _BORESIGHT, _TARGET_VEC, _B_ECI, cfg)
    grad_fd = _projected_qblock_fd_grad(sat, x, u, cfg, _TARGET_VEC)
    np.testing.assert_allclose(grad_ana[0:3], grad_fd[0:3], atol=5e-4)
    np.testing.assert_allclose(grad_ana[3:7], grad_fd[3:7], atol=5e-4)


def test_vector_roll_and_crossterm_hessian_blocks_match_fd():
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
# Vector-mode ANGLE COST refactor — ang_cost_func_type values {0,1,3}.
# (Type 2 (raw acos) was removed: concave + singular at both poles; migrate
#  to type 3 or 0.  Type 4 ((1-c)^2) was removed: exactly type 1 with doubled
#  angle weight.)
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
    cfg.RWh_knee_frac = 1.0
    cfg.RWh_desat_mult = 0.0
    cfg.RWh_stiction_mult = 0.0
    cfg.use_cost_hess = True
    cfg.ang_cost_func_type = ang_cost_func_type
    cfg.setTerminalEmphasis(1.0)
    return cfg


@pytest.mark.parametrize("act", [0, 1, 3])
def test_vec_ang_cost_grad_fd(act):
    sat = _make_satellite()
    x, u = _nominal_state(sat)
    cfg = _vec_only_cfg(act)
    grad_ana, _, _ = sat.stageCostJacobians(0, 100, x, u,
                                            _BORESIGHT, _TARGET_VEC, _B_ECI, cfg)
    grad_fd = _projected_qblock_fd_grad(sat, x, u, cfg, target=_TARGET_VEC)
    np.testing.assert_allclose(grad_ana[0:3], 0, atol=1e-10,
                                err_msg=f"[act={act}] ω-grad nonzero (should be 0)")
    np.testing.assert_allclose(grad_ana[7:], 0, atol=1e-10,
                                err_msg=f"[act={act}] RW-grad nonzero (should be 0)")
    np.testing.assert_allclose(grad_ana[3:7], grad_fd[3:7], atol=5e-4,
                                err_msg=f"[act={act}] q-grad mismatch")


@pytest.mark.parametrize("act", [0, 1, 3])
def test_vec_ang_cost_hess_qq_fd(act):
    """(q,q) Hessian projected to tangent plane vs FD."""
    sat = _make_satellite()
    x, u = _nominal_state(sat)
    x[0:3] = 0.0
    x[7:10] = 0.0
    th = np.radians(30.0) / 2.0
    x[3:7] = np.array([np.cos(th), 0.0, 0.0, np.sin(th)])

def test_vec_ang_cost_aligned_zero_at_target():
    sat = _make_satellite()
    nx = 7 + sat.numRW
    nu = sat.numMTQ + sat.numRW
    x = np.zeros(nx)
    th = np.radians(30) / 2
    x[3:7] = np.array([np.cos(th), 0, 0, np.sin(th)])
    u = np.zeros(nu)
    for act in (0, 1, 3):
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


# ============================================================================
# afc=3 Taylor protection at c=+1: cost matches ½·θ² across full angle range.
# ============================================================================

def test_afc3_taylor_matches_half_theta_squared_near_alignment():
    """afc=3 cost should equal ½·θ² for θ ∈ [0.001°, 179°]. Taylor protection
    handles the c→1 numerical edge; outside that range the exact formula
    is fine via cancellation. The test verifies the Taylor doesn't introduce
    error in the regime where the exact formula already works."""
    sat = _make_satellite()
    cfg = _vec_only_cfg(3)
    cfg.angle = 1.0
    cfg.angle_N = 1.0

    x = np.zeros(sat.stateDim)
    x[saltro.Satellite.QUAT_INDEX] = 1.0
    boresight = np.array([0., 0., 1.])
    B_eci = np.array([0., 0., 0.])
    u = np.zeros(sat.controlDim)

    def vgoal(v): return np.array([np.nan, v[0], v[1], v[2]])

    # θ values spanning the realistic pointing-error range plus extremes
    # near the c = +1 singularity to verify Taylor protection.
    for theta_deg in [0.001, 0.01, 0.1, 1.0, 10.0, 60.0, 120.0, 179.0]:
        theta = np.deg2rad(theta_deg)
        r_eci = np.array([np.sin(theta), 0.0, np.cos(theta)])
        cost = sat.stageCost(0, 100, x, u, boresight, vgoal(r_eci), B_eci, cfg)
        expected = 0.5 * theta * theta
        np.testing.assert_allclose(
            cost, expected, rtol=1e-6, atol=1e-20,
            err_msg=f"afc=3 cost at θ={theta_deg}° differs from ½·θ²: "
                    f"got {cost:.6e}, expected {expected:.6e}")


def test_afc3_taylor_matches_half_theta_squared_near_alignment_quat_mode():
    """Quaternion-goal twin of the test above.  In quat mode the inner scalar
    is d = |q_goal·q| = cos(θ/2) (post-hemisphere-alignment, d ∈ [0, 1]), so
    the afc=3 cost is ½·acos²(d) = ½·(θ/2)² and d → +1 (alignment) is exactly
    the Taylor-protected region.  Beyond the cost value, this also pins the
    analytic Taylor limits of the quat-mode derivative branches:
    dh/dd → −1 and d²h/dd² → 1/3 at d = 1 (the unprotected expressions gave
    dh/dd → 0 and d²h/dd² ≈ 1e12 there)."""
    sat = _make_satellite()
    cfg = _vec_only_cfg(3)
    cfg.angle = 1.0
    cfg.angle_N = 1.0

    QI = saltro.Satellite.QUAT_INDEX
    x = np.zeros(sat.stateDim)
    x[QI] = 1.0  # identity attitude
    boresight = np.array([0., 0., 1.])
    B_eci = np.array([0., 0., 0.])
    u = np.zeros(sat.controlDim)

    def qgoal(theta):
        return np.array([np.cos(theta / 2), np.sin(theta / 2), 0.0, 0.0])

    # Cost sweep: rotation angle θ about +x; cost = ½·(θ/2)².  atol floor
    # absorbs the ~0.5-ulp rounding of cos(θ/2) at the smallest angle.
    for theta_deg in [0.001, 0.01, 0.1, 1.0, 10.0, 60.0, 120.0, 179.0]:
        theta = np.deg2rad(theta_deg)
        cost = sat.stageCost(0, 100, x, u, boresight, qgoal(theta), B_eci, cfg)
        expected = 0.5 * (theta / 2) ** 2
        np.testing.assert_allclose(
            cost, expected, rtol=1e-6, atol=1e-15,
            err_msg=f"afc=3 quat-mode cost at θ={theta_deg}° differs from "
                    f"½·(θ/2)²: got {cost:.6e}, expected {expected:.6e}")

    # Analytic Taylor limits near d = 1 (θ = 1e-4 rad, deep in the Taylor
    # zone where the unprotected c-formula is already catastrophic).
    theta = 1e-4
    s = np.sin(theta / 2)  # ‖(I − qqᵀ)·q_goal‖
    lx, _, _ = sat.stageCostJacobians(0, 100, x, u, boresight, qgoal(theta),
                                      B_eci, cfg)
    # lx_q = w·(dh/dd)·(q_goal − d·q) = (dh/dd)·[0, s, 0, 0] with w = 1.
    dh_dd = lx[QI + 1] / s
    np.testing.assert_allclose(dh_dd, -1.0, atol=1e-6,
                               err_msg="quat-mode afc=3 dh/dd limit at d→1")
    lxx, _, _ = sat.stageCostHessians(0, 100, x, u, boresight, qgoal(theta),
                                      B_eci, cfg)
    # H_qq = P·(d²h/dd²·q_g·q_gᵀ − (dh/dd)·d·I)·P with P = diag(0,1,1,1):
    #   H[1,1] = d²h/dd²·s² − (dh/dd)·d,   H[2,2] = H[3,3] = −(dh/dd)·d.
    np.testing.assert_allclose(lxx[QI + 2, QI + 2], 1.0, atol=1e-6,
                               err_msg="quat-mode afc=3 PwA term −(dh/dd)·d")
    d2h_dd2 = (lxx[QI + 1, QI + 1] - lxx[QI + 2, QI + 2]) / s**2
    np.testing.assert_allclose(d2h_dd2, 1.0 / 3.0, atol=1e-3,
                               err_msg="quat-mode afc=3 d²h/dd² limit at d→1")

    # Exactly aligned (d = 1): gradient projects to zero and the Hessian
    # q-block reduces to the PwA tangent projector +P (not 0 / 1e12 garbage).
    lx0, _, _ = sat.stageCostJacobians(0, 100, x, u, boresight, qgoal(0.0),
                                       B_eci, cfg)
    np.testing.assert_allclose(lx0[QI:QI + 4], 0.0, atol=1e-12)
    lxx0, _, _ = sat.stageCostHessians(0, 100, x, u, boresight, qgoal(0.0),
                                       B_eci, cfg)
    np.testing.assert_allclose(lxx0[QI:QI + 4, QI:QI + 4],
                               np.diag([0., 1., 1., 1.]), atol=1e-9)


# ============================================================================
# afc=3 bounded antipodal clamp at c = −1 ("big but not infinite").
# ============================================================================
# The c = −1 cusp of ½·acos²(c) is GENUINE (Puiseux: φ = π − √(2u)·(1+u/12+…),
# u = 1+c), so it cannot be Taylor-removed like the c = +1 side.  Instead the
# shape clamps below u < 1e-6: (f', f'') are the exact-formula pair evaluated
# at the seam c_eff = −1 + 1e-6 and the value is extended linearly, keeping f
# strictly increasing toward the antipode.  Documented bounds (weight = 1):
#   |f'| ≤ 2220.442…,  f'' ≤ 1.110720…e9,
#   assembled GN q-block max-eig ≤ f''·4·(1−c_eff²) ≈ 8885.76.
# Twin of TEST SECTION 12 in tests/unit/pybind/test_satellite_cost.cpp.

_AC_U_EFF = 1e-6
_AC_C_EFF = -1.0 + _AC_U_EFF
_AC_OMC2_EFF = 1.0 - _AC_C_EFF * _AC_C_EFF          # = 2·u_eff − u_eff²
_AC_S_EFF = np.sqrt(_AC_OMC2_EFF)
_AC_PHI_EFF = np.arccos(_AC_C_EFF)                  # ≈ π − √(2e-6)
_AC_FP_CLAMP = -_AC_PHI_EFF / _AC_S_EFF             # ≈ −2220.442
_AC_FPP_CLAMP = (1.0 / _AC_OMC2_EFF
                 - _AC_PHI_EFF * _AC_C_EFF / (_AC_OMC2_EFF * _AC_S_EFF))  # ≈ 1.1107e9
_AC_GN_EIG_BOUND = _AC_FPP_CLAMP * 4.0 * _AC_OMC2_EFF  # ≈ 8885.76 (× weight)


def _ac_cfg(gn=False):
    cfg = _vec_only_cfg(3)
    cfg.angle = 1.0
    cfg.angle_N = 1.0
    cfg.cost_hess_gauss_newton = gn
    return cfg


def _ac_probe(sat, c):
    """Vec-mode probe at cosine c: boresight +z, target in the x-z plane.

    Returns (c_n, cost, lx, lxx_gn, lxx_fn) where c_n is the cosine the code
    actually sees after it normalizes the target vector (replicated here so
    exact-formula comparisons are bit-honest)."""
    s = np.sqrt(max(1.0 - c * c, 0.0))
    r = np.array([s, 0.0, c])
    c_n = (r / np.linalg.norm(r))[2]  # replicate the code's .normalized()
    tgt = np.array([np.nan, r[0], r[1], r[2]])
    x = np.zeros(sat.stateDim)
    x[saltro.Satellite.QUAT_INDEX] = 1.0
    u = np.zeros(sat.controlDim)
    bs = np.array([0.0, 0.0, 1.0])
    b0 = np.zeros(3)
    cost = sat.stageCost(0, 100, x, u, bs, tgt, b0, _ac_cfg())
    lx, _, _ = sat.stageCostJacobians(0, 100, x, u, bs, tgt, b0, _ac_cfg())
    lxx_gn, _, _ = sat.stageCostHessians(0, 100, x, u, bs, tgt, b0, _ac_cfg(gn=True))
    lxx_fn, _, _ = sat.stageCostHessians(0, 100, x, u, bs, tgt, b0, _ac_cfg(gn=False))
    return c_n, cost, lx, lxx_gn, lxx_fn


def _ac_qblock_maxeig(sat, lxx):
    QI = saltro.Satellite.QUAT_INDEX
    P = np.diag([0.0, 1.0, 1.0, 1.0])  # tangent projector at identity
    return np.linalg.eigvalsh(P @ lxx[QI:QI + 4, QI:QI + 4] @ P).max()


def test_afc3_antipode_exact_above_clamp():
    """(1) Above the clamp (u = 1+c ≥ 1e-6) the raw exact formula is in
    effect: cost equals ½·acos²(c) with no clamping."""
    sat = _make_satellite()
    for u in (2e-6, 1e-5, 1e-4, 1e-2, 0.5):
        c_n, cost, lx, _, _ = _ac_probe(sat, -1.0 + u)
        expected = 0.5 * np.arccos(c_n) ** 2
        np.testing.assert_allclose(
            cost, expected, rtol=1e-14,
            err_msg=f"afc=3 cost at u={u} deviates from the raw exact formula")
        assert np.isfinite(lx).all()


def test_afc3_antipode_clamped_below_threshold():
    """(2) Below the clamp: f' and f'' equal the documented seam values,
    the value is the linear extension (still monotone toward c = −1), and
    everything is finite."""
    sat = _make_satellite()
    QI = saltro.Satellite.QUAT_INDEX
    costs = []
    for u in (9.9e-7, 1e-7, 1e-9, 1e-12, 0.0):
        c = -1.0 + u
        c_n, cost, lx, lxx_gn, lxx_fn = _ac_probe(sat, c)
        # Value: linear extension f = f(c_eff) + f'_clamp·(c − c_eff).
        expected = 0.5 * _AC_PHI_EFF ** 2 + _AC_FP_CLAMP * (c_n - _AC_C_EFF)
        np.testing.assert_allclose(cost, expected, rtol=1e-12,
                                   err_msg=f"clamped value at u={u}")
        # f' via the assembled gradient: |lx_q| = |f'|·|∂c/∂θ| = |f'|·2·sinθ.
        s_n = np.sqrt(max(1.0 - c_n * c_n, 0.0))
        gnorm = np.linalg.norm(lx[QI:QI + 4])
        np.testing.assert_allclose(gnorm, abs(_AC_FP_CLAMP) * 2.0 * s_n,
                                   rtol=1e-9, atol=1e-12,
                                   err_msg=f"clamped f' at u={u}")
        # f'' via the assembled GN outer product: max-eig = f''·4·(1−c²).
        np.testing.assert_allclose(_ac_qblock_maxeig(sat, lxx_gn),
                                   _AC_FPP_CLAMP * 4.0 * (1.0 - c_n * c_n),
                                   rtol=1e-9, atol=1e-12,
                                   err_msg=f"clamped f'' at u={u}")
        assert np.isfinite(cost) and np.isfinite(lx).all()
        assert np.isfinite(lxx_gn).all() and np.isfinite(lxx_fn).all()
        costs.append(cost)
    # Monotone: f strictly increases as c decreases toward the antipode,
    # including across the seam from the exact side.
    _, cost_above, _, _, _ = _ac_probe(sat, -1.0 + 2e-6)
    assert cost_above < costs[0]
    for a, b in zip(costs, costs[1:]):
        assert a < b, "clamped f must stay strictly increasing toward c = −1"


def test_afc3_antipode_assembled_gn_bound_and_gradient():
    """(3) Assembled check: GN q-block max-eig at θ = 179.999° is ≤ the
    documented bound ≈ 8885.8·weight (measured ~7.2e5·weight unclamped), and
    the escape gradient at θ = 179.9° (outside the micro-clamp) is the
    unchanged ≈ 2θ ≈ 2π."""
    sat = _make_satellite()
    QI = saltro.Satellite.QUAT_INDEX
    # θ = 179.999° (u ≈ 1.5e-10, deep inside the clamp).
    theta = np.deg2rad(179.999)
    c_n, cost, lx, lxx_gn, lxx_fn = _ac_probe(sat, np.cos(theta))
    gmax = _ac_qblock_maxeig(sat, lxx_gn)
    assert 0.0 < gmax <= _AC_GN_EIG_BOUND * (1.0 + 1e-9), \
        f"GN max-eig {gmax} exceeds documented clamp bound {_AC_GN_EIG_BOUND}"
    assert np.isfinite(lxx_fn).all()
    # The bound holds across the whole antipodal approach (grow-then-fall-off
    # of the assembled GN curvature, peak at the seam).
    for theta_deg in np.linspace(179.0, 180.0, 41):
        _, _, _, lxx_gn_i, _ = _ac_probe(sat, np.cos(np.deg2rad(theta_deg)))
        assert _ac_qblock_maxeig(sat, lxx_gn_i) <= _AC_GN_EIG_BOUND * (1.0 + 1e-9)
    # θ = 179.9° (u ≈ 1.52e-6 > 1e-6: outside the clamp): |g| = 2θ unchanged.
    theta9 = np.deg2rad(179.9)
    c_n9, _, lx9, _, _ = _ac_probe(sat, np.cos(theta9))
    gnorm9 = np.linalg.norm(lx9[QI:QI + 4])
    np.testing.assert_allclose(gnorm9, 2.0 * np.arccos(c_n9), rtol=1e-9,
                               err_msg="antipode-escape gradient must be "
                                       "unchanged outside the clamp")


def test_afc3_antipode_fd_consistency_above_seam():
    """(4) FD-consistency of f' (= df/dc) vs f just above the seam, and slope
    continuity across the seam (the linear extension starts at exactly the
    seam slope)."""
    sat = _make_satellite()
    QI = saltro.Satellite.QUAT_INDEX
    # Just above the seam: central FD of the cost in c vs assembled f'.
    c0 = -1.0 + 2e-6
    delta = 1e-9
    _, f_p, _, _, _ = _ac_probe(sat, c0 + delta)
    _, f_m, _, _, _ = _ac_probe(sat, c0 - delta)
    fp_fd = (f_p - f_m) / (2.0 * delta)
    c_n0, _, lx0, _, _ = _ac_probe(sat, c0)
    s_n0 = np.sqrt(1.0 - c_n0 * c_n0)
    fp_ana = -np.linalg.norm(lx0[QI:QI + 4]) / (2.0 * s_n0)  # f' < 0 here
    np.testing.assert_allclose(fp_ana, fp_fd, rtol=1e-4,
                               err_msg="f' vs FD(f) just above the seam")
    # Across the seam: FD slope ≈ f'_clamp (C¹ in f/f' by construction).
    _, f_sp, _, _, _ = _ac_probe(sat, _AC_C_EFF + delta)
    _, f_sm, _, _, _ = _ac_probe(sat, _AC_C_EFF - delta)
    fp_seam_fd = (f_sp - f_sm) / (2.0 * delta)
    np.testing.assert_allclose(fp_seam_fd, _AC_FP_CLAMP, rtol=1e-2,
                               err_msg="slope continuity across the clamp seam")
