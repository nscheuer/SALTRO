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
    cfg.RWh_max_mult = 1.0
    cfg.RWh_ok_mult = 0.0
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
    hxx, _, _ = sat.stageCostHessians(0, 100, x, u, _BORESIGHT, _TARGET_VEC, _B_ECI, cfg)
    hxx_fd = _fd_hess_x(sat, x, u, cfg, _TARGET_VEC)
    expected = cfg.ang_vel * np.eye(3) - cfg.ang_vel * 0.5 * np.outer(_BORESIGHT, _BORESIGHT)
    np.testing.assert_allclose(hxx[0:3, 0:3], expected, atol=1e-8)
    np.testing.assert_allclose(hxx[0:3, 3:7], hxx_fd[0:3, 3:7], atol=2e-2)
    np.testing.assert_allclose(hxx[3:7, 0:3], hxx[0:3, 3:7].T, atol=1e-10)


def test_vector_angle_cost_has_pointing_semantics():
    sat = _make_satellite()
    x, u = _nominal_state(sat)
    x[0:3] = 0.0
    x[7:10] = 0.0
    th = np.radians(30.0) / 2.0
    x[3:7] = np.array([np.cos(th), 0.0, 0.0, np.sin(th)])

    for act in range(4):
        cfg = _cost_cfg(angle=1e2, ang_vel=0.0)
        cfg.ang_cost_func_type = act
        cost = sat.stageCost(0, 100, x, u, _BORESIGHT, _TARGET_VEC, _B_ECI, cfg)
        assert abs(cost) < 1e-8

    cfg = _cost_cfg(angle=1e2, ang_vel=0.0)
    cfg.ang_cost_func_type = 3
    roll_half = np.radians(45.0) / 2.0
    q_roll = np.array([np.cos(roll_half), np.sin(roll_half), 0.0, 0.0])
    q_rolled = _quat_mult(x[3:7], q_roll)
    x_rolled = x.copy()
    x_rolled[3:7] = q_rolled / np.linalg.norm(q_rolled)
    cost_rolled = sat.stageCost(0, 100, x_rolled, u, _BORESIGHT, _TARGET_VEC, _B_ECI, cfg)
    assert abs(cost_rolled) < 1e-8
