"""Tests for the disturbance-aware TVLQR gains (McKeen 2025, eq. 7.40).

When settings.tvlqr.disturbance_aware is set, trajOpt augments each per-step
tracking gain to [K_x | K_tau] of width reducedStateDim + 3: in addition to the
reduced-state error it feeds back the disturbance-torque error. The augmented
gain is the optimal feedback of the LQR backward pass on the state augmented
with a constant disturbance-torque channel, so:

  * the K_x block is byte-for-byte identical to the un-augmented gain (the
    disturbance channel has no control authority of its own and is unpenalized,
    so the state recursion is untouched);
  * the K_tau block feeds back (tau_estimate - tau_expected), consistent with
    the forward-pass reduced-state error convention (estimate - plan).

The behavioral test exercises the feature's purpose: the planner bakes a dipole
into the plan, but the real dipole DRIFTS away from it during the maneuver.
Feeding the live mismatch through K_tau holds the planned trajectory better than
state feedback alone -- improving both the RMS and the peak (worst-case)
attitude-tracking error throughout the maneuver, not merely at the end. The win
is largest on disturbance-dominated trajectories (where the controller fights
the disturbance the whole way) and is a no-op when the real disturbance matches
the plan.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(ROOT / "tests" / "debug" / "optimizer" / "configs"))

import saltro_py
from sat_0_3_rw import create_satellite as create_satellite_rw

SEC_PER_CENTURY = 36525.0 * 86400.0
NRED = 9  # reducedStateDim for the 3-RW satellite: 6 + 3


def _make_settings(dt_seconds, disturbance_aware, res_dipole=None):
    ps = saltro_py.PlannerSettings()
    ps.init_traj.initcontroller = 2
    ps.num_passes = 1
    ps.passes[0].dt = dt_seconds
    ps.passes[0].ilqr.cost_tol = 1e-5
    ps.passes[0].ilqr.max_iters = 40
    ps.passes[0].auglag.max_outer_iters = 12
    ps.passes[0].auglag.constraint_tol = 1e-3
    c = ps.passes[0].cost
    # Balanced running/terminal weights (the PlannerSettings defaults have
    # terminal ~10^4x running, which makes any LQR terminal-biased). With
    # comparable weights the gains track the whole trajectory, not just the end.
    c.angle = 1e2
    c.ang_vel = 1e2
    c.angle_N = 1e2
    c.ang_vel_N = 1e2
    c.control_mult = 1.0
    c.mtq_control_weight = 1e-2
    c.rw_control_weight = 1.0
    c.ang_cost_func_type = 3
    c.use_cost_hess = True
    for f in ("plan_for_aero", "plan_for_gg", "plan_for_srp", "plan_for_prop",
              "plan_for_gendist"):
        setattr(ps.disturbances, f, False)
    ps.disturbances.plan_for_resdipole = res_dipole is not None
    if res_dipole is not None:
        ps.disturbances.res_dipole = np.asarray(res_dipole, dtype=float)
    ps.passes[0].reg.reg_init = 1e-6
    ps.passes[0].reg.reg_max = 1e10
    ps.passes[0].reg.reg_scale = 10.0
    ps.passes[0].reg.use_dynamics_hess = False
    ps.passes[0].reg.use_constraint_hess = False
    ps.passes[0].linesearch.max_iters = 24
    ps.passes[0].linesearch.beta1 = 1e-10
    ps.passes[0].linesearch.beta2 = 5000.0
    ps.tvlqr.tvlqr_len = 1000.0
    ps.tvlqr.tvlqr_overlap = 0.0
    ps.tvlqr.disturbance_aware = disturbance_aware
    return ps


def _solve(dt=10.0, tf=200.0, disturbance_aware=False, res_dipole=None):
    ps = _make_settings(dt, disturbance_aware, res_dipole)
    sat = create_satellite_rw(ps)
    jtime = np.array([0.22, 0.22 + tf / SEC_PER_CENTURY])
    r2 = np.sqrt(2.0) / 2.0
    qgoal = np.array([[r2, r2], [0.0, 0.0], [0.0, 0.0], [r2, r2]])
    boresight = np.array([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    x0 = np.hstack(([-0.01, 0.02, 0.03], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0]))
    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, 7.5e3, 0.0])
    ok, X, U, K = saltro_py.trajOpt(ps, sat, x0, r0, v0, jtime, qgoal, boresight)
    return dict(ok=ok, X=X, U=U, K=K, ps=ps, sat=sat, x0=x0, r0=r0, v0=v0, dt=dt)


# --- small quaternion helpers (math module is not bound on this branch) ------
def _rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _quat_err(q_goal, q):
    qgi = np.array([q_goal[0], -q_goal[1], -q_goal[2], -q_goal[3]])
    e = np.array([
        qgi[0] * q[0] - qgi[1] * q[1] - qgi[2] * q[2] - qgi[3] * q[3],
        qgi[0] * q[1] + qgi[1] * q[0] + qgi[2] * q[3] - qgi[3] * q[2],
        qgi[0] * q[2] - qgi[1] * q[3] + qgi[2] * q[0] + qgi[3] * q[1],
        qgi[0] * q[3] + qgi[1] * q[2] - qgi[2] * q[1] + qgi[3] * q[0],
    ])
    return e if e[0] >= 0 else -e


def _reduced_err(x_cur, x_plan, nRW):
    dz = np.zeros(6 + nRW)
    dz[0:3] = x_cur[0:3] - x_plan[0:3]
    e = _quat_err(x_plan[3:7], x_cur[3:7])
    dz[3:6] = 2.0 * e[1:] / (1.0 + e[0])
    dz[6:] = x_cur[7:] - x_plan[7:]
    return dz


def _att_err(x_cur, x_plan):
    e = _quat_err(x_plan[3:7], x_cur[3:7])
    return np.linalg.norm(2.0 * e[1:] / (1.0 + e[0]))


def _gain_blocks(K, w, N):
    return [K[:, k * w:(k + 1) * w] for k in range(N)]


# ---------------------------------------------------------------------------
# Structural: shape, K_x invariance, K_tau presence, determinism
# ---------------------------------------------------------------------------
def test_disturbance_aware_widens_gains_and_preserves_state_feedback():
    off = _solve(disturbance_aware=False, res_dipole=(0.2, -0.1, 0.15))
    on = _solve(disturbance_aware=True, res_dipole=(0.2, -0.1, 0.15))
    assert off["ok"] and on["ok"]
    N = off["X"].shape[1]
    nu = off["sat"].controlDim
    # Width: n_red off, n_red+3 on.
    assert off["K"].shape == (nu, NRED * N)
    assert on["K"].shape == (nu, (NRED + 3) * N)

    Boff = _gain_blocks(off["K"], NRED, N)
    Bon = _gain_blocks(on["K"], NRED + 3, N)
    # K_x block is byte-for-byte identical (augmentation does not touch state FB).
    kx_diff = max(np.max(np.abs(Bon[k][:, :NRED] - Boff[k])) for k in range(N))
    assert kx_diff == 0.0
    # K_tau block is finite and not all zero (a disturbance Jacobian is present).
    assert np.all(np.isfinite(on["K"]))
    ktau = max(np.max(np.abs(Bon[k][:, NRED:])) for k in range(N))
    assert ktau > 1e-9


def test_disturbance_aware_gains_are_deterministic():
    a = _solve(disturbance_aware=True, res_dipole=(0.2, -0.1, 0.15))["K"]
    b = _solve(disturbance_aware=True, res_dipole=(0.2, -0.1, 0.15))["K"]
    assert np.array_equal(a, b)


# Planner bakes in this (large) constant residual dipole; U_plan counteracts it.
# A dynamically-significant disturbance is where the feedforward earns its keep:
# on a disturbance-dominated trajectory the controller fights it the whole way,
# whereas on a fast slew the maneuver dominates and feedback alone copes.
M_PLAN = (1.0, 0.6, -0.8)
# At runtime the real dipole DRIFTS linearly away from the planned value, ending
#100% off (i.e. doubling) -- the changing-disturbance case the feature targets.
M_DRIFT = np.array(M_PLAN, dtype=float)
# Longer horizon so the sustained disturbance accumulates and matters.
BEHAVIORAL_TF = 600.0


def _drift_rollout(r, B_orbit_eval, drift):
    """Roll out under a real dipole that drifts from M_PLAN by `drift` over the
    maneuver. Returns (rms, peak) attitude-tracking error for state feedback
    alone and with the disturbance feedforward.

    B_orbit_eval supplies (R, V, B, S, rho) along the fine trajectory.
    """
    sat, X, U, dt = r["sat"], r["X"], r["U"], r["dt"]
    N = X.shape[1]
    nRW = sat.stateDim - 7
    Kb = _gain_blocks(r["K"], NRED + 3, N)
    R, V, B, S, rho = B_orbit_eval
    m_plan = np.asarray(M_PLAN, dtype=float)

    def rollout(use_disturbance_fb):
        x = r["x0"].copy()
        errs = []
        for k in range(N - 1):
            m_real = m_plan + drift * (k / (N - 1))
            dist_real = saltro_py.DisturbanceConfig()
            dist_real.plan_for_resdipole = True
            dist_real.res_dipole = m_real
            u = U[:, k].copy() + Kb[k][:, :NRED] @ _reduced_err(x, X[:, k], nRW)
            if use_disturbance_fb:
                # K_tau feeds back the live mismatch tau_est - tau_planned.
                B_body = _rotmat(x[3:7]).T @ B[:, k]
                d_tau = np.cross(m_real, B_body) - np.cross(m_plan, B_body)
                u = u + Kb[k][:, NRED:] @ d_tau

            def f(xx):
                return np.array(sat.dynamics(xx, u, dist_real,
                                             R[:, k], B[:, k], S[:, k], V[:, k], int(rho[0, k])))
            k1 = f(x); k2 = f(x + 0.5 * dt * k1); k3 = f(x + 0.5 * dt * k2); k4 = f(x + dt * k3)
            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            x[3:7] = x[3:7] / np.linalg.norm(x[3:7])
            errs.append(_att_err(x, X[:, k + 1]))
        errs = np.asarray(errs)
        return np.sqrt(np.mean(errs ** 2)), errs.max()

    return rollout(False), rollout(True)


def _orbit_for(r):
    N = r["X"].shape[1]
    jt_fine = 0.22 + np.arange(N) * (r["dt"] / SEC_PER_CENTURY)
    ok_orbit, R, V, B, S, rho = saltro_py.generate_orbit(
        r["r0"], r["v0"], jt_fine, 0, 0, 0, 0, 0)
    assert ok_orbit
    return R, V, B, S, rho.reshape(1, -1)


# ---------------------------------------------------------------------------
# Behavioral: the feature's purpose -- track the plan when the real-time
# disturbance differs from / drifts away from the one baked into the plan.
# ---------------------------------------------------------------------------
def test_disturbance_feedback_improves_tracking_under_a_changing_disturbance():
    # Plan bakes in M_PLAN; the real dipole drifts away from it over the maneuver.
    r = _solve(tf=BEHAVIORAL_TF, disturbance_aware=True, res_dipole=M_PLAN)
    assert r["ok"]
    orbit = _orbit_for(r)
    (rms_so, peak_so), (rms_da, peak_da) = _drift_rollout(r, orbit, M_DRIFT)

    assert np.isfinite(rms_so) and np.isfinite(rms_da)
    # Feeding back the live mismatch improves tracking of the planned trajectory
    # in BOTH RMS and peak (worst-case) attitude error. The gain is plan-dependent
    # and toolchain-sensitive, so assert only a conservative margin (ratio <= 0.9);
    # the realized improvement is comfortably larger than the threshold.
    assert rms_da < 0.9 * rms_so
    assert peak_da < 0.92 * peak_so


def test_disturbance_feedback_is_a_noop_when_the_disturbance_matches_the_plan():
    # When the real dipole equals the planned one, the mismatch tau_est-tau_exp
    # is identically zero, so the disturbance feedforward contributes nothing and
    # the closed loop is bit-identical to state feedback alone -- no penalty for
    # enabling the feature when the plan is right.
    r = _solve(tf=BEHAVIORAL_TF, disturbance_aware=True, res_dipole=M_PLAN)
    assert r["ok"]
    orbit = _orbit_for(r)
    (rms_so, peak_so), (rms_da, peak_da) = _drift_rollout(r, orbit, np.zeros(3))
    assert rms_da == pytest.approx(rms_so, rel=1e-9, abs=1e-12)
    assert peak_da == pytest.approx(peak_so, rel=1e-9, abs=1e-12)
