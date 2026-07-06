"""Characterization tests for the chunked TVLQR feedback gains (trajOpt's K output).

trajOpt computes time-varying LQR tracking gains via compute_gains_chunked: it
runs the LQR backward pass over overlapping windows (tvlqr_len / tvlqr_overlap
seconds, stepping at the plan dt) and stitches them, dropping the overlap. This
is the design from McKeen (2025), Sec. 7.4.1 / Fig. 7.4: each chunk's gains
minimize the *remaining-chunk* cost (eq. 7.39), so the chunked gains are a
deliberate local-window approximation -- NOT expected to equal a single
full-horizon TVLQR. These tests therefore pin (a) structural validity, (b) the
no-op limit (a window covering the whole horizon reproduces the un-chunked
gains), (c) that shorter windows actually change the gains while staying
bounded, and (d) the gains' actual job: applying eq. 7.39 to a perturbed
closed-loop rollout tracks the plan far better than open loop.

These run in CI (the file name is not "test_alilqr", which CI deselects). The
trajectories are short (~20 steps) so the solves are quick, and every assertion
is structural or relative -- no absolute convergence thresholds.
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

# A window longer than the (200 s) trajectories below, with zero overlap, yields
# a single chunk spanning the whole horizon -- i.e. chunking is a no-op. (A
# nonzero overlap re-chunks even past the horizon, because the loop's next start
# = end + 1 - overlap_steps lands back inside the trajectory; and a window/dt
# ratio beyond INT_MAX overflows the step count, so keep the magnitude modest.)
SINGLE_CHUNK_LEN = 1000.0


def _make_settings(dt_seconds, tvlqr_len, tvlqr_overlap):
    ps = saltro_py.PlannerSettings()
    ps.init_traj.initcontroller = 2
    ps.num_passes = 1
    ps.passes[0].dt = dt_seconds
    ps.passes[0].ilqr.cost_tol = 1e-5
    ps.passes[0].ilqr.max_iters = 20
    ps.passes[0].auglag.max_outer_iters = 10
    ps.passes[0].auglag.constraint_tol = 1e-3
    c = ps.passes[0].cost
    c.angle = 1.0
    c.ang_vel = 1e1
    c.control_mult = 1.0
    c.mtq_control_weight = 1e-2
    c.rw_control_weight = 1.0
    c.ang_cost_func_type = 3
    c.use_cost_hess = True
    for f in ("plan_for_aero", "plan_for_gg", "plan_for_srp", "plan_for_prop",
              "plan_for_gendist", "plan_for_resdipole"):
        setattr(ps.disturbances, f, False)
    ps.passes[0].reg.reg_init = 1e-6
    ps.passes[0].reg.reg_max = 1e10
    ps.passes[0].reg.reg_scale = 10.0
    ps.passes[0].reg.use_dynamics_hess = False
    ps.passes[0].reg.use_constraint_hess = False
    ps.passes[0].linesearch.max_iters = 24
    ps.passes[0].linesearch.beta1 = 1e-10
    ps.passes[0].linesearch.beta2 = 5000.0
    ps.tvlqr.tvlqr_len = tvlqr_len
    ps.tvlqr.tvlqr_overlap = tvlqr_overlap
    return ps


def _solve(dt=10.0, tf=200.0, tvlqr_len=SINGLE_CHUNK_LEN, tvlqr_overlap=0.0):
    ps = _make_settings(dt, tvlqr_len, tvlqr_overlap)
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


def _gain_list(K, n_red, N):
    """Split the (input_dim, n_red*N) column-stacked K into N per-step blocks."""
    return [K[:, k * n_red:(k + 1) * n_red] for k in range(N)]


def _reduced_error(x_cur, x_plan, nRW):
    """Reduced-state error [dw, MRP(quatError), dh] -- the forward-pass convention."""
    dz = np.zeros(6 + nRW)
    dz[0:3] = x_cur[0:3] - x_plan[0:3]
    q_err = np.array(saltro_py.quatError(x_plan[3:7], x_cur[3:7]))
    dz[3:6] = np.array(saltro_py.quatToMRP(q_err))
    for i in range(nRW):
        dz[6 + i] = x_cur[7 + i] - x_plan[7 + i]
    return dz


def _rk4_step(sat, x, u, dist, R, B, S, V, rho, dt):
    def f(xx):
        return np.array(sat.dynamics(xx, u, dist, R, B, S, V, int(rho)))
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    xn = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    xn[3:7] = xn[3:7] / np.linalg.norm(xn[3:7])  # stay on the unit sphere
    return xn


def _rollout(sat, ps, X, U, Klist, x0_perturbed, R, B, S, V, rho, dt, use_feedback):
    """Roll the (possibly perturbed) state forward, optionally applying eq. 7.39.

    Returns the per-step reduced tracking error ||x_k - x_plan_k||.
    """
    nRW = sat.stateDim - 7
    N = X.shape[1]
    x = x0_perturbed.copy()
    errs = []
    for k in range(N - 1):
        u = U[:, k].copy()
        if use_feedback:
            u = u + Klist[k] @ _reduced_error(x, X[:, k], nRW)
        x = _rk4_step(sat, x, u, ps.disturbances,
                      R[:, k], B[:, k], S[:, k], V[:, k], rho[0, k], dt)
        errs.append(np.linalg.norm(_reduced_error(x, X[:, k + 1], nRW)))
    return np.array(errs)


# ---------------------------------------------------------------------------
# Structural validity
# ---------------------------------------------------------------------------
def test_gains_have_correct_shape_and_are_finite_nonzero():
    r = _solve()
    assert r["ok"]
    n_red = r["sat"].reducedStateDim
    nu = r["sat"].controlDim
    N = r["X"].shape[1]
    K = r["K"]
    assert K.shape == (nu, n_red * N)
    assert np.all(np.isfinite(K))
    assert np.any(K != 0.0)


def test_gains_are_deterministic():
    K1 = _solve()["K"]
    K2 = _solve()["K"]
    assert np.array_equal(K1, K2)


# ---------------------------------------------------------------------------
# No-op limit: a window covering the whole horizon == the un-chunked gains
# ---------------------------------------------------------------------------
def test_window_covering_horizon_is_a_single_chunk_noop():
    # Two zero-overlap windows that both exceed the trajectory duration produce
    # the identical stitched gains: each is a single chunk spanning the whole
    # horizon, so the window length cannot change anything.
    Ka = _solve(tvlqr_len=400.0, tvlqr_overlap=0.0)["K"]
    Kb = _solve(tvlqr_len=5000.0, tvlqr_overlap=0.0)["K"]
    assert np.array_equal(Ka, Kb)


# ---------------------------------------------------------------------------
# Shorter windows actually change the gains, but stay bounded (no stitch blowup)
# ---------------------------------------------------------------------------
def test_short_windows_change_gains_but_stay_bounded():
    K_full = _solve(tvlqr_len=SINGLE_CHUNK_LEN, tvlqr_overlap=0.0)["K"]
    K_chunked = _solve(tvlqr_len=30.0, tvlqr_overlap=10.0)["K"]
    assert K_full.shape == K_chunked.shape
    assert np.all(np.isfinite(K_chunked))
    # Chunking must do something...
    assert np.max(np.abs(K_chunked - K_full)) > 1e-6
    # ...without blowing up at chunk boundaries (local-window gains are no larger
    # than a small multiple of the full-horizon gains).
    assert np.max(np.abs(K_chunked)) <= 5.0 * np.max(np.abs(K_full))


# ---------------------------------------------------------------------------
# The gains' purpose (eq. 7.39): feedback tracks the plan under a perturbation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tvlqr_len, tvlqr_overlap", [(SINGLE_CHUNK_LEN, 0.0), (60.0, 15.0)])
def test_feedback_gains_track_plan_better_than_open_loop(tvlqr_len, tvlqr_overlap):
    r = _solve(tvlqr_len=tvlqr_len, tvlqr_overlap=tvlqr_overlap)
    assert r["ok"]
    sat, ps, X, U, dt = r["sat"], r["ps"], r["X"], r["U"], r["dt"]
    n_red = sat.reducedStateDim
    N = X.shape[1]
    Klist = _gain_list(r["K"], n_red, N)

    jt_fine = 0.22 + np.arange(N) * (dt / SEC_PER_CENTURY)
    ok_orbit, R, V, B, S, rho = saltro_py.generate_orbit(
        r["r0"], r["v0"], jt_fine, 0, 0, 0, 0, 0)
    assert ok_orbit
    rho = rho.reshape(1, -1)

    # Perturb the initial state: ~1 deg about +z and a small rate error.
    x0p = r["x0"].copy()
    x0p[0:3] += np.array([0.002, -0.0015, 0.001])
    half = 0.5 * np.deg2rad(1.0)
    dq = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
    q0 = r["x0"][3:7]
    x0p[3:7] = np.array([
        q0[0] * dq[0] - q0[1] * dq[1] - q0[2] * dq[2] - q0[3] * dq[3],
        q0[0] * dq[1] + q0[1] * dq[0] + q0[2] * dq[3] - q0[3] * dq[2],
        q0[0] * dq[2] - q0[1] * dq[3] + q0[2] * dq[0] + q0[3] * dq[1],
        q0[0] * dq[3] + q0[1] * dq[2] - q0[2] * dq[1] + q0[3] * dq[0],
    ])

    err_open = _rollout(sat, ps, X, U, Klist, x0p, R, B, S, V, rho, dt, use_feedback=False)
    err_closed = _rollout(sat, ps, X, U, Klist, x0p, R, B, S, V, rho, dt, use_feedback=True)

    assert np.all(np.isfinite(err_open))
    assert np.all(np.isfinite(err_closed))
    # Feedback must contain the perturbation: final tracking error at least 2x
    # smaller than open loop (empirically ~35x for the full-window gains).
    assert err_closed[-1] < 0.5 * err_open[-1]
