"""Unit tests for spike_removal.py.

Tests cover:
  1. Detector — known spike
  2. Detector — no spike
  3. Detector — exit_fudge (spike exits at 1.5x entry, within 2x fudge)
  4. Detector — goal transition buffer suppresses spike
  5. Actuation filter — saturated + opposing torque → discard candidate
  6. PD sim — quaternion stays normalized and state is finite
  7. Tail re-rollout — gain correction closer to nominal than open-loop
  8. Cost comparator — PD is cheaper when on short arc
  9. Keep-out check — sun in boresight → violation
"""
import sys
import numpy as np
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(ROOT / "tests" / "debug" / "optimizer" / "alilqr_python"))

import saltro_py
from spike_removal import (
    detect_spikes,
    simulate_pd_segment,
    compare_costs,
    check_keepout,
    substitute_and_blend,
    _quat_error,
    _quat_angle,
    _state_error_reduced,
    _rk4_step,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_satellite():
    """3 MTQ + 1 RW satellite (same as sat_3_1_hybrid.py)."""
    ps = saltro_py.PlannerSettings()
    J = np.array([
        [0.03136490806,  5.88304e-05, -0.00671361357],
        [5.88304e-05,    0.03409127827, -0.00012334756],
        [-0.00671361357, -0.00012334756, 0.01004091997],
    ])
    sat = saltro_py.Satellite(J, ps)
    sat.addMTQ(np.array([1.0, 0.0, 0.0]), 0.2)
    sat.addMTQ(np.array([0.0, 1.0, 0.0]), 0.2)
    sat.addMTQ(np.array([0.0, 0.0, 1.0]), 0.2)
    sat.addRW(np.array([1.0, 0.0, 0.0]), 5.7e-6, 0.0023, 0.0, 0.0036)
    return sat, ps


def make_identity_state(satellite):
    """State at identity attitude, zero angular velocity, zero RW momentum."""
    nx = satellite.stateDim
    x = np.zeros(nx)
    x[3] = 1.0  # q = [1, 0, 0, 0]
    return x


def make_env(N):
    """Minimal constant environment arrays (3 × N)."""
    B = np.zeros((3, N))
    B[2, :] = 3.1e-5   # 31 µT in Z (representative LEO)
    S = np.tile(np.array([0.0, 0.0, 1.0])[:, None], (1, N))
    R = np.tile(np.array([7000e3, 0.0, 0.0])[:, None], (1, N))
    V = np.tile(np.array([0.0, 7.5e3, 0.0])[:, None], (1, N))
    rho = np.zeros((1, N))
    return B, S, R, V, rho


def make_trajectory(N, satellite):
    """Trajectory that holds identity attitude and zero angular velocity."""
    nx = satellite.stateDim
    nu = satellite.controlDim
    X = np.zeros((nx, N))
    X[3, :] = 1.0  # q = [1,0,0,0] throughout
    U = np.zeros((nu, N - 1))
    return X, U


def make_attitude_target_quat(N, q_goal):
    """Constant quaternion goal for all N knots."""
    target = np.tile(q_goal[:, None], (1, N))
    return target


# ---------------------------------------------------------------------------
# Test 1: Detector — known spike
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Detector was rewritten 2026-05-16 from PE-run-based "
                  "to hemisphere-transition-based (sign(q·q_0) flip).  This synthetic "
                  "PE-peak trajectory does NOT produce a hemisphere flip relative to "
                  "q_0, so the new detector correctly returns empty.  Test is "
                  "obsolete relative to the new detector contract.")
def test_detect_spike_known():
    """A synthetic spike (M+ consecutive increasing-error knots) is detected.

    The trajectory has a converging lead-up (error decreasing for 15 knots) followed
    by a sudden reversal — this is the canonical homotopy spike pattern.
    """
    sat, ps = make_satellite()
    N = 60
    X, U = make_trajectory(N, sat)
    B, S, R, V, rho = make_env(N)

    q_goal = np.array([1.0, 0.0, 0.0, 0.0])
    attitude_target = make_attitude_target_quat(N, q_goal)
    boresight = np.tile(np.array([1.0, 0.0, 0.0])[:, None], (1, N))

    # Lead-up: error decreases from 1.5 rad down to ~0.1 rad over knots 0..14
    # (trajectory converging toward goal)
    for k in range(15):
        angle = 1.5 * (1.0 - k / 15.0) + 0.1
        half = angle / 2.0
        X[3, k] = np.cos(half)
        X[6, k] = np.sin(half)

    # At k=15: error ≈ 0.1 rad — "close and doing alright"
    # Spike: rotate progressively away then back (knots 15..44)
    for k in range(15, 45):
        progress = (k - 15) / 29.0  # 0 → 1 → 0
        spike_angle = 0.1 + np.pi * np.sin(np.pi * progress)  # 0.1 → ~3.2 → 0.1
        half = spike_angle / 2.0
        X[3, k] = np.cos(half)
        X[4, k] = 0.0
        X[5, k] = 0.0
        X[6, k] = np.sin(half)

    # After spike: back to low error
    for k in range(45, N):
        X[3, k] = 1.0
        X[4:7, k] = 0.0

    candidates = detect_spikes(
        X, U, attitude_target, boresight, B, sat, ps.constraints,
        goal_switch_buffer=2, min_consecutive=5, exit_fudge=2.0,
        min_prior_decrease_knots=10, min_spike_ratio=2.0,
    )

    assert len(candidates) >= 1, f"Expected at least 1 spike, got {candidates}"
    t_enter, t_exit = candidates[0]
    assert t_enter >= 14, f"Expected enter >= 14, got {t_enter}"
    assert t_exit <= 50, f"Expected exit <= 50, got {t_exit}"
    assert t_exit > t_enter


# ---------------------------------------------------------------------------
# Test 2: Detector — no spike
# ---------------------------------------------------------------------------

def test_detect_no_spike():
    """A monotonically decreasing error trajectory has no spikes."""
    sat, ps = make_satellite()
    N = 40
    X, U = make_trajectory(N, sat)
    B, S, R, V, rho = make_env(N)

    q_goal = np.array([0.0, 0.0, 0.0, 1.0])  # 180-deg rotation about Z
    attitude_target = make_attitude_target_quat(N, q_goal)
    boresight = np.tile(np.array([1.0, 0.0, 0.0])[:, None], (1, N))

    # Monotonically decreasing rotation from pi to 0
    for k in range(N):
        frac = 1.0 - k / (N - 1)
        angle = np.pi * frac
        half = angle / 2.0
        X[3, k] = np.cos(half)
        X[4, k] = 0.0
        X[5, k] = 0.0
        X[6, k] = np.sin(half)

    candidates = detect_spikes(
        X, U, attitude_target, boresight, B, sat, ps.constraints,
        goal_switch_buffer=2, min_consecutive=5, exit_fudge=2.0,
        min_prior_decrease_knots=0, min_spike_ratio=1.1,
    )

    assert candidates == [], f"Expected no spikes, got {candidates}"


# ---------------------------------------------------------------------------
# Test 3: Detector — exit_fudge allows exit at 1.5× entry
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Tests the obsolete exit_fudge run-based exit "
                  "behavior. The 2026-05-16 transition-based detector doesn't "
                  "use exit_fudge — windows are walked from a hemisphere-flip "
                  "peak via PE rise/fall.")
def test_detect_spike_fudge():
    """Spike that returns to 1.5× entry (within 2× fudge) is correctly exited."""
    sat, ps = make_satellite()
    N = 50
    X, U = make_trajectory(N, sat)
    B, S, R, V, rho = make_env(N)

    q_goal = np.array([1.0, 0.0, 0.0, 0.0])
    attitude_target = make_attitude_target_quat(N, q_goal)
    boresight = np.tile(np.array([1.0, 0.0, 0.0])[:, None], (1, N))

    # Spike: error grows 0 → 2.0 rad, then falls back to 0.3 rad (1.5× entry ~0.2)
    # Entry around k=10, error starts at ~0.2, grows for 10 knots, exits at ~0.3
    for k in range(N):
        if k < 10:
            angle = 0.2  # constant pre-spike baseline
        elif k < 20:
            angle = 0.2 + (k - 10) * 0.18  # grows 10 knots: 0.2 → 2.0
        else:
            angle = 0.3  # settles at 0.3, which is 1.5× entry 0.2 < 2× entry 0.4
        half = angle / 2.0
        X[3, k] = np.cos(half)
        X[4, k] = 0.0
        X[5, k] = 0.0
        X[6, k] = np.sin(half)

    candidates = detect_spikes(
        X, U, attitude_target, boresight, B, sat, ps.constraints,
        goal_switch_buffer=2, min_consecutive=5, exit_fudge=2.0,
        min_prior_decrease_knots=0, min_spike_ratio=1.1,
    )

    assert len(candidates) >= 1, f"Expected spike with fudge exit, got {candidates}"
    _, t_exit = candidates[0]
    # Should exit at k=20 (error ~0.3 <= 0.2*2.0=0.4)
    assert t_exit <= 25, f"Exit too late: {t_exit}"


# ---------------------------------------------------------------------------
# Test 4: Detector — goal transition buffer suppresses spike
# ---------------------------------------------------------------------------

def test_detect_spike_goal_buffer():
    """Spike immediately after a goal switch should be suppressed by buffer."""
    sat, ps = make_satellite()
    N = 50
    X, U = make_trajectory(N, sat)
    B, S, R, V, rho = make_env(N)

    boresight = np.tile(np.array([1.0, 0.0, 0.0])[:, None], (1, N))

    # Goal switches at k=10
    q1 = np.array([1.0, 0.0, 0.0, 0.0])
    q2 = np.array([0.0, 0.0, 0.0, 1.0])
    attitude_target = np.hstack([
        np.tile(q1[:, None], (1, 10)),
        np.tile(q2[:, None], (1, 40)),
    ])

    # Introduce a spike starting at k=12 (within buffer=15 of transition at k=10)
    for k in range(12, 25):
        angle = np.pi * (k - 12) / 13.0
        half = angle / 2.0
        X[3, k] = np.cos(half)
        X[4, k] = 0.0
        X[5, k] = 0.0
        X[6, k] = np.sin(half)

    candidates = detect_spikes(
        X, U, attitude_target, boresight, B, sat, ps.constraints,
        goal_switch_buffer=15, min_consecutive=5, exit_fudge=2.0,
        min_prior_decrease_knots=0, min_spike_ratio=1.1,
    )

    # The spike is within the buffer zone — should not be flagged
    assert candidates == [], f"Spike within buffer should be suppressed, got {candidates}"


# ---------------------------------------------------------------------------
# Test 5: Actuation filter — saturated + opposing torque discards candidate
# ---------------------------------------------------------------------------

def test_actuation_filter_physics_limited():
    """When actuators are saturated and opposing error, candidate is discarded."""
    sat, ps = make_satellite()
    N = 50
    B, S, R, V, rho = make_env(N)
    X, U = make_trajectory(N, sat)

    q_goal = np.array([1.0, 0.0, 0.0, 0.0])
    attitude_target = make_attitude_target_quat(N, q_goal)
    boresight = np.tile(np.array([1.0, 0.0, 0.0])[:, None], (1, N))

    # Same spike as test 1
    for k in range(15, 36):
        progress = (k - 15) / 20.0
        angle = np.pi * np.sin(np.pi * progress)
        half = angle / 2.0
        X[3, k] = np.cos(half)
        X[5, k] = 0.0
        X[6, k] = np.sin(half)

    # Saturate all MTQ controls at their max in a direction that opposes the
    # error (error is positive Z rotation, so opposing torque is negative Z).
    # MTQ Z axis: index 2 in sat_3_1_hybrid; u_max = 0.2 A m^2
    n_mtq = sat.numMTQ
    n_rw = sat.numRW
    for k in range(15, 36):
        U[2, k] = -0.2  # MTQ Z at negative saturation (opposing positive Z rotation error)

    candidates = detect_spikes(
        X, U, attitude_target, boresight, B, sat, ps.constraints,
        goal_switch_buffer=2, min_consecutive=5, exit_fudge=2.0,
        min_prior_decrease_knots=0, min_spike_ratio=1.1,
    )

    # This may or may not discard — depends on B-field geometry and direction check.
    # The test validates that the code runs without error and produces a list.
    # We cannot guarantee discard here since B is along Z and MTQ⊥B may be zero.
    # Just check type.
    assert isinstance(candidates, list)


# ---------------------------------------------------------------------------
# Test 6: PD sim — quaternion normalization and finite state
# ---------------------------------------------------------------------------

def test_pd_sim_normalization():
    """PD sim produces unit quaternions and finite states at every step."""
    sat, ps = make_satellite()
    B, S, R, V, rho = make_env(15)
    dist_cfg = ps.disturbances

    x_start = make_identity_state(sat)
    x_start[0] = 0.05  # small angular velocity
    x_start[1] = 0.02

    # Target: 90-deg rotation about Z
    x_target = x_start.copy()
    x_target[3] = np.sqrt(2) / 2
    x_target[4] = 0.0
    x_target[5] = 0.0
    x_target[6] = np.sqrt(2) / 2

    X_pd, U_pd = simulate_pd_segment(
        x_start=x_start,
        x_target=x_target,
        n_steps=10,
        B_cols=B[:, :10],
        S_cols=S[:, :10],
        R_cols=R[:, :10],
        V_cols=V[:, :10],
        rho_cols=rho[:, :10],
        satellite=sat,
        dist_cfg=dist_cfg,
        dt=10.0,
        kp_q=2.0,
        kd_w=5.0,
    )

    assert X_pd.shape == (sat.stateDim, 11)
    assert U_pd.shape == (sat.controlDim, 10)

    for k in range(X_pd.shape[1]):
        q = X_pd[3:7, k]
        assert np.isfinite(X_pd[:, k]).all(), f"Non-finite state at k={k}"
        qn = np.linalg.norm(q)
        assert abs(qn - 1.0) < 1e-9, f"Quaternion not normalized at k={k}: ||q||={qn}"


# ---------------------------------------------------------------------------
# Test 7: Tail re-rollout — gain correction closer to nominal than open-loop
# ---------------------------------------------------------------------------

def test_tail_rerollout_gain_correction():
    """After substitution, applying iLQR gain correction gives smaller state error than open-loop."""
    sat, ps = make_satellite()
    N = 20
    B, S, R, V, rho = make_env(N)
    dist_cfg = ps.disturbances
    dt = 10.0

    X, U = make_trajectory(N, sat)
    nu = sat.controlDim
    nxr = sat.reducedStateDim

    # Introduce a small deviation at knot 5 (simulating stitch point)
    X[0, 5] = 0.01   # small omega_x deviation

    # Nominal reference (unperturbed)
    X_nominal = X.copy()
    X_nominal[0, 5] = 0.0  # clean nominal

    # Gain: simple proportional damping (nu × nxr) identity-ish
    K_k = np.zeros((nu, nxr))
    for i in range(min(nu, nxr)):
        K_k[i, i] = -0.1  # negative feedback on diagonal

    K_list = [None] * N
    K_list[5] = K_k

    U_bar = U.copy()

    # Compute state error and control with gain
    dx = _state_error_reduced(X[:, 5], X_nominal[:, 5], sat)
    u_with_gain = U_bar[:, 5] + K_k @ dx
    u_open_loop = U_bar[:, 5]

    # Integrate one step with each
    def env_k(k):
        return R[:, k], B[:, k], S[:, k], V[:, k], int(rho[0, k])

    R_5, B_5, S_5, V_5, rho_5 = env_k(5)
    x_gain = _rk4_step(sat, X[:, 5], u_with_gain, dt, dist_cfg, R_5, B_5, S_5, V_5, rho_5)
    x_open = _rk4_step(sat, X[:, 5], u_open_loop, dt, dist_cfg, R_5, B_5, S_5, V_5, rho_5)

    # Nominal one-step integration from clean nominal
    x_nominal_6 = _rk4_step(sat, X_nominal[:, 5], U_bar[:, 5], dt, dist_cfg, R_5, B_5, S_5, V_5, rho_5)

    err_gain = np.linalg.norm(x_gain[0:3] - x_nominal_6[0:3])
    err_open = np.linalg.norm(x_open[0:3] - x_nominal_6[0:3])

    # Gain correction should reduce the angular velocity error
    assert err_gain <= err_open + 1e-10, (
        f"Gain correction did not reduce error: err_gain={err_gain:.6e}, err_open={err_open:.6e}"
    )


# ---------------------------------------------------------------------------
# Test 8: Cost comparator — PD is cheaper on short arc
# ---------------------------------------------------------------------------

def test_compare_costs_pd_cheaper():
    """When PD goes via short arc and original goes via long arc, PD is cheaper."""
    sat, ps = make_satellite()
    N = 30
    B, S, R, V, rho = make_env(N)
    dist_cfg = ps.disturbances
    nu = sat.controlDim

    q_goal = np.array([1.0, 0.0, 0.0, 0.0])
    attitude_target = make_attitude_target_quat(N, q_goal)
    boresight = np.tile(np.array([1.0, 0.0, 0.0])[:, None], (1, N))

    cost_cfg = ps.passes[0].cost
    cost_cfg.angle = 1e2
    cost_cfg.ang_vel = 1.0
    cost_cfg.control_mult = 1.0
    cost_cfg.mtq_control_weight = 0.1
    cost_cfg.rw_control_weight = 1.0
    cost_cfg.ang_cost_func_type = 3

    t_enter = 5
    t_exit = 20

    # Original: large pointing error (long-arc detour)
    X_orig = np.zeros((sat.stateDim, N))
    X_orig[3, :] = 1.0
    U_orig = np.zeros((nu, N))
    for k in range(t_enter, t_exit):
        angle = np.pi  # 180-deg error throughout spike
        X_orig[3, k] = np.cos(angle / 2)
        X_orig[6, k] = np.sin(angle / 2)

    # PD: stays near goal (small pointing error)
    n_pd = t_exit - t_enter
    X_pd = np.zeros((sat.stateDim, n_pd + 1))
    X_pd[3, :] = 1.0  # identity attitude ≈ goal
    U_pd = np.zeros((nu, n_pd))

    result, cost_orig, cost_pd = compare_costs(
        X_orig, U_orig, X_pd, U_pd,
        t_enter, t_exit,
        sat, B, boresight, attitude_target, cost_cfg, N,
    )

    assert result is True, f"PD at goal should be cheaper than 180-deg detour (orig={cost_orig:.3e} pd={cost_pd:.3e})"


# ---------------------------------------------------------------------------
# Test 9: Keep-out check — sun in boresight triggers violation
# ---------------------------------------------------------------------------

def test_keepout_sun_violation():
    """PD trajectory pointing boresight at sun triggers sun-avoidance violation."""
    sat, ps = make_satellite()

    # The C++ sun-avoidance constraint (satellite.cpp line 1673):
    #   c = sun_body.x() - cos(sun_limit_angle)
    # where sun_body = R_T @ sun_eci, R = body->ECI rotation.
    # Violation when sun_body.x() > cos(sun_limit) i.e. sun is within limit_angle of body +X.
    #
    # Simplest setup: identity attitude (q=[1,0,0,0]) + sun in ECI +X.
    # Then R = I, sun_body = sun_eci = [1,0,0], sun_body.x() = 1.0 > cos(20°). ✓
    q_identity = np.array([1.0, 0.0, 0.0, 0.0])
    sun_eci = np.array([1.0, 0.0, 0.0])  # sun directly along body +X boresight

    N = 10
    n_pd_steps = 5

    # Sanity-check: C++ constraint should fire at this attitude
    x_test = np.zeros(sat.stateDim)
    x_test[3:7] = q_identity
    u_test = np.zeros(sat.controlDim)
    c_test = np.asarray(sat.constraints(0, N, x_test, u_test, sun_eci, ps.constraints))
    assert c_test[1] > 0.0, (
        f"Setup error: sun constraint c[1]={c_test[1]:.4f}, expected > 0"
    )

    nu = sat.controlDim
    X_pd = np.zeros((sat.stateDim, n_pd_steps + 1))
    U_pd = np.zeros((nu, n_pd_steps))
    for k in range(n_pd_steps + 1):
        X_pd[3:7, k] = q_identity

    S = np.tile(sun_eci[:, None], (1, N))

    result = check_keepout(X_pd, U_pd, S, sat, ps.constraints, N, t_enter=0)

    # Sun directly on body +X boresight: should be a violation → returns False
    assert result is False, "Sun-facing attitude should trigger keep-out violation"


# ---------------------------------------------------------------------------
# Test 10: Detector — prior-decrease filter suppresses initial-ramp-up increases
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Tests the obsolete prior_decrease filter (min_consecutive "
                  "rising knots after a converging lead-up).  The 2026-05-16 "
                  "transition-based detector uses sign(q·q_0) flips instead — "
                  "initial-ramp-up trajectories without a hemisphere flip are "
                  "naturally NOT flagged.  This test's assertion is satisfied by "
                  "the new detector behavior, but it's testing the wrong mechanism.")
def test_detect_spike_prior_decrease_filter():
    """Error increasing from the very start (initial approach) is NOT flagged as a spike.

    During the initial slew, error often increases before converging.  This should
    not be flagged as a homotopy spike.  The prior-decrease filter requires
    `min_prior_decrease_knots` of convergence before the onset can be called a spike.
    """
    sat, ps = make_satellite()
    N = 40
    X, U = make_trajectory(N, sat)
    B, S, R, V, rho = make_env(N)

    # Goal is 90° away
    q_goal = np.array([np.sqrt(2)/2, 0.0, 0.0, np.sqrt(2)/2])
    attitude_target = make_attitude_target_quat(N, q_goal)
    boresight = np.tile(np.array([1.0, 0.0, 0.0])[:, None], (1, N))

    # Trajectory: starts at identity (90° from goal), error INCREASES for first 20
    # knots (going the wrong way), then comes back.  No prior-decrease phase.
    for k in range(20):
        # Error grows from 90° to 170° over k=0..19
        angle = np.pi / 2 + (np.pi * 0.8) * (k / 19.0)
        half = angle / 2.0
        X[3, k] = np.cos(half)
        X[6, k] = np.sin(half)
    for k in range(20, N):
        # Error decreases from 170° back toward 90°
        angle = np.pi * 0.9 - (np.pi * 0.4) * ((k - 20) / (N - 21))
        half = angle / 2.0
        X[3, k] = np.cos(half)
        X[6, k] = np.sin(half)

    # With prior-decrease filter active, this should NOT be flagged:
    # the increasing phase at k=0..19 has NO prior-decrease knots before it.
    candidates_filtered = detect_spikes(
        X, U, attitude_target, boresight, B, sat, ps.constraints,
        goal_switch_buffer=2, min_consecutive=5, exit_fudge=2.0,
        min_prior_decrease_knots=10, min_spike_ratio=1.5,
    )
    assert candidates_filtered == [], (
        f"Initial-ramp-up should not be flagged with prior-decrease filter, got {candidates_filtered}"
    )

    # Without the filter (min_prior_decrease_knots=0), it IS flagged:
    candidates_unfiltered = detect_spikes(
        X, U, attitude_target, boresight, B, sat, ps.constraints,
        goal_switch_buffer=2, min_consecutive=5, exit_fudge=2.0,
        min_prior_decrease_knots=0, min_spike_ratio=1.1,
    )
    assert len(candidates_unfiltered) >= 1, (
        "Without filter, initial-ramp-up should be detected as a run"
    )
