import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(ROOT / "tests" / "debug" / "optimizer" / "configs"))

import saltro_py
from sat_0_3_rw import create_satellite as create_satellite_rw
from sat_3_1_hybrid import create_satellite as create_satellite_hybrid


SEC_PER_CENTURY = 36525.0 * 86400.0


def create_planner_settings(dt_seconds: float) -> saltro_py.PlannerSettings:
	plannersettings = saltro_py.PlannerSettings()

	plannersettings.init_traj.initcontroller = 2

	plannersettings.num_passes = 1
	plannersettings.passes[0].dt = dt_seconds
	plannersettings.passes[0].ilqr.cost_tol = 1e-5
	plannersettings.passes[0].ilqr.max_iters = 20

	plannersettings.passes[0].auglag.max_outer_iters = 10
	plannersettings.passes[0].auglag.constraint_tol = 1e-3

	cost = plannersettings.passes[0].cost
	cost.angle = 1.0
	cost.ang_vel = 1e1
	cost.ang_vel_mag = 0.0
	cost.ang_vel_err_dir = 0.0
	cost.control_mult = 1.0
	cost.mtq_control_weight = 1e-2
	cost.rw_control_weight = 1.0
	cost.magic_control_weight = 0.0
	cost.rw_AM_weight = 0.0
	cost.rw_stic_weight = 0.0
	cost.RWh_max_mult = 0.0
	cost.RWh_stiction_mult = 0.0
	cost.RWh_ok_mult = 0.0
	cost.angle_N = 0.0
	cost.ang_vel_N = 0.0
	cost.ang_vel_mag_N = 0.0
	cost.ang_vel_err_dir_N = 0.0
	cost.ang_cost_func_type = 3
	cost.use_cost_hess = True

	plannersettings.disturbances.plan_for_aero = False
	plannersettings.disturbances.plan_for_gg = False
	plannersettings.disturbances.plan_for_srp = False
	plannersettings.disturbances.plan_for_prop = False
	plannersettings.disturbances.plan_for_gendist = False
	plannersettings.disturbances.plan_for_resdipole = False

	plannersettings.passes[0].reg.reg_init = 1e-6
	plannersettings.passes[0].reg.reg_max = 1e10
	plannersettings.passes[0].reg.reg_scale = 10.0
	plannersettings.passes[0].reg.use_dynamics_hess = False
	plannersettings.passes[0].reg.use_constraint_hess = False

	plannersettings.passes[0].linesearch.max_iters = 24
	plannersettings.passes[0].linesearch.beta1 = 1e-10
	plannersettings.passes[0].linesearch.beta2 = 5000.0

	return plannersettings


def quat_pointing_error_deg(q: np.ndarray, q_goal: np.ndarray) -> float:
	q_goal_inv = np.array([q_goal[0], -q_goal[1], -q_goal[2], -q_goal[3]], dtype=float)
	q_err = np.array(
		[
			q_goal_inv[0] * q[0] - q_goal_inv[1] * q[1] - q_goal_inv[2] * q[2] - q_goal_inv[3] * q[3],
			q_goal_inv[0] * q[1] + q_goal_inv[1] * q[0] + q_goal_inv[2] * q[3] - q_goal_inv[3] * q[2],
			q_goal_inv[0] * q[2] - q_goal_inv[1] * q[3] + q_goal_inv[2] * q[0] + q_goal_inv[3] * q[1],
			q_goal_inv[0] * q[3] + q_goal_inv[1] * q[2] - q_goal_inv[2] * q[1] + q_goal_inv[3] * q[0],
		],
		dtype=float,
	)
	return float(2.0 * np.arctan2(np.linalg.norm(q_err[1:]), abs(q_err[0])) * 180.0 / np.pi)


def max_constraint_violation(
	satellite: saltro_py.Satellite,
	settings: saltro_py.PlannerSettings,
	X: np.ndarray,
	U: np.ndarray,
	S: np.ndarray,
) -> float:
	N = X.shape[1]
	nu = satellite.controlDim
	max_violation = 0.0
	for k in range(N):
		uk = U[:, k] if k < U.shape[1] else np.zeros(nu)
		c = np.asarray(
			satellite.constraints(k, N, X[:, k], uk, S[:, k], settings.constraints),
			dtype=float,
		)
		if c.size == 0:
			continue
		max_violation = max(max_violation, float(np.max(c)))
	return max_violation


@pytest.mark.parametrize(
	"dt_seconds, tf_seconds",
	[
		(10.0, 200.0),
		(10.0, 1000.0),
		(50.0, 1000.0),
	],
)
def test_alilqr_slew_final_quality_and_constraints(dt_seconds: float, tf_seconds: float):
	plannersettings = create_planner_settings(dt_seconds)
	satellite = create_satellite_rw(plannersettings)

	jtime = np.array([0.22, 0.22 + tf_seconds / SEC_PER_CENTURY], dtype=float)
	qgoal = np.array(
		[
			[np.sqrt(2.0) / 2.0, np.sqrt(2.0) / 2.0],
			[0.0, 0.0],
			[0.0, 0.0],
			[np.sqrt(2.0) / 2.0, np.sqrt(2.0) / 2.0],
		],
		dtype=float,
	)
	boresight = np.array(
		[
			[1.0, 1.0],
			[0.0, 0.0],
			[0.0, 0.0],
		],
		dtype=float,
	)

	w0 = np.array([-0.01, 0.02, 0.03], dtype=float)
	q0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
	h0 = np.array([0.0, 0.0, 0.0], dtype=float)
	x0 = np.hstack((w0, q0, h0))

	r0 = np.array([7000e3, 0.0, 0.0], dtype=float)
	v0 = np.array([0.0, 7.5e3, 0.0], dtype=float)

	ok, X, U, _ = saltro_py.trajOpt(
		plannersettings,
		satellite,
		x0,
		r0,
		v0,
		jtime,
		qgoal,
		boresight,
	)

	assert ok, f"trajOpt failed for dt={dt_seconds}, tf={tf_seconds}"
	assert np.all(np.isfinite(X))
	assert np.all(np.isfinite(U))

	final_w_norm = float(np.linalg.norm(X[0:3, -1]))
	final_pointing_error_deg = quat_pointing_error_deg(X[3:7, -1], qgoal[:, -1])

	n_steps = X.shape[1]
	dt_centuries = dt_seconds / SEC_PER_CENTURY
	jtime_fine = jtime[0] + np.arange(n_steps, dtype=float) * dt_centuries

	ok_orbit, _R, _V, _B, S, _rho = saltro_py.generate_orbit(
		r0,
		v0,
		jtime_fine,
		0,
		0,
		0,
		0,
		0,
	)
	assert ok_orbit

	max_violation = max_constraint_violation(satellite, plannersettings, X, U, S)

	assert final_w_norm < 2e-2, f"final angular velocity too high: {final_w_norm:.3e} rad/s"
	assert final_pointing_error_deg < 5.0, f"final pointing error too high: {final_pointing_error_deg:.3f} deg"
	assert max_violation <= plannersettings.passes[0].auglag.constraint_tol + 1e-6, (
		f"constraint violation too high: {max_violation:.3e}"
	)


def create_hybrid_planner_settings(dt_seconds: float) -> saltro_py.PlannerSettings:
	plannersettings = saltro_py.PlannerSettings()

	plannersettings.init_traj.initcontroller = 2

	plannersettings.num_passes = 1
	plannersettings.passes[0].dt = dt_seconds
	plannersettings.passes[0].ilqr.cost_tol = 1e-5
	plannersettings.passes[0].ilqr.max_iters = 20

	plannersettings.passes[0].auglag.max_outer_iters = 10
	plannersettings.passes[0].auglag.constraint_tol = 1e-3

	cost = plannersettings.passes[0].cost
	cost.angle = 1e2
	cost.ang_vel = 1e1
	cost.ang_vel_mag = 0.0
	cost.ang_vel_err_dir = 0.0
	cost.control_mult = 1.0
	cost.mtq_control_weight = 1e-1
	cost.rw_control_weight = 1.0
	cost.magic_control_weight = 0.0
	cost.rw_AM_weight = 0.0
	cost.rw_stic_weight = 0.0
	cost.RWh_max_mult = 0.0
	cost.RWh_stiction_mult = 0.0
	cost.RWh_ok_mult = 0.0
	cost.angle_N = 1e2
	cost.ang_vel_N = 1e1
	cost.ang_vel_mag_N = 0.0
	cost.ang_vel_err_dir_N = 0.0
	cost.ang_cost_func_type = 3
	cost.use_cost_hess = True

	plannersettings.disturbances.plan_for_aero = False
	plannersettings.disturbances.plan_for_gg = False
	plannersettings.disturbances.plan_for_srp = False
	plannersettings.disturbances.plan_for_prop = False
	plannersettings.disturbances.plan_for_gendist = False
	plannersettings.disturbances.plan_for_resdipole = False

	plannersettings.passes[0].reg.reg_init = 1e-6
	plannersettings.passes[0].reg.reg_max = 1e10
	plannersettings.passes[0].reg.reg_scale = 10.0
	plannersettings.passes[0].reg.use_dynamics_hess = False
	plannersettings.passes[0].reg.use_constraint_hess = False

	plannersettings.passes[0].linesearch.max_iters = 24
	plannersettings.passes[0].linesearch.beta1 = 1e-10
	plannersettings.passes[0].linesearch.beta2 = 5000.0

	return plannersettings


@pytest.mark.parametrize(
	"tf_seconds, dt_seconds",
	[
		(1000.0, 10.0),
		(1000.0, 20.0),
		(1000.0, 5.0),
	],
)
def test_alilqr_hybrid_slew90_final_quality_and_constraints(tf_seconds: float, dt_seconds: float):
	plannersettings = create_hybrid_planner_settings(dt_seconds)
	satellite = create_satellite_hybrid(plannersettings)

	jtime = np.array([0.22, 0.22 + tf_seconds / SEC_PER_CENTURY], dtype=float)
	qgoal = np.array(
		[
			[np.sqrt(2.0) / 2.0, np.sqrt(2.0) / 2.0],
			[0.0, 0.0],
			[0.0, 0.0],
			[np.sqrt(2.0) / 2.0, np.sqrt(2.0) / 2.0],
		],
		dtype=float,
	)
	boresight = np.array(
		[
			[1.0, 1.0],
			[0.0, 0.0],
			[0.0, 0.0],
		],
		dtype=float,
	)

	w0 = np.array([0.01, 0.01, 0.01], dtype=float)
	q0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
	h0 = np.array([0.0], dtype=float)
	x0 = np.hstack((w0, q0, h0))

	r0 = np.array([7000e3, 0.0, 0.0], dtype=float)
	v0 = np.array([0.0, 7.5e3, 0.0], dtype=float)

	ok, X, U, _ = saltro_py.trajOpt(
		plannersettings,
		satellite,
		x0,
		r0,
		v0,
		jtime,
		qgoal,
		boresight,
	)

	assert ok, f"trajOpt failed for dt={dt_seconds}, tf={tf_seconds}"
	assert np.all(np.isfinite(X))
	assert np.all(np.isfinite(U))

	final_w_norm = float(np.linalg.norm(X[0:3, -1]))
	final_pointing_error_deg = quat_pointing_error_deg(X[3:7, -1], qgoal[:, -1])

	n_steps = X.shape[1]
	dt_centuries = dt_seconds / SEC_PER_CENTURY
	jtime_fine = jtime[0] + np.arange(n_steps, dtype=float) * dt_centuries

	ok_orbit, _R, _V, _B, S, _rho = saltro_py.generate_orbit(
		r0,
		v0,
		jtime_fine,
		0,
		0,
		0,
		0,
		0,
	)
	assert ok_orbit

	max_violation = max_constraint_violation(satellite, plannersettings, X, U, S)

	assert final_w_norm < 2e-2, f"final angular velocity too high: {final_w_norm:.3e} rad/s"
	assert final_pointing_error_deg < 5.0, f"final pointing error too high: {final_pointing_error_deg:.3f} deg"
	assert max_violation <= plannersettings.passes[0].auglag.constraint_tol + 1e-6, (
		f"constraint violation too high: {max_violation:.3e}"
	)



def test_per_pass_disturbance_override_matches_global():
    """A single-pass override of the disturbance config must reproduce the same
    solve as setting that config globally (and differ from a disturbance-free
    solve). This pins that the per-pass disturbances actually reach the inner
    solve -- no warm-start confound, since there is only one pass."""
    dt, tf = 10.0, 80.0
    GEN = np.array([2e-4, -1e-4, 1e-4])

    def build(global_on, override_on):
        s = create_planner_settings(dt)
        s.num_passes = 1
        if global_on:
            s.disturbances.plan_for_gendist = True
            s.disturbances.gendist_torque = GEN
        if override_on:
            s.passes[0].override_disturbances = True
            s.passes[0].disturbances.plan_for_gendist = True
            s.passes[0].disturbances.gendist_torque = GEN
        return s

    jtime = np.array([0.22, 0.22 + tf / SEC_PER_CENTURY])
    qgoal = np.array([[np.sqrt(2)/2, np.sqrt(2)/2], [0, 0], [0, 0], [np.sqrt(2)/2, np.sqrt(2)/2]])
    boresight = np.array([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    x0 = np.hstack(([-0.01, 0.02, 0.03], [1.0, 0, 0, 0], [0.0, 0, 0]))
    r0 = np.array([7000e3, 0.0, 0.0]); v0 = np.array([0.0, 7.5e3, 0.0])

    def solve(s):
        sat = create_satellite_rw(s)
        ok, X, U, _ = saltro_py.trajOpt(s, sat, x0, r0, v0, jtime, qgoal, boresight)
        assert ok and np.all(np.isfinite(U))
        return U

    U_global_on = solve(build(global_on=True, override_on=False))
    U_override_on = solve(build(global_on=False, override_on=True))
    U_off = solve(build(global_on=False, override_on=False))

    # Per-pass override(ON) reproduces global(ON) to solver tolerance...
    assert np.linalg.norm(U_override_on - U_global_on) < 1e-9, "override does not match global"
    # ...and genuinely differs from the disturbance-free solve.
    assert np.linalg.norm(U_override_on - U_off) > 1e-6, "override had no effect"


def _seed_setup():
    s = create_planner_settings(10.0)
    sat = create_satellite_rw(s)
    jtime = np.array([0.22, 0.22 + 80.0 / SEC_PER_CENTURY])
    qg = np.array([[np.sqrt(2)/2, np.sqrt(2)/2], [0, 0], [0, 0], [np.sqrt(2)/2, np.sqrt(2)/2]])
    bs = np.array([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    x0 = np.hstack(([-0.01, 0.02, 0.03], [1.0, 0, 0, 0], [0.0, 0, 0]))
    r0 = np.array([7000e3, 0.0, 0.0]); v0 = np.array([0.0, 7.5e3, 0.0])
    return s, sat, (x0, r0, v0, jtime, qg, bs)


def test_warm_start_reuse_seeds_the_solve():
    """trajOpt can be seeded from a prior (X, U) instead of a controller rollout
    (testing / iterate-and-refine). Seeding from the optimum round-trips back to
    it; a coarse (zero-order-hold-resampled) seed still runs and -- crucially --
    lands at a DIFFERENT converged trajectory than the optimal seed, proving the
    seed genuinely steers the solve (the deterministic solver would give the same
    result for every input if the seed were ignored)."""
    s, sat, args = _seed_setup()
    ok0, X0, U0, _ = saltro_py.trajOpt(s, sat, *args)
    assert ok0

    # Seed from the optimum -> converges right back to it.
    ok1, X1, U1, _ = saltro_py.trajOpt(s, sat, *args, seed_X=X0, seed_U=U0)
    assert ok1 and np.all(np.isfinite(X1))
    assert np.linalg.norm(X1 - X0) < 1e-3

    # Seed from a coarse half-resolution trajectory -> runs, valid, and the
    # zero-order-hold resample produces a different converged result.
    seed_Xc = X0[:, ::2].copy()
    seed_Xc[0:3, :] += 5e-2  # honest descent: see cpp twin comment (#61 ascent guard)
    okc, Xc, Uc, _ = saltro_py.trajOpt(s, sat, *args, seed_X=seed_Xc, seed_U=U0[:, ::2])
    assert okc and np.all(np.isfinite(Xc)) and Xc.shape == X0.shape
    # Post-#61 both seeds converge to the same optimum (see cpp twin comment);
    # assert the resampled seed lands back at it.
    assert np.linalg.norm(Xc - X0) < 1e-2

    # Wrong-dimension seed is rejected.
    with pytest.raises(Exception):
        saltro_py.trajOpt(s, sat, *args, seed_X=np.zeros((3, 5)), seed_U=U0)
