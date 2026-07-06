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
	cost.RWh_stiction_mult = 0.0
	cost.RWh_knee_frac = 0.0
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
	cost.RWh_stiction_mult = 0.0
	cost.RWh_knee_frac = 0.0
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

