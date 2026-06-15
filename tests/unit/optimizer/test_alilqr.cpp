#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

#include <Eigen/Dense>

#include <saltro/limits.h>
#include <saltro/optimizer/alilqr.h>
#include <saltro/optimizer/trajOpt.h>
#include <saltro/optimizer/warm_start.h>
#include <saltro/orbit_generation/generate_orbit.h>
#include <saltro/pybind/satellite.h>

using namespace saltro;

namespace {

constexpr double SEC_PER_CENTURY = 36525.0 * 86400.0;
constexpr double PI = 3.14159265358979323846;

PlannerSettings createRWPlannerSettings(double dt_seconds) {
	PlannerSettings plannersettings;

	plannersettings.init_traj.initcontroller = 2;

	plannersettings.num_passes = 1;
	plannersettings.passes[0].dt = dt_seconds;
	plannersettings.passes[0].ilqr.cost_tol = 1e-5;
	plannersettings.passes[0].ilqr.max_iters = 20;

	plannersettings.passes[0].auglag.max_outer_iters = 10;
	plannersettings.passes[0].auglag.constraint_tol = 1e-3;

	auto& cost = plannersettings.passes[0].cost;
	cost.angle = 1.0;
	cost.ang_vel = 1e1;
	cost.ang_vel_mag = 0.0;
	cost.ang_vel_err_dir = 0.0;
	cost.control_mult = 1.0;
	cost.mtq_control_weight = 1e-2;
	cost.rw_control_weight = 1.0;
	cost.magic_control_weight = 0.0;
	cost.rw_AM_weight = 0.0;
	cost.rw_stic_weight = 0.0;
	cost.RWh_stiction_mult = 0.0;
	cost.RWh_ok_mult = 0.0;
	cost.angle_N = 0.0;
	cost.ang_vel_N = 0.0;
	cost.ang_vel_mag_N = 0.0;
	cost.ang_vel_err_dir_N = 0.0;
	cost.ang_cost_func_type = 3;
	cost.use_cost_hess = true;

	plannersettings.disturbances.plan_for_aero = false;
	plannersettings.disturbances.plan_for_gg = false;
	plannersettings.disturbances.plan_for_srp = false;
	plannersettings.disturbances.plan_for_prop = false;
	plannersettings.disturbances.plan_for_gendist = false;
	plannersettings.disturbances.plan_for_resdipole = false;

	plannersettings.passes[0].reg.reg_init = 1e-6;
	plannersettings.passes[0].reg.reg_max = 1e10;
	plannersettings.passes[0].reg.reg_scale = 10.0;
	plannersettings.passes[0].reg.use_dynamics_hess = false;
	plannersettings.passes[0].reg.use_constraint_hess = false;

	plannersettings.passes[0].linesearch.max_iters = 24;
	plannersettings.passes[0].linesearch.beta1 = 1e-10;
	plannersettings.passes[0].linesearch.beta2 = 5000.0;

	return plannersettings;
}

PlannerSettings createHybridPlannerSettings(double dt_seconds) {
	PlannerSettings plannersettings;

	plannersettings.init_traj.initcontroller = 2;

	plannersettings.num_passes = 1;
	plannersettings.passes[0].dt = dt_seconds;
	plannersettings.passes[0].ilqr.cost_tol = 1e-5;
	plannersettings.passes[0].ilqr.max_iters = 20;

	plannersettings.passes[0].auglag.max_outer_iters = 10;
	plannersettings.passes[0].auglag.constraint_tol = 1e-3;

	auto& cost = plannersettings.passes[0].cost;
	cost.angle = 1e2;
	cost.ang_vel = 1e1;
	cost.ang_vel_mag = 0.0;
	cost.ang_vel_err_dir = 0.0;
	cost.control_mult = 1.0;
	cost.mtq_control_weight = 1e-1;
	cost.rw_control_weight = 1.0;
	cost.magic_control_weight = 0.0;
	cost.rw_AM_weight = 0.0;
	cost.rw_stic_weight = 0.0;
	cost.RWh_stiction_mult = 0.0;
	cost.RWh_ok_mult = 0.0;
	cost.angle_N = 1e2;
	cost.ang_vel_N = 1e1;
	cost.ang_vel_mag_N = 0.0;
	cost.ang_vel_err_dir_N = 0.0;
	cost.ang_cost_func_type = 3;
	cost.use_cost_hess = true;

	plannersettings.disturbances.plan_for_aero = false;
	plannersettings.disturbances.plan_for_gg = false;
	plannersettings.disturbances.plan_for_srp = false;
	plannersettings.disturbances.plan_for_prop = false;
	plannersettings.disturbances.plan_for_gendist = false;
	plannersettings.disturbances.plan_for_resdipole = false;

	plannersettings.passes[0].reg.reg_init = 1e-6;
	plannersettings.passes[0].reg.reg_max = 1e10;
	plannersettings.passes[0].reg.reg_scale = 10.0;
	plannersettings.passes[0].reg.use_dynamics_hess = false;
	plannersettings.passes[0].reg.use_constraint_hess = false;

	plannersettings.passes[0].linesearch.max_iters = 24;
	plannersettings.passes[0].linesearch.beta1 = 1e-10;
	plannersettings.passes[0].linesearch.beta2 = 5000.0;

	return plannersettings;
}

Eigen::VectorXd resampleJtimeZeroOrderHold(const Eigen::VectorXd& jtime_coarse, double dt_seconds) {
	const double t0 = jtime_coarse(0);
	const double tN = jtime_coarse(jtime_coarse.size() - 1);
	const double dt_centuries = dt_seconds / SEC_PER_CENTURY;
	const int max_samples = static_cast<int>(std::ceil((tN - t0) / dt_centuries)) + 1;

	Eigen::VectorXd tmp = Eigen::VectorXd::Zero(limits::MAX_LENGTH_TRAJ);
	int n_fine = 0;
	double t = t0;
	for (int k = 0; k < max_samples; ++k) {
		if (t > tN + 1e-12) {
			break;
		}
		tmp(n_fine++) = t;
		t += dt_centuries;
	}

	if (n_fine > 0 && std::abs(tmp(n_fine - 1) - tN) > 1e-12) {
		tmp(n_fine++) = tN;
	}

	return tmp.head(n_fine);
}

double quatPointingErrorDeg(const Eigen::Vector4d& q, const Eigen::Vector4d& q_goal) {
	const Eigen::Vector4d q_goal_inv(q_goal(0), -q_goal(1), -q_goal(2), -q_goal(3));
	Eigen::Vector4d q_err;
	q_err(0) = q_goal_inv(0) * q(0) - q_goal_inv(1) * q(1) - q_goal_inv(2) * q(2) - q_goal_inv(3) * q(3);
	q_err(1) = q_goal_inv(0) * q(1) + q_goal_inv(1) * q(0) + q_goal_inv(2) * q(3) - q_goal_inv(3) * q(2);
	q_err(2) = q_goal_inv(0) * q(2) - q_goal_inv(1) * q(3) + q_goal_inv(2) * q(0) + q_goal_inv(3) * q(1);
	q_err(3) = q_goal_inv(0) * q(3) + q_goal_inv(1) * q(2) - q_goal_inv(2) * q(1) + q_goal_inv(3) * q(0);

	return 2.0 * std::atan2(q_err.tail<3>().norm(), std::abs(q_err(0))) * 180.0 / PI;
}

double maxConstraintViolation(
	const Satellite& satellite,
	const PlannerSettings& settings,
	const Eigen::MatrixXd& X,
	const Eigen::MatrixXd& U,
	const Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ>& S,
	int N
) {
	double max_violation = 0.0;
	for (int k = 0; k < N; ++k) {
		Satellite::VecX xk(satellite.stateDim());
		xk = X.col(k);

		Satellite::VecX uk(satellite.controlDim());
		if (k < U.cols()) {
			uk = U.col(k);
		} else {
			uk.setZero();
		}

		const Satellite::VecX c = satellite.constraints(k, N, xk, uk, S.col(k), settings.constraints);
		if (c.size() > 0) {
			max_violation = std::max(max_violation, c.maxCoeff());
		}
	}
	return max_violation;
}

void runAndCheckCase(
	const PlannerSettings& settings,
	const Satellite& satellite,
	const Satellite::VecX& x0,
	double tf_seconds,
	double dt_seconds
) {
	const Eigen::Vector3d r0(7000e3, 0.0, 0.0);
	const Eigen::Vector3d v0(0.0, 7.5e3, 0.0);

	Eigen::VectorXd jtime(2);
	jtime(0) = 0.22;
	jtime(1) = 0.22 + tf_seconds / SEC_PER_CENTURY;

	Eigen::MatrixXd q_goal(4, 2);
	q_goal << std::sqrt(2.0) / 2.0, std::sqrt(2.0) / 2.0,
			  0.0, 0.0,
			  0.0, 0.0,
			  std::sqrt(2.0) / 2.0, std::sqrt(2.0) / 2.0;

	Eigen::MatrixXd boresight(3, 2);
	boresight << 1.0, 1.0,
				 0.0, 0.0,
				 0.0, 0.0;

	const int state_dim = satellite.stateDim();
	const int input_dim = satellite.controlDim();
	const int reduced_state_dim = satellite.reducedStateDim();

	Eigen::MatrixXd X = Eigen::MatrixXd::Zero(state_dim, limits::MAX_LENGTH_TRAJ);
	Eigen::MatrixXd U = Eigen::MatrixXd::Zero(input_dim, limits::MAX_LENGTH_TRAJ);
	Eigen::MatrixXd K = Eigen::MatrixXd::Zero(input_dim, reduced_state_dim * limits::MAX_LENGTH_TRAJ);
	int N = static_cast<int>(jtime.size());

	bool ok = false;
	REQUIRE_NOTHROW(ok = optimizer::trajOpt(
		settings,
		satellite,
		x0,
		r0,
		v0,
		jtime,
		q_goal,
		boresight,
		X,
		U,
		K,
		state_dim,
		input_dim,
		N
	));

	REQUIRE(ok);
	REQUIRE(N > 0);
	REQUIRE(N <= limits::MAX_LENGTH_TRAJ);

	for (int i = 0; i < X.rows(); ++i) {
		for (int k = 0; k < N; ++k) {
			REQUIRE(std::isfinite(X(i, k)));
		}
	}
	for (int i = 0; i < U.rows(); ++i) {
		for (int k = 0; k < N; ++k) {
			REQUIRE(std::isfinite(U(i, k)));
		}
	}

	const double final_w_norm = X.block(0, N - 1, 3, 1).norm();
	const double final_pointing_error_deg = quatPointingErrorDeg(X.block(3, N - 1, 4, 1), q_goal.col(q_goal.cols() - 1));

	const Eigen::VectorXd jtime_fine = resampleJtimeZeroOrderHold(jtime, dt_seconds);
	REQUIRE(static_cast<int>(jtime_fine.size()) == N);

	Eigen::Matrix<double, 1, limits::MAX_LENGTH_TRAJ> jtime_full;
	jtime_full.setZero();
	jtime_full.leftCols(N) = jtime_fine.transpose();

	Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> R;
	Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> V;
	Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> B;
	Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> S;
	Eigen::Matrix<double, 1, limits::MAX_LENGTH_TRAJ> rho;
	R.setZero();
	V.setZero();
	B.setZero();
	S.setZero();
	rho.setZero();

	const bool ok_orbit = orbits::generate_orbit(r0, v0, jtime_full, N, 0, 0, 0, 0, 0, R, V, B, S, rho);
	REQUIRE(ok_orbit);

	const double max_violation = maxConstraintViolation(satellite, settings, X.leftCols(N), U.leftCols(N), S, N);

	REQUIRE(final_w_norm < 2e-2);
	REQUIRE(final_pointing_error_deg < 5.0);
	REQUIRE(max_violation <= settings.passes[0].auglag.constraint_tol + 1e-6);
}

void runRWCase(double tf_seconds, double dt_seconds) {
	const PlannerSettings settings = createRWPlannerSettings(dt_seconds);
	Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
	J(0, 0) = 0.067;
	J(1, 1) = 0.071;
	J(2, 2) = 0.069;
	Satellite satellite(J, settings);
	satellite.addRW(Eigen::Vector3d::UnitX(), 0.001, 1e-5, 0.0, 0.02);
	satellite.addRW(Eigen::Vector3d::UnitY(), 0.001, 1e-5, 0.0, 0.02);
	satellite.addRW(Eigen::Vector3d::UnitZ(), 0.001, 1e-5, 0.0, 0.02);

	Satellite::VecX x0(satellite.stateDim());
	x0.setZero();
	x0.segment<3>(Satellite::AV_INDEX) = Eigen::Vector3d(-0.01, 0.02, 0.03);
	x0.segment<4>(Satellite::QUAT_INDEX) = Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
	x0.segment(satellite.RW_MOMENTUM_INDEX, 3) = Eigen::Vector3d::Zero();

	runAndCheckCase(settings, satellite, x0, tf_seconds, dt_seconds);
}

void runHybridCase(double tf_seconds, double dt_seconds) {
	const PlannerSettings settings = createHybridPlannerSettings(dt_seconds);
	Eigen::Matrix3d J;
	J << 0.03136490806, 5.88304e-05, -0.00671361357,
		 5.88304e-05, 0.03409127827, -0.00012334756,
		 -0.00671361357, -0.00012334756, 0.01004091997;
	Satellite satellite(J, settings);
	satellite.addMTQ(Eigen::Vector3d::UnitX(), 0.2);
	satellite.addMTQ(Eigen::Vector3d::UnitY(), 0.2);
	satellite.addMTQ(Eigen::Vector3d::UnitZ(), 0.2);
	satellite.addRW(Eigen::Vector3d::UnitX(), 5.7e-6, 0.0023, 0.0, 0.0036);

	Satellite::VecX x0(satellite.stateDim());
	x0.setZero();
	x0.segment<3>(Satellite::AV_INDEX) = Eigen::Vector3d(0.01, 0.01, 0.01);
	x0.segment<4>(Satellite::QUAT_INDEX) = Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
	x0(satellite.RW_MOMENTUM_INDEX) = 0.0;

	runAndCheckCase(settings, satellite, x0, tf_seconds, dt_seconds);
}

// ----------------------------------------------------------------------------
// Per-family AL penalty schedule tests
// ----------------------------------------------------------------------------

struct ALRunResult {
	Eigen::MatrixXd X;
	Eigen::MatrixXd U;
	bool ok = false;
	double max_c = 0.0;
};

// Run alilqr() directly (bypassing trajOpt's convergence-or-throw policy) on
// the 3-RW satellite with a tightened wmax so the AngularVelocity constraint
// is ACTIVE from the start (|w0| = 0.0374 rad/s > wmax). The initial state
// violates the constraint, so penalty ramping is genuinely exercised and the
// chosen per-family penalties influence the solution.
ALRunResult runALILQRDirectRWCase(
	const std::function<void(PlannerSettings&)>& mutate_settings,
	const Eigen::Vector3d& w0 = Eigen::Vector3d(-0.01, 0.02, 0.03)
) {
	const double dt_seconds = 10.0;
	const double tf_seconds = 200.0;

	PlannerSettings settings = createRWPlannerSettings(dt_seconds);
	settings.constraints.wmax = 0.02;
	settings.passes[0].auglag.max_outer_iters = 3;
	settings.passes[0].ilqr.max_iters = 10;
	mutate_settings(settings);

	Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
	J(0, 0) = 0.067;
	J(1, 1) = 0.071;
	J(2, 2) = 0.069;
	Satellite satellite(J, settings);
	satellite.addRW(Eigen::Vector3d::UnitX(), 0.001, 1e-5, 0.0, 0.02);
	satellite.addRW(Eigen::Vector3d::UnitY(), 0.001, 1e-5, 0.0, 0.02);
	satellite.addRW(Eigen::Vector3d::UnitZ(), 0.001, 1e-5, 0.0, 0.02);

	Satellite::VecX x0(satellite.stateDim());
	x0.setZero();
	x0.segment<3>(Satellite::AV_INDEX) = w0;
	x0.segment<4>(Satellite::QUAT_INDEX) = Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);

	const int N = static_cast<int>(tf_seconds / dt_seconds) + 1;
	REQUIRE(N <= limits::MAX_LENGTH_TRAJ);

	Eigen::Matrix<double, 1, limits::MAX_LENGTH_TRAJ> jtime_full;
	jtime_full.setZero();
	for (int k = 0; k < N; ++k) {
		jtime_full(k) = 0.22 + k * dt_seconds / SEC_PER_CENTURY;
	}

	Eigen::MatrixXd q_goal = Eigen::MatrixXd::Zero(4, N);
	Eigen::MatrixXd boresight = Eigen::MatrixXd::Zero(3, N);
	for (int k = 0; k < N; ++k) {
		q_goal.col(k) = Eigen::Vector4d(std::sqrt(2.0) / 2.0, 0.0, 0.0, std::sqrt(2.0) / 2.0);
		boresight.col(k) = Eigen::Vector3d::UnitX();
	}

	const Eigen::Vector3d r0(7000e3, 0.0, 0.0);
	const Eigen::Vector3d v0(0.0, 7.5e3, 0.0);

	Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> R;
	Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> V;
	Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> B;
	Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> S;
	Eigen::Matrix<double, 1, limits::MAX_LENGTH_TRAJ> rho;
	R.setZero();
	V.setZero();
	B.setZero();
	S.setZero();
	rho.setZero();
	REQUIRE(orbits::generate_orbit(r0, v0, jtime_full, N, 1, 2, 0, 0, 0, R, V, B, S, rho));

	Eigen::MatrixXd X = Eigen::MatrixXd::Zero(satellite.stateDim(), limits::MAX_LENGTH_TRAJ);
	Eigen::MatrixXd U = Eigen::MatrixXd::Zero(satellite.controlDim(), limits::MAX_LENGTH_TRAJ);

	const Eigen::VectorXd jtime_vec = jtime_full.leftCols(N).transpose();
	REQUIRE(optimizer::warm_start(settings, satellite, x0, jtime_vec, q_goal, boresight, N, R, V, B, S, rho, X, U));

	ALRunResult result;
	optimizer::ALILQRStatus status = optimizer::ALILQRStatus::MaxOuterIterations;
	double max_c = 0.0;
	REQUIRE_NOTHROW(result.ok = optimizer::alilqr(
		settings,
		0,
		satellite,
		X.leftCols(N),
		U.leftCols(N),
		R.leftCols(N),
		V.leftCols(N),
		B.leftCols(N),
		S.leftCols(N),
		rho.leftCols(N),
		jtime_vec,
		boresight,
		q_goal,
		status,
		max_c
	));

	result.X = X.leftCols(N);
	result.U = U.leftCols(N);
	result.max_c = max_c;

	for (int i = 0; i < result.X.rows(); ++i) {
		for (int k = 0; k < N; ++k) {
			REQUIRE(std::isfinite(result.X(i, k)));
		}
	}
	for (int i = 0; i < result.U.rows(); ++i) {
		for (int k = 0; k < N; ++k) {
			REQUIRE(std::isfinite(result.U(i, k)));
		}
	}

	return result;
}

} // namespace

TEST_CASE("AL per-family penalties: empty vectors match scalar path exactly", "[optimizer][alilqr][per_family]") {
	// Scalar baseline (per-family vectors left empty).
	const ALRunResult base = runALILQRDirectRWCase([](PlannerSettings&) {});

	// Per-family vectors explicitly filled with the scalar values must take
	// the per-family code path yet reproduce the scalar behavior bit-exactly.
	const ALRunResult per_family = runALILQRDirectRWCase([](PlannerSettings& s) {
		auto& aug = s.passes[0].auglag;
		const size_t nf = static_cast<size_t>(ConstraintFamily::NumFamilies);
		aug.penalty_init_per_family.assign(nf, aug.penalty_init);
		aug.penalty_max_per_family.assign(nf, aug.penalty_max);
		aug.penalty_scale_per_family.assign(nf, aug.penalty_scale);
	});

	REQUIRE(base.ok == per_family.ok);
	REQUIRE((base.X - per_family.X).cwiseAbs().maxCoeff() == 0.0);
	REQUIRE((base.U - per_family.U).cwiseAbs().maxCoeff() == 0.0);
}

TEST_CASE("AL per-family penalties: wrong-size vectors fall back to scalar path", "[optimizer][alilqr][per_family]") {
	const ALRunResult base = runALILQRDirectRWCase([](PlannerSettings&) {});

	// Vectors of the wrong length are ignored (scalar fallback) by contract.
	const ALRunResult wrong_size = runALILQRDirectRWCase([](PlannerSettings& s) {
		auto& aug = s.passes[0].auglag;
		aug.penalty_init_per_family.assign(3, 12345.0);
		aug.penalty_max_per_family.assign(2, 0.5);
		aug.penalty_scale_per_family.assign(4, 99.0);
	});

	REQUIRE(base.ok == wrong_size.ok);
	REQUIRE((base.X - wrong_size.X).cwiseAbs().maxCoeff() == 0.0);
	REQUIRE((base.U - wrong_size.U).cwiseAbs().maxCoeff() == 0.0);
}

TEST_CASE("AL per-family penalties: family-specific penalty_init is consumed", "[optimizer][alilqr][per_family]") {
	const ALRunResult base = runALILQRDirectRWCase([](PlannerSettings&) {});

	// Give the AngularVelocity family (which is ACTIVE in this scenario, see
	// runALILQRDirectRWCase) a much larger initial penalty than the scalar
	// default (1e-1). If the per-family value is actually consumed, the AL
	// landscape of the very first inner solve changes and the optimized
	// trajectory must differ from the scalar baseline.
	const ALRunResult hot_family = runALILQRDirectRWCase([](PlannerSettings& s) {
		auto& aug = s.passes[0].auglag;
		const size_t nf = static_cast<size_t>(ConstraintFamily::NumFamilies);
		aug.penalty_init_per_family.assign(nf, aug.penalty_init);
		aug.penalty_init_per_family[static_cast<size_t>(ConstraintFamily::AngularVelocity)] = 1e3;
	});

	REQUIRE((base.U - hot_family.U).cwiseAbs().maxCoeff() > 1e-12);
}

TEST_CASE("AL per-family penalties: contraction ratio 0 means always ramp (same as base)", "[optimizer][alilqr][per_family]") {
	const ALRunResult base = runALILQRDirectRWCase([](PlannerSettings&) {});

	// family_contraction_ratio = 0 disables conditional ramping; combined
	// with per-family vectors equal to the scalars, the result must be
	// identical to the scalar baseline.
	const ALRunResult ratio_zero = runALILQRDirectRWCase([](PlannerSettings& s) {
		auto& aug = s.passes[0].auglag;
		const size_t nf = static_cast<size_t>(ConstraintFamily::NumFamilies);
		aug.penalty_init_per_family.assign(nf, aug.penalty_init);
		aug.penalty_max_per_family.assign(nf, aug.penalty_max);
		aug.penalty_scale_per_family.assign(nf, aug.penalty_scale);
		aug.family_contraction_ratio = 0.0;
	});

	REQUIRE(base.ok == ratio_zero.ok);
	REQUIRE((base.X - ratio_zero.X).cwiseAbs().maxCoeff() == 0.0);
	REQUIRE((base.U - ratio_zero.U).cwiseAbs().maxCoeff() == 0.0);
}

TEST_CASE("State slack: dormant knobs are bit-identical to baseline", "[optimizer][alilqr][slack]") {
	const ALRunResult base = runALILQRDirectRWCase([](PlannerSettings&) {});

	// With use_state_slack=false the slack knobs must be completely inert.
	const ALRunResult dormant = runALILQRDirectRWCase([](PlannerSettings& s) {
		auto& aug = s.passes[0].auglag;
		aug.use_state_slack = false;
		aug.slack_rho = 0.1;
		aug.slack_sigma = 7.0;
		aug.slack_off_tol = 0.5;
	});

	REQUIRE(base.ok == dormant.ok);
	REQUIRE((base.X - dormant.X).cwiseAbs().maxCoeff() == 0.0);
	REQUIRE((base.U - dormant.U).cwiseAbs().maxCoeff() == 0.0);
}

TEST_CASE("State slack: uneconomical slack price reproduces baseline bit-exactly", "[optimizer][alilqr][slack]") {
	// With slack_rho far above any lambda + mu*c reached in this short run,
	// s* = 0 at every site, so all four AL sites (BP stage, BP terminal seed,
	// FP merit, lambda update) must produce baseline numbers exactly. Any
	// site applying the slack inconsistently (wrong family gate, wrong
	// formula) breaks this equality. The run is capped at 3 outer iters with
	// an active AngularVelocity violation, so the phase switch never fires.
	const ALRunResult base = runALILQRDirectRWCase([](PlannerSettings&) {});

	const ALRunResult priced_out = runALILQRDirectRWCase([](PlannerSettings& s) {
		auto& aug = s.passes[0].auglag;
		aug.use_state_slack = true;
		aug.slack_rho = 1e30;
		aug.slack_off_tol = 1e-9;  // never "reasonably satisfied" mid-run
	});

	REQUIRE(base.ok == priced_out.ok);
	REQUIRE((base.X - priced_out.X).cwiseAbs().maxCoeff() == 0.0);
	REQUIRE((base.U - priced_out.U).cwiseAbs().maxCoeff() == 0.0);
}

TEST_CASE("State slack: cheap slack changes the solve", "[optimizer][alilqr][slack]") {
	// A slack price low enough to be economical against the active
	// AngularVelocity violation must actually alter the AL landscape (the
	// relaxation is consumed, not just plumbed through).
	const ALRunResult base = runALILQRDirectRWCase([](PlannerSettings&) {});

	const ALRunResult slack = runALILQRDirectRWCase([](PlannerSettings& s) {
		auto& aug = s.passes[0].auglag;
		aug.use_state_slack = true;
		aug.slack_rho = 1e-2;
	});

	REQUIRE((base.U - slack.U).cwiseAbs().maxCoeff() > 1e-12);
}

TEST_CASE("State slack: two-phase solve converges and meets the TRUE constraint tol", "[optimizer][alilqr][slack]") {
	// Full slack -> polish cycle on a FEASIBLE but binding problem (the
	// default harness x0 violates wmax at the fixed initial knot, which no
	// optimizer can repair): start at rest inside the limit with a 90 deg
	// slew whose time budget forces |w| close to wmax along the way.
	// Convergence is declared only in the polish phase, so ok=true proves at
	// least one slack-free inner solve ran, and max_c is the true
	// (unslacked) violation.
	const auto mutate = [](PlannerSettings& s) {
		s.constraints.wmax = 0.012;
		s.passes[0].auglag.max_outer_iters = 15;
		s.passes[0].ilqr.max_iters = 30;
	};

	const ALRunResult slack = runALILQRDirectRWCase([&](PlannerSettings& s) {
		mutate(s);
		auto& aug = s.passes[0].auglag;
		aug.use_state_slack = true;
		aug.slack_rho = 50.0;
		aug.slack_off_tol = 0.02;
	}, Eigen::Vector3d::Zero());

	CAPTURE(slack.max_c);
	REQUIRE(slack.ok);
	REQUIRE(slack.max_c <= 1e-3 + 1e-9);
}

TEST_CASE("State slack: stall fallback rescues an underpriced slack phase", "[optimizer][alilqr][slack]") {
	// The auto-switch is OPT-IN: a wrong trigger does not cleanly revert to
	// baseline (the polish inherits the slack-phase trajectory, ramped mu,
	// and lambda clipped at slack_rho), so the default must stay disabled.
	REQUIRE(AugLagConfig{}.slack_stall_iters == 0);

	// slack_rho far below the binding constraint's multiplier: the slack
	// phase "buys" the violation and can never reach slack_off_tol. With the
	// stall fallback disabled the run must burn the outer budget and fail;
	// with it enabled the solver drops the slacks after slack_stall_iters
	// non-contracting iterations and converges exactly like baseline AL.
	const auto base_mutate = [](PlannerSettings& s) {
		s.constraints.wmax = 0.012;
		s.passes[0].auglag.max_outer_iters = 15;
		s.passes[0].ilqr.max_iters = 30;
		auto& aug = s.passes[0].auglag;
		aug.use_state_slack = true;
		aug.slack_rho = 1e-3;     // absurdly cheap: violation is always "bought"
		aug.slack_off_tol = 1e-9; // tol-based switch can never fire mid-run
	};

	const ALRunResult no_fallback = runALILQRDirectRWCase([&](PlannerSettings& s) {
		base_mutate(s);
		s.passes[0].auglag.slack_stall_iters = 0;  // disabled
	}, Eigen::Vector3d::Zero());

	const ALRunResult with_fallback = runALILQRDirectRWCase([&](PlannerSettings& s) {
		base_mutate(s);
		s.passes[0].auglag.slack_stall_iters = 3;
	}, Eigen::Vector3d::Zero());

	CAPTURE(no_fallback.max_c, with_fallback.max_c);
	REQUIRE_FALSE(no_fallback.ok);
	REQUIRE(with_fallback.ok);
	REQUIRE(with_fallback.max_c <= 1e-3 + 1e-9);
}

TEST_CASE("State slack: rho continuation default is off (bit-identical to fixed-rho)", "[optimizer][alilqr][slack]") {
	// slack_rho_scale = 1.0 must disable the ramp entirely. A run with an
	// explicit scale of 1.0 and a generous rho ceiling must reproduce the
	// plain fixed-rho slack solve to the last bit.
	const ALRunResult fixed = runALILQRDirectRWCase([](PlannerSettings& s) {
		auto& aug = s.passes[0].auglag;
		aug.use_state_slack = true;
		aug.slack_rho = 1e-2;  // economical: actually changes the solve
	});
	REQUIRE(AugLagConfig{}.slack_rho_scale == 1.0);

	const ALRunResult scale_one = runALILQRDirectRWCase([](PlannerSettings& s) {
		auto& aug = s.passes[0].auglag;
		aug.use_state_slack = true;
		aug.slack_rho = 1e-2;
		aug.slack_rho_scale = 1.0;  // explicit no-ramp
		aug.slack_rho_max = 1e12;
	});

	REQUIRE(fixed.ok == scale_one.ok);
	REQUIRE((fixed.X - scale_one.X).cwiseAbs().maxCoeff() == 0.0);
	REQUIRE((fixed.U - scale_one.U).cwiseAbs().maxCoeff() == 0.0);
}

TEST_CASE("State slack: rho continuation rescues an underpriced slack phase", "[optimizer][alilqr][slack]") {
	// Same underpriced setup as the stall-fallback test: slack_rho far below
	// the binding constraint's multiplier so a fixed-rho slack phase "buys"
	// the violation forever and never reaches slack_off_tol. Continuation
	// (slack_rho_scale > 1) anneals the cap upward until it exceeds the
	// shadow price, then hands off CONTINUOUSLY to exact AL and converges —
	// with the stall fallback left disabled, so this exercises the ramp, not
	// the abrupt drop. slack_sigma > 0 keeps the bought region conditioned.
	const ALRunResult cont = runALILQRDirectRWCase([](PlannerSettings& s) {
		s.constraints.wmax = 0.012;
		s.passes[0].auglag.max_outer_iters = 15;
		s.passes[0].ilqr.max_iters = 30;
		auto& aug = s.passes[0].auglag;
		aug.use_state_slack = true;
		aug.slack_rho = 1e-2;        // absurdly cheap to start
		aug.slack_sigma = 1.0;       // Huber curvature floor
		aug.slack_off_tol = 0.02;
		aug.slack_rho_scale = 10.0;  // continuation
		aug.slack_rho_max = 1e6;
		aug.slack_stall_iters = 0;   // fallback OFF: only continuation in play
	}, Eigen::Vector3d::Zero());

	CAPTURE(cont.max_c);
	REQUIRE(cont.ok);
	REQUIRE(cont.max_c <= 1e-3 + 1e-9);
}

TEST_CASE("AL per-family penalties: trajOpt converges with per-family config and conditional ramping", "[optimizer][alilqr][per_family]") {
	// End-to-end sanity: the full planner still converges with per-family
	// penalties and conditional ramping enabled (constraints here are easily
	// satisfiable, so gating must not break convergence).
	PlannerSettings settings = createRWPlannerSettings(10.0);
	auto& aug = settings.passes[0].auglag;
	const size_t nf = static_cast<size_t>(ConstraintFamily::NumFamilies);
	aug.penalty_init_per_family.assign(nf, aug.penalty_init);
	aug.penalty_max_per_family.assign(nf, aug.penalty_max);
	aug.penalty_scale_per_family.assign(nf, aug.penalty_scale);
	aug.penalty_init_per_family[static_cast<size_t>(ConstraintFamily::RWMomentum)] = 1.0;
	aug.family_contraction_ratio = 0.5;

	Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
	J(0, 0) = 0.067;
	J(1, 1) = 0.071;
	J(2, 2) = 0.069;
	Satellite satellite(J, settings);
	satellite.addRW(Eigen::Vector3d::UnitX(), 0.001, 1e-5, 0.0, 0.02);
	satellite.addRW(Eigen::Vector3d::UnitY(), 0.001, 1e-5, 0.0, 0.02);
	satellite.addRW(Eigen::Vector3d::UnitZ(), 0.001, 1e-5, 0.0, 0.02);

	Satellite::VecX x0(satellite.stateDim());
	x0.setZero();
	x0.segment<3>(Satellite::AV_INDEX) = Eigen::Vector3d(-0.01, 0.02, 0.03);
	x0.segment<4>(Satellite::QUAT_INDEX) = Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
	x0.segment(satellite.RW_MOMENTUM_INDEX, 3) = Eigen::Vector3d::Zero();

	runAndCheckCase(settings, satellite, x0, 200.0, 10.0);
}

TEST_CASE("AL-iLQR RW case tf=200 dt=10", "[optimizer][alilqr][rw]") {
	runRWCase(200.0, 10.0);
}

TEST_CASE("AL-iLQR RW case tf=1000 dt=10", "[optimizer][alilqr][rw]") {
	runRWCase(1000.0, 10.0);
}

TEST_CASE("AL-iLQR RW case tf=1000 dt=50", "[optimizer][alilqr][rw]") {
	runRWCase(1000.0, 50.0);
}

TEST_CASE("AL-iLQR hybrid case tf=1000 dt=10", "[optimizer][alilqr][hybrid]") {
	runHybridCase(1000.0, 10.0);
}

TEST_CASE("AL-iLQR hybrid case tf=1000 dt=20", "[optimizer][alilqr][hybrid]") {
	runHybridCase(1000.0, 20.0);
}

TEST_CASE("AL-iLQR hybrid case tf=1000 dt=5", "[optimizer][alilqr][hybrid]") {
	runHybridCase(1000.0, 5.0);
}
