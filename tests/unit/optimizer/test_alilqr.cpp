#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>

#include <Eigen/Dense>

#include <saltro/limits.h>
#include <saltro/optimizer/trajOpt.h>
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

} // namespace

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
