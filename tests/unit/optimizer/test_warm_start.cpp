#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>

#include <Eigen/Dense>

#include <saltro/limits.h>
#include <saltro/math/integrators/rk4.h>
#include <saltro/optimizer/warm_start.h>
#include <saltro/pybind/satellite.h>

using namespace saltro;

namespace {

constexpr double PI = 3.14159265358979323846;
constexpr double SEC_PER_CENTURY = 36525.0 * 86400.0;

class WarmStartFixture {
public:
	static constexpr int N = 20;

	PlannerSettings settings;
	Satellite satellite;

	Satellite::VecX x0;
	Eigen::VectorXd jtime;
	Eigen::MatrixXd q_goal;
	Eigen::MatrixXd boresight;

	Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> R;
	Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> V;
	Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> B;
	Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> S;
	Eigen::Matrix<double, 1, limits::MAX_LENGTH_TRAJ> rho;

	WarmStartFixture()
		: settings(),
		  satellite(makeInertia(), settings),
		  x0(Satellite::VecX::Zero(satellite.stateDim())),
		  jtime(Eigen::VectorXd::Zero(N)),
		  q_goal(Eigen::MatrixXd::Zero(4, N)),
		  boresight(Eigen::MatrixXd::Zero(3, N)) {
		satellite.addMTQ(Eigen::Vector3d::UnitX(), 0.2);
		satellite.addMTQ(Eigen::Vector3d::UnitY(), 0.2);
		satellite.addMTQ(Eigen::Vector3d::UnitZ(), 0.2);

		satellite.addRW(Eigen::Vector3d::UnitX(), 0.001, 1e-5, 0.0, 0.02);
		satellite.addRW(Eigen::Vector3d::UnitY(), 0.001, 1e-5, 0.0, 0.02);
		satellite.addRW(Eigen::Vector3d::UnitZ(), 0.001, 1e-5, 0.0, 0.02);

		x0 = Satellite::VecX::Zero(satellite.stateDim());
		x0.segment<3>(Satellite::AV_INDEX) = Eigen::Vector3d(0.02, -0.01, 0.015);
		x0.segment<4>(Satellite::QUAT_INDEX) = Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);

		const double dt_seconds = 0.5;
		const double dt_centuries = dt_seconds / SEC_PER_CENTURY;
		for (int k = 0; k < N; ++k) {
			jtime(k) = 0.25 + k * dt_centuries;
			q_goal.col(k) = Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
			boresight.col(k) = Eigen::Vector3d::UnitX();

			R.col(k) = Eigen::Vector3d(7000e3, 100.0 * k, -50.0 * k);
			V.col(k) = Eigen::Vector3d(0.0, 7500.0, 0.0);
			B.col(k) = Eigen::Vector3d(2.5e-5, -1.5e-5, 3.0e-5);
			S.col(k) = Eigen::Vector3d(1.0, 0.1, -0.05);
			rho(k) = 0.0;
		}

		settings.disturbances.plan_for_aero = false;
		settings.disturbances.plan_for_gg = false;
		settings.disturbances.plan_for_srp = false;
		settings.disturbances.plan_for_prop = false;
		settings.disturbances.plan_for_gendist = false;
		settings.disturbances.plan_for_resdipole = false;
		settings.num_passes = 1;
		settings.passes[0].dt = dt_seconds;
	}

private:
	static Eigen::Matrix3d makeInertia() {
		Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
		J(0, 0) = 0.067;
		J(1, 1) = 0.071;
		J(2, 2) = 0.069;
		return J;
	}
};

} // namespace

TEST_CASE_METHOD(WarmStartFixture, "warm_start returns correct output dimensions", "[warm_start][dimensions]") {
	settings.init_traj.initcontroller = 0;

	Eigen::MatrixXd X = Eigen::MatrixXd::Zero(satellite.stateDim(), N);
	Eigen::MatrixXd U = Eigen::MatrixXd::Zero(satellite.controlDim(), N);

	const bool ok = optimizer::warm_start(
		settings, satellite, x0, jtime, q_goal, boresight, N,
		R, V, B, S, rho,
		X, U
	);

	REQUIRE(ok);
	REQUIRE(X.rows() == satellite.stateDim());
	REQUIRE(X.cols() == N);
	REQUIRE(U.rows() == satellite.controlDim());
	REQUIRE(U.cols() == N);
}

TEST_CASE_METHOD(WarmStartFixture, "warm_start uses RK4 propagation consistently for zero controller", "[warm_start][rk4]") {
	settings.init_traj.initcontroller = 0;

	Eigen::MatrixXd X = Eigen::MatrixXd::Zero(satellite.stateDim(), N);
	Eigen::MatrixXd U = Eigen::MatrixXd::Zero(satellite.controlDim(), N);

	const bool ok = optimizer::warm_start(
		settings, satellite, x0, jtime, q_goal, boresight, N,
		R, V, B, S, rho,
		X, U
	);

	REQUIRE(ok);

	REQUIRE_THAT(U.norm(), Catch::Matchers::WithinAbs(0.0, 1e-14));

	Satellite::VecX x_manual_next;
	const Satellite::VecX u0 = Satellite::VecX::Zero(satellite.controlDim());
	const double dt = (jtime(1) - jtime(0)) * SEC_PER_CENTURY;
	const int rho0 = static_cast<int>(std::max(0.0, std::round(rho(0))));

	rk4_step<Satellite::VecX>(
		[&](double, const Satellite::VecX& x_state, Satellite::VecX& dxdt) {
			dxdt = satellite.dynamics(
				x_state,
				u0,
				settings.disturbances,
				R.col(0),
				B.col(0),
				S.col(0),
				V.col(0),
				rho0
			);
		},
		x0,
		0.0,
		dt,
		x_manual_next
	);

	Eigen::Vector4d q = x_manual_next.segment<4>(Satellite::QUAT_INDEX);
	q.normalize();
	x_manual_next.segment<4>(Satellite::QUAT_INDEX) = q;

	const Eigen::VectorXd x1 = X.col(1);
	REQUIRE((x1 - x_manual_next).norm() < 1e-11);

	for (int k = 0; k < N; ++k) {
		const double qn = X.col(k).segment<4>(Satellite::QUAT_INDEX).norm();
		REQUIRE_THAT(qn, Catch::Matchers::WithinAbs(1.0, 1e-10));
	}
}

TEST_CASE_METHOD(WarmStartFixture, "controllers produce expected warm_start behavior", "[warm_start][controllers]") {
	Eigen::MatrixXd X_zero = Eigen::MatrixXd::Zero(satellite.stateDim(), N);
	Eigen::MatrixXd U_zero = Eigen::MatrixXd::Zero(satellite.controlDim(), N);
	Eigen::MatrixXd X_exc = Eigen::MatrixXd::Zero(satellite.stateDim(), N);
	Eigen::MatrixXd U_exc = Eigen::MatrixXd::Zero(satellite.controlDim(), N);

	settings.init_traj.initcontroller = 0;
	const bool ok_zero = optimizer::warm_start(
		settings, satellite, x0, jtime, q_goal, boresight, N,
		R, V, B, S, rho,
		X_zero, U_zero
	);

	settings.init_traj.initcontroller = 1;
	const bool ok_exc = optimizer::warm_start(
		settings, satellite, x0, jtime, q_goal, boresight, N,
		R, V, B, S, rho,
		X_exc, U_exc
	);

	REQUIRE(ok_zero);
	REQUIRE(ok_exc);

	REQUIRE_THAT(U_zero.norm(), Catch::Matchers::WithinAbs(0.0, 1e-14));
	REQUIRE(U_exc.norm() > 1e-10);

	const double trajectory_delta = (X_exc - X_zero).norm();
	REQUIRE(trajectory_delta > 1e-10);
}

