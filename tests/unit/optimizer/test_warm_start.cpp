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

// ----------------------------------------------------------------------------
// Magic-actuator PD warm start (post-#28 rebase).
//
// PDController::find_u now allocates across the FULL control dimension
// (nu = controlDim(), Magic columns included in the actuator Jacobian), so a
// PD warm start must (a) pass warm_start's uk.size() == controlDim() gate for
// mixed actuator sets, and (b) actively command Magic actuators — including
// the Magic-ONLY case, which previously risked a silent U ≡ 0 "success"
// indistinguishable from initcontroller=0.
// ----------------------------------------------------------------------------
namespace {

// Shared trajectory inputs for the custom-satellite warm-start tests below.
struct TrajInputs {
	Eigen::VectorXd jtime;
	Eigen::MatrixXd q_goal;
	Eigen::MatrixXd boresight;
	Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> R;
	Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> V;
	Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> B;
	Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> S;
	Eigen::Matrix<double, 1, limits::MAX_LENGTH_TRAJ> rho;

	explicit TrajInputs(int n, double dt_seconds)
		: jtime(Eigen::VectorXd::Zero(n)),
		  q_goal(Eigen::MatrixXd::Zero(4, n)),
		  boresight(Eigen::MatrixXd::Zero(3, n)) {
		R.setZero(); V.setZero(); B.setZero(); S.setZero(); rho.setZero();
		for (int k = 0; k < n; ++k) {
			jtime(k) = 0.25 + k * dt_seconds / SEC_PER_CENTURY;
			q_goal.col(k) = Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
			boresight.col(k) = Eigen::Vector3d::UnitX();
			R.col(k) = Eigen::Vector3d(7000e3, 100.0 * k, -50.0 * k);
			V.col(k) = Eigen::Vector3d(0.0, 7500.0, 0.0);
			B.col(k) = Eigen::Vector3d(2.5e-5, -1.5e-5, 3.0e-5);
			S.col(k) = Eigen::Vector3d(1.0, 0.1, -0.05);
		}
	}
};

void disableDisturbances(PlannerSettings& settings) {
	settings.disturbances.plan_for_aero = false;
	settings.disturbances.plan_for_gg = false;
	settings.disturbances.plan_for_srp = false;
	settings.disturbances.plan_for_prop = false;
	settings.disturbances.plan_for_gendist = false;
	settings.disturbances.plan_for_resdipole = false;
}

Eigen::Matrix3d smallInertia() {
	Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
	J(0, 0) = 0.067;
	J(1, 1) = 0.071;
	J(2, 2) = 0.069;
	return J;
}

} // namespace

TEST_CASE("PD warm start sizes and drives Magic channels on a mixed MTQ+RW+Magic satellite",
          "[warm_start][pd][magic]") {
	const int N = 20;
	const double dt = 0.5;

	PlannerSettings settings;
	disableDisturbances(settings);
	settings.num_passes = 1;
	settings.passes[0].dt = dt;
	settings.init_traj.initcontroller = 3;

	Satellite satellite(smallInertia(), settings);
	satellite.addMTQ(Eigen::Vector3d::UnitX(), 0.2);
	satellite.addMTQ(Eigen::Vector3d::UnitY(), 0.2);
	satellite.addMTQ(Eigen::Vector3d::UnitZ(), 0.2);
	satellite.addRW(Eigen::Vector3d::UnitZ(), 0.001, 1e-5, 0.0, 0.02);
	satellite.addMagic(Eigen::Vector3d::UnitX(), 1.0e-3);
	REQUIRE(satellite.controlDim() == 5);

	TrajInputs in(N, dt);
	Satellite::VecX x0 = Satellite::VecX::Zero(satellite.stateDim());
	x0.segment<3>(Satellite::AV_INDEX) = Eigen::Vector3d(0.02, -0.01, 0.015);
	x0.segment<4>(Satellite::QUAT_INDEX) = Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);

	Eigen::MatrixXd X = Eigen::MatrixXd::Zero(satellite.stateDim(), N);
	Eigen::MatrixXd U = Eigen::MatrixXd::Zero(satellite.controlDim(), N);

	const bool ok = optimizer::warm_start(
		settings, satellite, x0, in.jtime, in.q_goal, in.boresight, N,
		in.R, in.V, in.B, in.S, in.rho, X, U);

	REQUIRE(ok);                                   // uk.size() gate accepts controlDim
	REQUIRE(U.rows() == satellite.controlDim());
	REQUIRE(U.allFinite());
	REQUIRE(U.norm() > 1e-10);                     // PD actually commands something
	// Post-#28 PD allocation spans Magic columns: the Magic row is driven too.
	REQUIRE(U.row(4).cwiseAbs().maxCoeff() > 1e-12);
}

TEST_CASE("PD warm start on a Magic-only satellite produces nonzero control (no silent U=0)",
          "[warm_start][pd][magic]") {
	const int N = 20;
	const double dt = 0.5;

	PlannerSettings settings;
	disableDisturbances(settings);
	settings.num_passes = 1;
	settings.passes[0].dt = dt;
	settings.init_traj.initcontroller = 3;

	Satellite satellite(smallInertia(), settings);
	satellite.addMagic(Eigen::Vector3d::UnitX(), 1.0e-3);
	satellite.addMagic(Eigen::Vector3d::UnitY(), 1.0e-3);
	satellite.addMagic(Eigen::Vector3d::UnitZ(), 1.0e-3);
	REQUIRE(satellite.controlDim() == 3);

	TrajInputs in(N, dt);
	Satellite::VecX x0 = Satellite::VecX::Zero(satellite.stateDim());
	const Eigen::Vector3d omega0(0.02, -0.01, 0.015);
	x0.segment<3>(Satellite::AV_INDEX) = omega0;
	x0.segment<4>(Satellite::QUAT_INDEX) = Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);

	Eigen::MatrixXd X = Eigen::MatrixXd::Zero(satellite.stateDim(), N);
	Eigen::MatrixXd U = Eigen::MatrixXd::Zero(satellite.controlDim(), N);

	const bool ok = optimizer::warm_start(
		settings, satellite, x0, in.jtime, in.q_goal, in.boresight, N,
		in.R, in.V, in.B, in.S, in.rho, X, U);

	REQUIRE(ok);
	REQUIRE(U.allFinite());
	// The audited failure mode was a "successful" warm start with U ≡ 0
	// (indistinguishable from initcontroller=0).  Post-#28, find_u allocates
	// over Magic columns, so the PD must command every Magic channel...
	REQUIRE(U.norm() > 1e-10);
	for (int i = 0; i < 3; ++i) {
		REQUIRE(U.row(i).cwiseAbs().maxCoeff() > 1e-12);
	}
	// ...and the rollout must actually damp the tumble (rate-to-zero PD).
	const Eigen::Vector3d omegaN = X.col(N - 1).segment<3>(Satellite::AV_INDEX);
	REQUIRE(omegaN.norm() < 0.5 * omega0.norm());
}

// ----------------------------------------------------------------------------
// Goal-rate feedforward: exercise the ACTUAL ω_des computation in warm_start
// (not setGoalRate directly) with a slowly rotating quaternion-goal sequence.
// For goals q_g(k) = [cos(θk/2), 0, 0, sin(θk/2)], θk = ω·k·dt (constant body
// rate ω about +z), the FF should command a spin-up toward ω ẑ and track far
// more tightly than the rate-to-zero PD.  A sign or frame error in the ω_des
// finite difference makes the FF FIGHT the maneuver (tracks worse than no-FF),
// which this test catches.
// ----------------------------------------------------------------------------
TEST_CASE("PD goal-rate feedforward tracks a rotating quaternion goal (sign/frame)",
          "[warm_start][pd][ff]") {
	const int N = 40;
	const double dt = 1.0;
	const double w_goal = 0.02;  // rad/s about body +z

	PlannerSettings settings;
	disableDisturbances(settings);
	settings.num_passes = 1;
	settings.passes[0].dt = dt;
	settings.init_traj.initcontroller = 3;

	Satellite satellite(smallInertia(), settings);
	satellite.addRW(Eigen::Vector3d::UnitX(), 0.001, 1e-5, 0.0, 0.02);
	satellite.addRW(Eigen::Vector3d::UnitY(), 0.001, 1e-5, 0.0, 0.02);
	satellite.addRW(Eigen::Vector3d::UnitZ(), 0.001, 1e-5, 0.0, 0.02);

	TrajInputs in(N, dt);
	for (int k = 0; k < N; ++k) {
		const double th = w_goal * k * dt;
		in.q_goal.col(k) =
			Eigen::Vector4d(std::cos(0.5 * th), 0.0, 0.0, std::sin(0.5 * th));
	}

	// Start ON the initial goal, at rest: all subsequent error comes from the
	// goal moving, which is exactly what the FF is supposed to anticipate.
	Satellite::VecX x0 = Satellite::VecX::Zero(satellite.stateDim());
	x0.segment<4>(Satellite::QUAT_INDEX) = Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);

	auto run = [&](bool ff_enabled, Eigen::MatrixXd& X, Eigen::MatrixXd& U) {
		settings.init_traj.pd_goal_rate_ff_enabled = ff_enabled;
		return optimizer::warm_start(
			settings, satellite, x0, in.jtime, in.q_goal, in.boresight, N,
			in.R, in.V, in.B, in.S, in.rho, X, U);
	};

	Eigen::MatrixXd X_off = Eigen::MatrixXd::Zero(satellite.stateDim(), N);
	Eigen::MatrixXd U_off = Eigen::MatrixXd::Zero(satellite.controlDim(), N);
	Eigen::MatrixXd X_on = Eigen::MatrixXd::Zero(satellite.stateDim(), N);
	Eigen::MatrixXd U_on = Eigen::MatrixXd::Zero(satellite.controlDim(), N);
	REQUIRE(run(false, X_off, U_off));
	REQUIRE(run(true, X_on, U_on));
	REQUIRE(X_on.allFinite());
	REQUIRE(U_on.allFinite());

	auto attitude_err = [&](const Eigen::MatrixXd& X, int k) {
		Eigen::Vector4d q = X.col(k).segment<4>(Satellite::QUAT_INDEX);
		q.normalize();
		const double d =
			std::min(1.0, std::abs(q.dot(in.q_goal.col(k).head<4>())));
		return 2.0 * std::acos(d);
	};

	double mean_off = 0.0, mean_on = 0.0;
	for (int k = 0; k < N; ++k) {
		mean_off += attitude_err(X_off, k);
		mean_on += attitude_err(X_on, k);
	}
	mean_off /= N;
	mean_on /= N;

	// No-FF lags the moving goal substantially; FF must cut the mean tracking
	// error by well over 2x (measured: ~14.4 deg -> ~2.4 deg) and beat it at
	// the final knot too.  A sign-flipped ω_des FAILS these (error grows).
	REQUIRE(mean_off > 0.05);                       // sanity: goal does move
	REQUIRE(mean_on < 0.5 * mean_off);
	REQUIRE(attitude_err(X_on, N - 1) < attitude_err(X_off, N - 1));

	// The FF rollout must acquire the analytic goal rate ω ẑ (constant-rate
	// rotation), not its negative.
	const Eigen::Vector3d omegaN = X_on.col(N - 1).segment<3>(Satellite::AV_INDEX);
	REQUIRE(omegaN(2) > 0.5 * w_goal);
	REQUIRE(std::abs(omegaN(2) - w_goal) < 0.25 * w_goal);
}

