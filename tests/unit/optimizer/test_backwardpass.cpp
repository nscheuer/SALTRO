#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <vector>
#include <functional>
#include <algorithm>
#include <limits>

#include <Eigen/Dense>

#include <saltro/limits.h>
#include <saltro/math/integrators/rk4.h>
#include <saltro/math/mrp.h>
#include <saltro/optimizer/backwardpass.h>
#include <saltro/pybind/satellite.h>

using namespace saltro;

namespace {

constexpr double PI = 3.14159265358979323846;
constexpr double SEC_PER_CENTURY = 36525.0 * 86400.0;

Eigen::MatrixXd makeAttitudeTraj(const Eigen::Vector4d& att, int N_cols) {
	Eigen::MatrixXd traj(4, N_cols);
	for (int k = 0; k < N_cols; ++k) {
		traj.col(k) = att;
	}
	return traj;
}

class BackwardPassFixture {
public:
	static constexpr int N = 2;  // Minimal case: 2 timesteps, 1 backward iteration

	PlannerSettings settings;
	Satellite satellite;

	Satellite::VecX x0;
	Eigen::VectorXd jtime;
	Eigen::Vector4d attitude_target;
	Eigen::MatrixXd attitude_target_traj;
	Eigen::MatrixXd boresight;

	Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> R;
	Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> V;
	Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> B;
	Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> S;
	Eigen::Matrix<double, 1, limits::MAX_LENGTH_TRAJ> rho;

	BackwardPassFixture()
		: settings(),
		  satellite(makeInertia(), settings),
		  x0(Satellite::VecX::Zero(satellite.stateDim())),
		  jtime(Eigen::VectorXd::Zero(N)),
		  attitude_target(Eigen::Vector4d::Zero()),
		  boresight(Eigen::MatrixXd::Zero(3, N)) {
		satellite.addMTQ(Eigen::Vector3d::UnitX(), 0.2);
		satellite.addMTQ(Eigen::Vector3d::UnitY(), 0.2);
		satellite.addMTQ(Eigen::Vector3d::UnitZ(), 0.2);

		satellite.addRW(Eigen::Vector3d::UnitX(), 0.001, 1e-5, 0.0, 0.02);
		satellite.addRW(Eigen::Vector3d::UnitY(), 0.001, 1e-5, 0.0, 0.02);
		satellite.addRW(Eigen::Vector3d::UnitZ(), 0.001, 1e-5, 0.0, 0.02);

		// Initial state: near identity quaternion with small angular velocity
		x0 = Satellite::VecX::Zero(satellite.stateDim());
		x0.segment<3>(Satellite::AV_INDEX) = Eigen::Vector3d(0.01, -0.005, 0.008);
		x0.segment<4>(Satellite::QUAT_INDEX) = Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);

		// Goal: identity quaternion (ECI format)
		attitude_target << std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0;
		attitude_target_traj = makeAttitudeTraj(attitude_target, N);

		// Time setup: 2 timesteps with dt=0.5 seconds
		const double dt_seconds = 0.5;
		const double dt_centuries = dt_seconds / SEC_PER_CENTURY;
		for (int k = 0; k < N; ++k) {
			jtime(k) = 0.25 + k * dt_centuries;
			boresight.col(k) = Eigen::Vector3d::UnitX();

			// Orbital environment
			R.col(k) = Eigen::Vector3d(7000e3, 0.0, 0.0);
			V.col(k) = Eigen::Vector3d(0.0, 7500.0, 0.0);
			B.col(k) = Eigen::Vector3d(2.5e-5, -1.5e-5, 3.0e-5);
			S.col(k) = Eigen::Vector3d(1.0, 0.1, -0.05).normalized();
			rho(0, k) = 0.0;
		}

		// Disable disturbances for cleaner test
		settings.disturbances.plan_for_aero = false;
		settings.disturbances.plan_for_gg = false;
		settings.disturbances.plan_for_srp = false;
		settings.disturbances.plan_for_prop = false;
		settings.disturbances.plan_for_gendist = false;
		settings.disturbances.plan_for_resdipole = false;
		settings.num_passes = 1;
		settings.passes[0].dt = dt_seconds;

		// Regularization settings
		settings.passes[0].reg.reg_init = 1e-8;
		settings.passes[0].reg.reg_scale = 10.0;
		settings.passes[0].reg.reg_max = 1e4;
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

}  // namespace

TEST_CASE_METHOD(BackwardPassFixture, "backward_pass N=1 edge case (no backward pass)", "[backward_pass][n1_edge_case]") {
	// N=1 means only terminal timestep: no backward pass loop needed
	// K and d should be empty (no controls)
	constexpr int N_test = 1;

	PlannerSettings settings_test = settings;
	Satellite satellite_test(
		Eigen::Vector3d(0.067, 0.071, 0.069).asDiagonal(), settings_test
	);
	satellite_test.addMTQ(Eigen::Vector3d::UnitX(), 0.2);
	satellite_test.addMTQ(Eigen::Vector3d::UnitY(), 0.2);
	satellite_test.addMTQ(Eigen::Vector3d::UnitZ(), 0.2);
	satellite_test.addRW(Eigen::Vector3d::UnitX(), 0.001, 1e-5, 0.0, 0.02);
	satellite_test.addRW(Eigen::Vector3d::UnitY(), 0.001, 1e-5, 0.0, 0.02);
	satellite_test.addRW(Eigen::Vector3d::UnitZ(), 0.001, 1e-5, 0.0, 0.02);

	Eigen::MatrixXd X(satellite_test.stateDim(), N_test);
	Eigen::MatrixXd U(satellite_test.controlDim(), 0);  // No controls for N=1
	Eigen::MatrixXd R_test(3, N_test);
	Eigen::MatrixXd V_test(3, N_test);
	Eigen::MatrixXd B_test(3, N_test);
	Eigen::MatrixXd S_test(3, N_test);
	Eigen::MatrixXd rho_test(1, N_test);
	Eigen::MatrixXd boresight_test(3, N_test);
	Eigen::Vector4d attitude_target_test;

	X.col(0) = x0;
	R_test.col(0) = Eigen::Vector3d(7000e3, 0.0, 0.0);
	V_test.col(0) = Eigen::Vector3d(0.0, 7500.0, 0.0);
	B_test.col(0) = Eigen::Vector3d(2.5e-5, -1.5e-5, 3.0e-5);
	S_test.col(0) = Eigen::Vector3d(1.0, 0.1, -0.05).normalized();
	rho_test(0, 0) = 0.0;
	boresight_test.col(0) = Eigen::Vector3d::UnitX();
	attitude_target_test << std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0;
	Eigen::MatrixXd attitude_target_test_traj = makeAttitudeTraj(attitude_target_test, N_test);

	std::vector<Eigen::MatrixXd> K(0);
	std::vector<Eigen::VectorXd> d(0);
	Eigen::Vector2d deltaV = Eigen::Vector2d::Zero();

	const bool ok = optimizer::backwardPass(
		satellite_test, X, U, R_test, V_test, B_test, S_test, rho_test, boresight_test, attitude_target_test_traj, settings_test, settings_test.passes[0].reg.reg_init, K, d, deltaV
	);

	REQUIRE(ok);
	REQUIRE(K.size() == 0);  // N-1 = 0, so no gains
	REQUIRE(d.size() == 0);
}

TEST_CASE_METHOD(BackwardPassFixture, "backward_pass N=2 hand-verified computation", "[backward_pass][n2_hand_verified]") {
	// N=2: Terminal state at k=1, backward iteration at k=0
	// Hand-verify K[0] and d[0] computation
	
	const int N_test = 2;

	PlannerSettings settings_test = settings;
	Satellite satellite_test(
		Eigen::Vector3d(0.067, 0.071, 0.069).asDiagonal(), settings_test
	);
	satellite_test.addMTQ(Eigen::Vector3d::UnitX(), 0.2);
	satellite_test.addMTQ(Eigen::Vector3d::UnitY(), 0.2);
	satellite_test.addMTQ(Eigen::Vector3d::UnitZ(), 0.2);
	satellite_test.addRW(Eigen::Vector3d::UnitX(), 0.001, 1e-5, 0.0, 0.02);
	satellite_test.addRW(Eigen::Vector3d::UnitY(), 0.001, 1e-5, 0.0, 0.02);
	satellite_test.addRW(Eigen::Vector3d::UnitZ(), 0.001, 1e-5, 0.0, 0.02);

	Eigen::MatrixXd X(satellite_test.stateDim(), N_test);
	Eigen::MatrixXd U(satellite_test.controlDim(), N_test - 1);
	Eigen::MatrixXd R_test(3, N_test);
	Eigen::MatrixXd V_test(3, N_test);
	Eigen::MatrixXd B_test(3, N_test);
	Eigen::MatrixXd S_test(3, N_test);
	Eigen::MatrixXd rho_test(1, N_test);
	Eigen::MatrixXd boresight_test(3, N_test);
	Eigen::Vector4d attitude_target_test;

	// Simple case: identical states at k=0 and k=1
	X.col(0) = x0;
	X.col(1) = x0;  // Terminal state same as initial

	// Zero control (simplified case for hand verification)
	U.col(0).setZero();

	// Orbital environment
	for (int k = 0; k < N_test; ++k) {
		R_test.col(k) = Eigen::Vector3d(7000e3, 0.0, 0.0);
		V_test.col(k) = Eigen::Vector3d(0.0, 7500.0, 0.0);
		B_test.col(k) = Eigen::Vector3d(2.5e-5, -1.5e-5, 3.0e-5);
		S_test.col(k) = Eigen::Vector3d(1.0, 0.1, -0.05).normalized();
		rho_test(0, k) = 0.0;
		boresight_test.col(k) = Eigen::Vector3d::UnitX();
	}

	attitude_target_test << std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0;
	Eigen::MatrixXd attitude_target_test_traj = makeAttitudeTraj(attitude_target_test, N_test);

	int nu_test = satellite_test.controlDim();
	int nx_test = satellite_test.reducedStateDim();
	std::vector<Eigen::MatrixXd> K(N_test - 1);
	std::vector<Eigen::VectorXd> d(N_test - 1);
	for (int kk = 0; kk < N_test - 1; ++kk) {
		K[kk] = Eigen::MatrixXd::Zero(nu_test, nx_test);
		d[kk] = Eigen::VectorXd::Zero(nu_test);
	}
	Eigen::Vector2d deltaV = Eigen::Vector2d::Zero();

	const bool ok = optimizer::backwardPass(
		satellite_test, X, U, R_test, V_test, B_test, S_test, rho_test, boresight_test, attitude_target_test_traj, settings_test, settings_test.passes[0].reg.reg_init, K, d, deltaV
	);

	REQUIRE(ok);

	// Verify output structure
	REQUIRE(K.size() == 1);
	REQUIRE(d.size() == 1);
	REQUIRE(K[0].rows() == satellite_test.controlDim());
	REQUIRE(K[0].cols() == satellite_test.reducedStateDim());
	REQUIRE(d[0].rows() == satellite_test.controlDim());

	// K[0] and d[0] should be finite
	REQUIRE(K[0].allFinite());
	REQUIRE(d[0].allFinite());

	// Hand verification: manually compute expected values
	const Eigen::VectorXd x_0 = X.col(0);
	const Eigen::VectorXd u_0 = U.col(0);
	const Eigen::Vector3d B_0 = B_test.col(0);
	const Eigen::Vector3d boresight_0 = boresight_test.col(0);

	// Extract terminal cost-to-go (p_1, P_1)
	auto [p_1_expected, _, __] = satellite_test.terminalCostJacobians(
		x_0, boresight_0, attitude_target_test, B_0, settings_test.passes[0].cost
	);
	auto [P_1_expected, ___, ____] = satellite_test.terminalCostHessians(
		x_0, boresight_0, attitude_target_test, B_0, settings_test.passes[0].cost
	);

	// Compute stage cost derivatives at k=0
	const Eigen::Vector3d R_0 = R_test.col(0);
	const Eigen::Vector3d V_0 = V_test.col(0);
	const Eigen::Vector3d S_0 = S_test.col(0);

	auto [l_x_0, l_u_0_mat, l_ux_grad_0] = satellite_test.stageCostJacobians(
		0, N_test, x_0, u_0, boresight_0, attitude_target_test, B_0, settings_test.passes[0].cost
	);
	auto [l_xx_0, l_uu_0, l_ux_hess_0] = satellite_test.stageCostHessians(
		0, N_test, x_0, u_0, boresight_0, attitude_target_test, B_0, settings_test.passes[0].cost
	);

	Eigen::VectorXd l_u_0 = l_u_0_mat.row(0);
	const double dt = settings_test.passes[0].dt;
	(void)dt;
	// BP no longer scales stage cost derivatives by dt: Satellite::totalCost
	// SUMS stageCost without a dt factor, so gradients/Hessians are already
	// in the right scale.  This manual computation must mirror that — do not
	// reintroduce the dt multiplication.

	// Compute discrete-time dynamics Jacobians at k=0 (matches backward pass).
	DisturbanceConfig dist_config;
	Eigen::MatrixXd A_0 = Eigen::MatrixXd::Zero(satellite_test.stateDim(), satellite_test.stateDim());
	Eigen::MatrixXd B_0_dyn = Eigen::MatrixXd::Zero(satellite_test.stateDim(), satellite_test.controlDim());
	auto dynamics_jac_wrapper = [&](double, const Eigen::Ref<const Eigen::VectorXd>& x_local,
	                                const Eigen::Ref<const Eigen::VectorXd>& u_local,
	                                Eigen::Ref<Eigen::MatrixXd> A_c_out,
	                                Eigen::Ref<Eigen::MatrixXd> B_c_out,
	                                Eigen::Ref<Eigen::VectorXd> k_out) {
		auto [A_c, B_c, ___unused] = satellite_test.dynamicsJacobians(
			x_local, u_local, dist_config, R_0, B_0, S_0, V_0
		);
		A_c_out = A_c;
		B_c_out = B_c;
		k_out = satellite_test.dynamics(x_local, u_local, dist_config, R_0, B_0, S_0, V_0, 0);
	};
	rk4_jacobians(dynamics_jac_wrapper, x_0, u_0, 0.0, dt, A_0, B_0_dyn);

	// Project derivatives and dynamics into reduced state space (MRP-based).
	const int nRW = satellite_test.numRW();
	const Eigen::Vector4d q_0 = X.col(0).segment<4>(Satellite::QUAT_INDEX);
	const Eigen::Vector4d q_1 = X.col(1).segment<4>(Satellite::QUAT_INDEX);
	Eigen::MatrixXd G_0 = saltro::math::findGMat(q_0, nRW);
	Eigen::MatrixXd G_1 = saltro::math::findGMat(q_1, nRW);

	Eigen::VectorXd p_1_reduced = G_1 * p_1_expected;
	Eigen::MatrixXd P_1_reduced = G_1 * P_1_expected * G_1.transpose();

	Eigen::VectorXd l_x_0_reduced = G_0 * l_x_0;
	Eigen::MatrixXd l_xx_0_reduced = G_0 * l_xx_0 * G_0.transpose();
	Eigen::MatrixXd l_ux_0_reduced = l_ux_hess_0 * G_0.transpose();

	Eigen::MatrixXd A_0_reduced = G_1 * A_0 * G_0.transpose();
	Eigen::MatrixXd B_0_reduced = G_1 * B_0_dyn;

	// Assemble Q matrices
	Eigen::MatrixXd Q_xx_0 = l_xx_0_reduced + A_0_reduced.transpose() * P_1_reduced * A_0_reduced;
	Eigen::MatrixXd Q_uu_0 = l_uu_0 + B_0_reduced.transpose() * P_1_reduced * B_0_reduced;
	Eigen::MatrixXd Q_ux_0 = l_ux_0_reduced + B_0_reduced.transpose() * P_1_reduced * A_0_reduced;
	Eigen::VectorXd Q_x_0 = l_x_0_reduced + A_0_reduced.transpose() * p_1_reduced;
	Eigen::VectorXd Q_u_0 = l_u_0 + B_0_reduced.transpose() * p_1_reduced;

	// Compute expected K[0] and d[0]
	// K_k = -(Q_uu + ρI)^{-1} * Q_ux
	// d_k = -(Q_uu + ρI)^{-1} * Q_u
	// For this test, regularization should not be needed (natural PD)
	double rho_reg = settings_test.passes[0].reg.reg_init;
	Eigen::MatrixXd Q_uu_reg_0 = Q_uu_0 + rho_reg * Eigen::MatrixXd::Identity(Q_uu_0.rows(), Q_uu_0.cols());
	Eigen::LLT<Eigen::MatrixXd> llt(Q_uu_reg_0);

	Eigen::MatrixXd K_0_expected = -llt.solve(Q_ux_0);
	Eigen::VectorXd d_0_expected = -llt.solve(Q_u_0);

	// Verify K[0] and d[0] match expected values (within numerical tolerance).
	REQUIRE((K[0] - K_0_expected).norm() < 1e-10);
	REQUIRE((d[0] - d_0_expected).norm() < 1e-10);

	// Verify p_1 and P_1 were used (deltaV should be non-zero if cost matrix is non-zero)
	REQUIRE(deltaV.allFinite());
}

TEST_CASE_METHOD(BackwardPassFixture, "backward_pass returns correct output dimensions (N=2)", "[backward_pass][dimensions]") {
	// Create simple trajectory: x0 at k=0, x1 at k=1
	Eigen::MatrixXd X(satellite.stateDim(), N);
	Eigen::MatrixXd U(satellite.controlDim(), N - 1);

	X.col(0) = x0;
	X.col(1) = x0;  // Simple case: no dynamics change
	U.col(0).setZero();

	// Output containers - preallocate
	std::vector<Eigen::MatrixXd> K(N - 1);
	std::vector<Eigen::VectorXd> d(N - 1);
	for (int kk = 0; kk < N - 1; ++kk) {
		K[kk] = Eigen::MatrixXd::Zero(satellite.controlDim(), satellite.reducedStateDim());
		d[kk] = Eigen::VectorXd::Zero(satellite.controlDim());
	}
	Eigen::Vector2d deltaV = Eigen::Vector2d::Zero();

	const bool ok = optimizer::backwardPass(
		satellite, X, U, R, V, B, S, rho, boresight, attitude_target_traj, settings, settings.passes[0].reg.reg_init, K, d, deltaV
	);

	REQUIRE(ok);

	// Dimension checks
	REQUIRE(K.size() == N - 1);  // K[k] for k=0 only
	REQUIRE(d.size() == N - 1);

	REQUIRE(K[0].rows() == satellite.controlDim());
	REQUIRE(K[0].cols() == satellite.reducedStateDim());
	REQUIRE(d[0].rows() == satellite.controlDim());
}

TEST_CASE_METHOD(BackwardPassFixture, "backward_pass computes terminal cost-to-go correctly (N=2)", "[backward_pass][terminal]") {
	// Setup: two identical states (zero dynamics for simplicity)
	Eigen::MatrixXd X(satellite.stateDim(), N);
	Eigen::MatrixXd U(satellite.controlDim(), N - 1);

	X.col(0) = x0;
	X.col(1) = x0;  // Terminal state same as initial
	U.col(0).setZero();

	std::vector<Eigen::MatrixXd> K(N - 1);
	std::vector<Eigen::VectorXd> d(N - 1);
	for (int kk = 0; kk < N - 1; ++kk) {
		K[kk] = Eigen::MatrixXd::Zero(satellite.controlDim(), satellite.reducedStateDim());
		d[kk] = Eigen::VectorXd::Zero(satellite.controlDim());
	}
	Eigen::Vector2d deltaV = Eigen::Vector2d::Zero();

	const bool ok = optimizer::backwardPass(
		satellite, X, U, R, V, B, S, rho, boresight, attitude_target_traj, settings, settings.passes[0].reg.reg_init, K, d, deltaV
	);

	REQUIRE(ok);

	// For N=2, backward pass should:
	// 1. Extract p_N and P_N from terminal cost at k=1
	// 2. Compute Q matrices at k=0
	// 3. Solve for K[0] and d[0]
	//
	// Basic sanity check: K[0] should be finite and reasonably sized
	REQUIRE(K[0].allFinite());
	REQUIRE(d[0].allFinite());

	// Feedback gain magnitude should be moderate (not exploding)
	REQUIRE(K[0].norm() < 100.0);
}

TEST_CASE_METHOD(BackwardPassFixture, "backward_pass accumulates cost reduction terms", "[backward_pass][deltav]") {
	Eigen::MatrixXd X(satellite.stateDim(), N);
	Eigen::MatrixXd U(satellite.controlDim(), N - 1);

	X.col(0) = x0;
	X.col(1) = x0;
	U.col(0).setZero();

	std::vector<Eigen::MatrixXd> K(N - 1);
	std::vector<Eigen::VectorXd> d(N - 1);
	for (int kk = 0; kk < N - 1; ++kk) {
		K[kk] = Eigen::MatrixXd::Zero(satellite.controlDim(), satellite.reducedStateDim());
		d[kk] = Eigen::VectorXd::Zero(satellite.controlDim());
	}
	Eigen::Vector2d deltaV = Eigen::Vector2d::Zero();

	const bool ok = optimizer::backwardPass(
		satellite, X, U, R, V, B, S, rho, boresight, attitude_target_traj, settings, settings.passes[0].reg.reg_init, K, d, deltaV
	);

	REQUIRE(ok);

	// deltaV should be accumulated across all timesteps
	// For a descent direction (typical iLQR), V_1 should be negative (cost decrease)
	REQUIRE(deltaV.allFinite());

	// At least accumulate non-zero terms (unless by accident they cancel)
	REQUIRE((std::abs(deltaV(0)) > 1e-15 || std::abs(deltaV(1)) > 1e-15));
}

TEST_CASE_METHOD(BackwardPassFixture, "backward_pass regularization loop converges", "[backward_pass][regularization]") {
	Eigen::MatrixXd X(satellite.stateDim(), N);
	Eigen::MatrixXd U(satellite.controlDim(), N - 1);

	// Perturb state slightly to create richer dynamics
	X.col(0) = x0;
	X.col(1) = x0 + 0.001 * Eigen::VectorXd::Random(satellite.stateDim());

	// Non-zero control
	U.col(0) = 0.001 * Eigen::VectorXd::Random(satellite.controlDim());

	std::vector<Eigen::MatrixXd> K(N - 1);
	std::vector<Eigen::VectorXd> d(N - 1);
	for (int kk = 0; kk < N - 1; ++kk) {
		K[kk] = Eigen::MatrixXd::Zero(satellite.controlDim(), satellite.reducedStateDim());
		d[kk] = Eigen::VectorXd::Zero(satellite.controlDim());
	}
	Eigen::Vector2d deltaV = Eigen::Vector2d::Zero();

	const bool ok = optimizer::backwardPass(
		satellite, X, U, R, V, B, S, rho, boresight, attitude_target_traj, settings, settings.passes[0].reg.reg_init, K, d, deltaV
	);

	// Should succeed (regularization loop finds positive definite Q_uu)
	REQUIRE(ok);
	REQUIRE(K.size() == 1);
	REQUIRE(d.size() == 1);
	REQUIRE(K[0].allFinite());
	REQUIRE(d[0].allFinite());
}

TEST_CASE_METHOD(BackwardPassFixture, "backward_pass handles longer trajectory N=5", "[backward_pass][longer_trajectory]") {
	constexpr int N_test = 5;

	PlannerSettings settings_test = settings;
	settings_test.num_passes = 1;
	settings_test.passes[0].dt = 0.5;

	Satellite satellite_test(
		Eigen::Vector3d(0.067, 0.071, 0.069).asDiagonal(), settings_test
	);
	satellite_test.addMTQ(Eigen::Vector3d::UnitX(), 0.2);
	satellite_test.addMTQ(Eigen::Vector3d::UnitY(), 0.2);
	satellite_test.addMTQ(Eigen::Vector3d::UnitZ(), 0.2);
	satellite_test.addRW(Eigen::Vector3d::UnitX(), 0.001, 1e-5, 0.0, 0.02);
	satellite_test.addRW(Eigen::Vector3d::UnitY(), 0.001, 1e-5, 0.0, 0.02);
	satellite_test.addRW(Eigen::Vector3d::UnitZ(), 0.001, 1e-5, 0.0, 0.02);

	Eigen::MatrixXd X(satellite_test.stateDim(), N_test);
	Eigen::MatrixXd U(satellite_test.controlDim(), N_test - 1);
	Eigen::MatrixXd R_test(3, N_test);
	Eigen::MatrixXd V_test(3, N_test);
	Eigen::MatrixXd B_test(3, N_test);
	Eigen::MatrixXd S_test(3, N_test);
	Eigen::MatrixXd rho_test(1, N_test);
	Eigen::MatrixXd boresight_test(3, N_test);
	Eigen::Vector4d attitude_target_test;

	for (int k = 0; k < N_test; ++k) {
		X.col(k) = x0;
		if (k < N_test - 1) U.col(k) = 0.001 * Eigen::VectorXd::Random(satellite_test.controlDim());

		R_test.col(k) = Eigen::Vector3d(7000e3, 0.0, 0.0);
		V_test.col(k) = Eigen::Vector3d(0.0, 7500.0, 0.0);
		B_test.col(k) = Eigen::Vector3d(2.5e-5, -1.5e-5, 3.0e-5);
		S_test.col(k) = Eigen::Vector3d(1.0, 0.1, -0.05).normalized();
		rho_test(0, k) = 0.0;
		boresight_test.col(k) = Eigen::Vector3d::UnitX();
	}

	attitude_target_test << std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0;
	Eigen::MatrixXd attitude_target_test_traj = makeAttitudeTraj(attitude_target_test, N_test);

	std::vector<Eigen::MatrixXd> K(N_test - 1);
	std::vector<Eigen::VectorXd> d(N_test - 1);
	int nu_test_5 = satellite_test.controlDim();
	int nx_test_5 = satellite_test.reducedStateDim();
	for (int kk = 0; kk < N_test - 1; ++kk) {
		K[kk] = Eigen::MatrixXd::Zero(nu_test_5, nx_test_5);
		d[kk] = Eigen::VectorXd::Zero(nu_test_5);
	}
	Eigen::Vector2d deltaV = Eigen::Vector2d::Zero();

	const bool ok = optimizer::backwardPass(
		satellite_test, X, U, R_test, V_test, B_test, S_test, rho_test, boresight_test, attitude_target_test_traj, settings_test, settings_test.passes[0].reg.reg_init, K, d, deltaV
	);

	REQUIRE(ok);
	REQUIRE(K.size() == N_test - 1);  // K[0] through K[N-2]
	REQUIRE(d.size() == N_test - 1);

	for (int k = 0; k < N_test - 1; ++k) {
		REQUIRE(K[k].allFinite());
		REQUIRE(d[k].allFinite());
	}
}

// ============================================================================
// DDP second-order terms (G12 dynamics Hessian / G13 constraint Hessian)
// ============================================================================

namespace {

// Run the backward pass on a fixed N=3 rotated-attitude problem and return the
// stacked feedback gains + feedforward + deltaV. The rotation makes ∂²f/∂q²
// nonzero and a nonzero cost-to-go gradient p makes the DDP term active.
struct BPResult {
	std::vector<Eigen::MatrixXd> K;
	std::vector<Eigen::VectorXd> d;
	Eigen::Vector2d deltaV;
	bool ok;
};

BPResult runDDPScenario(const std::function<void(PlannerSettings&)>& configure,
                        bool with_constraint_lambda) {
	constexpr int N_test = 3;
	PlannerSettings settings_test;
	settings_test.disturbances.plan_for_aero = false;
	settings_test.disturbances.plan_for_gg = false;
	settings_test.disturbances.plan_for_srp = false;
	settings_test.disturbances.plan_for_prop = false;
	settings_test.disturbances.plan_for_gendist = false;
	settings_test.disturbances.plan_for_resdipole = false;
	settings_test.num_passes = 1;
	settings_test.passes[0].dt = 0.5;
	settings_test.passes[0].reg.reg_init = 1e-6;
	settings_test.passes[0].reg.reg_scale = 10.0;
	settings_test.passes[0].reg.reg_max = 1e6;
	// Make the cost-to-go gradient large so the DDP pᵀ·∂²f term is non-trivial.
	settings_test.passes[0].cost.angle = 50.0;
	settings_test.passes[0].cost.ang_vel = 10.0;
	configure(settings_test);

	Satellite sat(Eigen::Vector3d(0.067, 0.071, 0.069).asDiagonal(), settings_test);
	sat.addMTQ(Eigen::Vector3d::UnitX(), 0.2);
	sat.addMTQ(Eigen::Vector3d::UnitY(), 0.2);
	sat.addMTQ(Eigen::Vector3d::UnitZ(), 0.2);
	sat.addRW(Eigen::Vector3d::UnitX(), 0.001, 1e-5, 0.0, 0.02);
	sat.addRW(Eigen::Vector3d::UnitY(), 0.001, 1e-5, 0.0, 0.02);
	sat.addRW(Eigen::Vector3d::UnitZ(), 0.001, 1e-5, 0.0, 0.02);

	const int nx = sat.stateDim();
	const int nu = sat.controlDim();
	const int nxr = sat.reducedStateDim();

	// Rotated state: 60 deg about a tilted axis → ∂²f/∂q² nonzero, p nonzero.
	Eigen::Vector3d axis = Eigen::Vector3d(0.3, -0.6, 0.74).normalized();
	const double half = 0.5 * (60.0 * PI / 180.0);
	Eigen::Vector4d q;
	q << std::cos(half), std::sin(half) * axis;
	Satellite::VecX xs = Satellite::VecX::Zero(nx);
	xs.segment<3>(Satellite::AV_INDEX) = Eigen::Vector3d(0.05, -0.03, 0.04);
	xs.segment<4>(Satellite::QUAT_INDEX) = q;
	if (sat.numRW() > 0) {
		for (int i = 0; i < sat.numRW(); ++i) xs(Satellite::RW_MOMENTUM_INDEX + i) = 0.005;
	}

	Eigen::MatrixXd X(nx, N_test), U(nu, N_test - 1);
	Eigen::MatrixXd R_t(3, N_test), V_t(3, N_test), B_t(3, N_test), S_t(3, N_test);
	Eigen::MatrixXd rho_t(1, N_test), bore_t(3, N_test);
	for (int k = 0; k < N_test; ++k) {
		X.col(k) = xs;
		if (k < N_test - 1) U.col(k) = Eigen::VectorXd::Constant(nu, 0.01);
		R_t.col(k) = Eigen::Vector3d(7000e3, 0.0, 0.0);
		V_t.col(k) = Eigen::Vector3d(0.0, 7500.0, 0.0);
		B_t.col(k) = Eigen::Vector3d(2.5e-5, -1.5e-5, 3.0e-5);
		S_t.col(k) = Eigen::Vector3d(1.0, 0.1, -0.05).normalized();
		rho_t(0, k) = 0.0;
		bore_t.col(k) = Eigen::Vector3d::UnitX();
	}
	// Goal: identity quaternion (so there is real attitude error → nonzero p).
	Eigen::Vector4d att;
	att << std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0;
	Eigen::MatrixXd att_traj = makeAttitudeTraj(att, N_test);

	std::vector<Eigen::VectorXd> lambda_aug, mu_aug;
	if (with_constraint_lambda) {
		// Drive an angular-velocity-limit constraint active by lowering wmax
		// below the state's |omega|, then arm a positive lambda so the active
		// set (c>0 OR lambda>0) is hit and the constraint Hessian contributes.
		settings_test.constraints.wmax = 0.01;  // |omega| ~ 0.07 > wmax → c > 0
		const Eigen::VectorXd c0 = sat.constraints(0, N_test, X.col(0), U.col(0),
		                                            S_t.col(0), settings_test.constraints);
		lambda_aug.assign(static_cast<std::size_t>(N_test), Eigen::VectorXd::Constant(c0.size(), 1.0));
		mu_aug.assign(static_cast<std::size_t>(N_test), Eigen::VectorXd::Constant(c0.size(), 100.0));
	}

	BPResult res;
	res.K.assign(static_cast<std::size_t>(N_test - 1), Eigen::MatrixXd::Zero(nu, nxr));
	res.d.assign(static_cast<std::size_t>(N_test - 1), Eigen::VectorXd::Zero(nu));
	res.deltaV = Eigen::Vector2d::Zero();
	res.ok = optimizer::backwardPass(
		sat, X, U, R_t, V_t, B_t, S_t, rho_t, bore_t, att_traj, settings_test,
		settings_test.passes[0].reg.reg_init, res.K, res.d, res.deltaV, lambda_aug, mu_aug);
	return res;
}

double maxGainDiff(const BPResult& a, const BPResult& b) {
	double m = 0.0;
	for (std::size_t k = 0; k < a.K.size(); ++k) {
		m = std::max(m, (a.K[k] - b.K[k]).cwiseAbs().maxCoeff());
		m = std::max(m, (a.d[k] - b.d[k]).cwiseAbs().maxCoeff());
	}
	return m;
}

bool allFinite(const BPResult& r) {
	for (std::size_t k = 0; k < r.K.size(); ++k) {
		if (!r.K[k].allFinite() || !r.d[k].allFinite()) return false;
	}
	return r.deltaV.allFinite();
}

}  // namespace

TEST_CASE("DDP regression guard: knobs OFF reproduce Gauss-Newton bitwise", "[backward_pass][ddp][regression]") {
	// Default (both DDP knobs off) MUST be bitwise identical to a run where we
	// explicitly disable them — i.e. the DDP code path adds exactly zero.
	BPResult gn = runDDPScenario([](PlannerSettings&) {}, false);
	BPResult gn2 = runDDPScenario([](PlannerSettings& s) {
		s.passes[0].reg.use_dynamics_hess = false;
		s.passes[0].reg.use_constraint_hess = false;
		s.passes[0].reg.psd_clip_quu_ddp = false;
	}, false);
	REQUIRE(gn.ok);
	REQUIRE(gn2.ok);
	// Bitwise identical: no tolerance.
	for (std::size_t k = 0; k < gn.K.size(); ++k) {
		REQUIRE(gn.K[k] == gn2.K[k]);
		REQUIRE(gn.d[k] == gn2.d[k]);
	}
	REQUIRE(gn.deltaV == gn2.deltaV);
}

TEST_CASE("DDP regression guard with active constraint: knobs OFF bitwise", "[backward_pass][ddp][regression]") {
	BPResult gn = runDDPScenario([](PlannerSettings&) {}, true);
	BPResult gn2 = runDDPScenario([](PlannerSettings& s) {
		s.passes[0].reg.use_dynamics_hess = false;
		s.passes[0].reg.use_constraint_hess = false;
	}, true);
	REQUIRE(gn.ok);
	REQUIRE(gn2.ok);
	for (std::size_t k = 0; k < gn.K.size(); ++k) {
		REQUIRE(gn.K[k] == gn2.K[k]);
		REQUIRE(gn.d[k] == gn2.d[k]);
	}
}

TEST_CASE("DDP dynamics Hessian changes the gains vs Gauss-Newton", "[backward_pass][ddp][dynamics_hess]") {
	BPResult gn  = runDDPScenario([](PlannerSettings&) {}, false);
	BPResult ddp = runDDPScenario([](PlannerSettings& s) {
		s.passes[0].reg.use_dynamics_hess = true;
		s.passes[0].reg.psd_clip_quu_ddp = true;  // keep Q_uu PD for the LLT
	}, false);
	REQUIRE(gn.ok);
	REQUIRE(ddp.ok);
	REQUIRE(allFinite(ddp));
	// The second-order dynamics term must actually move the solution.
	REQUIRE(maxGainDiff(gn, ddp) > 1e-9);
}

TEST_CASE("DDP constraint Hessian changes the gains vs Gauss-Newton", "[backward_pass][ddp][constraint_hess]") {
	BPResult gn  = runDDPScenario([](PlannerSettings&) {}, true);
	BPResult ddp = runDDPScenario([](PlannerSettings& s) {
		s.passes[0].reg.use_constraint_hess = true;
		s.passes[0].reg.psd_clip_quu_ddp = true;
	}, true);
	REQUIRE(gn.ok);
	REQUIRE(ddp.ok);
	REQUIRE(allFinite(ddp));
	REQUIRE(maxGainDiff(gn, ddp) > 1e-9);
}

TEST_CASE("DDP psd_clip yields a PSD-clipped matrix (descent-safe)", "[backward_pass][ddp][psd_clip]") {
	// Engineer an indefinite symmetric matrix and confirm the psd_clip used in
	// the backward pass produces a PSD matrix (eigenvalues >= 0) close to the
	// original on the positive subspace — the property that keeps Q_uu's LLT
	// solvable and the step a descent direction.
	Eigen::MatrixXd M(3, 3);
	M << 2.0, 0.0, 0.0,
	     0.0, -5.0, 1.0,
	     0.0, 1.0, 3.0;
	M = 0.5 * (M + M.transpose());
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es0(M);
	REQUIRE(es0.eigenvalues().minCoeff() < 0.0);  // confirm it really is indefinite

	// Mirror the in-BP psd_clip exactly.
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(M);
	Eigen::VectorXd lam = es.eigenvalues().cwiseMax(0.0);
	Eigen::MatrixXd Mc = es.eigenvectors() * lam.asDiagonal() * es.eigenvectors().transpose();

	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es1(Mc);
	REQUIRE(es1.eigenvalues().minCoeff() >= -1e-12);
	// With a small absolute regularization the clipped matrix is solvable.
	Eigen::MatrixXd Mreg = Mc + 1e-6 * Eigen::MatrixXd::Identity(3, 3);
	Eigen::LLT<Eigen::MatrixXd> llt(Mreg);
	REQUIRE(llt.info() == Eigen::Success);
}

TEST_CASE("rk4_hessians matches finite-difference of rk4_jacobians", "[backward_pass][ddp][fd_sanity]") {
	// FD-sanity: the discrete dynamics Hessian F_xx[l] from rk4_hessians must
	// equal ∂A_discrete(l-th row)/∂x from finite-differencing rk4_jacobians,
	// confirming the second-order RK4 composition is consistent with the
	// first-order chain rule the backward pass already trusts.
	PlannerSettings settings_test;
	settings_test.num_passes = 1;
	// Small dt + modest |ω| keep the central-difference truncation error
	// (∝ dt³·∂³f) well below the analytic-vs-FD tolerance, so the comparison
	// validates the rk4_hessians composition rather than FD truncation.
	settings_test.passes[0].dt = 0.05;
	Satellite sat(Eigen::Vector3d(0.067, 0.071, 0.069).asDiagonal(), settings_test);
	sat.addMTQ(Eigen::Vector3d::UnitX(), 0.2);
	sat.addRW(Eigen::Vector3d::UnitX(), 0.001, 1e-5, 0.0, 0.02);

	const int nx = sat.stateDim();
	const int nu = sat.controlDim();
	const double dt = settings_test.passes[0].dt;

	Eigen::Vector3d axis = Eigen::Vector3d(0.2, 0.5, -0.84).normalized();
	const double half = 0.5 * (40.0 * PI / 180.0);
	Eigen::Vector4d q; q << std::cos(half), std::sin(half) * axis;
	Eigen::VectorXd x = Eigen::VectorXd::Zero(nx);
	x.segment<3>(Satellite::AV_INDEX) = Eigen::Vector3d(0.08, -0.05, 0.06);
	x.segment<4>(Satellite::QUAT_INDEX) = q;
	if (sat.numRW() > 0) x(Satellite::RW_MOMENTUM_INDEX) = 0.004;
	Eigen::VectorXd u = Eigen::VectorXd::Constant(nu, 0.02);

	DisturbanceConfig dist;
	Eigen::Vector3d R0(7000e3, 0, 0), V0(0, 7500, 0), B0(2.5e-5, -1.5e-5, 3.0e-5);
	Eigen::Vector3d S0 = Eigen::Vector3d(1.0, 0.1, -0.05).normalized();

	auto jac_wrapper = [&](double, const Eigen::Ref<const Eigen::VectorXd>& xl,
	                       const Eigen::Ref<const Eigen::VectorXd>& ul,
	                       Eigen::Ref<Eigen::MatrixXd> A_out,
	                       Eigen::Ref<Eigen::MatrixXd> B_out,
	                       Eigen::Ref<Eigen::VectorXd> k_out) {
		auto [Ac, Bc, Cu] = sat.dynamicsJacobians(xl, ul, dist, R0, B0, S0, V0);
		A_out = Ac; B_out = Bc;
		k_out = sat.dynamics(xl, ul, dist, R0, B0, S0, V0, 0);
	};
	auto hess_wrapper = [&](double, const Eigen::Ref<const Eigen::VectorXd>& xl,
	                        const Eigen::Ref<const Eigen::VectorXd>& ul,
	                        Eigen::Ref<Eigen::MatrixXd> A_out,
	                        Eigen::Ref<Eigen::MatrixXd> B_out,
	                        Eigen::Ref<Eigen::VectorXd> k_out,
	                        std::vector<Eigen::MatrixXd>& fxx,
	                        std::vector<Eigen::MatrixXd>& fux,
	                        std::vector<Eigen::MatrixXd>& fuu) {
		auto [Ac, Bc, Cu] = sat.dynamicsJacobians(xl, ul, dist, R0, B0, S0, V0);
		A_out = Ac; B_out = Bc;
		k_out = sat.dynamics(xl, ul, dist, R0, B0, S0, V0, 0);
		auto [hxx, hux, huu] = sat.dynamicsHessians(xl, ul, dist, R0, B0, S0, V0);
		const int nxl = static_cast<int>(xl.size());
		const int nul = static_cast<int>(ul.size());
		fxx.assign(static_cast<std::size_t>(nxl), Eigen::MatrixXd::Zero(nxl, nxl));
		fux.assign(static_cast<std::size_t>(nxl), Eigen::MatrixXd::Zero(nul, nxl));
		fuu.assign(static_cast<std::size_t>(nxl), Eigen::MatrixXd::Zero(nul, nul));
		for (int l = 0; l < nxl; ++l) {
			fxx[static_cast<std::size_t>(l)] = hxx.slice(l).topLeftCorner(nxl, nxl);
			fux[static_cast<std::size_t>(l)] = hux.slice(l).topLeftCorner(nul, nxl);
			fuu[static_cast<std::size_t>(l)] = huu.slice(l).topLeftCorner(nul, nul);
		}
	};

	std::vector<Eigen::MatrixXd> Fxx, Fux, Fuu;
	rk4_hessians(hess_wrapper, x, u, 0.0, dt, Fxx, Fux, Fuu);

	// FD: d/dx_j of A_discrete(l, m) ≈ (A(x+e_j)(l,m) - A(x-e_j)(l,m)) / 2eps
	//     should equal Fxx[l](m, j).
	//
	// NOTE: dynamicsJacobians/dynamicsHessians internally normalize the
	// quaternion, while base's rk4_jacobians does NOT renormalize substeps (the
	// G6 / PR #41 chain-rule work is a separate, not-yet-landed change). So a
	// finite difference that perturbs a RAW quaternion component picks up the
	// normalization null-direction, which the analytic Hessian (evaluated at the
	// normalized q) does not model. We therefore FD-validate the NON-quaternion
	// state directions (ω indices 0..2 and the RW-momentum tail), where the
	// model is unambiguous. This still exercises the full RK4 second-order
	// composition (every stage, all output rows).
	auto is_quat = [&](int idx) { return idx >= Satellite::QUAT_INDEX && idx < Satellite::QUAT_INDEX + 4; };
	const double eps = 1e-6;
	double max_err = 0.0;
	int checked = 0;
	for (int j = 0; j < nx; ++j) {
		if (is_quat(j)) continue;
		Eigen::VectorXd xp = x; xp(j) += eps;
		Eigen::VectorXd xm = x; xm(j) -= eps;
		Eigen::MatrixXd Ap(nx, nx), Bp(nx, nu), Am(nx, nx), Bm(nx, nu);
		rk4_jacobians(jac_wrapper, xp, u, 0.0, dt, Ap, Bp);
		rk4_jacobians(jac_wrapper, xm, u, 0.0, dt, Am, Bm);
		Eigen::MatrixXd dA = (Ap - Am) / (2.0 * eps);  // dA(l,m)/dx_j
		for (int l = 0; l < nx; ++l) {
			for (int m = 0; m < nx; ++m) {
				if (is_quat(m)) continue;
				max_err = std::max(max_err, std::abs(dA(l, m) - Fxx[static_cast<std::size_t>(l)](m, j)));
				++checked;
			}
		}
	}
	INFO("checked " << checked << " entries, max |rk4_hessians - FD(rk4_jacobians)| = " << max_err);
	REQUIRE(checked > 0);
	REQUIRE(max_err < 1e-4);
}

TEST_CASE("DDP both knobs on with active constraint stays finite", "[backward_pass][ddp][combined]") {
	BPResult ddp = runDDPScenario([](PlannerSettings& s) {
		s.passes[0].reg.use_dynamics_hess = true;
		s.passes[0].reg.use_constraint_hess = true;
		s.passes[0].reg.psd_clip_quu_ddp = true;
	}, true);
	REQUIRE(ddp.ok);
	REQUIRE(allFinite(ddp));
}

TEST_CASE_METHOD(BackwardPassFixture, "backward_pass K and d have consistent norms across timesteps", "[backward_pass][gain_magnitudes]") {
	constexpr int N_test = 3;

	PlannerSettings settings_test = settings;
	Satellite satellite_test(
		Eigen::Vector3d(0.067, 0.071, 0.069).asDiagonal(), settings_test
	);
	satellite_test.addMTQ(Eigen::Vector3d::UnitX(), 0.2);
	satellite_test.addMTQ(Eigen::Vector3d::UnitY(), 0.2);
	satellite_test.addMTQ(Eigen::Vector3d::UnitZ(), 0.2);
	satellite_test.addRW(Eigen::Vector3d::UnitX(), 0.001, 1e-5, 0.0, 0.02);
	satellite_test.addRW(Eigen::Vector3d::UnitY(), 0.001, 1e-5, 0.0, 0.02);
	satellite_test.addRW(Eigen::Vector3d::UnitZ(), 0.001, 1e-5, 0.0, 0.02);

	Eigen::MatrixXd X(satellite_test.stateDim(), N_test);
	Eigen::MatrixXd U(satellite_test.controlDim(), N_test - 1);
	Eigen::MatrixXd R_test(3, N_test);
	Eigen::MatrixXd V_test(3, N_test);
	Eigen::MatrixXd B_test(3, N_test);
	Eigen::MatrixXd S_test(3, N_test);
	Eigen::MatrixXd rho_test(1, N_test);
	Eigen::MatrixXd boresight_test(3, N_test);
	Eigen::Vector4d attitude_target_test;

	// Uniform trajectory
	for (int k = 0; k < N_test; ++k) {
		X.col(k) = x0;
		if (k < N_test - 1) U.col(k).setZero();

		R_test.col(k) = Eigen::Vector3d(7000e3, 0.0, 0.0);
		V_test.col(k) = Eigen::Vector3d(0.0, 7500.0, 0.0);
		B_test.col(k) = Eigen::Vector3d(2.5e-5, -1.5e-5, 3.0e-5);
		S_test.col(k) = Eigen::Vector3d(1.0, 0.1, -0.05).normalized();
		rho_test(0, k) = 0.0;
		boresight_test.col(k) = Eigen::Vector3d::UnitX();
	}

	attitude_target_test << std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0;
	Eigen::MatrixXd attitude_target_test_traj = makeAttitudeTraj(attitude_target_test, N_test);

	std::vector<Eigen::MatrixXd> K(N_test - 1);
	std::vector<Eigen::VectorXd> d(N_test - 1);
	int nu_test_3 = satellite_test.controlDim();
	int nx_test_3 = satellite_test.reducedStateDim();
	for (int kk = 0; kk < N_test - 1; ++kk) {
		K[kk] = Eigen::MatrixXd::Zero(nu_test_3, nx_test_3);
		d[kk] = Eigen::VectorXd::Zero(nu_test_3);
	}
	Eigen::Vector2d deltaV = Eigen::Vector2d::Zero();

	const bool ok = optimizer::backwardPass(
		satellite_test, X, U, R_test, V_test, B_test, S_test, rho_test, boresight_test, attitude_target_test_traj, settings_test, settings_test.passes[0].reg.reg_init, K, d, deltaV
	);

	REQUIRE(ok);

	// For a uniform trajectory, gain magnitudes should be similar across timesteps
	double K0_norm = K[0].norm();
	double d0_norm = d[0].norm();
	REQUIRE(K0_norm >= 0.0);
	REQUIRE(d0_norm >= 0.0);

	// Gains should not explode or vanish unexpectedly
	for (size_t k = 1; k < K.size(); ++k) {
		double Kk_norm = K[k].norm();
		double dk_norm = d[k].norm();
		REQUIRE(std::isfinite(Kk_norm));
		REQUIRE(std::isfinite(dk_norm));
		REQUIRE(Kk_norm < 1e6);
		REQUIRE(dk_norm < 1e6);
	}
}

TEST_CASE_METHOD(BackwardPassFixture,
	"backward_pass linearizes with settings.disturbances (regression: BP ignored the config)",
	"[backward_pass][disturbances]") {
	// The forward pass rolls out with settings.disturbances; the backward pass
	// must linearize with the same config, or A_k/B_k disagree with the actual
	// rollout and the expected-decrease prediction is biased. This test runs
	// the identical trajectory through the BP with all disturbances off and
	// with gravity-gradient planning on: the gains must differ. (With the old
	// default-constructed DisturbanceConfig inside backwardPass, both runs
	// produced bit-identical K/d.)
	constexpr int N_test = 5;

	PlannerSettings settings_off = settings;  // fixture: all plan_for_* false
	PlannerSettings settings_gg = settings;
	settings_gg.disturbances.plan_for_gg = true;

	Satellite satellite_test(
		Eigen::Vector3d(0.067, 0.071, 0.069).asDiagonal(), settings_off
	);
	satellite_test.addMTQ(Eigen::Vector3d::UnitX(), 0.2);
	satellite_test.addMTQ(Eigen::Vector3d::UnitY(), 0.2);
	satellite_test.addMTQ(Eigen::Vector3d::UnitZ(), 0.2);
	satellite_test.addRW(Eigen::Vector3d::UnitX(), 0.001, 1e-5, 0.0, 0.02);
	satellite_test.addRW(Eigen::Vector3d::UnitY(), 0.001, 1e-5, 0.0, 0.02);
	satellite_test.addRW(Eigen::Vector3d::UnitZ(), 0.001, 1e-5, 0.0, 0.02);

	Eigen::MatrixXd X(satellite_test.stateDim(), N_test);
	Eigen::MatrixXd U(satellite_test.controlDim(), N_test - 1);
	Eigen::MatrixXd R_test(3, N_test);
	Eigen::MatrixXd V_test(3, N_test);
	Eigen::MatrixXd B_test(3, N_test);
	Eigen::MatrixXd S_test(3, N_test);
	Eigen::MatrixXd rho_test(1, N_test);
	Eigen::MatrixXd boresight_test(3, N_test);

	// Attitude rotated 30 deg about z so the gravity-gradient torque and its
	// quaternion Jacobian are nonzero (at identity with r along body x the GG
	// torque vanishes for a diagonal inertia).
	Satellite::VecX x_rot = x0;
	x_rot.segment<4>(Satellite::QUAT_INDEX) =
		Eigen::Vector4d(std::cos(M_PI / 12.0), 0.0, 0.0, std::sin(M_PI / 12.0));

	for (int k = 0; k < N_test; ++k) {
		X.col(k) = x_rot;
		if (k < N_test - 1) U.col(k) = 0.001 * Eigen::VectorXd::Random(satellite_test.controlDim());

		R_test.col(k) = Eigen::Vector3d(7000e3, 0.0, 0.0);
		V_test.col(k) = Eigen::Vector3d(0.0, 7500.0, 0.0);
		B_test.col(k) = Eigen::Vector3d(2.5e-5, -1.5e-5, 3.0e-5);
		S_test.col(k) = Eigen::Vector3d(1.0, 0.1, -0.05).normalized();
		rho_test(0, k) = 0.0;
		boresight_test.col(k) = Eigen::Vector3d::UnitX();
	}

	Eigen::Vector4d attitude_target_test;
	attitude_target_test << std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0;
	Eigen::MatrixXd attitude_target_test_traj = makeAttitudeTraj(attitude_target_test, N_test);

	const int nu = satellite_test.controlDim();
	const int nxr = satellite_test.reducedStateDim();

	auto run_bp = [&](const PlannerSettings& s,
	                  std::vector<Eigen::MatrixXd>& K,
	                  std::vector<Eigen::VectorXd>& d,
	                  Eigen::Vector2d& deltaV) {
		K.assign(N_test - 1, Eigen::MatrixXd::Zero(nu, nxr));
		d.assign(N_test - 1, Eigen::VectorXd::Zero(nu));
		deltaV.setZero();
		return optimizer::backwardPass(
			satellite_test, X, U, R_test, V_test, B_test, S_test, rho_test,
			boresight_test, attitude_target_test_traj, s,
			s.passes[0].reg.reg_init, K, d, deltaV
		);
	};

	std::vector<Eigen::MatrixXd> K_off, K_gg;
	std::vector<Eigen::VectorXd> d_off, d_gg;
	Eigen::Vector2d deltaV_off, deltaV_gg;

	REQUIRE(run_bp(settings_off, K_off, d_off, deltaV_off));
	REQUIRE(run_bp(settings_gg, K_gg, d_gg, deltaV_gg));

	double diff = 0.0;
	for (int k = 0; k < N_test - 1; ++k) {
		REQUIRE(K_gg[k].allFinite());
		REQUIRE(d_gg[k].allFinite());
		diff += (K_gg[k] - K_off[k]).norm() + (d_gg[k] - d_off[k]).norm();
	}
	REQUIRE(diff > 0.0);
}

TEST_CASE_METHOD(BackwardPassFixture,
	"backward_pass rejects non-finite trajectory states instead of emitting NaN gains",
	"[backward_pass][nan_guard]") {
	// Eigen's LLT does not flag NaN input, so before the guards a NaN state
	// produced NaN K/d "successfully" and the failure surfaced only when the
	// forward-pass rollout died -- a full FP later, misattributed to the line
	// search. The BP must fail fast instead.
	constexpr int N_test = 5;

	PlannerSettings settings_test = settings;
	Satellite satellite_test(
		Eigen::Vector3d(0.067, 0.071, 0.069).asDiagonal(), settings_test
	);
	satellite_test.addMTQ(Eigen::Vector3d::UnitX(), 0.2);
	satellite_test.addMTQ(Eigen::Vector3d::UnitY(), 0.2);
	satellite_test.addMTQ(Eigen::Vector3d::UnitZ(), 0.2);
	satellite_test.addRW(Eigen::Vector3d::UnitX(), 0.001, 1e-5, 0.0, 0.02);
	satellite_test.addRW(Eigen::Vector3d::UnitY(), 0.001, 1e-5, 0.0, 0.02);
	satellite_test.addRW(Eigen::Vector3d::UnitZ(), 0.001, 1e-5, 0.0, 0.02);

	Eigen::MatrixXd X(satellite_test.stateDim(), N_test);
	Eigen::MatrixXd U(satellite_test.controlDim(), N_test - 1);
	Eigen::MatrixXd R_t(3, N_test), V_t(3, N_test), B_t(3, N_test), S_t(3, N_test);
	Eigen::MatrixXd rho_t(1, N_test), bs_t(3, N_test);
	for (int k = 0; k < N_test; ++k) {
		X.col(k) = x0;
		if (k < N_test - 1) U.col(k) = 1e-4 * Eigen::VectorXd::Ones(satellite_test.controlDim());
		R_t.col(k) = Eigen::Vector3d(7000e3, 0.0, 0.0);
		V_t.col(k) = Eigen::Vector3d(0.0, 7500.0, 0.0);
		B_t.col(k) = Eigen::Vector3d(2.5e-5, -1.5e-5, 3.0e-5);
		S_t.col(k) = Eigen::Vector3d(1.0, 0.1, -0.05).normalized();
		rho_t(0, k) = 0.0;
		bs_t.col(k) = Eigen::Vector3d::UnitX();
	}
	Eigen::Vector4d tgt;
	tgt << std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0;
	Eigen::MatrixXd tgt_traj = makeAttitudeTraj(tgt, N_test);

	// Poison one mid-trajectory angular velocity entry.
	X(0, 2) = std::numeric_limits<double>::quiet_NaN();

	const int nu = satellite_test.controlDim();
	const int nxr = satellite_test.reducedStateDim();
	std::vector<Eigen::MatrixXd> K(N_test - 1, Eigen::MatrixXd::Zero(nu, nxr));
	std::vector<Eigen::VectorXd> d(N_test - 1, Eigen::VectorXd::Zero(nu));
	Eigen::Vector2d deltaV = Eigen::Vector2d::Zero();

	const bool ok = optimizer::backwardPass(
		satellite_test, X, U, R_t, V_t, B_t, S_t, rho_t, bs_t, tgt_traj,
		settings_test, settings_test.passes[0].reg.reg_init, K, d, deltaV
	);

	REQUIRE_FALSE(ok);
	// And no NaN gains may have been handed back for the knots it did process.
	for (const auto& Kk : K) {
		REQUIRE((Kk.allFinite() || Kk.isZero()));
	}
}

TEST_CASE_METHOD(BackwardPassFixture, "backward_pass psd_clamp_lxx diagnostic flag", "[backward_pass][psd_clamp_lxx]") {
	// psd_clamp_lxx is a TESTING/DIAGNOSTIC knob (default false). This test
	// guards:
	//  1. Regression: with the flag at its default (untouched settings) the
	//     result is bitwise-identical to the flag explicitly set to false.
	//  2. Flag on, well-conditioned problem: backward pass still succeeds
	//     and results stay finite.
	//  3. Flag on, indefinite-lxx scenario (analytic cost Hessian with the
	//     concave raw-acos shape, heavy stage angle weight, large attitude
	//     error): backward pass succeeds and results stay finite.
	constexpr int N_test = 5;

	PlannerSettings settings_test = settings;
	settings_test.num_passes = 1;
	settings_test.passes[0].dt = 0.5;

	Satellite satellite_test(
		Eigen::Vector3d(0.067, 0.071, 0.069).asDiagonal(), settings_test
	);
	satellite_test.addMTQ(Eigen::Vector3d::UnitX(), 0.2);
	satellite_test.addMTQ(Eigen::Vector3d::UnitY(), 0.2);
	satellite_test.addMTQ(Eigen::Vector3d::UnitZ(), 0.2);
	satellite_test.addRW(Eigen::Vector3d::UnitX(), 0.001, 1e-5, 0.0, 0.02);
	satellite_test.addRW(Eigen::Vector3d::UnitY(), 0.001, 1e-5, 0.0, 0.02);
	satellite_test.addRW(Eigen::Vector3d::UnitZ(), 0.001, 1e-5, 0.0, 0.02);

	Eigen::MatrixXd X(satellite_test.stateDim(), N_test);
	Eigen::MatrixXd U(satellite_test.controlDim(), N_test - 1);
	Eigen::MatrixXd R_test(3, N_test);
	Eigen::MatrixXd V_test(3, N_test);
	Eigen::MatrixXd B_test(3, N_test);
	Eigen::MatrixXd S_test(3, N_test);
	Eigen::MatrixXd rho_test(1, N_test);
	Eigen::MatrixXd boresight_test(3, N_test);

	// Deterministic trajectory (no Random) so repeated runs are comparable.
	for (int k = 0; k < N_test; ++k) {
		X.col(k) = x0;
		if (k < N_test - 1) U.col(k).setZero();

		R_test.col(k) = Eigen::Vector3d(7000e3, 0.0, 0.0);
		V_test.col(k) = Eigen::Vector3d(0.0, 7500.0, 0.0);
		B_test.col(k) = Eigen::Vector3d(2.5e-5, -1.5e-5, 3.0e-5);
		S_test.col(k) = Eigen::Vector3d(1.0, 0.1, -0.05).normalized();
		rho_test(0, k) = 0.0;
		boresight_test.col(k) = Eigen::Vector3d::UnitX();
	}

	// Quaternion attitude target 120 deg about x away from the (identity)
	// state quaternion, so d = q_goal . q = 0.5 and the raw-acos shape
	// (type 2) has strongly negative curvature f''(d) < 0.
	Eigen::Vector4d attitude_target_test(0.5, std::sqrt(3.0) / 2.0, 0.0, 0.0);
	Eigen::MatrixXd attitude_target_test_traj = makeAttitudeTraj(attitude_target_test, N_test);

	const int nu_t = satellite_test.controlDim();
	const int nxr_t = satellite_test.reducedStateDim();

	auto runBP = [&](bool set_flag, bool flag_value, bool indefinite_cost,
	                 std::vector<Eigen::MatrixXd>& K, std::vector<Eigen::VectorXd>& d) {
		PlannerSettings s = settings_test;
		if (set_flag) {
			s.passes[0].reg.psd_clamp_lxx = flag_value;
		}
		if (indefinite_cost) {
			s.passes[0].cost.use_cost_hess = true;
			s.passes[0].cost.ang_cost_func_type = 2;  // raw acos: concave in d
			s.passes[0].cost.angle = 1e5;             // heavy stage angle weight
			s.passes[0].cost.angle_N = 0.0;           // keep terminal P_N PSD (clamp covers stage lxx only)
		}
		K.assign(N_test - 1, Eigen::MatrixXd::Zero(nu_t, nxr_t));
		d.assign(N_test - 1, Eigen::VectorXd::Zero(nu_t));
		Eigen::Vector2d deltaV = Eigen::Vector2d::Zero();
		return optimizer::backwardPass(
			satellite_test, X, U, R_test, V_test, B_test, S_test, rho_test, boresight_test, attitude_target_test_traj, s, s.passes[0].reg.reg_init, K, d, deltaV
		);
	};

	// 1. Regression guard: default flag (untouched RegularizationConfig)
	//    must be bitwise-identical to flag explicitly off.
	std::vector<Eigen::MatrixXd> K_default, K_off, K_on, K_diag;
	std::vector<Eigen::VectorXd> d_default, d_off, d_on, d_diag;

	REQUIRE(runBP(false, false, false, K_default, d_default));
	REQUIRE(runBP(true, false, false, K_off, d_off));
	for (int k = 0; k < N_test - 1; ++k) {
		REQUIRE((K_default[k].array() == K_off[k].array()).all());
		REQUIRE((d_default[k].array() == d_off[k].array()).all());
	}

	// 2. Flag on, well-conditioned problem: succeeds, finite.
	REQUIRE(runBP(true, true, false, K_on, d_on));
	for (int k = 0; k < N_test - 1; ++k) {
		REQUIRE(K_on[k].allFinite());
		REQUIRE(d_on[k].allFinite());
	}

	// 3. Flag on, indefinite-lxx scenario: succeeds, finite.
	REQUIRE(runBP(true, true, true, K_diag, d_diag));
	for (int k = 0; k < N_test - 1; ++k) {
		REQUIRE(K_diag[k].allFinite());
		REQUIRE(d_diag[k].allFinite());
	}
}
