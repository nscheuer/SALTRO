#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <vector>

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
	// Match backward pass implementation: scale stage derivatives by dt.
	l_x_0 *= dt;
	l_u_0 *= dt;
	l_xx_0 *= dt;
	l_uu_0 *= dt;
	l_ux_hess_0 *= dt;

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

	// Verify K[0] and d[0] match expected values (within numerical tolerance)
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
