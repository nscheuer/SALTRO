/**
 * @file test_properties.cpp
 * @brief Property-based unit tests for SALTRO's optimizer (C++ twins).
 *
 * Catch2 twins of tests/unit/optimizer/test_properties.py, mirroring the
 * python configs and tolerances exactly (same satellite, same
 * PlannerSettings, same assertions). See the python file's module docstring
 * for the full rationale; briefly:
 *
 * - Warmstart determinism: two identical trajOpt calls give bit-identical
 *   trajectories (all warmstart controllers are deterministic by design).
 *
 * - Eigenaxis preservation: single y-axis RW + initial rotation purely
 *   about y => the trajectory stays in the y-axial subspace
 *   (omega_x = omega_z = q_x = q_z = 0) to machine precision. The 90 deg
 *   and ~180 deg cases are tagged [!mayfail] (the Catch2 analog of the
 *   python xfail(strict=False)) pending the large-angle cost-tuning
 *   investigation; if they ever start passing, promote them.
 *
 * - At-goal-commands-zero: if x(0) is the goal with zero rate, the unique
 *   optimum is u = 0 and the planner must converge to it.
 */
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>

#include <Eigen/Dense>

#include <saltro/limits.h>
#include <saltro/optimizer/trajOpt.h>
#include <saltro/pybind/plannersettings.h>
#include <saltro/pybind/satellite.h>

using namespace saltro;

namespace {

constexpr double PI = 3.14159265358979323846;
constexpr double SEC_PER_CENTURY = 36525.0 * 86400.0;

/// Twin of test_properties.py::create_default_planner_settings.
/// Standard single-pass PlannerSettings tuned for the property tests:
/// deterministic IntegratedBdotController warmstart + moderate tolerances.
PlannerSettings createDefaultPlannerSettings(double dt_seconds) {
	PlannerSettings ps;
	ps.init_traj.initcontroller = 2;         // IntegratedBdotController

	ps.num_passes = 1;
	ps.passes[0].dt = dt_seconds;
	ps.passes[0].ilqr.max_iters = 30;
	ps.passes[0].ilqr.cost_tol = 1e-6;
	ps.passes[0].auglag.max_outer_iters = 15;
	ps.passes[0].auglag.constraint_tol = 1e-3;

	CostConfig& c = ps.passes[0].cost;
	c.angle = 1e3;
	c.angle_N = 1e6;
	c.ang_vel = 1e3;
	c.ang_vel_N = 1e5;
	c.ang_vel_mag = 0.0;
	c.ang_vel_err_dir = 0.0;
	c.control_mult = 1.0;
	c.mtq_control_weight = 1e3;
	c.rw_control_weight = 1.0;
	c.magic_control_weight = 0.0;
	c.rw_AM_weight = 0.0;
	c.rw_stic_weight = 0.0;
	c.RWh_max_mult = 0.0;
	c.RWh_stiction_mult = 0.0;
	c.RWh_ok_mult = 0.0;
	c.ang_vel_mag_N = 0.0;
	c.ang_vel_err_dir_N = 0.0;
	c.ang_cost_func_type = 2;
	c.use_cost_hess = true;

	ps.disturbances.plan_for_aero = false;
	ps.disturbances.plan_for_gg = false;
	ps.disturbances.plan_for_srp = false;
	ps.disturbances.plan_for_prop = false;
	ps.disturbances.plan_for_gendist = false;
	ps.disturbances.plan_for_resdipole = false;

	ps.passes[0].reg.reg_init = 1e-6;
	ps.passes[0].reg.reg_max = 1e10;
	ps.passes[0].reg.reg_scale = 10.0;
	ps.passes[0].reg.use_dynamics_hess = false;
	ps.passes[0].reg.use_constraint_hess = false;

	ps.passes[0].linesearch.max_iters = 24;
	ps.passes[0].linesearch.beta1 = 1e-10;
	ps.passes[0].linesearch.beta2 = 5000.0;

	return ps;
}

struct TrajOptResult {
	bool ok = false;
	Eigen::MatrixXd X;  ///< (state_dim, N)
	Eigen::MatrixXd U;  ///< (input_dim, N)
};

/// Twin of test_properties.py::_run_trajopt: 0 MTQ + 1 y-axis RW satellite
/// with isotropic 0.1 kg m^2 inertia (tests/debug/optimizer/configs/
/// sat_0_1_rw_y.py), planning from a pure-y rotation to identity.
TrajOptResult runTrajopt(double initial_y_angle_rad,
                         double omega0_y = 0.0,
                         double dt_seconds = 1.0,
                         int N = 31) {
	PlannerSettings ps = createDefaultPlannerSettings(dt_seconds);

	const Eigen::Matrix3d J = 0.1 * Eigen::Matrix3d::Identity();
	Satellite sat(J, ps);
	// addRW signature: (axis, max_torque, J_wheel, h0, h_max).
	sat.addRW(Eigen::Vector3d::UnitY(), 1.0, 1e-6, 0.0, 10.0);

	const int state_dim = sat.stateDim();
	const int input_dim = sat.controlDim();

	Eigen::VectorXd jtime(2);
	jtime << 0.22, 0.22 + (N - 1) * dt_seconds / SEC_PER_CENTURY;

	// Goal: identity attitude held throughout the horizon; qgoal / boresight
	// are given at the start and end of jtime and resampled inside trajOpt.
	Eigen::MatrixXd q_goal(4, 2);
	q_goal.col(0) = Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
	q_goal.col(1) = Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
	// Boresight = body z-axis, held in body frame.
	Eigen::MatrixXd boresight(3, 2);
	boresight.col(0) = Eigen::Vector3d(0.0, 0.0, 1.0);
	boresight.col(1) = Eigen::Vector3d(0.0, 0.0, 1.0);

	Satellite::VecX x0 = Satellite::VecX::Zero(state_dim);
	x0.segment<3>(Satellite::AV_INDEX) = Eigen::Vector3d(0.0, omega0_y, 0.0);
	Eigen::Vector4d q0(std::cos(initial_y_angle_rad / 2.0), 0.0,
	                   std::sin(initial_y_angle_rad / 2.0), 0.0);
	q0.normalize();
	x0.segment<4>(Satellite::QUAT_INDEX) = q0;
	// RW momentum (last entry) stays 0.

	const Eigen::Vector3d r0(7000e3, 0.0, 0.0);
	const Eigen::Vector3d v0(0.0, 7.5e3, 0.0);

	Eigen::MatrixXd X(state_dim, limits::MAX_LENGTH_TRAJ);
	Eigen::MatrixXd U(input_dim, limits::MAX_LENGTH_TRAJ);
	const int reduced_state_dim = sat.reducedStateDim();
	Eigen::MatrixXd K(input_dim, reduced_state_dim * limits::MAX_LENGTH_TRAJ);
	int N_out = static_cast<int>(jtime.size());

	TrajOptResult result;
	result.ok = optimizer::trajOpt(
		ps, sat, x0, r0, v0, jtime, q_goal, boresight,
		X, U, K,
		state_dim, input_dim, N_out
	);

	const int N_cols = std::max(0, std::min(N_out, limits::MAX_LENGTH_TRAJ));
	result.X = X.leftCols(N_cols);
	result.U = U.leftCols(N_cols);
	return result;
}

/// Twin of test_y_axis_rw_preserves_y_axial_subspace's assertion block.
/// State layout: [omega_x, omega_y, omega_z, q_w, q_x, q_y, q_z, h_RW];
/// off-axis omega_x, omega_z, q_x, q_z must stay zero to machine precision.
void requireYAxialSubspace(double init_angle, int N) {
	const TrajOptResult res = runTrajopt(init_angle, 0.0, 1.0, N);
	REQUIRE(res.ok);

	constexpr int off_axis_idx[4] = {0, 2, 4, 6};  // omega_x, omega_z, q_x, q_z
	for (const int idx : off_axis_idx) {
		const double max_off = res.X.row(idx).cwiseAbs().maxCoeff();
		INFO("off-axis state row " << idx << " deviated from zero: max = "
		     << max_off << " (init angle = " << init_angle << " rad)");
		REQUIRE(max_off < 1e-9);
	}
}

} // namespace

// ---------------------------------------------------------------------------
// Warmstart determinism
// ---------------------------------------------------------------------------

TEST_CASE("trajOpt warmstart is deterministic", "[properties][trajOpt][determinism]") {
	// Two identical trajOpt calls must give bit-identical trajectories.
	// The warmstart controllers (Zero / Excitation / IntegratedBdot) are
	// deterministic by design; this catches hidden randomness (e.g., an
	// unseeded randn, a la OldPlanner's OldPlanner.cpp:276).
	const TrajOptResult r1 = runTrajopt(0.1);
	const TrajOptResult r2 = runTrajopt(0.1);
	REQUIRE(r1.ok);
	REQUIRE(r2.ok);

	REQUIRE(r1.X.rows() == r2.X.rows());
	REQUIRE(r1.X.cols() == r2.X.cols());
	REQUIRE(r1.U.rows() == r2.U.rows());
	REQUIRE(r1.U.cols() == r2.U.cols());

	INFO("State trajectory not bit-identical across identical runs: max diff = "
	     << (r1.X - r2.X).cwiseAbs().maxCoeff());
	REQUIRE((r1.X.array() == r2.X.array()).all());

	INFO("Control trajectory not bit-identical: max diff = "
	     << (r1.U - r2.U).cwiseAbs().maxCoeff());
	REQUIRE((r1.U.array() == r2.U.array()).all());
}

// ---------------------------------------------------------------------------
// Eigenaxis preservation (symmetry-protected)
// ---------------------------------------------------------------------------

TEST_CASE("y-axis RW preserves y-axial subspace (small angle)",
          "[properties][trajOpt][symmetry]") {
	// Single y-axis RW + initial rotation purely about y MUST give a
	// trajectory confined to (omega_y, delta_q_y, h_RW): the dynamics
	// conserve this subspace exactly when only a y-axis torque is available
	// and the cost is rotation-axis-agnostic.
	requireYAxialSubspace(0.1, 31);
}

// The two large-angle cases are the twins of the python xfail(strict=False)
// params: AL-iLQR with the default-tuned costs doesn't converge for large
// rest-to-rest slews via a single y-axis RW (needs cost retuning or a longer
// horizon; tracked separately). [!mayfail] reports them without failing the
// suite; if SALTRO ever converges these, they show up as "failed as expected"
// no longer -- promote them to regular tests.

TEST_CASE("y-axis RW preserves y-axial subspace (90 deg, convergence pending)",
          "[properties][trajOpt][symmetry][!mayfail]") {
	requireYAxialSubspace(PI / 2.0, 121);
}

TEST_CASE("y-axis RW preserves y-axial subspace (~180 deg, convergence pending)",
          "[properties][trajOpt][symmetry][!mayfail]") {
	// Pre-180 deg is harder still (cost-function near-singularity).
	requireYAxialSubspace(PI - 0.05, 201);
}

// ---------------------------------------------------------------------------
// At-goal commands zero
// ---------------------------------------------------------------------------

TEST_CASE("at goal with zero rate commands near-zero control",
          "[properties][trajOpt][atgoal]") {
	// If x(0) = goal with zero rate, the optimal control is u = 0 by
	// construction (zero state cost everywhere along u = 0; the control
	// cost has its unique global minimum at u = 0). Any sustained nonzero
	// control is a goal-handling bug or a spurious local solution.
	const TrajOptResult res = runTrajopt(0.0, 0.0, 1.0, 20);
	REQUIRE(res.ok);

	const double max_u = res.U.cwiseAbs().maxCoeff();
	const double rms_u = std::sqrt(res.U.array().square().mean());
	INFO("At-goal commanded nonzero control: max|u| = " << max_u
	     << " (rms|u| = " << rms_u << "); the optimum is u = 0.");
	REQUIRE(max_u < 1e-3);
}
