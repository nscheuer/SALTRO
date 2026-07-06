// C++ twin of tests/unit/optimizer/test_tvlqr_disturbance.py.
//
// Disturbance-aware TVLQR (McKeen 2025, eq. 7.40): with
// settings.tvlqr.disturbance_aware set, trajOpt augments each per-step gain to
// [K_x | K_tau] of width reducedStateDim + 3. K_x is unchanged; K_tau feeds back
// the disturbance-torque error. See the .py twin for the full rationale.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include <saltro/limits.h>
#include <saltro/math/mrp.h>
#include <saltro/math/quaternion.h>
#include <saltro/optimizer/trajOpt.h>
#include <saltro/orbit_generation/generate_orbit.h>
#include <saltro/pybind/satellite.h>

using namespace saltro;

namespace {

constexpr double SEC_PER_CENTURY = 36525.0 * 86400.0;
constexpr int NRED = 9;  // reducedStateDim for the 3-RW satellite

PlannerSettings makeSettings(double dt, bool disturbance_aware, bool plan_dist,
                             const Eigen::Vector3d& res_dipole,
                             bool use_sqrt_bp = false) {
	PlannerSettings s;
	s.init_traj.initcontroller = 2;
	s.num_passes = 1;
	s.passes[0].dt = dt;
	s.passes[0].ilqr.cost_tol = 1e-5;
	s.passes[0].ilqr.max_iters = 40;
	s.passes[0].auglag.max_outer_iters = 12;
	s.passes[0].auglag.constraint_tol = 1e-3;
	auto& c = s.passes[0].cost;
	// Balanced running/terminal weights (defaults are terminal ~1e4x running,
	// which makes any LQR terminal-biased).
	c.angle = 1e2;
	c.ang_vel = 1e2;
	c.angle_N = 1e2;
	c.ang_vel_N = 1e2;
	c.control_mult = 1.0;
	c.mtq_control_weight = 1e-2;
	c.rw_control_weight = 1.0;
	c.ang_cost_func_type = 3;
	c.use_cost_hess = true;
	// The re-keyed RW momentum cost (post-#54) has an always-on desat
	// quadratic whose default weight (rw_AM_weight = 1e4) dominates this
	// disturbance-dominated scenario and reshapes the plan the behavioral
	// margins were tuned on. Pin a moderate weight: wheels stay in-band
	// without the momentum cost strangling the disturbance-fighting plan.
	c.rw_AM_weight = 1e2;
	s.disturbances.plan_for_resdipole = plan_dist;
	if (plan_dist) {
		s.disturbances.res_dipole = res_dipole;
	}
	s.passes[0].reg.reg_init = 1e-6;
	s.passes[0].reg.reg_max = 1e10;
	s.passes[0].reg.reg_scale = 10.0;
	s.passes[0].reg.use_dynamics_hess = false;
	s.passes[0].reg.use_constraint_hess = false;
	s.passes[0].reg.use_sqrt_bp = use_sqrt_bp;
	s.passes[0].linesearch.max_iters = 24;
	s.passes[0].linesearch.beta1 = 1e-10;
	s.passes[0].linesearch.beta2 = 5000.0;
	s.tvlqr.tvlqr_len = 1000.0;
	s.tvlqr.tvlqr_overlap = 0.0;
	s.tvlqr.disturbance_aware = disturbance_aware;
	return s;
}

Eigen::Matrix3d rwInertia() {
	Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
	J(0, 0) = 0.067;
	J(1, 1) = 0.071;
	J(2, 2) = 0.069;
	return J;
}

void configureRW(Satellite& s) {
	s.addRW(Eigen::Vector3d::UnitX(), 0.001, 1e-5, 0.0, 0.02);
	s.addRW(Eigen::Vector3d::UnitY(), 0.001, 1e-5, 0.0, 0.02);
	s.addRW(Eigen::Vector3d::UnitZ(), 0.001, 1e-5, 0.0, 0.02);
}

Satellite::VecX makeX0(const Satellite& s) {
	Satellite::VecX x0(s.stateDim());
	x0.setZero();
	x0.segment<3>(Satellite::AV_INDEX) = Eigen::Vector3d(-0.01, 0.02, 0.03);
	x0.segment<4>(Satellite::QUAT_INDEX) = Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
	return x0;
}

struct Solve {
	bool ok = false;
	int N = 0;
	int w = 0;  // per-step gain width
	Eigen::MatrixXd X, U, K;
};

Solve solveRW(const PlannerSettings& settings, double tf = 200.0) {
	Satellite sat(rwInertia(), settings);
	configureRW(sat);
	const Satellite::VecX x0 = makeX0(sat);
	const Eigen::Vector3d r0(7000e3, 0.0, 0.0), v0(0.0, 7.5e3, 0.0);
	Eigen::VectorXd jtime(2);
	jtime << 0.22, 0.22 + tf / SEC_PER_CENTURY;
	Eigen::MatrixXd q_goal(4, 2);
	q_goal << std::sqrt(2.0) / 2.0, std::sqrt(2.0) / 2.0, 0.0, 0.0, 0.0, 0.0,
	          std::sqrt(2.0) / 2.0, std::sqrt(2.0) / 2.0;
	Eigen::MatrixXd boresight(3, 2);
	boresight << 1.0, 1.0, 0.0, 0.0, 0.0, 0.0;

	const int sd = sat.stateDim(), id = sat.controlDim(), rd = sat.reducedStateDim();
	const int w = settings.tvlqr.disturbance_aware ? rd + 3 : rd;
	Eigen::MatrixXd X = Eigen::MatrixXd::Zero(sd, limits::MAX_LENGTH_TRAJ);
	Eigen::MatrixXd U = Eigen::MatrixXd::Zero(id, limits::MAX_LENGTH_TRAJ);
	Eigen::MatrixXd K = Eigen::MatrixXd::Zero(id, w * limits::MAX_LENGTH_TRAJ);
	int N = static_cast<int>(jtime.size());
	Solve r;
	REQUIRE_NOTHROW(r.ok = optimizer::trajOpt(settings, sat, x0, r0, v0, jtime, q_goal,
	                                           boresight, X, U, K, sd, id, N));
	r.N = N;
	r.w = w;
	r.X = X.leftCols(N);
	r.U = U.leftCols(N);
	r.K = K.leftCols(w * N);
	return r;
}

Eigen::VectorXd reducedError(const Eigen::VectorXd& xc, const Eigen::VectorXd& xp, int nRW) {
	Eigen::VectorXd dz = Eigen::VectorXd::Zero(6 + nRW);
	dz.head<3>() = xc.head<3>() - xp.head<3>();
	dz.segment<3>(3) = math::quatToMRP(math::quatError(xp.segment<4>(3), xc.segment<4>(3)));
	for (int i = 0; i < nRW; ++i) dz(6 + i) = xc(7 + i) - xp(7 + i);
	return dz;
}

double attErr(const Eigen::VectorXd& xc, const Eigen::VectorXd& xp) {
	return math::quatToMRP(math::quatError(xp.segment<4>(3), xc.segment<4>(3))).norm();
}

using OrbitCols = Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ>;
using ScalarRow = Eigen::Matrix<double, 1, limits::MAX_LENGTH_TRAJ>;

// Roll out under a real dipole that drifts from m_plan by `drift` over the
// maneuver; returns (rms, peak) attitude-tracking error. With use_dist_fb, the
// disturbance feedforward K_tau acts on the live mismatch (tau_est - tau_plan).
std::pair<double, double> driftRollout(
		const Satellite& sat, const Solve& r,
		const OrbitCols& R, const OrbitCols& V, const OrbitCols& B, const OrbitCols& S,
		const ScalarRow& rho, const Eigen::Vector3d& m_plan, const Eigen::Vector3d& drift,
		double dt, bool use_dist_fb) {
	const int N = r.N;
	const int nRW = sat.stateDim() - 7;
	Eigen::VectorXd x = makeX0(sat);
	double sumsq = 0.0;
	double peak = 0.0;
	for (int k = 0; k < N - 1; ++k) {
		const Eigen::Vector3d m_real = m_plan + drift * (static_cast<double>(k) / (N - 1));
		DisturbanceConfig dist_real;
		dist_real.plan_for_resdipole = true;
		dist_real.res_dipole = m_real;
		Eigen::VectorXd u = r.U.col(k)
			+ r.K.block(0, k * r.w, r.K.rows(), NRED) * reducedError(x, r.X.col(k), nRW);
		if (use_dist_fb) {
			const Eigen::Vector3d B_body =
				math::rotationMatrix(x.segment<4>(3)).transpose() * B.col(k);
			const Eigen::Vector3d d_tau = m_real.cross(B_body) - m_plan.cross(B_body);
			u += r.K.block(0, k * r.w + NRED, r.K.rows(), 3) * d_tau;
		}
		const auto f = [&](const Eigen::VectorXd& xx) {
			return sat.dynamics(xx, u, dist_real, R.col(k), B.col(k), S.col(k), V.col(k),
			                    static_cast<int>(rho(0, k)));
		};
		const Eigen::VectorXd k1 = f(x);
		const Eigen::VectorXd k2 = f(x + 0.5 * dt * k1);
		const Eigen::VectorXd k3 = f(x + 0.5 * dt * k2);
		const Eigen::VectorXd k4 = f(x + dt * k3);
		x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
		x.segment<4>(3).normalize();
		const double e = attErr(x, r.X.col(k + 1));
		sumsq += e * e;
		peak = std::max(peak, e);
	}
	return {std::sqrt(sumsq / (N - 1)), peak};
}

}  // namespace

TEST_CASE("disturbance-aware TVLQR widens gains and preserves state feedback",
          "[optimizer][tvlqr][disturbance]") {
	const Eigen::Vector3d m(0.2, -0.1, 0.15);
	const Solve off = solveRW(makeSettings(10.0, false, true, m));
	const Solve on = solveRW(makeSettings(10.0, true, true, m));
	REQUIRE(off.ok);
	REQUIRE(on.ok);
	REQUIRE(off.K.cols() == NRED * off.N);
	REQUIRE(on.K.cols() == (NRED + 3) * on.N);

	double kx_diff = 0.0, ktau = 0.0;
	for (int k = 0; k < on.N; ++k) {
		const Eigen::MatrixXd kx_on = on.K.block(0, k * (NRED + 3), on.K.rows(), NRED);
		const Eigen::MatrixXd kx_off = off.K.block(0, k * NRED, off.K.rows(), NRED);
		kx_diff = std::max(kx_diff, (kx_on - kx_off).cwiseAbs().maxCoeff());
		ktau = std::max(ktau,
		                on.K.block(0, k * (NRED + 3) + NRED, on.K.rows(), 3).cwiseAbs().maxCoeff());
	}
	REQUIRE(kx_diff == 0.0);     // K_x identical to the un-augmented gain
	REQUIRE(on.K.allFinite());
	REQUIRE(ktau > 1e-9);        // K_tau present
}

TEST_CASE("disturbance-aware TVLQR gains are deterministic",
          "[optimizer][tvlqr][disturbance]") {
	const Eigen::Vector3d m(0.2, -0.1, 0.15);
	const Solve a = solveRW(makeSettings(10.0, true, true, m));
	const Solve b = solveRW(makeSettings(10.0, true, true, m));
	REQUIRE(a.K.cols() == b.K.cols());
	REQUIRE((a.K - b.K).cwiseAbs().maxCoeff() == 0.0);
}

TEST_CASE("disturbance feedback improves tracking under a changing disturbance",
          "[optimizer][tvlqr][disturbance]") {
	const double dt = 10.0;
	const double tf = 600.0;  // longer horizon so the sustained disturbance matters
	// A large (dynamically-significant) planned dipole: the controller fights it
	// the whole way, so the feedforward earns its keep -- unlike a fast slew
	// where the maneuver dominates and feedback alone copes.
	const Eigen::Vector3d m_plan(1.0, 0.6, -0.8);

	const PlannerSettings settings = makeSettings(dt, true, true, m_plan);  // plan WITH disturbance
	Satellite sat(rwInertia(), settings);
	configureRW(sat);

	const Solve r = solveRW(settings, tf);
	REQUIRE(r.ok);

	const Eigen::Vector3d r0(7000e3, 0.0, 0.0), v0(0.0, 7.5e3, 0.0);
	ScalarRow jtime_full;
	jtime_full.setZero();
	for (int k = 0; k < r.N; ++k) jtime_full(0, k) = 0.22 + k * (dt / SEC_PER_CENTURY);
	OrbitCols R, V, B, S;
	ScalarRow rho;
	R.setZero(); V.setZero(); B.setZero(); S.setZero(); rho.setZero();
	REQUIRE(orbits::generate_orbit(r0, v0, jtime_full, r.N, 0, 0, 0, 0, 0, R, V, B, S, rho));

	// Changing disturbance: the real dipole drifts to 100% off the planned value.
	const Eigen::Vector3d drift = m_plan;
	const auto [rms_so, peak_so] = driftRollout(sat, r, R, V, B, S, rho, m_plan, drift, dt, false);
	const auto [rms_da, peak_da] = driftRollout(sat, r, R, V, B, S, rho, m_plan, drift, dt, true);
	REQUIRE(std::isfinite(rms_so));
	REQUIRE(std::isfinite(rms_da));
	// Feeding back the live mismatch improves tracking in BOTH RMS and peak. The
	// gain is plan-dependent and toolchain-sensitive, so assert only conservative
	// margins (ratios well below 1); the realized improvement is comfortably larger.
	REQUIRE(rms_da < 0.9 * rms_so);
	REQUIRE(peak_da < 0.92 * peak_so);
}

TEST_CASE("disturbance feedback is a no-op when the disturbance matches the plan",
          "[optimizer][tvlqr][disturbance]") {
	// Twin of test_disturbance_feedback_is_a_noop_when_the_disturbance_matches
	// _the_plan (.py). When the real dipole equals the planned one, the mismatch
	// tau_est - tau_exp is identically zero, so the disturbance feedforward
	// contributes nothing and the closed loop is bit-identical to state feedback
	// alone -- no penalty for enabling the feature when the plan is right.
	const double dt = 10.0;
	const double tf = 600.0;
	const Eigen::Vector3d m_plan(1.0, 0.6, -0.8);

	const PlannerSettings settings = makeSettings(dt, true, true, m_plan);  // plan WITH disturbance
	Satellite sat(rwInertia(), settings);
	configureRW(sat);

	const Solve r = solveRW(settings, tf);
	REQUIRE(r.ok);

	const Eigen::Vector3d r0(7000e3, 0.0, 0.0), v0(0.0, 7.5e3, 0.0);
	ScalarRow jtime_full;
	jtime_full.setZero();
	for (int k = 0; k < r.N; ++k) jtime_full(0, k) = 0.22 + k * (dt / SEC_PER_CENTURY);
	OrbitCols R, V, B, S;
	ScalarRow rho;
	R.setZero(); V.setZero(); B.setZero(); S.setZero(); rho.setZero();
	REQUIRE(orbits::generate_orbit(r0, v0, jtime_full, r.N, 0, 0, 0, 0, 0, R, V, B, S, rho));

	// No mismatch: the real dipole stays at the planned value (zero drift).
	const auto [rms_so, peak_so] =
		driftRollout(sat, r, R, V, B, S, rho, m_plan, Eigen::Vector3d::Zero(), dt, false);
	const auto [rms_da, peak_da] =
		driftRollout(sat, r, R, V, B, S, rho, m_plan, Eigen::Vector3d::Zero(), dt, true);
	// Same tolerance as the .py twin: pytest.approx(rel=1e-9, abs=1e-12).
	using Catch::Matchers::WithinAbs;
	using Catch::Matchers::WithinRel;
	REQUIRE_THAT(rms_da, WithinRel(rms_so, 1e-9) || WithinAbs(rms_so, 1e-12));
	REQUIRE_THAT(peak_da, WithinRel(peak_so, 1e-9) || WithinAbs(peak_so, 1e-12));
}

TEST_CASE("disturbance feedback improves tracking with the sqrt backward pass",
          "[optimizer][tvlqr][disturbance][sqrt]") {
	// Twin of "disturbance feedback improves tracking under a changing
	// disturbance" with use_sqrt_bp=true: the disturbance-aware machinery is
	// wired through backwardPassSqrt, so the sqrt-mode solve must converge and
	// its [K_x | K_tau] gains must deliver tracking equivalent to the dense
	// pass.
	const double dt = 10.0;
	const double tf = 600.0;
	const Eigen::Vector3d m_plan(1.0, 0.6, -0.8);

	const PlannerSettings settings = makeSettings(dt, true, true, m_plan, /*use_sqrt_bp=*/true);
	Satellite sat(rwInertia(), settings);
	configureRW(sat);

	const Solve r = solveRW(settings, tf);
	REQUIRE(r.ok);
	REQUIRE(r.K.allFinite());

	const Eigen::Vector3d r0(7000e3, 0.0, 0.0), v0(0.0, 7.5e3, 0.0);
	ScalarRow jtime_full;
	jtime_full.setZero();
	for (int k = 0; k < r.N; ++k) jtime_full(0, k) = 0.22 + k * (dt / SEC_PER_CENTURY);
	OrbitCols R, V, B, S;
	ScalarRow rho;
	R.setZero(); V.setZero(); B.setZero(); S.setZero(); rho.setZero();
	REQUIRE(orbits::generate_orbit(r0, v0, jtime_full, r.N, 0, 0, 0, 0, 0, R, V, B, S, rho));

	const Eigen::Vector3d drift = m_plan;
	const auto [rms_so, peak_so] = driftRollout(sat, r, R, V, B, S, rho, m_plan, drift, dt, false);
	const auto [rms_da, peak_da] = driftRollout(sat, r, R, V, B, S, rho, m_plan, drift, dt, true);
	REQUIRE(std::isfinite(rms_so));
	REQUIRE(std::isfinite(rms_da));
	// Same conservative margins as the dense-mode test.
	REQUIRE(rms_da < 0.9 * rms_so);
	REQUIRE(peak_da < 0.92 * peak_so);

	// Equivalent tracking to the dense-mode solve: per-iteration sqrt-vs-dense
	// backward-pass parity is ~1e-8 relative, so the closed-loop tracking
	// errors of the two solves agree far tighter than the behavioral margins.
	const PlannerSettings settings_dense = makeSettings(dt, true, true, m_plan);
	const Solve r_dense = solveRW(settings_dense, tf);
	REQUIRE(r_dense.ok);
	const auto [rms_so_d, peak_so_d] =
		driftRollout(sat, r_dense, R, V, B, S, rho, m_plan, drift, dt, false);
	const auto [rms_da_d, peak_da_d] =
		driftRollout(sat, r_dense, R, V, B, S, rho, m_plan, drift, dt, true);
	(void)rms_so_d;
	(void)peak_so_d;
	using Catch::Matchers::WithinRel;
	REQUIRE_THAT(rms_da, WithinRel(rms_da_d, 5e-2));
	REQUIRE_THAT(peak_da, WithinRel(peak_da_d, 5e-2));
}

TEST_CASE("disturbance feedback is a no-op when matching, sqrt backward pass",
          "[optimizer][tvlqr][disturbance][sqrt]") {
	// Twin of "disturbance feedback is a no-op when the disturbance matches
	// the plan" with use_sqrt_bp=true.
	const double dt = 10.0;
	const double tf = 600.0;
	const Eigen::Vector3d m_plan(1.0, 0.6, -0.8);

	const PlannerSettings settings = makeSettings(dt, true, true, m_plan, /*use_sqrt_bp=*/true);
	Satellite sat(rwInertia(), settings);
	configureRW(sat);

	const Solve r = solveRW(settings, tf);
	REQUIRE(r.ok);

	const Eigen::Vector3d r0(7000e3, 0.0, 0.0), v0(0.0, 7.5e3, 0.0);
	ScalarRow jtime_full;
	jtime_full.setZero();
	for (int k = 0; k < r.N; ++k) jtime_full(0, k) = 0.22 + k * (dt / SEC_PER_CENTURY);
	OrbitCols R, V, B, S;
	ScalarRow rho;
	R.setZero(); V.setZero(); B.setZero(); S.setZero(); rho.setZero();
	REQUIRE(orbits::generate_orbit(r0, v0, jtime_full, r.N, 0, 0, 0, 0, 0, R, V, B, S, rho));

	const auto [rms_so, peak_so] =
		driftRollout(sat, r, R, V, B, S, rho, m_plan, Eigen::Vector3d::Zero(), dt, false);
	const auto [rms_da, peak_da] =
		driftRollout(sat, r, R, V, B, S, rho, m_plan, Eigen::Vector3d::Zero(), dt, true);
	using Catch::Matchers::WithinAbs;
	using Catch::Matchers::WithinRel;
	REQUIRE_THAT(rms_da, WithinRel(rms_so, 1e-9) || WithinAbs(rms_so, 1e-12));
	REQUIRE_THAT(peak_da, WithinRel(peak_so, 1e-9) || WithinAbs(peak_so, 1e-12));
}
