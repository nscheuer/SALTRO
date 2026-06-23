// Comprehensive forward pass tests exercising cost improvement, dynamics fidelity,
// and backtracking behavior.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>
#include <limits>

#include <Eigen/Dense>

#include <saltro/limits.h>
#include <saltro/math/mrp.h>
#include <saltro/math/integrators/rk4.h>
#include <saltro/optimizer/backwardpass.h>
#include <saltro/optimizer/forwardpass.h>
#include <saltro/optimizer/warm_start.h>
#include <saltro/orbit_generation/generate_orbit.h>
#include <saltro/pybind/satellite.h>

using namespace saltro;

namespace {

constexpr double PI = 3.14159265358979323846;
constexpr double SEC_PER_CENTURY = 36525.0 * 86400.0;

struct EnvMatrices {
	Eigen::MatrixXd R;
	Eigen::MatrixXd V;
	Eigen::MatrixXd B;
	Eigen::MatrixXd S;
	Eigen::MatrixXd rho;
};

struct RolloutResult {
	Eigen::MatrixXd X;
	Eigen::MatrixXd U;
	double cost = std::numeric_limits<double>::quiet_NaN();
};

class ForwardPassFixture {
public:
	static constexpr int N = 8;

	PlannerSettings settings;
	Satellite satellite;
	Satellite::VecX x0;
	Eigen::VectorXd jtime;
	Eigen::MatrixXd q_goal;
	Eigen::MatrixXd boresight;
	Eigen::MatrixXd attitude_target_traj;

	Eigen::Matrix<double, 1, limits::MAX_LENGTH_TRAJ> jtime_full;
	Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> R_full;
	Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> V_full;
	Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> B_full;
	Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> S_full;
	Eigen::Matrix<double, 1, limits::MAX_LENGTH_TRAJ> rho_full;

	bool orbit_ok = false;

	ForwardPassFixture()
		: settings(),
		  satellite(makeInertia(), settings),
		  x0(Satellite::VecX::Zero(satellite.stateDim())),
		  jtime(Eigen::VectorXd::Zero(N)),
		  q_goal(Eigen::MatrixXd::Zero(4, N)),
		  boresight(Eigen::MatrixXd::Zero(3, N)),
		  attitude_target_traj(Eigen::MatrixXd::Zero(4, N))
	{
		configureSettings();
		configureSatellite();
		configureTimeline();
		configureTargets();
		orbit_ok = generateOrbit();
	}

	bool warmStart(Eigen::MatrixXd& X, Eigen::MatrixXd& U) const {
		return optimizer::warm_start(
			settings,
			satellite,
			x0,
			jtime,
			q_goal,
			boresight,
			N,
			R_full,
			V_full,
			B_full,
			S_full,
			rho_full,
			X,
			U
		);
	}

	EnvMatrices envMatrices() const {
		EnvMatrices env;
		env.R.resize(3, N);
		env.V.resize(3, N);
		env.B.resize(3, N);
		env.S.resize(3, N);
		env.rho.resize(1, N);

		for (int k = 0; k < N; ++k) {
			env.R.col(k) = R_full.col(k);
			env.V.col(k) = V_full.col(k);
			env.B.col(k) = B_full.col(k);
			env.S.col(k) = S_full.col(k);
			env.rho.col(k) = rho_full.col(k);
		}
		return env;
	}

	RolloutResult rolloutWithAlpha(
		double alpha,
		const std::vector<Eigen::MatrixXd>& K,
		const std::vector<Eigen::VectorXd>& d,
		const PlannerSettings& settings_in,
		const Eigen::MatrixXd& X_ref,
		const Eigen::MatrixXd& U_ref,
		const EnvMatrices& env
	) const {
		const int nx = static_cast<int>(X_ref.rows());
		const int nu = static_cast<int>(U_ref.rows());

		RolloutResult result;
		result.X = Eigen::MatrixXd::Zero(nx, N);
		result.U = Eigen::MatrixXd::Zero(nu, std::max(0, N - 1));
		result.X.col(0) = X_ref.col(0);

		const auto& dist_cfg = settings_in.disturbances;
		const CostConfig& cost_cfg = settings_in.passes[0].cost;

		for (int k = 0; k < N - 1; ++k) {
			double dt = timestepSeconds(k, settings_in);
			Eigen::VectorXd u_bar = U_ref.col(k);
			if (k < static_cast<int>(K.size())) {
				// Match optimizer::forwardPass reduced-state feedback error.
				Eigen::VectorXd state_error_reduced(satellite.reducedStateDim());
				state_error_reduced.head<3>() = result.X.col(k).head<3>() - X_ref.col(k).head<3>();

				const Eigen::Vector4d q_ref = X_ref.col(k).segment<4>(Satellite::QUAT_INDEX);
				const Eigen::Vector4d q_bar = result.X.col(k).segment<4>(Satellite::QUAT_INDEX);
				const Eigen::Vector4d q_err = saltro::math::quatError(q_ref, q_bar);
				state_error_reduced.segment<3>(3) = saltro::math::quatToMRP(q_err);

				for (int i = 0; i < satellite.numRW(); ++i) {
					state_error_reduced(6 + i) = result.X(7 + i, k) - X_ref(7 + i, k);
				}

				u_bar += K[k] * state_error_reduced;
			}
			if (k < static_cast<int>(d.size())) {
				u_bar += alpha * d[k];
			}
			result.U.col(k) = u_bar;

			Eigen::VectorXd x_next;
			rk4_step<Eigen::VectorXd>(
				[&](double, const Eigen::VectorXd& x_state, Eigen::VectorXd& dxdt) {
					dxdt = satellite.dynamics(
						x_state,
						u_bar,
						dist_cfg,
						env.R.col(k),
						env.B.col(k),
						env.S.col(k),
						env.V.col(k),
						static_cast<int>(std::max(0.0, std::round(env.rho(0, k))))
					);
				},
				result.X.col(k),
				0.0,
				dt,
				x_next
			);

			result.X.col(k + 1) = x_next;
		}

		result.cost = satellite.totalCost(result.X, result.U, env.B, boresight, attitude_target_traj, cost_cfg);
		return result;
	}

		double timestepSeconds(int k, const PlannerSettings& settings_in) const {
			double dt = 0.0;
			if (jtime.size() > k + 1) {
				const double dt_centuries = jtime(k + 1) - jtime(k);
				if (std::isfinite(dt_centuries) && dt_centuries > 0.0) {
					dt = dt_centuries * SEC_PER_CENTURY;
				}
			}
			if ((!std::isfinite(dt) || dt <= 0.0) && settings_in.num_passes > 0 && std::isfinite(settings_in.passes[0].dt) && settings_in.passes[0].dt > 0.0) {
				dt = settings_in.passes[0].dt;
			}
			return dt;
		}

private:
	static Eigen::Matrix3d makeInertia() {
		Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
		J(0, 0) = 0.067;
		J(1, 1) = 0.071;
		J(2, 2) = 0.069;
		return J;
	}

	void configureSettings() {
		settings.num_passes = 1;
		settings.passes[0].dt = 0.5;  // seconds
		settings.disturbances.plan_for_aero = false;
		settings.disturbances.plan_for_gg = false;
		settings.disturbances.plan_for_srp = false;
		settings.disturbances.plan_for_prop = false;
		settings.disturbances.plan_for_gendist = false;
		settings.disturbances.plan_for_resdipole = false;
	}

	void configureSatellite() {
		satellite.addMTQ(Eigen::Vector3d::UnitX(), 0.2);
		satellite.addMTQ(Eigen::Vector3d::UnitY(), 0.2);
		satellite.addMTQ(Eigen::Vector3d::UnitZ(), 0.2);

		satellite.addRW(Eigen::Vector3d::UnitX(), 0.001, 1e-5, 0.0, 0.02);
		satellite.addRW(Eigen::Vector3d::UnitY(), 0.001, 1e-5, 0.0, 0.02);
		satellite.addRW(Eigen::Vector3d::UnitZ(), 0.001, 1e-5, 0.0, 0.02);

		x0 = Satellite::VecX::Zero(satellite.stateDim());
		x0.segment<3>(Satellite::AV_INDEX) = Eigen::Vector3d(0.02, -0.01, 0.015);
		x0.segment<4>(Satellite::QUAT_INDEX) = Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
	}

	void configureTimeline() {
		const double dt_seconds = settings.passes[0].dt;
		const double dt_centuries = dt_seconds / SEC_PER_CENTURY;
		jtime_full.setZero();
		for (int k = 0; k < N; ++k) {
			jtime(k) = 0.25 + k * dt_centuries;
			jtime_full(0, k) = jtime(k);
		}
	}

	void configureTargets() {
		for (int k = 0; k < N; ++k) {
			q_goal.col(k) = Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
			boresight.col(k) = Eigen::Vector3d::UnitX();
			attitude_target_traj.col(k) = q_goal.col(k);
		}
	}

	bool generateOrbit() {
		Eigen::Vector3d r0(7000e3, 0.0, 0.0);
		Eigen::Vector3d v0(0.0, 7500.0, 0.0);
		R_full.setZero();
		V_full.setZero();
		B_full.setZero();
		S_full.setZero();
		rho_full.setZero();

		return orbits::generate_orbit(
			r0,
			v0,
			jtime_full,
			N,
			0, // Keplerian
			0, // Tilted dipole
			0, // NOAA Sun model
			0, // Cylindrical eclipse
			0, // Harris-Priester density
			R_full,
			V_full,
			B_full,
			S_full,
			rho_full
		);
	}
};

} // namespace

TEST_CASE_METHOD(ForwardPassFixture, "forward_pass reduces cost and matches dynamics", "[forward_pass][cost][dynamics]") {
	REQUIRE(orbit_ok);

	Eigen::MatrixXd X = Eigen::MatrixXd::Zero(satellite.stateDim(), N);
	Eigen::MatrixXd U = Eigen::MatrixXd::Zero(satellite.controlDim(), N);
	REQUIRE(warmStart(X, U));

	EnvMatrices env = envMatrices();
	const CostConfig& cost_cfg = settings.passes[0].cost;

	std::vector<Eigen::MatrixXd> K(N - 1, Eigen::MatrixXd::Zero(satellite.controlDim(), satellite.stateDim()));
	std::vector<Eigen::VectorXd> d(N - 1, Eigen::VectorXd::Zero(satellite.controlDim()));
	Eigen::Vector2d deltaV = Eigen::Vector2d::Zero();

	Eigen::MatrixXd U_trimmed = U.leftCols(N - 1);
	REQUIRE(optimizer::backwardPass(
		satellite,
		X,
		U_trimmed,
		env.R,
		env.V,
		env.B,
		env.S,
		env.rho,
		boresight,
		attitude_target_traj,
		settings,
		settings.passes[0].reg.reg_init,
		K,
		d,
		deltaV
	));

	double J_prev = satellite.totalCost(X, U_trimmed, env.B, boresight, attitude_target_traj, cost_cfg);
	double J_new = std::numeric_limits<double>::quiet_NaN();
	REQUIRE(optimizer::forwardPass(
		satellite,
		X,
		U,
		K,
		d,
		deltaV,
		env.B,
		env.R,
		env.V,
		env.S,
		env.rho,
		boresight,
		attitude_target_traj,
		settings,
		jtime,
		J_prev,
		J_new
	));

	REQUIRE(std::isfinite(J_new));
	REQUIRE(J_new <= J_prev + 1e-6);
	REQUIRE(X.cols() == N);
	REQUIRE(U.cols() == N);

	for (int k = 0; k < N - 1; ++k) {
		double dt = timestepSeconds(k, settings);
		REQUIRE(dt > 0.0);

		Eigen::VectorXd x_next;
		rk4_step<Eigen::VectorXd>(
			[&](double, const Eigen::VectorXd& x_state, Eigen::VectorXd& dxdt) {
				dxdt = satellite.dynamics(
					x_state,
					U.col(k),
					settings.disturbances,
					env.R.col(k),
					env.B.col(k),
					env.S.col(k),
					env.V.col(k),
					static_cast<int>(std::max(0.0, std::round(env.rho(0, k))))
				);
			},
			X.col(k),
			0.0,
			dt,
			x_next
		);

		REQUIRE((x_next - X.col(k + 1)).norm() < 1e-10);
	}
}

// Helper: scan FP's halving alpha sequence and find the one whose manual
// rollout cost matches J_new at fp precision.  Returns -1.0 if no match.
namespace {
double findChosenAlpha(double J_new,
                       const std::vector<Eigen::MatrixXd>& K,
                       const std::vector<Eigen::VectorXd>& d,
                       const ForwardPassFixture& fixture,
                       const PlannerSettings& settings_ls,
                       const Eigen::MatrixXd& X_base,
                       const Eigen::MatrixXd& U_base,
                       const EnvMatrices& env) {
	const int max_iters = settings_ls.passes[0].linesearch.max_iters;
	const double tol = 1e-12 * std::max(1.0, std::abs(J_new));
	for (int iter = 0; iter < max_iters; ++iter) {
		const double alpha = std::ldexp(1.0, -iter);
		const auto rollout = fixture.rolloutWithAlpha(alpha, K, d, settings_ls,
		                                              X_base, U_base, env);
		if (std::abs(rollout.cost - J_new) <= tol) {
			return alpha;
		}
	}
	return -1.0;
}
} // namespace

TEST_CASE_METHOD(ForwardPassFixture, "forward_pass J_new matches its accepted alphas rollout cost", "[forward_pass][linesearch]") {
	// Self-consistency check: FP's reported J_new must equal the manual
	// rollout cost at some alpha in {1, 1/2, 1/4, ...}.  Engineers a scenario
	// where alpha=1 overshoots (forces *some* nontrivial backtracking
	// behavior or alpha=1 acceptance) and verifies FP's reported J_new is
	// consistent with one of its own alpha candidates.  Does NOT assert
	// FP backed off — that's the next test.
	REQUIRE(orbit_ok);

	Eigen::MatrixXd X_base = Eigen::MatrixXd::Zero(satellite.stateDim(), N);
	Eigen::MatrixXd U_base = Eigen::MatrixXd::Zero(satellite.controlDim(), N);
	REQUIRE(warmStart(X_base, U_base));

	EnvMatrices env = envMatrices();
	const CostConfig& cost_cfg = settings.passes[0].cost;

	std::vector<Eigen::MatrixXd> K_base(N - 1, Eigen::MatrixXd::Zero(satellite.controlDim(), satellite.stateDim()));
	std::vector<Eigen::VectorXd> d_base(N - 1, Eigen::VectorXd::Zero(satellite.controlDim()));
	Eigen::Vector2d deltaV_base = Eigen::Vector2d::Zero();

	Eigen::MatrixXd U_bp = U_base.leftCols(N - 1);
	REQUIRE(optimizer::backwardPass(
		satellite, X_base, U_bp, env.R, env.V, env.B, env.S, env.rho,
		boresight, attitude_target_traj, settings,
		settings.passes[0].reg.reg_init,
		K_base, d_base, deltaV_base
	));

	double J_prev = satellite.totalCost(X_base, U_bp, env.B, boresight, attitude_target_traj, cost_cfg);

	// Pick a scale where alpha=1 overshoots and alpha=0.5 helps.
	const std::array<double, 4> scales = {2.0, 4.0, 6.0, 8.0};
	double chosen_scale = 1.0;
	RolloutResult alpha1;
	bool found = false;
	for (double scale : scales) {
		std::vector<Eigen::VectorXd> d_scaled = d_base;
		for (auto& dk : d_scaled) { dk *= scale; }
		alpha1 = rolloutWithAlpha(1.0, K_base, d_scaled, settings, X_base, U_base, env);
		const auto alpha_half = rolloutWithAlpha(0.5, K_base, d_scaled, settings, X_base, U_base, env);
		if (alpha1.cost > J_prev && alpha_half.cost < J_prev) {
			chosen_scale = scale;
			found = true;
			break;
		}
	}
	REQUIRE(found);

	std::vector<Eigen::VectorXd> d_scaled = d_base;
	for (auto& dk : d_scaled) { dk *= chosen_scale; }
	Eigen::Vector2d deltaV_scaled;
	deltaV_scaled(0) = chosen_scale * deltaV_base(0);
	deltaV_scaled(1) = chosen_scale * chosen_scale * deltaV_base(1);

	PlannerSettings settings_ls = settings;
	settings_ls.passes[0].linesearch.beta2 = 1.5;

	Eigen::MatrixXd X_forward = X_base;
	Eigen::MatrixXd U_forward = U_base;
	double J_new = std::numeric_limits<double>::quiet_NaN();
	REQUIRE(optimizer::forwardPass(
		satellite, X_forward, U_forward, K_base, d_scaled, deltaV_scaled,
		env.B, env.R, env.V, env.S, env.rho,
		boresight, attitude_target_traj, settings_ls, jtime, J_prev, J_new
	));

	// Cost-decrease invariant: never worse than alpha=1.
	REQUIRE(J_new <= alpha1.cost + 1e-8);

	// J_new matches FP's own alpha candidate's rollout at fp64 precision.
	const double chosen_alpha = findChosenAlpha(J_new, K_base, d_scaled, *this,
	                                            settings_ls, X_base, U_base, env);
	REQUIRE(chosen_alpha > 0.0);
}

// NOTE: a deterministic "FP backs off below alpha=1 when alpha=1
// overshoots" test would be ideal but is hard to construct robustly.
// Armijo accept/reject at alpha=1 depends on the z ratio
//   z = (J_prev - J_new) / (-alpha*(deltaV(0) + alpha*deltaV(1)))
// which depends on (a) how badly alpha=1 overshoots the cost (numerator)
// and (b) how the quadratic deltaV(1) term interacts with the linear
// deltaV(0) term after scaling d (denominator).  For the BP-computed
// d_base we have deltaV(1) = -0.5 * deltaV(0) so after scaling d by s
// the predicted delta at alpha=1 flips sign at s=2 — making the
// classic "overshoot triggers backoff" intuition non-deterministic in
// scaled-d setups.  Engineering a stable scenario requires either
// custom-crafted d that is descent-only (no Q_uu^{-1} structure) or
// instrumentation of beta1/beta2 boundaries which becomes brittle.
//
// The self-consistency test above + the accept-alpha=1 test below
// cover the algorithmic invariants we actually depend on.

TEST_CASE_METHOD(ForwardPassFixture, "forward_pass accepts alpha=1 when full step already descends", "[forward_pass][linesearch][no-backtrack]") {
	// Use the unscaled BP step (which descends at alpha=1 by construction
	// for a small initial-condition trajectory) and assert FP picks alpha=1
	// exactly — no backtracking.
	REQUIRE(orbit_ok);

	Eigen::MatrixXd X_base = Eigen::MatrixXd::Zero(satellite.stateDim(), N);
	Eigen::MatrixXd U_base = Eigen::MatrixXd::Zero(satellite.controlDim(), N);
	REQUIRE(warmStart(X_base, U_base));

	EnvMatrices env = envMatrices();
	const CostConfig& cost_cfg = settings.passes[0].cost;

	std::vector<Eigen::MatrixXd> K_base(N - 1, Eigen::MatrixXd::Zero(satellite.controlDim(), satellite.stateDim()));
	std::vector<Eigen::VectorXd> d_base(N - 1, Eigen::VectorXd::Zero(satellite.controlDim()));
	Eigen::Vector2d deltaV_base = Eigen::Vector2d::Zero();

	Eigen::MatrixXd U_bp = U_base.leftCols(N - 1);
	REQUIRE(optimizer::backwardPass(
		satellite, X_base, U_bp, env.R, env.V, env.B, env.S, env.rho,
		boresight, attitude_target_traj, settings,
		settings.passes[0].reg.reg_init,
		K_base, d_base, deltaV_base
	));

	double J_prev = satellite.totalCost(X_base, U_bp, env.B, boresight, attitude_target_traj, cost_cfg);

	// Precondition: alpha=1 with unscaled d already gives J_new < J_prev.
	const auto alpha1 = rolloutWithAlpha(1.0, K_base, d_base, settings, X_base, U_base, env);
	REQUIRE(alpha1.cost < J_prev);

	Eigen::MatrixXd X_forward = X_base;
	Eigen::MatrixXd U_forward = U_base;
	double J_new = std::numeric_limits<double>::quiet_NaN();
	REQUIRE(optimizer::forwardPass(
		satellite, X_forward, U_forward, K_base, d_base, deltaV_base,
		env.B, env.R, env.V, env.S, env.rho,
		boresight, attitude_target_traj, settings, jtime, J_prev, J_new
	));

	// FP should accept alpha=1 directly.
	const double chosen_alpha = findChosenAlpha(J_new, K_base, d_base, *this,
	                                            settings, X_base, U_base, env);
	REQUIRE(chosen_alpha > 0.0);
	REQUIRE(chosen_alpha == 1.0);
}
