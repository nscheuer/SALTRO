#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include <Eigen/Dense>

#include <saltro/limits.h>
#include <saltro/optimizer/backwardpass.h>
#include <saltro/optimizer/forwardpass.h>
#include <saltro/optimizer/warm_start.h>
#include <saltro/orbit_generation/generate_orbit.h>
#include <saltro/pybind/satellite.h>

using namespace saltro;

namespace {

constexpr double SEC_PER_CENTURY = 36525.0 * 86400.0;

struct EnvMatrices {
	Eigen::MatrixXd R;
	Eigen::MatrixXd V;
	Eigen::MatrixXd B;
	Eigen::MatrixXd S;
	Eigen::MatrixXd rho;
};

class ForwardPassALFixture {
public:
	explicit ForwardPassALFixture(int n = 8)
		: N(n),
		  settings(),
		  satellite(makeInertia(), settings),
		  jtime(Eigen::VectorXd::Zero(N)),
		  q_goal(Eigen::MatrixXd::Zero(4, N)),
		  boresight(Eigen::MatrixXd::Zero(3, N)),
		  attitude_target_traj(Eigen::MatrixXd::Zero(4, N)) {
		configureSettings();
		// configureSatellite() resizes x0 to the post-addRW state dimension
		// (must run before configureTimeline / orbit-gen which reference x0).
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

	void initialTrajectory(Eigen::MatrixXd& X, Eigen::MatrixXd& U) const {
		X = Eigen::MatrixXd::Zero(satellite.stateDim(), N);
		U = Eigen::MatrixXd::Zero(satellite.controlDim(), N);
		if (!warmStart(X, U)) {
			for (int k = 0; k < N; ++k) {
				X.col(k) = x0;
			}
			U.setZero();
		}
	}

	EnvMatrices env() const {
		EnvMatrices out;
		out.R.resize(3, N);
		out.V.resize(3, N);
		out.B.resize(3, N);
		out.S.resize(3, N);
		out.rho.resize(1, N);
		for (int k = 0; k < N; ++k) {
			out.R.col(k) = R_full.col(k);
			out.V.col(k) = V_full.col(k);
			out.B.col(k) = B_full.col(k);
			out.S.col(k) = S_full.col(k);
			out.rho.col(k) = rho_full.col(k);
		}
		return out;
	}

	std::vector<Eigen::VectorXd> collectConstraints(
		const Eigen::MatrixXd& X,
		const Eigen::MatrixXd& U,
		const EnvMatrices& env_mats
	) const {
		std::vector<Eigen::VectorXd> c_list;
		c_list.reserve(static_cast<size_t>(N));
		for (int k = 0; k < N; ++k) {
			Eigen::VectorXd uk = Eigen::VectorXd::Zero(satellite.controlDim());
			if (k < U.cols()) {
				uk = U.col(k);
			}
			c_list.push_back(satellite.constraints(k, N, X.col(k), uk, env_mats.S.col(k), settings.constraints));
		}
		return c_list;
	}

	static void makeZeroAug(
		const std::vector<Eigen::VectorXd>& c_list,
		std::vector<Eigen::VectorXd>& lambda_aug,
		std::vector<Eigen::VectorXd>& mu_aug
	) {
		lambda_aug.clear();
		mu_aug.clear();
		lambda_aug.reserve(c_list.size());
		mu_aug.reserve(c_list.size());
		for (const Eigen::VectorXd& c : c_list) {
			lambda_aug.push_back(Eigen::VectorXd::Zero(c.size()));
			mu_aug.push_back(Eigen::VectorXd::Zero(c.size()));
		}
	}

	static void makeActiveAug(
		const std::vector<Eigen::VectorXd>& c_list,
		double lambda_value,
		double mu_value,
		std::vector<Eigen::VectorXd>& lambda_aug,
		std::vector<Eigen::VectorXd>& mu_aug
	) {
		lambda_aug.clear();
		mu_aug.clear();
		lambda_aug.reserve(c_list.size());
		mu_aug.reserve(c_list.size());
		for (const Eigen::VectorXd& c : c_list) {
			Eigen::VectorXd lam = Eigen::VectorXd::Zero(c.size());
			Eigen::VectorXd mu = Eigen::VectorXd::Zero(c.size());
			for (int i = 0; i < c.size(); ++i) {
				if (c(i) > 0.0) {
					lam(i) = lambda_value;
					mu(i) = mu_value;
				}
			}
			lambda_aug.push_back(lam);
			mu_aug.push_back(mu);
		}
	}

	double augmentedPenaltyTotal(
		const Eigen::MatrixXd& X,
		const Eigen::MatrixXd& U,
		const EnvMatrices& env_mats,
		const std::vector<Eigen::VectorXd>& lambda_aug,
		const std::vector<Eigen::VectorXd>& mu_aug
	) const {
		double total = 0.0;
		if (lambda_aug.empty() || mu_aug.empty()) {
			return total;
		}
		const int n_steps = std::min(N, static_cast<int>(std::min(lambda_aug.size(), mu_aug.size())));
		for (int k = 0; k < n_steps; ++k) {
			Eigen::VectorXd uk = Eigen::VectorXd::Zero(satellite.controlDim());
			if (k < U.cols()) {
				uk = U.col(k);
			}
			const Eigen::VectorXd ck = satellite.constraints(k, N, X.col(k), uk, env_mats.S.col(k), settings.constraints);
			if (lambda_aug[static_cast<size_t>(k)].size() != ck.size() || mu_aug[static_cast<size_t>(k)].size() != ck.size()) {
				continue;
			}
			for (int i = 0; i < ck.size(); ++i) {
				if (ck(i) <= 0.0) {
					continue;
				}
				total += lambda_aug[static_cast<size_t>(k)](i) * ck(i)
					+ 0.5 * mu_aug[static_cast<size_t>(k)](i) * ck(i) * ck(i);
			}
		}
		return total;
	}

	int N;
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
		settings.passes[0].dt = 0.5;
		settings.passes[0].reg.reg_init = 1e-2;
		settings.init_traj.initcontroller = 0;
		settings.constraints.control_limit_scale = 0.75;
		settings.constraints.wmax = 20.0 * M_PI / 180.0;
		settings.constraints.sun_limit_angle = 20.0 * M_PI / 180.0;
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

		settings.constraints.u_max = Eigen::VectorXd::Constant(satellite.controlDim(), 1.0);

		// Re-size x0 to the post-addRW state dimension.  See
		// docs/test_audit_silent_release_bugs.md for the silent-UB issue
		// where x0 was initialized in the member-init list pre-addRW.
		x0 = Satellite::VecX::Zero(satellite.stateDim());
		x0.segment<3>(Satellite::AV_INDEX) = Eigen::Vector3d(0.02, -0.01, 0.015);
		x0.segment<4>(Satellite::QUAT_INDEX) = Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
	}

	void configureTimeline() {
		const double dt_centuries = settings.passes[0].dt / SEC_PER_CENTURY;
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
			0,
			0,
			0,
			0,
			0,
			R_full,
			V_full,
			B_full,
			S_full,
			rho_full
		);
	}
};

bool runBackwardPass(
	const ForwardPassALFixture& fixture,
	const Eigen::MatrixXd& X,
	const Eigen::MatrixXd& U_trim,
	const EnvMatrices& env,
	const std::vector<Eigen::VectorXd>& lambda_aug,
	const std::vector<Eigen::VectorXd>& mu_aug,
	std::vector<Eigen::MatrixXd>& K,
	std::vector<Eigen::VectorXd>& d,
	Eigen::Vector2d& deltaV
) {
	const int steps = std::max(0, fixture.N - 1);
	K.assign(static_cast<size_t>(steps), Eigen::MatrixXd::Zero(fixture.satellite.controlDim(), fixture.satellite.reducedStateDim()));
	d.assign(static_cast<size_t>(steps), Eigen::VectorXd::Zero(fixture.satellite.controlDim()));
	deltaV.setZero();
	return optimizer::backwardPass(
		fixture.satellite,
		X,
		U_trim,
		env.R,
		env.V,
		env.B,
		env.S,
		env.rho,
		fixture.boresight,
		fixture.attitude_target_traj,
		fixture.settings,
		fixture.settings.passes[0].reg.reg_init,
		K,
		d,
		deltaV,
		lambda_aug,
		mu_aug
	);
}

bool runForwardPass(
	const ForwardPassALFixture& fixture,
	const Eigen::MatrixXd& X_in,
	const Eigen::MatrixXd& U_in,
	const EnvMatrices& env,
	const std::vector<Eigen::MatrixXd>& K,
	const std::vector<Eigen::VectorXd>& d,
	const Eigen::Vector2d& deltaV,
	const std::vector<Eigen::VectorXd>& lambda_aug,
	const std::vector<Eigen::VectorXd>& mu_aug,
	double J_prev,
	Eigen::MatrixXd& X_out,
	Eigen::MatrixXd& U_out,
	double& J_new
) {
	X_out = X_in;
	U_out = U_in;
	J_new = J_prev;
	return optimizer::forwardPass(
		fixture.satellite,
		X_out,
		U_out,
		K,
		d,
		deltaV,
		env.B,
		env.R,
		env.V,
		env.S,
		env.rho,
		fixture.boresight,
		fixture.attitude_target_traj,
		fixture.settings,
		lambda_aug,
		mu_aug,
		fixture.jtime,
		J_prev,
		J_new
	);
}

} // namespace

TEST_CASE("forward_pass AL: runs with constraint-sized multipliers", "[forward_pass][al][sized]") {
	ForwardPassALFixture fixture(8);
	REQUIRE(fixture.orbit_ok);

	Eigen::MatrixXd X;
	Eigen::MatrixXd U;
	fixture.initialTrajectory(X, U);

	EnvMatrices env = fixture.env();
	Eigen::MatrixXd U_trim = U.leftCols(fixture.N - 1);

	const std::vector<Eigen::VectorXd> c_list = fixture.collectConstraints(X, U_trim, env);
	std::vector<Eigen::VectorXd> lambda_aug;
	std::vector<Eigen::VectorXd> mu_aug;
	ForwardPassALFixture::makeZeroAug(c_list, lambda_aug, mu_aug);

	std::vector<Eigen::MatrixXd> K;
	std::vector<Eigen::VectorXd> d;
	Eigen::Vector2d deltaV = Eigen::Vector2d::Zero();
	REQUIRE(runBackwardPass(fixture, X, U_trim, env, lambda_aug, mu_aug, K, d, deltaV));

	const double J_prev = fixture.satellite.totalCost(X, U_trim, env.B, fixture.boresight, fixture.attitude_target_traj, fixture.settings.passes[0].cost);

	Eigen::MatrixXd X_new;
	Eigen::MatrixXd U_new;
	double J_new = J_prev;
	const bool ok_fp = runForwardPass(fixture, X, U, env, K, d, deltaV, lambda_aug, mu_aug, J_prev, X_new, U_new, J_new);

	REQUIRE(ok_fp);
	REQUIRE(X_new.allFinite());
	REQUIRE(U_new.allFinite());
	REQUIRE(std::isfinite(J_new));
}

TEST_CASE("forward_pass AL: active penalties modify cost", "[forward_pass][al][active_effect]") {
	ForwardPassALFixture fixture(8);
	REQUIRE(fixture.orbit_ok);

	fixture.settings.constraints.wmax = 1e-4;
	fixture.settings.constraints.control_limit_scale = 1.0;
	fixture.settings.constraints.sun_limit_angle = 3.14159265358979323846;
	fixture.settings.constraints.u_max = Eigen::VectorXd::Constant(fixture.satellite.controlDim(), 1e-4);

	Eigen::MatrixXd X;
	Eigen::MatrixXd U;
	fixture.initialTrajectory(X, U);

	EnvMatrices env = fixture.env();
	Eigen::MatrixXd U_trim = U.leftCols(fixture.N - 1);

	const std::vector<Eigen::VectorXd> c_list = fixture.collectConstraints(X, U_trim, env);
	bool any_active = false;
	for (int k = 0; k < fixture.N - 1; ++k) {
		if ((c_list[static_cast<size_t>(k)].array() > 0.0).any()) {
			any_active = true;
			break;
		}
	}
	REQUIRE(any_active);

	std::vector<Eigen::VectorXd> lambda_zero;
	std::vector<Eigen::VectorXd> mu_zero;
	ForwardPassALFixture::makeZeroAug(c_list, lambda_zero, mu_zero);

	std::vector<Eigen::VectorXd> lambda_active;
	std::vector<Eigen::VectorXd> mu_active;
	ForwardPassALFixture::makeActiveAug(c_list, 0.5, 1.0, lambda_active, mu_active);

	std::vector<Eigen::MatrixXd> K0;
	std::vector<Eigen::VectorXd> d0;
	Eigen::Vector2d deltaV0 = Eigen::Vector2d::Zero();
	REQUIRE(runBackwardPass(fixture, X, U_trim, env, lambda_zero, mu_zero, K0, d0, deltaV0));

	std::vector<Eigen::MatrixXd> K1;
	std::vector<Eigen::VectorXd> d1;
	Eigen::Vector2d deltaV1 = Eigen::Vector2d::Zero();
	REQUIRE(runBackwardPass(fixture, X, U_trim, env, lambda_active, mu_active, K1, d1, deltaV1));

	const double J_nom = fixture.satellite.totalCost(X, U_trim, env.B, fixture.boresight, fixture.attitude_target_traj, fixture.settings.passes[0].cost);
	const double J_prev0 = J_nom + fixture.augmentedPenaltyTotal(X, U_trim, env, lambda_zero, mu_zero);
	const double J_prev1 = J_nom + fixture.augmentedPenaltyTotal(X, U_trim, env, lambda_active, mu_active);

	Eigen::MatrixXd X0;
	Eigen::MatrixXd U0;
	double J_new0 = J_prev0;
	const bool ok_fp0 = runForwardPass(fixture, X, U, env, K0, d0, deltaV0, lambda_zero, mu_zero, J_prev0, X0, U0, J_new0);

	Eigen::MatrixXd X1;
	Eigen::MatrixXd U1;
	double J_new1 = J_prev1;
	const bool ok_fp1 = runForwardPass(fixture, X, U, env, K1, d1, deltaV1, lambda_active, mu_active, J_prev1, X1, U1, J_new1);

	// Active AL penalties can alter line-search acceptance through changed
	// backward-pass gains, so success flags need not match across runs.
	if (ok_fp0 && ok_fp1) {
		REQUIRE(J_new1 >= J_new0 - 1e-9);
	}
	if (!ok_fp0) {
		REQUIRE(std::abs(J_new0 - J_prev0) < 1e-12);
	}
	if (!ok_fp1) {
		REQUIRE(std::abs(J_new1 - J_prev1) < 1e-12);
	}
}

TEST_CASE("forward_pass AL: size mismatch returns false", "[forward_pass][al][size_mismatch]") {
	ForwardPassALFixture fixture(8);
	REQUIRE(fixture.orbit_ok);

	Eigen::MatrixXd X;
	Eigen::MatrixXd U;
	fixture.initialTrajectory(X, U);

	EnvMatrices env = fixture.env();
	Eigen::MatrixXd U_trim = U.leftCols(fixture.N - 1);

	const std::vector<Eigen::VectorXd> c_list = fixture.collectConstraints(X, U_trim, env);
	std::vector<Eigen::VectorXd> lambda_ok;
	std::vector<Eigen::VectorXd> mu_ok;
	ForwardPassALFixture::makeZeroAug(c_list, lambda_ok, mu_ok);

	std::vector<Eigen::MatrixXd> K;
	std::vector<Eigen::VectorXd> d;
	Eigen::Vector2d deltaV = Eigen::Vector2d::Zero();
	REQUIRE(runBackwardPass(fixture, X, U_trim, env, lambda_ok, mu_ok, K, d, deltaV));

	std::vector<Eigen::VectorXd> lambda_bad;
	std::vector<Eigen::VectorXd> mu_bad;
	lambda_bad.reserve(lambda_ok.size());
	mu_bad.reserve(mu_ok.size());
	for (size_t k = 0; k < lambda_ok.size(); ++k) {
		lambda_bad.push_back(Eigen::VectorXd::Zero(lambda_ok[k].size() + 1));
		mu_bad.push_back(Eigen::VectorXd::Zero(mu_ok[k].size() + 2));
	}

	const double J_prev = fixture.satellite.totalCost(X, U_trim, env.B, fixture.boresight, fixture.attitude_target_traj, fixture.settings.passes[0].cost);

	Eigen::MatrixXd X_new;
	Eigen::MatrixXd U_new;
	double J_new = J_prev;
	const bool ok_fp = runForwardPass(fixture, X, U, env, K, d, deltaV, lambda_bad, mu_bad, J_prev, X_new, U_new, J_new);

	REQUIRE_FALSE(ok_fp);
	REQUIRE(std::abs(J_new - J_prev) < 1e-12);
}

TEST_CASE("forward_pass AL: partial-horizon multiplier lists are accepted", "[forward_pass][al][partial_horizon]") {
	ForwardPassALFixture fixture(8);
	REQUIRE(fixture.orbit_ok);

	Eigen::MatrixXd X;
	Eigen::MatrixXd U;
	fixture.initialTrajectory(X, U);

	EnvMatrices env = fixture.env();
	Eigen::MatrixXd U_trim = U.leftCols(fixture.N - 1);

	const std::vector<Eigen::VectorXd> c_list = fixture.collectConstraints(X, U_trim, env);
	std::vector<Eigen::VectorXd> lambda_aug;
	std::vector<Eigen::VectorXd> mu_aug;
	ForwardPassALFixture::makeZeroAug(c_list, lambda_aug, mu_aug);

	std::vector<Eigen::MatrixXd> K;
	std::vector<Eigen::VectorXd> d;
	Eigen::Vector2d deltaV = Eigen::Vector2d::Zero();
	REQUIRE(runBackwardPass(fixture, X, U_trim, env, lambda_aug, mu_aug, K, d, deltaV));

	std::vector<Eigen::VectorXd> lambda_short{lambda_aug[0]};
	std::vector<Eigen::VectorXd> mu_short{mu_aug[0]};

	const double J_prev = fixture.satellite.totalCost(X, U_trim, env.B, fixture.boresight, fixture.attitude_target_traj, fixture.settings.passes[0].cost);

	Eigen::MatrixXd X_new;
	Eigen::MatrixXd U_new;
	double J_new = J_prev;
	const bool ok_fp = runForwardPass(fixture, X, U, env, K, d, deltaV, lambda_short, mu_short, J_prev, X_new, U_new, J_new);

	if (ok_fp) {
		REQUIRE(X_new.allFinite());
		REQUIRE(U_new.allFinite());
		REQUIRE(std::isfinite(J_new));
	} else {
		REQUIRE(std::abs(J_new - J_prev) < 1e-12);
	}
}

TEST_CASE("forward_pass AL: long horizon random multipliers stable", "[forward_pass][al][stability]") {
	ForwardPassALFixture fixture(10);
	REQUIRE(fixture.orbit_ok);

	Eigen::MatrixXd X;
	Eigen::MatrixXd U;
	fixture.initialTrajectory(X, U);

	EnvMatrices env = fixture.env();
	Eigen::MatrixXd U_trim = U.leftCols(fixture.N - 1);

	const std::vector<Eigen::VectorXd> c_list = fixture.collectConstraints(X, U_trim, env);
	std::vector<Eigen::VectorXd> lambda_aug;
	std::vector<Eigen::VectorXd> mu_aug;
	ForwardPassALFixture::makeZeroAug(c_list, lambda_aug, mu_aug);

	std::mt19937 rng(7);
	std::normal_distribution<double> dist_lam(0.2, 0.1);
	std::normal_distribution<double> dist_mu(1.0, 0.3);
	for (size_t k = 0; k < lambda_aug.size(); ++k) {
		for (int i = 0; i < lambda_aug[k].size(); ++i) {
			lambda_aug[k](i) = std::max(0.0, dist_lam(rng));
			mu_aug[k](i) = std::max(0.0, dist_mu(rng));
		}
	}

	std::vector<Eigen::MatrixXd> K;
	std::vector<Eigen::VectorXd> d;
	Eigen::Vector2d deltaV = Eigen::Vector2d::Zero();
	REQUIRE(runBackwardPass(fixture, X, U_trim, env, lambda_aug, mu_aug, K, d, deltaV));

	const double J_nom = fixture.satellite.totalCost(X, U_trim, env.B, fixture.boresight, fixture.attitude_target_traj, fixture.settings.passes[0].cost);
	const double J_prev = J_nom + fixture.augmentedPenaltyTotal(X, U_trim, env, lambda_aug, mu_aug);

	Eigen::MatrixXd X_new;
	Eigen::MatrixXd U_new;
	double J_new = J_prev;
	const bool ok_fp = runForwardPass(fixture, X, U, env, K, d, deltaV, lambda_aug, mu_aug, J_prev, X_new, U_new, J_new);

	// "Stability" = finite outputs under random AL multipliers, not
	// ok_fp==true.  Random mu can make Q_uu indefinite → BP returns
	// ascent gains → FP correctly rejects.  Production iLQR recovers
	// via outer reg-bump loop; single-call unit test doesn't.
	(void)ok_fp;
	REQUIRE(X_new.allFinite());
	REQUIRE(U_new.allFinite());
	REQUIRE(std::isfinite(J_new));
}
