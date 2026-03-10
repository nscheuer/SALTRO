#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include <Eigen/Dense>

#include <saltro/optimizer/backwardpass.h>
#include <saltro/pybind/satellite.h>

using namespace saltro;

namespace {

Eigen::MatrixXd makeAttitudeTraj(const Eigen::Vector4d& att, int n_cols) {
	Eigen::MatrixXd traj(4, n_cols);
	for (int k = 0; k < n_cols; ++k) {
		traj.col(k) = att;
	}
	return traj;
}

struct TrajectoryData {
	Eigen::MatrixXd X;
	Eigen::MatrixXd U;
	Eigen::MatrixXd R;
	Eigen::MatrixXd V;
	Eigen::MatrixXd B;
	Eigen::MatrixXd S;
	Eigen::MatrixXd rho;
	Eigen::MatrixXd boresight;
	Eigen::MatrixXd attitude_target;
};

class BackwardPassALFixture {
public:
	explicit BackwardPassALFixture(int n = 6)
		: N(n),
		  settings(),
		  satellite(makeInertia(), settings),
		  reg(1e-8) {
		configureSatellite();
		configureSettings();
	}

	TrajectoryData makeTrajectory(bool force_active) {
		const int nx = satellite.stateDim();
		const int nu = satellite.controlDim();

		TrajectoryData td;
		td.X = Eigen::MatrixXd::Zero(nx, N);
		td.U = Eigen::MatrixXd::Zero(nu, std::max(0, N - 1));
		td.R = Eigen::MatrixXd::Zero(3, N);
		td.V = Eigen::MatrixXd::Zero(3, N);
		td.B = Eigen::MatrixXd::Zero(3, N);
		td.S = Eigen::MatrixXd::Zero(3, N);
		td.rho = Eigen::MatrixXd::Zero(1, N);
		td.boresight = Eigen::MatrixXd::Zero(3, N);

		Satellite::VecX x0 = Satellite::VecX::Zero(nx);
		x0.segment<3>(Satellite::AV_INDEX) = Eigen::Vector3d(0.01, -0.005, 0.008);
		x0.segment<4>(Satellite::QUAT_INDEX) = Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);

		for (int k = 0; k < N; ++k) {
			td.X.col(k) = x0;
			if (force_active && k < N - 1) {
				td.U.col(k) << 0.15, -0.12, 0.10, 0.01, -0.01, 0.01;
			}

			td.R.col(k) = Eigen::Vector3d(7000e3, 0.0, 0.0);
			td.V.col(k) = Eigen::Vector3d(0.0, 7500.0, 0.0);
			td.B.col(k) = Eigen::Vector3d(2.5e-5, -1.5e-5, 3.0e-5);
			td.S.col(k) = Eigen::Vector3d(1.0, 0.1, -0.05).normalized();
			td.boresight.col(k) = Eigen::Vector3d::UnitX();
		}

		Eigen::Vector4d att;
		att << std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0;
		td.attitude_target = makeAttitudeTraj(att, N);

		if (force_active) {
			settings.constraints.wmax = 1e-4;
			settings.constraints.control_limit_scale = 1.0;
			settings.constraints.sun_limit_angle = 3.14159265358979323846;
			settings.constraints.u_max = Eigen::VectorXd::Constant(satellite.controlDim(), 1e-4);
		}

		return td;
	}

	std::vector<Eigen::VectorXd> collectConstraints(const TrajectoryData& td) const {
		std::vector<Eigen::VectorXd> c_list;
		c_list.reserve(static_cast<size_t>(N));

		for (int k = 0; k < N; ++k) {
			Eigen::VectorXd uk = Eigen::VectorXd::Zero(satellite.controlDim());
			if (k < td.U.cols()) {
				uk = td.U.col(k);
			}
			c_list.push_back(satellite.constraints(k, N, td.X.col(k), uk, td.S.col(k), settings.constraints));
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

	bool runBackward(
		const TrajectoryData& td,
		const std::vector<Eigen::VectorXd>& lambda_aug,
		const std::vector<Eigen::VectorXd>& mu_aug,
		std::vector<Eigen::MatrixXd>& K,
		std::vector<Eigen::VectorXd>& d,
		Eigen::Vector2d& deltaV
	) const {
		initializeGains(K, d);
		deltaV.setZero();
		return optimizer::backwardPass(
			satellite,
			td.X,
			td.U,
			td.R,
			td.V,
			td.B,
			td.S,
			td.rho,
			td.boresight,
			td.attitude_target,
			settings,
			reg,
			K,
			d,
			deltaV,
			lambda_aug,
			mu_aug
		);
	}

	void initializeGains(std::vector<Eigen::MatrixXd>& K, std::vector<Eigen::VectorXd>& d) const {
		const int steps = std::max(0, N - 1);
		K.assign(static_cast<size_t>(steps), Eigen::MatrixXd::Zero(satellite.controlDim(), satellite.reducedStateDim()));
		d.assign(static_cast<size_t>(steps), Eigen::VectorXd::Zero(satellite.controlDim()));
	}

	int N;
	PlannerSettings settings;
	Satellite satellite;
	double reg;

private:
	static Eigen::Matrix3d makeInertia() {
		Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
		J(0, 0) = 0.067;
		J(1, 1) = 0.071;
		J(2, 2) = 0.069;
		return J;
	}

	void configureSatellite() {
		satellite.addMTQ(Eigen::Vector3d::UnitX(), 0.2);
		satellite.addMTQ(Eigen::Vector3d::UnitY(), 0.2);
		satellite.addMTQ(Eigen::Vector3d::UnitZ(), 0.2);
		satellite.addRW(Eigen::Vector3d::UnitX(), 0.001, 1e-5, 0.0, 0.02);
		satellite.addRW(Eigen::Vector3d::UnitY(), 0.001, 1e-5, 0.0, 0.02);
		satellite.addRW(Eigen::Vector3d::UnitZ(), 0.001, 1e-5, 0.0, 0.02);
	}

	void configureSettings() {
		settings.disturbances.plan_for_aero = false;
		settings.disturbances.plan_for_gg = false;
		settings.disturbances.plan_for_srp = false;
		settings.disturbances.plan_for_prop = false;
		settings.disturbances.plan_for_gendist = false;
		settings.disturbances.plan_for_resdipole = false;
		settings.num_passes = 1;
		settings.passes[0].dt = 0.5;
		settings.passes[0].reg.reg_init = reg;
		settings.passes[0].reg.reg_scale = 10.0;
		settings.passes[0].reg.reg_max = 1e4;
	}
};

double gainDeltaNorm(const std::vector<Eigen::MatrixXd>& A, const std::vector<Eigen::MatrixXd>& B) {
	double acc = 0.0;
	const size_t n = std::min(A.size(), B.size());
	for (size_t i = 0; i < n; ++i) {
		acc += (A[i] - B[i]).squaredNorm();
	}
	return std::sqrt(acc);
}

double feedforwardDeltaNorm(const std::vector<Eigen::VectorXd>& A, const std::vector<Eigen::VectorXd>& B) {
	double acc = 0.0;
	const size_t n = std::min(A.size(), B.size());
	for (size_t i = 0; i < n; ++i) {
		acc += (A[i] - B[i]).squaredNorm();
	}
	return std::sqrt(acc);
}

} // namespace

TEST_CASE("backward_pass AL: runs with constraint-sized multipliers", "[backward_pass][al][sized]") {
	BackwardPassALFixture fixture(6);
	TrajectoryData td = fixture.makeTrajectory(false);
	const std::vector<Eigen::VectorXd> c_list = fixture.collectConstraints(td);

	std::vector<Eigen::VectorXd> lambda_aug;
	std::vector<Eigen::VectorXd> mu_aug;
	BackwardPassALFixture::makeZeroAug(c_list, lambda_aug, mu_aug);

	std::vector<Eigen::MatrixXd> K;
	std::vector<Eigen::VectorXd> d;
	Eigen::Vector2d deltaV = Eigen::Vector2d::Zero();

	const bool ok = fixture.runBackward(td, lambda_aug, mu_aug, K, d, deltaV);

	REQUIRE(ok);
	REQUIRE(K.size() == static_cast<size_t>(fixture.N - 1));
	REQUIRE(d.size() == static_cast<size_t>(fixture.N - 1));
	for (size_t k = 0; k < K.size(); ++k) {
		REQUIRE(K[k].allFinite());
		REQUIRE(d[k].allFinite());
	}
	REQUIRE(deltaV.allFinite());
}

TEST_CASE("backward_pass AL: active penalties modify gains/feedforward", "[backward_pass][al][active_effect]") {
	BackwardPassALFixture fixture(6);
	TrajectoryData td = fixture.makeTrajectory(true);
	const std::vector<Eigen::VectorXd> c_list = fixture.collectConstraints(td);

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
	BackwardPassALFixture::makeZeroAug(c_list, lambda_zero, mu_zero);

	std::vector<Eigen::VectorXd> lambda_aug;
	std::vector<Eigen::VectorXd> mu_aug;
	lambda_aug.reserve(c_list.size());
	mu_aug.reserve(c_list.size());
	for (const Eigen::VectorXd& c : c_list) {
		Eigen::VectorXd lam = Eigen::VectorXd::Zero(c.size());
		Eigen::VectorXd mu = Eigen::VectorXd::Zero(c.size());
		for (int i = 0; i < c.size(); ++i) {
			if (c(i) > 0.0) {
				lam(i) = 5.0;
				mu(i) = 25.0;
			}
		}
		lambda_aug.push_back(lam);
		mu_aug.push_back(mu);
	}

	std::vector<Eigen::MatrixXd> K0;
	std::vector<Eigen::VectorXd> d0;
	Eigen::Vector2d deltaV0 = Eigen::Vector2d::Zero();
	const bool ok0 = fixture.runBackward(td, lambda_zero, mu_zero, K0, d0, deltaV0);

	std::vector<Eigen::MatrixXd> K1;
	std::vector<Eigen::VectorXd> d1;
	Eigen::Vector2d deltaV1 = Eigen::Vector2d::Zero();
	const bool ok1 = fixture.runBackward(td, lambda_aug, mu_aug, K1, d1, deltaV1);

	REQUIRE(ok0);
	REQUIRE(ok1);
	const double k_diff = gainDeltaNorm(K1, K0);
	const double d_diff = feedforwardDeltaNorm(d1, d0);
	const double dv_diff = (deltaV1 - deltaV0).norm();
	REQUIRE((k_diff > 1e-12 || d_diff > 1e-12));
	REQUIRE(dv_diff > 1e-12);
}

TEST_CASE("backward_pass AL: size mismatch lists are ignored", "[backward_pass][al][size_mismatch]") {
	BackwardPassALFixture fixture(6);
	TrajectoryData td = fixture.makeTrajectory(true);
	const std::vector<Eigen::VectorXd> c_list = fixture.collectConstraints(td);

	std::vector<Eigen::VectorXd> lambda_zero;
	std::vector<Eigen::VectorXd> mu_zero;
	BackwardPassALFixture::makeZeroAug(c_list, lambda_zero, mu_zero);

	std::vector<Eigen::VectorXd> lambda_bad;
	std::vector<Eigen::VectorXd> mu_bad;
	lambda_bad.reserve(c_list.size());
	mu_bad.reserve(c_list.size());
	for (const Eigen::VectorXd& c : c_list) {
		lambda_bad.push_back(Eigen::VectorXd::Zero(c.size() + 1));
		mu_bad.push_back(Eigen::VectorXd::Ones(c.size() + 2));
	}

	std::vector<Eigen::MatrixXd> K0;
	std::vector<Eigen::VectorXd> d0;
	Eigen::Vector2d deltaV0 = Eigen::Vector2d::Zero();
	const bool ok0 = fixture.runBackward(td, lambda_zero, mu_zero, K0, d0, deltaV0);

	std::vector<Eigen::MatrixXd> Kb;
	std::vector<Eigen::VectorXd> db;
	Eigen::Vector2d deltaVb = Eigen::Vector2d::Zero();
	const bool okb = fixture.runBackward(td, lambda_bad, mu_bad, Kb, db, deltaVb);

	REQUIRE(ok0);
	REQUIRE(okb);
	REQUIRE(gainDeltaNorm(Kb, K0) < 1e-12);
	REQUIRE(feedforwardDeltaNorm(db, d0) < 1e-12);
	REQUIRE((deltaVb - deltaV0).norm() < 1e-12);
}

TEST_CASE("backward_pass AL: partial-horizon multiplier lists are accepted", "[backward_pass][al][partial_horizon]") {
	BackwardPassALFixture fixture(6);
	TrajectoryData td = fixture.makeTrajectory(true);
	const std::vector<Eigen::VectorXd> c_list = fixture.collectConstraints(td);

	std::vector<Eigen::VectorXd> lambda_short;
	std::vector<Eigen::VectorXd> mu_short;
	lambda_short.reserve(1);
	mu_short.reserve(1);

	Eigen::VectorXd lam = Eigen::VectorXd::Zero(c_list[0].size());
	Eigen::VectorXd mu = Eigen::VectorXd::Zero(c_list[0].size());
	for (int i = 0; i < c_list[0].size(); ++i) {
		if (c_list[0](i) > 0.0) {
			lam(i) = 2.0;
			mu(i) = 10.0;
		}
	}
	lambda_short.push_back(lam);
	mu_short.push_back(mu);

	std::vector<Eigen::MatrixXd> K;
	std::vector<Eigen::VectorXd> d;
	Eigen::Vector2d deltaV = Eigen::Vector2d::Zero();

	const bool ok = fixture.runBackward(td, lambda_short, mu_short, K, d, deltaV);

	REQUIRE(ok);
	for (size_t k = 0; k < K.size(); ++k) {
		REQUIRE(K[k].allFinite());
		REQUIRE(d[k].allFinite());
	}
	REQUIRE(deltaV.allFinite());
}

TEST_CASE("backward_pass AL: longer horizon random nonnegative multipliers remain stable", "[backward_pass][al][stability]") {
	BackwardPassALFixture fixture(10);
	TrajectoryData td = fixture.makeTrajectory(true);
	const std::vector<Eigen::VectorXd> c_list = fixture.collectConstraints(td);

	std::mt19937 rng(7);
	std::normal_distribution<double> dist_lam(0.2, 0.1);
	std::normal_distribution<double> dist_mu(5.0, 2.0);

	std::vector<Eigen::VectorXd> lambda_aug;
	std::vector<Eigen::VectorXd> mu_aug;
	lambda_aug.reserve(c_list.size());
	mu_aug.reserve(c_list.size());

	for (const Eigen::VectorXd& c : c_list) {
		Eigen::VectorXd lam(c.size());
		Eigen::VectorXd mu(c.size());
		for (int i = 0; i < c.size(); ++i) {
			lam(i) = std::max(0.0, dist_lam(rng));
			mu(i) = std::max(0.0, dist_mu(rng));
		}
		lambda_aug.push_back(lam);
		mu_aug.push_back(mu);
	}

	std::vector<Eigen::MatrixXd> K;
	std::vector<Eigen::VectorXd> d;
	Eigen::Vector2d deltaV = Eigen::Vector2d::Zero();

	const bool ok = fixture.runBackward(td, lambda_aug, mu_aug, K, d, deltaV);

	REQUIRE(ok);
	REQUIRE(K.size() == static_cast<size_t>(fixture.N - 1));
	REQUIRE(d.size() == static_cast<size_t>(fixture.N - 1));
	for (size_t k = 0; k < K.size(); ++k) {
		REQUIRE(K[k].allFinite());
		REQUIRE(d[k].allFinite());
	}
	REQUIRE(deltaV.allFinite());
}
