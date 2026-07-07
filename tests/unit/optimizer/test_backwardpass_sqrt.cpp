#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
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

class BackwardPassSqrtFixture {
public:
	explicit BackwardPassSqrtFixture(int n = 6)
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
			// Vary the state slightly along the trajectory so gains differ
			// between timesteps.
			td.X.col(k).segment<3>(Satellite::AV_INDEX) += 0.001 * k * Eigen::Vector3d(1.0, -0.5, 0.25);
			if (k < N - 1) {
				if (force_active) {
					td.U.col(k) << 0.15, -0.12, 0.10, 0.01, -0.01, 0.01;
				} else {
					td.U.col(k) = 0.001 * Eigen::VectorXd::Ones(nu) * (k + 1);
				}
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

	static void makeConstAug(
		const std::vector<Eigen::VectorXd>& c_list,
		double lambda_val,
		double mu_val,
		std::vector<Eigen::VectorXd>& lambda_aug,
		std::vector<Eigen::VectorXd>& mu_aug
	) {
		lambda_aug.clear();
		mu_aug.clear();
		lambda_aug.reserve(c_list.size());
		mu_aug.reserve(c_list.size());
		for (const Eigen::VectorXd& c : c_list) {
			lambda_aug.push_back(Eigen::VectorXd::Constant(c.size(), lambda_val));
			mu_aug.push_back(Eigen::VectorXd::Constant(c.size(), mu_val));
		}
	}

	bool runDense(
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
			satellite, td.X, td.U, td.R, td.V, td.B, td.S, td.rho,
			td.boresight, td.attitude_target, settings, reg,
			K, d, deltaV, lambda_aug, mu_aug
		);
	}

	bool runSqrt(
		const TrajectoryData& td,
		const std::vector<Eigen::VectorXd>& lambda_aug,
		const std::vector<Eigen::VectorXd>& mu_aug,
		std::vector<Eigen::MatrixXd>& K,
		std::vector<Eigen::VectorXd>& d,
		Eigen::Vector2d& deltaV
	) const {
		initializeGains(K, d);
		deltaV.setZero();
		return optimizer::backwardPassSqrt(
			satellite, td.X, td.U, td.R, td.V, td.B, td.S, td.rho,
			td.boresight, td.attitude_target, settings, reg,
			K, d, deltaV, lambda_aug, mu_aug
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

double relGainDelta(const std::vector<Eigen::MatrixXd>& A, const std::vector<Eigen::MatrixXd>& B) {
	double num = 0.0;
	double den = 0.0;
	const size_t n = std::min(A.size(), B.size());
	for (size_t i = 0; i < n; ++i) {
		num += (A[i] - B[i]).squaredNorm();
		den += A[i].squaredNorm();
	}
	return std::sqrt(num) / std::max(1e-300, std::sqrt(den));
}

double relFeedforwardDelta(const std::vector<Eigen::VectorXd>& A, const std::vector<Eigen::VectorXd>& B) {
	double num = 0.0;
	double den = 0.0;
	const size_t n = std::min(A.size(), B.size());
	for (size_t i = 0; i < n; ++i) {
		num += (A[i] - B[i]).squaredNorm();
		den += A[i].squaredNorm();
	}
	return std::sqrt(num) / std::max(1e-300, std::sqrt(den));
}

} // namespace

TEST_CASE("backward_pass sqrt: matches dense pass without constraints", "[backward_pass][sqrt][parity]") {
	BackwardPassSqrtFixture fixture(6);
	TrajectoryData td = fixture.makeTrajectory(false);

	const std::vector<Eigen::VectorXd> empty_aug;

	std::vector<Eigen::MatrixXd> K_dense, K_sqrt;
	std::vector<Eigen::VectorXd> d_dense, d_sqrt;
	Eigen::Vector2d dV_dense = Eigen::Vector2d::Zero();
	Eigen::Vector2d dV_sqrt = Eigen::Vector2d::Zero();

	REQUIRE(fixture.runDense(td, empty_aug, empty_aug, K_dense, d_dense, dV_dense));
	REQUIRE(fixture.runSqrt(td, empty_aug, empty_aug, K_sqrt, d_sqrt, dV_sqrt));

	REQUIRE(K_sqrt.size() == K_dense.size());
	REQUIRE(relGainDelta(K_dense, K_sqrt) < 1e-8);
	REQUIRE(relFeedforwardDelta(d_dense, d_sqrt) < 1e-8);
	REQUIRE(std::abs(dV_dense(0) - dV_sqrt(0)) <= 1e-8 * std::max(1.0, std::abs(dV_dense(0))));
	REQUIRE(std::abs(dV_dense(1) - dV_sqrt(1)) <= 1e-8 * std::max(1.0, std::abs(dV_dense(1))));
}

TEST_CASE("backward_pass sqrt: matches dense pass with active AL constraints", "[backward_pass][sqrt][al][parity]") {
	BackwardPassSqrtFixture fixture(6);
	TrajectoryData td = fixture.makeTrajectory(true);
	const std::vector<Eigen::VectorXd> c_list = fixture.collectConstraints(td);

	std::vector<Eigen::VectorXd> lambda_aug;
	std::vector<Eigen::VectorXd> mu_aug;
	BackwardPassSqrtFixture::makeConstAug(c_list, 1.0, 100.0, lambda_aug, mu_aug);

	std::vector<Eigen::MatrixXd> K_dense, K_sqrt;
	std::vector<Eigen::VectorXd> d_dense, d_sqrt;
	Eigen::Vector2d dV_dense = Eigen::Vector2d::Zero();
	Eigen::Vector2d dV_sqrt = Eigen::Vector2d::Zero();

	REQUIRE(fixture.runDense(td, lambda_aug, mu_aug, K_dense, d_dense, dV_dense));
	REQUIRE(fixture.runSqrt(td, lambda_aug, mu_aug, K_sqrt, d_sqrt, dV_sqrt));

	REQUIRE(relGainDelta(K_dense, K_sqrt) < 1e-8);
	REQUIRE(relFeedforwardDelta(d_dense, d_sqrt) < 1e-8);
	REQUIRE(std::abs(dV_dense(0) - dV_sqrt(0)) <= 1e-8 * std::max(1.0, std::abs(dV_dense(0))));
	REQUIRE(std::abs(dV_dense(1) - dV_sqrt(1)) <= 1e-8 * std::max(1.0, std::abs(dV_dense(1))));
}

TEST_CASE("backward_pass sqrt: stays finite with extreme AL penalties", "[backward_pass][sqrt][al][stress]") {
	BackwardPassSqrtFixture fixture(6);
	TrajectoryData td = fixture.makeTrajectory(true);
	const std::vector<Eigen::VectorXd> c_list = fixture.collectConstraints(td);

	// Penalty weights far past the point where forming mu * c^T c outer
	// products loses half the available precision.
	for (const double mu_val : {1e8, 1e12, 1e16}) {
		std::vector<Eigen::VectorXd> lambda_aug;
		std::vector<Eigen::VectorXd> mu_aug;
		BackwardPassSqrtFixture::makeConstAug(c_list, 10.0, mu_val, lambda_aug, mu_aug);

		std::vector<Eigen::MatrixXd> K_sqrt;
		std::vector<Eigen::VectorXd> d_sqrt;
		Eigen::Vector2d dV_sqrt = Eigen::Vector2d::Zero();

		const bool ok = fixture.runSqrt(td, lambda_aug, mu_aug, K_sqrt, d_sqrt, dV_sqrt);

		REQUIRE(ok);
		REQUIRE(dV_sqrt.allFinite());
		for (size_t k = 0; k < K_sqrt.size(); ++k) {
			REQUIRE(K_sqrt[k].allFinite());
			REQUIRE(d_sqrt[k].allFinite());
		}
	}
}

TEST_CASE("backward_pass sqrt: use_sqrt_bp flag routes backwardPass to the sqrt pass", "[backward_pass][sqrt][dispatch]") {
	BackwardPassSqrtFixture fixture(6);
	TrajectoryData td = fixture.makeTrajectory(false);

	const std::vector<Eigen::VectorXd> empty_aug;

	std::vector<Eigen::MatrixXd> K_sqrt, K_flag;
	std::vector<Eigen::VectorXd> d_sqrt, d_flag;
	Eigen::Vector2d dV_sqrt = Eigen::Vector2d::Zero();
	Eigen::Vector2d dV_flag = Eigen::Vector2d::Zero();

	REQUIRE(fixture.runSqrt(td, empty_aug, empty_aug, K_sqrt, d_sqrt, dV_sqrt));

	fixture.settings.passes[0].reg.use_sqrt_bp = true;
	REQUIRE(fixture.runDense(td, empty_aug, empty_aug, K_flag, d_flag, dV_flag));

	// The flagged dense entry point must produce bit-identical results to
	// calling backwardPassSqrt directly.
	REQUIRE(relGainDelta(K_sqrt, K_flag) == 0.0);
	REQUIRE(relFeedforwardDelta(d_sqrt, d_flag) == 0.0);
	REQUIRE(dV_sqrt == dV_flag);
}

TEST_CASE("backward_pass sqrt: DDP flags force the dense path", "[backward_pass][sqrt][dispatch][ddp]") {
	// backwardPassSqrt has no DDP second-order path. When use_sqrt_bp is set
	// together with use_dynamics_hess/use_constraint_hess (possible for
	// callers that skip validatePlannerSettings, e.g. the direct python
	// bindings), backwardPass must fall back to the dense DDP pass instead of
	// silently dropping the curvature terms.
	BackwardPassSqrtFixture fixture(6);
	TrajectoryData td = fixture.makeTrajectory(true);
	const std::vector<Eigen::VectorXd> c_list = fixture.collectConstraints(td);

	std::vector<Eigen::VectorXd> lambda_aug;
	std::vector<Eigen::VectorXd> mu_aug;
	BackwardPassSqrtFixture::makeConstAug(c_list, 1.0, 100.0, lambda_aug, mu_aug);

	// Baseline: dense pass with DDP second-order terms enabled.
	fixture.settings.passes[0].reg.use_sqrt_bp = false;
	fixture.settings.passes[0].reg.use_dynamics_hess = true;
	fixture.settings.passes[0].reg.use_constraint_hess = true;

	std::vector<Eigen::MatrixXd> K_dense, K_flag;
	std::vector<Eigen::VectorXd> d_dense, d_flag;
	Eigen::Vector2d dV_dense = Eigen::Vector2d::Zero();
	Eigen::Vector2d dV_flag = Eigen::Vector2d::Zero();

	REQUIRE(fixture.runDense(td, lambda_aug, mu_aug, K_dense, d_dense, dV_dense));

	// Same settings plus use_sqrt_bp: the dispatch must take the dense path
	// and produce bit-identical results.
	fixture.settings.passes[0].reg.use_sqrt_bp = true;
	REQUIRE(fixture.runDense(td, lambda_aug, mu_aug, K_flag, d_flag, dV_flag));

	REQUIRE(relGainDelta(K_dense, K_flag) == 0.0);
	REQUIRE(relFeedforwardDelta(d_dense, d_flag) == 0.0);
	REQUIRE(dV_dense == dV_flag);
}

TEST_CASE("backward_pass sqrt: N=1 edge case", "[backward_pass][sqrt][n1_edge_case]") {
	BackwardPassSqrtFixture fixture(1);
	TrajectoryData td = fixture.makeTrajectory(false);

	const std::vector<Eigen::VectorXd> empty_aug;

	std::vector<Eigen::MatrixXd> K;
	std::vector<Eigen::VectorXd> d;
	Eigen::Vector2d deltaV = Eigen::Vector2d::Zero();

	REQUIRE(fixture.runSqrt(td, empty_aug, empty_aug, K, d, deltaV));
	REQUIRE(K.size() == 0);
	REQUIRE(d.size() == 0);
}
