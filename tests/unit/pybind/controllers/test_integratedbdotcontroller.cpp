
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>

#include <Eigen/Dense>

#include <saltro/limits.h>
#include <saltro/optimizer/warm_start.h>
#include <saltro/pybind/satellite.h>

using namespace saltro;

namespace {

constexpr double PI = 3.14159265358979323846;
constexpr double DEG2RAD = PI / 180.0;
constexpr double SEC_PER_CENTURY = 36525.0 * 86400.0;

constexpr int NUM_SAMPLES = 5;
constexpr double DT_SECONDS = 10.0;
constexpr double SIM_SECONDS = 1000.0;
constexpr int N = static_cast<int>(SIM_SECONDS / DT_SECONDS) + 1;
constexpr double TUMBLE_STOP_THRESHOLD = 0.5 * DEG2RAD;
constexpr unsigned SEED = 20260302u;

// std::uniform_real_distribution / std::normal_distribution are
// implementation-defined: libc++ (macOS) and libstdc++ (Linux CI) draw
// DIFFERENT sequences from the same mt19937 seed, so this Monte Carlo was
// silently testing different satellites per platform (and one libc++ draw --
// 2 MTQ + 1 RW under the constant B field -- cannot reach the detumble
// threshold: MTQs produce no torque along a constant B, so the parallel
// component is the lone RW's job). These portable draws make the cases
// bit-identical everywhere.
double portableUniform01(std::mt19937& rng) {
	const std::uint64_t hi = rng();
	const std::uint64_t lo = rng();
	const std::uint64_t mantissa = ((hi << 21) ^ lo) & ((1ull << 53) - 1);
	return static_cast<double>(mantissa) * (1.0 / 9007199254740992.0);  // 2^-53
}

double portableUniform(std::mt19937& rng, double lo, double hi) {
	return lo + (hi - lo) * portableUniform01(rng);
}

int portableUniformInt(std::mt19937& rng, int lo, int hi) {
	const int span = hi - lo + 1;
	const int v = lo + static_cast<int>(portableUniform01(rng) * span);
	return std::min(v, hi);
}

double portableNormal(std::mt19937& rng) {
	// Box-Muller; u1 nudged away from 0 so log() stays finite.
	const double u1 = std::max(portableUniform01(rng), 1e-16);
	const double u2 = portableUniform01(rng);
	return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
}

Eigen::Vector3d randomUnitVector(std::mt19937& rng) {
	Eigen::Vector3d v(portableNormal(rng), portableNormal(rng), portableNormal(rng));
	const double n = v.norm();
	if (n < 1e-12) {
		return Eigen::Vector3d::UnitX();
	}
	return v / n;
}

Eigen::Matrix3d randomInertia(std::mt19937& rng) {
	const double mass = portableUniform(rng, 4.0, 60.0);
	const double lx = portableUniform(rng, 0.10, 0.55);
	const double ly = portableUniform(rng, 0.10, 0.55);
	const double lz = portableUniform(rng, 0.10, 0.55);

	Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
	J(0, 0) = (mass / 12.0) * (ly * ly + lz * lz);
	J(1, 1) = (mass / 12.0) * (lx * lx + lz * lz);
	J(2, 2) = (mass / 12.0) * (lx * lx + ly * ly);
	return J;
}

void makeRandomSatelliteCase(
	std::mt19937& rng,
	PlannerSettings& settings,
	Eigen::Matrix3d& J,
	std::unique_ptr<Satellite>& satellite
) {
	settings = PlannerSettings();
	settings.init_traj.initcontroller = 2;

	settings.disturbances.plan_for_aero = false;
	settings.disturbances.plan_for_gg = false;
	settings.disturbances.plan_for_srp = false;
	settings.disturbances.plan_for_prop = false;
	settings.disturbances.plan_for_gendist = false;
	settings.disturbances.plan_for_resdipole = false;

	settings.num_passes = 1;
	settings.passes[0].dt = DT_SECONDS;

	J = randomInertia(rng);
	satellite = std::make_unique<Satellite>(J, settings);

	const double Javg = std::max(1e-6, J.trace() / 3.0);

	const int n_mtq = portableUniformInt(rng, 2, 3);
	const int n_rw = portableUniformInt(rng, 1, 3);

	const double mtq_base = std::clamp(0.8 * std::sqrt(Javg), 0.03, 0.35);
	const double rw_torque_base = std::clamp(0.006 * Javg, 8e-5, 8e-3);
	const double rw_hmax_base = std::clamp(0.35 * Javg, 0.005, 0.12);

	for (int i = 0; i < n_mtq; ++i) {
		const Eigen::Vector3d axis = randomUnitVector(rng);
		const double max_dipole = std::clamp(mtq_base * portableUniform(rng, 0.7, 1.4), 0.02, 0.40);
		satellite->addMTQ(axis, max_dipole);
	}

	for (int i = 0; i < n_rw; ++i) {
		const Eigen::Vector3d axis = randomUnitVector(rng);
		const double max_torque = std::clamp(rw_torque_base * portableUniform(rng, 0.7, 1.4), 5e-5, 0.010);
		const double rw_J = std::clamp(0.015 * Javg * portableUniform(rng, 0.7, 1.4), 5e-6, 2e-3);
		const double h_max = std::clamp(rw_hmax_base * portableUniform(rng, 0.7, 1.4), 0.004, 0.16);
		satellite->addRW(axis, max_torque, rw_J, 0.0, h_max);
	}
}

Satellite::VecX randomInitialState(const Satellite& satellite, std::mt19937& rng) {
	Satellite::VecX x0 = Satellite::VecX::Zero(satellite.stateDim());

	const double w_mag = portableUniform(rng, 4.0 * DEG2RAD, 15.0 * DEG2RAD);
	x0.segment<3>(Satellite::AV_INDEX) = w_mag * randomUnitVector(rng);

	const double angle = portableUniform(rng, -PI, PI);
	const Eigen::Vector3d axis = randomUnitVector(rng);
	Eigen::Vector4d q;
	q(0) = std::cos(0.5 * angle);
	q.segment<3>(1) = axis * std::sin(0.5 * angle);
	q.normalize();
	x0.segment<4>(Satellite::QUAT_INDEX) = q;

	return x0;
}

void makeConstantEnvironment(
	Eigen::VectorXd& jtime,
	Eigen::MatrixXd& q_goal,
	Eigen::MatrixXd& boresight,
	Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ>& R,
	Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ>& V,
	Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ>& B,
	Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ>& S,
	Eigen::Matrix<double, 1, limits::MAX_LENGTH_TRAJ>& rho
) {
	jtime = Eigen::VectorXd::Zero(N);
	q_goal = Eigen::MatrixXd::Zero(4, N);
	boresight = Eigen::MatrixXd::Zero(3, N);

	R.setZero();
	V.setZero();
	B.setZero();
	S.setZero();
	rho.setZero();

	const Eigen::Vector3d B_const(2.2e-5, -1.6e-5, 3.1e-5);

	for (int k = 0; k < N; ++k) {
		const double t_sec = static_cast<double>(k) * DT_SECONDS;
		jtime(k) = 0.25 + t_sec / SEC_PER_CENTURY;
		q_goal(0, k) = 1.0;
		boresight.col(k) = Eigen::Vector3d::UnitX();

		B.col(k) = B_const;
	}
}

} // namespace

TEST_CASE("IntegratedBdotController warm_start Monte Carlo detumbles randomized satellites", "[controller][integratedbdot][warm_start][montecarlo]") {
	std::mt19937 rng(SEED);

	Eigen::VectorXd jtime;
	Eigen::MatrixXd q_goal;
	Eigen::MatrixXd boresight;
	Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> R;
	Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> V;
	Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> B;
	Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> S;
	Eigen::Matrix<double, 1, limits::MAX_LENGTH_TRAJ> rho;

	makeConstantEnvironment(jtime, q_goal, boresight, R, V, B, S, rho);

	for (int sample = 0; sample < NUM_SAMPLES; ++sample) {
		PlannerSettings settings;
		Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
		std::unique_ptr<Satellite> satellite;
		makeRandomSatelliteCase(rng, settings, J, satellite);

		const Satellite::VecX x0 = randomInitialState(*satellite, rng);

		Eigen::MatrixXd X = Eigen::MatrixXd::Zero(satellite->stateDim(), N);
		Eigen::MatrixXd U = Eigen::MatrixXd::Zero(satellite->controlDim(), N);

		bool ok = false;
		REQUIRE_NOTHROW(
			ok = optimizer::warm_start(
				settings,
				*satellite,
				x0,
				jtime,
				q_goal,
				boresight,
				N,
				R,
				V,
				B,
				S,
				rho,
				X,
				U
			)
		);
		REQUIRE(ok);

		const double initial_w = x0.segment<3>(Satellite::AV_INDEX).norm();
		const double final_w = X.col(N - 1).segment<3>(Satellite::AV_INDEX).norm();

		INFO("sample=" << sample
			 << " nMTQ=" << satellite->numMTQ()
			 << " nRW=" << satellite->numRW()
			 << " initial_w=" << initial_w
			 << " final_w=" << final_w
			 << " Jdiag=[" << J(0, 0) << ", " << J(1, 1) << ", " << J(2, 2) << "]");

		REQUIRE(std::isfinite(final_w));
		REQUIRE(final_w <= TUMBLE_STOP_THRESHOLD);
		REQUIRE(final_w < initial_w);
	}
}
