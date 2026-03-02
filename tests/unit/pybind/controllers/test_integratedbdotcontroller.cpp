
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <random>

#include <Eigen/Dense>

#include <saltro/math/integrators/rk4.h>
#include <saltro/pybind/controller/integratedbdotcontroller.h>
#include <saltro/pybind/satellite.h>

using namespace saltro;

namespace {

constexpr double PI = 3.14159265358979323846;
constexpr double DEG2RAD = PI / 180.0;

struct SatelliteCase {
	PlannerSettings settings;
	Satellite satellite;

	explicit SatelliteCase(const Eigen::Matrix3d& inertia)
		: settings(),
		  satellite(inertia, settings) {}

	SatelliteCase(const SatelliteCase&) = delete;
	SatelliteCase& operator=(const SatelliteCase&) = delete;
	SatelliteCase(SatelliteCase&&) = delete;
	SatelliteCase& operator=(SatelliteCase&&) = delete;
};

Eigen::Vector3d randomUnitVector(std::mt19937& rng) {
	std::normal_distribution<double> normal(0.0, 1.0);
	Eigen::Vector3d v(normal(rng), normal(rng), normal(rng));
	if (v.norm() < 1e-12) {
		return Eigen::Vector3d::UnitX();
	}
	return v.normalized();
}

Eigen::Matrix3d randomInertia(std::mt19937& rng) {
	std::uniform_real_distribution<double> mass_dist(4.0, 60.0);
	std::uniform_real_distribution<double> size_dist(0.10, 0.55);

	const double mass = mass_dist(rng);
	const double lx = size_dist(rng);
	const double ly = size_dist(rng);
	const double lz = size_dist(rng);

	Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
	J(0, 0) = (mass / 12.0) * (ly * ly + lz * lz);
	J(1, 1) = (mass / 12.0) * (lx * lx + lz * lz);
	J(2, 2) = (mass / 12.0) * (lx * lx + ly * ly);
	return J;
}

void configureRandomSatelliteCase(SatelliteCase& sc, std::mt19937& rng) {
	sc.settings.disturbances.plan_for_aero = false;
	sc.settings.disturbances.plan_for_gg = false;
	sc.settings.disturbances.plan_for_srp = false;
	sc.settings.disturbances.plan_for_prop = false;
	sc.settings.disturbances.plan_for_gendist = false;
	sc.settings.disturbances.plan_for_resdipole = false;

	sc.settings.init_traj.initcontroller = 2;
	sc.settings.num_passes = 1;
	sc.settings.passes[0].dt = 10.0;

	const double Javg = std::max(1e-6, sc.satellite.inertia().trace() / 3.0);

	std::uniform_int_distribution<int> n_mtq_dist(1, 3);
	std::uniform_int_distribution<int> n_rw_dist(1, 3);
	const int n_mtq = n_mtq_dist(rng);
	const int n_rw = n_rw_dist(rng);

	const double mtq_base = std::clamp(0.8 * std::sqrt(Javg), 0.03, 0.35);
	const double rw_torque_base = std::clamp(0.006 * Javg, 8e-5, 8e-3);
	const double rw_hmax_base = std::clamp(0.35 * Javg, 0.005, 0.12);

	std::uniform_real_distribution<double> scale_dist(0.7, 1.4);

	for (int i = 0; i < n_mtq; ++i) {
		const Eigen::Vector3d axis = randomUnitVector(rng);
		const double max_dipole = std::clamp(mtq_base * scale_dist(rng), 0.02, 0.40);
		sc.satellite.addMTQ(axis, max_dipole);
	}

	for (int i = 0; i < n_rw; ++i) {
		const Eigen::Vector3d axis = randomUnitVector(rng);
		const double max_torque = std::clamp(rw_torque_base * scale_dist(rng), 5e-5, 0.010);
		const double rw_J = std::clamp(0.015 * Javg * scale_dist(rng), 5e-6, 2e-3);
		const double h_max = std::clamp(rw_hmax_base * scale_dist(rng), 0.004, 0.16);
		sc.satellite.addRW(axis, max_torque, rw_J, 0.0, h_max);
	}
}

Satellite::VecX randomInitialState(const Satellite& sat, std::mt19937& rng) {
	Satellite::VecX x = Satellite::VecX::Zero(sat.stateDim());

	std::uniform_real_distribution<double> w_mag_dist(4.0 * DEG2RAD, 15.0 * DEG2RAD);
	const double w_mag = w_mag_dist(rng);
	const Eigen::Vector3d w_dir = randomUnitVector(rng);
	x.segment<3>(Satellite::AV_INDEX) = w_mag * w_dir;

	std::uniform_real_distribution<double> angle_dist(-PI, PI);
	const double angle = angle_dist(rng);
	const Eigen::Vector3d axis = randomUnitVector(rng);
	Eigen::Vector4d q;
	q(0) = std::cos(0.5 * angle);
	q.segment<3>(1) = axis * std::sin(0.5 * angle);
	q.normalize();
	x.segment<4>(Satellite::QUAT_INDEX) = q;

	return x;
}

void propagateOneStep(
	const Satellite& sat,
	const PlannerSettings& settings,
	controller::IntegratedBdotController& controller,
	Satellite::VecX& x,
	const Eigen::Vector3d& B_eci,
	double dt_seconds
) {
	controller.set_dt(dt_seconds);

	const Eigen::Vector4d q_goal = Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
	const Satellite::VecX u = controller.find_u(x, B_eci, q_goal);

	Satellite::VecX x_next(sat.stateDim());
	rk4_step<Satellite::VecX>(
		[&](double, const Satellite::VecX& x_state, Satellite::VecX& dxdt) {
			dxdt = sat.dynamics(
				x_state,
				u,
				settings.disturbances,
				Eigen::Vector3d::Zero(),
				B_eci,
				Eigen::Vector3d::Zero(),
				Eigen::Vector3d::Zero(),
				0
			);
		},
		x,
		0.0,
		dt_seconds,
		x_next
	);

	Eigen::Vector4d q = x_next.segment<4>(Satellite::QUAT_INDEX);
	const double qn = q.norm();
	REQUIRE(std::isfinite(qn));
	REQUIRE(qn > 1e-12);
	q /= qn;
	x_next.segment<4>(Satellite::QUAT_INDEX) = q;

	x = x_next;
}

} // namespace

TEST_CASE("IntegratedBdotController Monte Carlo detumbles randomized satellites", "[controller][integratedbdot][montecarlo]") {
	std::mt19937 rng(20260302);

	constexpr int num_samples = 80;
	constexpr double dt = 10.0;
	constexpr int steps = 100;

	const Eigen::Vector3d B_eci(2.2e-5, -1.6e-5, 3.1e-5);
	constexpr double tumble_stop_threshold = 0.5 * DEG2RAD;

	for (int sample = 0; sample < num_samples; ++sample) {
		SatelliteCase sc(randomInertia(rng));
		configureRandomSatelliteCase(sc, rng);
		controller::IntegratedBdotController controller(sc.satellite);

		Satellite::VecX x = randomInitialState(sc.satellite, rng);
		const double initial_w = x.segment<3>(Satellite::AV_INDEX).norm();

		for (int k = 0; k < steps; ++k) {
			propagateOneStep(sc.satellite, sc.settings, controller, x, B_eci, dt);
		}

		const double final_w = x.segment<3>(Satellite::AV_INDEX).norm();

		INFO("sample=" << sample
			 << " nMTQ=" << sc.satellite.numMTQ()
			 << " nRW=" << sc.satellite.numRW()
			 << " initial_w=" << initial_w
			 << " final_w=" << final_w);

		REQUIRE(std::isfinite(final_w));
		REQUIRE(final_w <= tumble_stop_threshold);
		REQUIRE(final_w < initial_w);
	}
}
