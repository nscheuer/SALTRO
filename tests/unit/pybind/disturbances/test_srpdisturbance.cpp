#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Dense>

#include <saltro/constants/constants.h>
#include <saltro/pybind/disturbances/srpdisturbance.h>
#include <saltro/pybind/plannersettings.h>

using namespace saltro::disturbances;
using Vec3 = Eigen::Vector3d;

namespace {

GeometryFace makeFace(double area, const Vec3& centroid, const Vec3& normal,
					  double eta_s, double eta_d, double eta_a) {
	return GeometryFace(area, centroid, normal, eta_s, eta_d, eta_a, 0.0);
}

DisturbanceConfig makeDistCfg(bool plan_for_srp) {
	DisturbanceConfig cfg;
	cfg.plan_for_srp = plan_for_srp;
	return cfg;
}

Vec3 torqueFromDq(const SRPDisturbance& srp,
				  const DisturbanceConfig& dist,
				  const Vec3& v0,
				  const SRPDisturbance::Mat34& dV_dq,
				  const Eigen::Vector4d& dq) {
	SRPDisturbance::BaseState x = SRPDisturbance::BaseState::Zero();
	const Vec3 v = v0 + dV_dq * dq;
	return srp.torque(x, dist, v);
}

} // namespace

TEST_CASE("SRPDisturbance returns zero when disabled", "[srpdisturbance]") {
	GeometryConfig config;
	config.addFace(makeFace(1.0, Vec3(1.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), 0.2, 0.3, 0.1));

	SRPDisturbance srp(config);
	DisturbanceConfig dist = makeDistCfg(false);

	SRPDisturbance::BaseState x = SRPDisturbance::BaseState::Zero();
	Vec3 v_body(1.0, 0.0, 0.0);

	Vec3 torque = srp.torque(x, dist, v_body);
	REQUIRE(torque.isZero());
}

TEST_CASE("SRPDisturbance returns zero when inactive", "[srpdisturbance]") {
	GeometryConfig config;
	config.addFace(makeFace(1.0, Vec3(1.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), 0.2, 0.3, 0.1));

	SRPDisturbance srp(config);
	srp.setActive(false);
	DisturbanceConfig dist = makeDistCfg(true);

	SRPDisturbance::BaseState x = SRPDisturbance::BaseState::Zero();
	Vec3 v_body(1.0, 0.0, 0.0);

	Vec3 torque = srp.torque(x, dist, v_body);
	REQUIRE(torque.isZero());
}

TEST_CASE("SRPDisturbance returns zero for near-zero sun vector", "[srpdisturbance]") {
	GeometryConfig config;
	config.addFace(makeFace(1.0, Vec3(1.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), 0.2, 0.3, 0.1));

	SRPDisturbance srp(config);
	DisturbanceConfig dist = makeDistCfg(true);

	SRPDisturbance::BaseState x = SRPDisturbance::BaseState::Zero();
	Vec3 v_body(0.0, 0.0, 0.0);

	Vec3 torque = srp.torque(x, dist, v_body);
	REQUIRE(torque.isZero());
}

TEST_CASE("SRPDisturbance ignores faces with zero area", "[srpdisturbance]") {
	GeometryConfig config;
	config.addFace(makeFace(0.0, Vec3(1.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), 0.2, 0.3, 0.1));

	SRPDisturbance srp(config);
	DisturbanceConfig dist = makeDistCfg(true);

	SRPDisturbance::BaseState x = SRPDisturbance::BaseState::Zero();
	Vec3 v_body(1.0, 0.0, 0.0);

	Vec3 torque = srp.torque(x, dist, v_body);
	REQUIRE(torque.isZero());
}

TEST_CASE("SRPDisturbance ignores faces with invalid normals", "[srpdisturbance]") {
	GeometryConfig config;
	config.addFace(makeFace(1.0, Vec3(1.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0), 0.2, 0.3, 0.1));

	SRPDisturbance srp(config);
	DisturbanceConfig dist = makeDistCfg(true);

	SRPDisturbance::BaseState x = SRPDisturbance::BaseState::Zero();
	Vec3 v_body(1.0, 0.0, 0.0);

	Vec3 torque = srp.torque(x, dist, v_body);
	REQUIRE(torque.isZero());
}

TEST_CASE("SRPDisturbance ignores negative incidence", "[srpdisturbance]") {
	GeometryConfig config;
	config.addFace(makeFace(1.0, Vec3(1.0, 0.0, 0.0), Vec3(-1.0, 0.0, 0.0), 0.2, 0.3, 0.1));

	SRPDisturbance srp(config);
	DisturbanceConfig dist = makeDistCfg(true);

	SRPDisturbance::BaseState x = SRPDisturbance::BaseState::Zero();
	Vec3 v_body(1.0, 0.0, 0.0);

	Vec3 torque = srp.torque(x, dist, v_body);
	REQUIRE(torque.isZero());
}

TEST_CASE("SRPDisturbance single face torque matches expected", "[srpdisturbance]") {
	GeometryConfig config;
	config.addFace(makeFace(2.0, Vec3(0.0, 1.0, 0.0), Vec3(1.0, 0.0, 0.0), 0.1, 0.3, 0.2));

	SRPDisturbance srp(config);
	DisturbanceConfig dist = makeDistCfg(true);

	SRPDisturbance::BaseState x = SRPDisturbance::BaseState::Zero();
	Vec3 v_body(1.0, 0.0, 0.0);

	const double P = saltro::constants::SOLAR_CONSTANT / saltro::constants::C_LIGHT;
	const double m_s = 2.0 * (0.2 + 0.3);
	const double m_n = 2.0 * (2.0 * 0.1 * 1.0 + (2.0 / 3.0) * 0.3);
	const double expected_z = P * (m_s + m_n);

	Vec3 torque = srp.torque(x, dist, v_body);

	REQUIRE_THAT(torque(0), Catch::Matchers::WithinAbs(0.0, 1e-12));
	REQUIRE_THAT(torque(1), Catch::Matchers::WithinAbs(0.0, 1e-12));
	REQUIRE_THAT(torque(2), Catch::Matchers::WithinRel(expected_z, 1e-12));
}

TEST_CASE("SRPDisturbance normalizes face normals", "[srpdisturbance]") {
	GeometryConfig config;
	config.addFace(makeFace(1.0, Vec3(0.0, 1.0, 0.0), Vec3(2.0, 0.0, 0.0), 0.0, 0.0, 1.0));

	SRPDisturbance srp(config);
	DisturbanceConfig dist = makeDistCfg(true);

	SRPDisturbance::BaseState x = SRPDisturbance::BaseState::Zero();
	Vec3 v_body(1.0, 0.0, 0.0);

	const double P = saltro::constants::SOLAR_CONSTANT / saltro::constants::C_LIGHT;
	const double m_s = 1.0 * (1.0);
	const double expected_z = P * m_s;

	Vec3 torque = srp.torque(x, dist, v_body);

	REQUIRE_THAT(torque(2), Catch::Matchers::WithinRel(expected_z, 1e-12));
}

TEST_CASE("SRPDisturbance sums torque across multiple faces", "[srpdisturbance]") {
	GeometryConfig config;
	config.addFace(makeFace(1.0, Vec3(0.0, 1.0, 0.0), Vec3(1.0, 0.0, 0.0), 0.0, 0.0, 1.0));
	config.addFace(makeFace(2.0, Vec3(0.0, 0.0, 1.0), Vec3(1.0, 0.0, 0.0), 0.0, 0.0, 1.0));

	SRPDisturbance srp(config);
	DisturbanceConfig dist = makeDistCfg(true);

	SRPDisturbance::BaseState x = SRPDisturbance::BaseState::Zero();
	Vec3 v_body(1.0, 0.0, 0.0);

	const double P = saltro::constants::SOLAR_CONSTANT / saltro::constants::C_LIGHT;
	const Vec3 t1 = Vec3(0.0, 1.0, 0.0).cross(Vec3(1.0, 0.0, 0.0));
	const Vec3 t2 = Vec3(0.0, 0.0, 1.0).cross(Vec3(1.0, 0.0, 0.0));
	const Vec3 expected = -P * (1.0 * t1 + 2.0 * t2);

	Vec3 torque = srp.torque(x, dist, v_body);

	REQUIRE_THAT(torque(0), Catch::Matchers::WithinRel(expected(0), 1e-12));
	REQUIRE_THAT(torque(1), Catch::Matchers::WithinRel(expected(1), 1e-12));
	REQUIRE_THAT(torque(2), Catch::Matchers::WithinRel(expected(2), 1e-12));
}

TEST_CASE("SRPDisturbance jacobian is zero when dV_dq is zero", "[srpdisturbance]") {
	GeometryConfig config;
	config.addFace(makeFace(1.0, Vec3(0.0, 1.0, 0.0), Vec3(1.0, 0.0, 0.0), 0.1, 0.2, 0.3));

	SRPDisturbance srp(config);
	DisturbanceConfig dist = makeDistCfg(true);

	SRPDisturbance::BaseState x = SRPDisturbance::BaseState::Zero();
	Vec3 v_body(1.0, 0.0, 0.0);
	SRPDisturbance::Mat34 dV_dq = SRPDisturbance::Mat34::Zero();

	SRPDisturbance::Mat34 J = srp.dtorque_dq(x, dist, v_body, dV_dq);
	REQUIRE(J.isZero());
}

TEST_CASE("SRPDisturbance jacobian matches finite difference", "[srpdisturbance]") {
	GeometryConfig config;
	config.addFace(makeFace(2.0, Vec3(0.0, 1.0, 0.0), Vec3(1.0, 0.0, 0.0), 0.2, 0.1, 0.1));

	SRPDisturbance srp(config);
	DisturbanceConfig dist = makeDistCfg(true);

	SRPDisturbance::BaseState x = SRPDisturbance::BaseState::Zero();
	Vec3 v_body(1.0, 0.2, -0.1);

	SRPDisturbance::Mat34 dV_dq = SRPDisturbance::Mat34::Zero();
	dV_dq(0, 0) = 0.1;
	dV_dq(1, 1) = -0.2;
	dV_dq(2, 2) = 0.15;
	dV_dq(0, 3) = -0.05;

	const double eps = 1e-6;
	SRPDisturbance::Mat34 J = srp.dtorque_dq(x, dist, v_body, dV_dq);

	for (int j = 0; j < 4; ++j) {
		Eigen::Vector4d dq = Eigen::Vector4d::Zero();
		dq(j) = eps;
		const Vec3 f_plus = torqueFromDq(srp, dist, v_body, dV_dq, dq);
		dq(j) = -eps;
		const Vec3 f_minus = torqueFromDq(srp, dist, v_body, dV_dq, dq);
		const Vec3 fd = (f_plus - f_minus) / (2.0 * eps);

		REQUIRE_THAT(J(0, j), Catch::Matchers::WithinRel(fd(0), 1e-5));
		REQUIRE_THAT(J(1, j), Catch::Matchers::WithinRel(fd(1), 1e-5));
		REQUIRE_THAT(J(2, j), Catch::Matchers::WithinRel(fd(2), 1e-5));
	}
}

TEST_CASE("SRPDisturbance hessian is zero when disabled", "[srpdisturbance]") {
	GeometryConfig config;
	config.addFace(makeFace(1.0, Vec3(0.0, 1.0, 0.0), Vec3(1.0, 0.0, 0.0), 0.1, 0.2, 0.3));

	SRPDisturbance srp(config);
	DisturbanceConfig dist = makeDistCfg(false);

	SRPDisturbance::BaseState x = SRPDisturbance::BaseState::Zero();
	Vec3 v_body(1.0, 0.0, 0.0);
	SRPDisturbance::Mat34 dV_dq = SRPDisturbance::Mat34::Zero();
	std::array<SRPDisturbance::Mat44, 3> d2V_dq2 = {
		SRPDisturbance::Mat44::Zero(),
		SRPDisturbance::Mat44::Zero(),
		SRPDisturbance::Mat44::Zero()
	};

	SRPDisturbance::T443 H = srp.ddtorque_dqdq(x, dist, v_body, dV_dq, d2V_dq2);
	for (int k = 0; k < 3; ++k) {
		REQUIRE(H.slice(k).isZero());
	}
}

TEST_CASE("SRPDisturbance hessian matches finite difference", "[srpdisturbance]") {
	GeometryConfig config;
	config.addFace(makeFace(1.5, Vec3(0.0, 1.0, 0.0), Vec3(1.0, 0.0, 0.0), 0.2, 0.2, 0.1));

	SRPDisturbance srp(config);
	DisturbanceConfig dist = makeDistCfg(true);

	SRPDisturbance::BaseState x = SRPDisturbance::BaseState::Zero();
	Vec3 v_body(1.2, -0.3, 0.4);

	SRPDisturbance::Mat34 dV_dq = SRPDisturbance::Mat34::Zero();
	dV_dq(0, 0) = 0.2;
	dV_dq(1, 1) = -0.1;
	dV_dq(2, 2) = 0.15;
	dV_dq(0, 3) = -0.05;

	std::array<SRPDisturbance::Mat44, 3> d2V_dq2 = {
		SRPDisturbance::Mat44::Zero(),
		SRPDisturbance::Mat44::Zero(),
		SRPDisturbance::Mat44::Zero()
	};

	const double eps = 2e-6;
	SRPDisturbance::T443 H = srp.ddtorque_dqdq(x, dist, v_body, dV_dq, d2V_dq2);

	auto torque_at = [&](const Eigen::Vector4d& dq) {
		return torqueFromDq(srp, dist, v_body, dV_dq, dq);
	};

	for (int i = 0; i < 2; ++i) {
		for (int j = 0; j < 2; ++j) {
			Eigen::Vector4d dq_pp = Eigen::Vector4d::Zero();
			Eigen::Vector4d dq_pm = Eigen::Vector4d::Zero();
			Eigen::Vector4d dq_mp = Eigen::Vector4d::Zero();
			Eigen::Vector4d dq_mm = Eigen::Vector4d::Zero();

			dq_pp(i) = eps; dq_pp(j) = eps;
			dq_pm(i) = eps; dq_pm(j) = -eps;
			dq_mp(i) = -eps; dq_mp(j) = eps;
			dq_mm(i) = -eps; dq_mm(j) = -eps;

			const Vec3 f_pp = torque_at(dq_pp);
			const Vec3 f_pm = torque_at(dq_pm);
			const Vec3 f_mp = torque_at(dq_mp);
			const Vec3 f_mm = torque_at(dq_mm);

			const Vec3 fd = (f_pp - f_pm - f_mp + f_mm) / (4.0 * eps * eps);

			const double tol_abs = 2e-7;
			const double tol_rel = 1e-4;

			if (std::abs(fd(0)) < tol_abs) {
				REQUIRE_THAT(H.slice(0)(i, j), Catch::Matchers::WithinAbs(fd(0), tol_abs));
			} else {
				REQUIRE_THAT(H.slice(0)(i, j), Catch::Matchers::WithinRel(fd(0), tol_rel));
			}

			if (std::abs(fd(1)) < tol_abs) {
				REQUIRE_THAT(H.slice(1)(i, j), Catch::Matchers::WithinAbs(fd(1), tol_abs));
			} else {
				REQUIRE_THAT(H.slice(1)(i, j), Catch::Matchers::WithinRel(fd(1), tol_rel));
			}

			if (std::abs(fd(2)) < tol_abs) {
				REQUIRE_THAT(H.slice(2)(i, j), Catch::Matchers::WithinAbs(fd(2), tol_abs));
			} else {
				REQUIRE_THAT(H.slice(2)(i, j), Catch::Matchers::WithinRel(fd(2), tol_rel));
			}
		}
	}
}
