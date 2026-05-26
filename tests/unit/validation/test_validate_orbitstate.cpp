#include <catch2/catch_test_macros.hpp>

#include <Eigen/Dense>

#include <saltro/constants/constants.h>
#include <saltro/validation/validate_orbitstate.h>

#include <cmath>
#include <limits>
#include <string>

using Vec3 = Eigen::Vector3d;

static void makeCircularLEO(double altitude_m, Vec3& r0, Vec3& v0) {
    const double radius = saltro::constants::R_EARTH + altitude_m;
    const double speed = std::sqrt(saltro::constants::MU_EARTH / radius);

    r0 = Vec3(radius, 0.0, 0.0);
    v0 = Vec3(0.0, speed, 0.0);
}

TEST_CASE("Valid circular LEO state passes validation", "[orbitstate][validation]") {
    Vec3 r0;
    Vec3 v0;
    makeCircularLEO(500e3, r0, v0);

    std::string error_msg;
    const bool ok = saltro::validation::validateOrbitState(r0, v0, error_msg);

    REQUIRE(ok);
    REQUIRE(error_msg.empty());
}

TEST_CASE("r0 likely in km fails with unit hint", "[orbitstate][validation][units]") {
    const Vec3 r0_km(6878.0, 0.0, 0.0);
    const Vec3 v0_kms(0.0, 7.6, 0.0);

    std::string error_msg;
    const bool ok = saltro::validation::validateOrbitState(r0_km, v0_kms, error_msg);

    REQUIRE_FALSE(ok);
    REQUIRE(error_msg == "r0 magnitude too small; expected meters (did you provide kilometers?)");
}

TEST_CASE("Non-finite state fails", "[orbitstate][validation][finite]") {
    Vec3 r0;
    Vec3 v0;
    makeCircularLEO(500e3, r0, v0);
    r0(1) = std::numeric_limits<double>::quiet_NaN();

    std::string error_msg;
    const bool ok = saltro::validation::validateOrbitState(r0, v0, error_msg);

    REQUIRE_FALSE(ok);
    REQUIRE(error_msg == "r0 contains non-finite values");
}

TEST_CASE("Highly elliptical orbit fails", "[orbitstate][validation][eccentricity]") {
    const double re = saltro::constants::R_EARTH;
    const double mu = saltro::constants::MU_EARTH;

    const double rp = re + 400e3;
    const double ra = re + 10000e3;
    const double a = 0.5 * (rp + ra);
    const double vp = std::sqrt(mu * (2.0 / rp - 1.0 / a));

    const Vec3 r0(rp, 0.0, 0.0);
    const Vec3 v0(0.0, vp, 0.0);

    std::string error_msg;
    const bool ok = saltro::validation::validateOrbitState(r0, v0, error_msg);

    REQUIRE_FALSE(ok);
    REQUIRE(error_msg == "orbit is too elliptical for LEO use (eccentricity > 0.3)");
}

TEST_CASE("Bound orbit above LEO ceiling fails", "[orbitstate][validation][leo]") {
    Vec3 r0;
    Vec3 v0;
    makeCircularLEO(3500e3, r0, v0);

    std::string error_msg;
    const bool ok = saltro::validation::validateOrbitState(r0, v0, error_msg);

    REQUIRE_FALSE(ok);
    REQUIRE(error_msg == "apogee altitude above LEO bounds");
}

TEST_CASE("Unbound orbit fails", "[orbitstate][validation][energy]") {
    Vec3 r0;
    Vec3 v0;
    makeCircularLEO(500e3, r0, v0);

    const double escape = std::sqrt(2.0 * saltro::constants::MU_EARTH / r0.norm());
    v0 = Vec3(0.0, 1.1 * escape, 0.0);

    std::string error_msg;
    const bool ok = saltro::validation::validateOrbitState(r0, v0, error_msg);

    REQUIRE_FALSE(ok);
    REQUIRE(error_msg == "specific orbital energy is non-negative (orbit not bound to Earth)");
}
