#include <catch2/catch_test_macros.hpp>
#include <cmath>

#include <saltro/math/angles.h>

using namespace saltro::math;

// ============================================================================
// wrap_to_2pi
// ============================================================================

TEST_CASE("wrap_to_2pi keeps in-range angles", "[math][angles][wrap]") {
    REQUIRE(std::abs(wrap_to_2pi(0.0) - 0.0) < 1e-12);
    REQUIRE(std::abs(wrap_to_2pi(1.0) - 1.0) < 1e-12);
    REQUIRE(std::abs(wrap_to_2pi(M_PI) - M_PI) < 1e-12);
}

TEST_CASE("wrap_to_2pi wraps above and below", "[math][angles][wrap]") {
    REQUIRE(std::abs(wrap_to_2pi(2.0 * M_PI + 0.5) - 0.5) < 1e-12);
    REQUIRE(std::abs(wrap_to_2pi(-0.5) - (2.0 * M_PI - 0.5)) < 1e-12);
    REQUIRE(std::abs(wrap_to_2pi(4.0 * M_PI + 1.0) - 1.0) < 1e-12);
}

TEST_CASE("wrap_to_2pi output is always in [0, 2pi)", "[math][angles][wrap]") {
    for (double a = -20.0; a <= 20.0; a += 0.37) {
        double w = wrap_to_2pi(a);
        REQUIRE(w >= 0.0);
        REQUIRE(w < 2.0 * M_PI);
    }
}

// ============================================================================
// wrap_to_360
// ============================================================================

TEST_CASE("wrap_to_360 known values", "[math][angles][wrap]") {
    REQUIRE(std::abs(wrap_to_360(45.0) - 45.0) < 1e-12);
    REQUIRE(std::abs(wrap_to_360(360.0) - 0.0) < 1e-12);
    REQUIRE(std::abs(wrap_to_360(450.0) - 90.0) < 1e-12);
    REQUIRE(std::abs(wrap_to_360(-90.0) - 270.0) < 1e-12);
}

TEST_CASE("wrap_to_360 output is always in [0, 360)", "[math][angles][wrap]") {
    for (double a = -1000.0; a <= 1000.0; a += 17.3) {
        double w = wrap_to_360(a);
        REQUIRE(w >= 0.0);
        REQUIRE(w < 360.0);
    }
}

// ============================================================================
// deg2rad / rad2deg
// ============================================================================

TEST_CASE("deg2rad known values", "[math][angles][convert]") {
    REQUIRE(std::abs(deg2rad(180.0) - M_PI) < 1e-12);
    REQUIRE(std::abs(deg2rad(90.0) - M_PI / 2.0) < 1e-12);
    REQUIRE(std::abs(deg2rad(0.0)) < 1e-12);
}

TEST_CASE("rad2deg known values", "[math][angles][convert]") {
    REQUIRE(std::abs(rad2deg(M_PI) - 180.0) < 1e-12);
    REQUIRE(std::abs(rad2deg(M_PI / 2.0) - 90.0) < 1e-12);
}

TEST_CASE("deg2rad and rad2deg are mutual inverses", "[math][angles][convert]") {
    for (double d = -350.0; d <= 350.0; d += 13.0) {
        REQUIRE(std::abs(rad2deg(deg2rad(d)) - d) < 1e-10);
    }
}
