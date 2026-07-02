#include <catch2/catch_test_macros.hpp>
#include <Eigen/Dense>
#include <cmath>

#include <saltro/math/frames.h>

using namespace saltro::math;

// ============================================================================
// gmst_rad
// ============================================================================

TEST_CASE("gmst_rad output is wrapped to [0, 2pi)", "[math][frames][gmst]") {
    for (double T = -0.5; T <= 0.5; T += 0.05) {
        double g = gmst_rad(T);
        REQUIRE(g >= 0.0);
        REQUIRE(g < 2.0 * M_PI);
    }
}

TEST_CASE("gmst_rad is deterministic", "[math][frames][gmst]") {
    REQUIRE(std::abs(gmst_rad(0.1) - gmst_rad(0.1)) < 1e-15);
}

// ============================================================================
// eci_to_ecef_dcm / ecef_to_eci_dcm
// ============================================================================

TEST_CASE("eci_to_ecef_dcm is orthonormal with det +1", "[math][frames][dcm]") {
    for (double T : {-0.2, 0.0, 0.1, 0.37}) {
        Eigen::Matrix3d C = eci_to_ecef_dcm(T);
        REQUIRE((C * C.transpose() - Eigen::Matrix3d::Identity()).norm() < 1e-12);
        REQUIRE(std::abs(C.determinant() - 1.0) < 1e-12);
    }
}

TEST_CASE("ecef_to_eci_dcm is orthonormal with det +1", "[math][frames][dcm]") {
    for (double T : {-0.2, 0.0, 0.1, 0.37}) {
        Eigen::Matrix3d C = ecef_to_eci_dcm(T);
        REQUIRE((C * C.transpose() - Eigen::Matrix3d::Identity()).norm() < 1e-12);
        REQUIRE(std::abs(C.determinant() - 1.0) < 1e-12);
    }
}

TEST_CASE("ECI<->ECEF DCMs are mutual transposes/inverses", "[math][frames][dcm]") {
    for (double T : {-0.2, 0.0, 0.1, 0.37}) {
        Eigen::Matrix3d C1 = eci_to_ecef_dcm(T);
        Eigen::Matrix3d C2 = ecef_to_eci_dcm(T);
        REQUIRE((C1 - C2.transpose()).norm() < 1e-14);
        REQUIRE((C1 * C2 - Eigen::Matrix3d::Identity()).norm() < 1e-12);
    }
}

TEST_CASE("ECI->ECEF DCM is a pure z-axis rotation", "[math][frames][dcm]") {
    double T = 0.123;
    Eigen::Matrix3d C = eci_to_ecef_dcm(T);
    // Third row/column should be the z-axis (rotation about z leaves z fixed).
    REQUIRE(std::abs(C(2, 2) - 1.0) < 1e-12);
    REQUIRE(std::abs(C(2, 0)) < 1e-12);
    REQUIRE(std::abs(C(2, 1)) < 1e-12);
    REQUIRE(std::abs(C(0, 2)) < 1e-12);
    REQUIRE(std::abs(C(1, 2)) < 1e-12);
}
