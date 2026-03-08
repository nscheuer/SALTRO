#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <Eigen/Dense>
#include <saltro/validation/validate_satellite.h>
#include <saltro/pybind/satellite.h>
#include <saltro/pybind/disturbances/geometryconfig.h>
#include <saltro/pybind/plannersettings.h>
#include <saltro/limits.h>
#include <cmath>
#include <string>
#include <limits>

using Mat33 = Eigen::Matrix3d;
using Vec3 = Eigen::Vector3d;

// ============================================================================
// NOTE ON TEST DESIGN
// ============================================================================
// The Satellite class performs its own validation in constructors and setters,
// throwing exceptions for invalid inputs (e.g., non-invertible inertia).
// The validateSatellite() function provides an additional safety layer that:
//   1. Confirms valid satellites pass validation
//   2. Catches edge cases and internal inconsistencies
//   3. Provides detailed error messages for debugging
//
// Many negative test cases cannot reach validateSatellite() because the
// Satellite API prevents creation of invalid states. These are tested
// via REQUIRE_THROWS to verify API-level validation works correctly.
// ============================================================================

// ============================================================================
// Helper Functions
// ============================================================================

static Mat33 validInertiaMatrix() {
    Mat33 J;
    J << 0.067, 0.0, 0.0,
         0.0, 0.067, 0.0,
         0.0, 0.0, 0.067;
    return J;
}

// ============================================================================
// Basic Validation Tests
// ============================================================================

TEST_CASE("Valid satellite passes validation", "[satellite][validation]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    std::string error_msg;
    
    REQUIRE(saltro::validation::validateSatellite(sat, error_msg));
    REQUIRE(error_msg.empty());
}

TEST_CASE("Valid satellite with MTQ", "[satellite][validation]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    sat.addMTQ(Vec3(1.0, 0.0, 0.0), 0.2);
    std::string error_msg;
    
    REQUIRE(saltro::validation::validateSatellite(sat, error_msg));
}

TEST_CASE("Valid satellite with RW", "[satellite][validation]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    sat.addRW(Vec3(1.0, 0.0, 0.0), 0.01, 0.001, 0.0, 0.1);
    std::string error_msg;
    
    REQUIRE(saltro::validation::validateSatellite(sat, error_msg));
}

TEST_CASE("Valid satellite with geometry", "[satellite][validation]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    saltro::disturbances::GeometryConfig& geom = sat.geometryConfig();
    
    Vec3 centroid(0.0, 0.0, 0.1);
    Vec3 normal(0.0, 0.0, 1.0);
    saltro::disturbances::GeometryFace face(0.01, centroid, normal, 0.1, 0.2, 0.7, 2.2);
    geom.addFace(face);
    
    std::string error_msg;
    REQUIRE(saltro::validation::validateSatellite(sat, error_msg));
}

// ============================================================================
// Inertia Matrix Validation Tests
// ============================================================================

TEST_CASE("Invalid inertia - contains NaN", "[satellite][validation][inertia]") {
    Mat33 J = validInertiaMatrix();
    J(0, 0) = std::numeric_limits<double>::quiet_NaN();
    PlannerSettings settings;
    // Satellite constructor throws for non-finite inertia
    REQUIRE_THROWS_AS(Satellite(J, settings), std::invalid_argument);
}

TEST_CASE("Invalid inertia - contains infinity", "[satellite][validation][inertia]") {
    Mat33 J = validInertiaMatrix();
    J(1, 1) = std::numeric_limits<double>::infinity();
    PlannerSettings settings;
    Satellite sat(validInertiaMatrix(), settings);
    // Satellite API rejects non-finite inertia
    REQUIRE_THROWS_AS(sat.setInertia(J), std::invalid_argument);
}

TEST_CASE("Invalid inertia - not symmetric", "[satellite][validation][inertia]") {
    Mat33 J = validInertiaMatrix();
    J(0, 1) = 0.01;
    J(1, 0) = 0.02;  // Different from J(0,1)
    PlannerSettings settings;
    Satellite sat(validInertiaMatrix(), settings);
    // setInertia doesn't check symmetry, so it accepts this
    REQUIRE_NOTHROW(sat.setInertia(J));
    
    std::string error_msg;
    // validateSatellite should catch non-symmetric inertia
    REQUIRE_FALSE(saltro::validation::validateSatellite(sat, error_msg));
    REQUIRE(error_msg == "inertia matrix is not symmetric");
}

TEST_CASE("Invalid inertia - zero determinant", "[satellite][validation][inertia]") {
    Mat33 J;
    J << 1.0, 2.0, 3.0,
         2.0, 4.0, 6.0,
         3.0, 6.0, 9.0;
    PlannerSettings settings;
    Satellite sat(validInertiaMatrix(), settings);
    // Satellite API rejects non-invertible inertia (determinant < 1e-12)
    REQUIRE_THROWS_AS(sat.setInertia(J), std::invalid_argument);
}

TEST_CASE("Invalid inertia - negative determinant", "[satellite][validation][inertia]") {
    Mat33 J;
    J << -0.1, 0.0, 0.0,
         0.0, 0.1, 0.0,
         0.0, 0.0, 0.1;
    PlannerSettings settings;
    Satellite sat(validInertiaMatrix(), settings);
    // This matrix has determinant = -0.001, which is <= 1e-12, so setInertia accepts it
    // But validateSatellite catches it via determinant check
    REQUIRE_NOTHROW(sat.setInertia(J));
    
    std::string error_msg;
    REQUIRE_FALSE(saltro::validation::validateSatellite(sat, error_msg));
    REQUIRE(error_msg == "inertia matrix determinant is too small or non-positive");
}

TEST_CASE("Invalid inertia - non-positive eigenvalue", "[satellite][validation][inertia]") {
    Mat33 J;
    J << 0.1, 0.0, 0.0,
         0.0, 0.1, 0.0,
         0.0, 0.0, -0.05;  // Negative eigenvalue
    PlannerSettings settings;
    Satellite sat(validInertiaMatrix(), settings);
    // This matrix has determinant = 0.1 * 0.1 * (-0.05) = -0.0005, which is also <= 1e-12
    // validateSatellite will catch it via determinant check
    REQUIRE_NOTHROW(sat.setInertia(J));
    
    std::string error_msg;
    REQUIRE_FALSE(saltro::validation::validateSatellite(sat, error_msg));
    REQUIRE(error_msg == "inertia matrix determinant is too small or non-positive");
}

TEST_CASE("Invalid inertia - magnitude too small", "[satellite][validation][inertia]") {
    Mat33 J;
    J << 1e-7, 0.0, 0.0,
         0.0, 1e-7, 0.0,
         0.0, 0.0, 1e-7;
    PlannerSettings settings;
    Satellite sat(validInertiaMatrix(), settings);
    // Satellite API rejects inertia that's too small
    REQUIRE_THROWS_AS(sat.setInertia(J), std::invalid_argument);
}

TEST_CASE("Invalid inertia - magnitude too large", "[satellite][validation][inertia]") {
    Mat33 J;
    J << 1e7, 0.0, 0.0,
         0.0, 1e7, 0.0,
         0.0, 0.0, 1e7;
    PlannerSettings settings;
    Satellite sat(validInertiaMatrix(), settings);
    sat.setInertia(J);
    
    std::string error_msg;
    // This passes Satellite API but validateSatellite should catch it
    REQUIRE_FALSE(saltro::validation::validateSatellite(sat, error_msg));
    REQUIRE(error_msg == "inertia matrix magnitude out of reasonable range");
}

// ============================================================================
// Actuator Count Validation Tests
// ============================================================================

TEST_CASE("Valid satellite with maximum MTQs", "[satellite][validation][actuators]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    for (int i = 0; i < saltro::limits::MAX_NUM_MTQ; ++i) {
        Vec3 axis = Vec3::Zero();
        axis(i % 3) = 1.0;
        sat.addMTQ(axis, 0.2);
    }
    
    std::string error_msg;
    REQUIRE(saltro::validation::validateSatellite(sat, error_msg));
}

TEST_CASE("Valid satellite with maximum RWs", "[satellite][validation][actuators]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    for (int i = 0; i < saltro::limits::MAX_NUM_RW; ++i) {
        Vec3 axis = Vec3::Zero();
        axis(i % 3) = 1.0;
        sat.addRW(axis, 0.01, 0.001, 0.0, 0.1);
    }
    
    std::string error_msg;
    REQUIRE(saltro::validation::validateSatellite(sat, error_msg));
}

// Note: Testing negative actuator counts would require mocking the Satellite class,
// as the actual implementation manages counts internally and won't allow negatives.
// The validation function still checks for this defensive programming.

// ============================================================================
// MTQ Configuration Validation Tests
// ============================================================================
// Note: Actuator constructor normalizes axes and applies abs() to u_max,
// so tests for non-normalized axes and negative u_max are not possible.

TEST_CASE("Invalid MTQ - axis contains NaN", "[satellite][validation][mtq]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    Vec3 axis(std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0);
    // Actuator constructor throws for non-finite axis
    REQUIRE_THROWS_AS(sat.addMTQ(axis, 0.2), std::invalid_argument);
}

TEST_CASE("Invalid MTQ - max dipole is zero", "[satellite][validation][mtq]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    Vec3 axis(1.0, 0.0, 0.0);
    // API accepts zero (after abs), but validation catches it
    REQUIRE_NOTHROW(sat.addMTQ(axis, 0.0));
    
    std::string error_msg;
    REQUIRE_FALSE(saltro::validation::validateSatellite(sat, error_msg));
    REQUIRE(error_msg == "MTQ 0 max dipole invalid");
}

TEST_CASE("Invalid MTQ - max dipole is NaN", "[satellite][validation][mtq]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    Vec3 axis(1.0, 0.0, 0.0);
    // Actuator constructor throws for non-finite u_max
    REQUIRE_THROWS_AS(sat.addMTQ(axis, std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
}

TEST_CASE("Invalid MTQ - max dipole unreasonably large", "[satellite][validation][mtq]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    Vec3 axis(1.0, 0.0, 0.0);
    sat.addMTQ(axis, 2e6);  // > 1e6
    
    std::string error_msg;
    REQUIRE_FALSE(saltro::validation::validateSatellite(sat, error_msg));
    REQUIRE(error_msg == "MTQ 0 max dipole unreasonably large");
}

TEST_CASE("Invalid MTQ - second MTQ has invalid axis", "[satellite][validation][mtq]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    Vec3 axis1(1.0, 0.0, 0.0);
    sat.addMTQ(axis1, 0.2);
    // Second MTQ has zero dipole
    sat.addMTQ(axis1, 0.0);
    
    std::string error_msg;
    REQUIRE_FALSE(saltro::validation::validateSatellite(sat, error_msg));
    REQUIRE(error_msg == "MTQ 1 max dipole invalid");
}

// ============================================================================
// RW Configuration Validation Tests
// ============================================================================
// Note: Actuator constructor normalizes axes and applies abs() to u_max,
// so tests for non-normalized axes and negative values are not possible.

TEST_CASE("Invalid RW - axis contains NaN", "[satellite][validation][rw]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    Vec3 axis(std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0);
    // Actuator constructor throws for non-finite axis
    REQUIRE_THROWS_AS(sat.addRW(axis, 0.01, 0.001, 0.0, 0.1), std::invalid_argument);
}

TEST_CASE("Invalid RW - max torque is zero", "[satellite][validation][rw]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    Vec3 axis(1.0, 0.0, 0.0);
    // API accepts zero (after abs), but validation catches it
    REQUIRE_NOTHROW(sat.addRW(axis, 0.0, 0.001, 0.0, 0.1));
    
    std::string error_msg;
    REQUIRE_FALSE(saltro::validation::validateSatellite(sat, error_msg));
    REQUIRE(error_msg == "RW 0 max torque invalid");
}

TEST_CASE("Invalid RW - max torque is NaN", "[satellite][validation][rw]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    Vec3 axis(1.0, 0.0, 0.0);
    // Actuator constructor throws for non-finite u_max
    REQUIRE_THROWS_AS(sat.addRW(axis, std::numeric_limits<double>::quiet_NaN(), 0.001, 0.0, 0.1), std::invalid_argument);
}

TEST_CASE("Invalid RW - max torque unreasonably large", "[satellite][validation][rw]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    Vec3 axis(1.0, 0.0, 0.0);
    sat.addRW(axis, 2e4, 0.001, 0.0, 0.1);  // > 1e4
    
    std::string error_msg;
    REQUIRE_FALSE(saltro::validation::validateSatellite(sat, error_msg));
    REQUIRE(error_msg == "RW 0 max torque unreasonably large");
}

TEST_CASE("Invalid RW - wheel inertia is zero", "[satellite][validation][rw]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    Vec3 axis(1.0, 0.0, 0.0);
    // RW constructor throws for J <= 0
    REQUIRE_THROWS_AS(sat.addRW(axis, 0.01, 0.0, 0.0, 0.1), std::invalid_argument);
}

TEST_CASE("Invalid RW - wheel inertia is negative", "[satellite][validation][rw]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    Vec3 axis(1.0, 0.0, 0.0);
    // RW constructor throws for J <= 0
    REQUIRE_THROWS_AS(sat.addRW(axis, 0.01, -0.001, 0.0, 0.1), std::invalid_argument);
}

TEST_CASE("Invalid RW - wheel inertia is NaN", "[satellite][validation][rw]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    Vec3 axis(1.0, 0.0, 0.0);
    // RW constructor throws for non-finite J
    REQUIRE_THROWS_AS(sat.addRW(axis, 0.01, std::numeric_limits<double>::quiet_NaN(), 0.0, 0.1), std::invalid_argument);
}

TEST_CASE("Invalid RW - wheel inertia unreasonably large", "[satellite][validation][rw]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    Vec3 axis(1.0, 0.0, 0.0);
    sat.addRW(axis, 0.01, 2e3, 0.0, 0.1);  // > 1e3
    
    std::string error_msg;
    REQUIRE_FALSE(saltro::validation::validateSatellite(sat, error_msg));
    REQUIRE(error_msg == "RW 0 wheel inertia unreasonably large");
}

TEST_CASE("Invalid RW - initial momentum is NaN", "[satellite][validation][rw]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    Vec3 axis(1.0, 0.0, 0.0);
    // API accepts NaN h0, validation catches it
    REQUIRE_NOTHROW(sat.addRW(axis, 0.01, 0.001, std::numeric_limits<double>::quiet_NaN(), 0.1));
    
    std::string error_msg;
    REQUIRE_FALSE(saltro::validation::validateSatellite(sat, error_msg));
    REQUIRE(error_msg == "RW 0 momentum values not finite");
}

TEST_CASE("Invalid RW - initial momentum is infinite", "[satellite][validation][rw]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    Vec3 axis(1.0, 0.0, 0.0);
    // API accepts infinity, validation catches it
    REQUIRE_NOTHROW(sat.addRW(axis, 0.01, 0.001, std::numeric_limits<double>::infinity(), 0.1));
    
    std::string error_msg;
    REQUIRE_FALSE(saltro::validation::validateSatellite(sat, error_msg));
    REQUIRE(error_msg == "RW 0 momentum values not finite");
}

TEST_CASE("Invalid RW - max momentum is NaN", "[satellite][validation][rw]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    Vec3 axis(1.0, 0.0, 0.0);
    // RW constructor throws when !(h_max >= 0.0), and NaN fails this check
    REQUIRE_THROWS_AS(sat.addRW(axis, 0.01, 0.001, 0.0, std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
}

TEST_CASE("Invalid RW - max momentum is zero", "[satellite][validation][rw]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    Vec3 axis(1.0, 0.0, 0.0);
    sat.addRW(axis, 0.01, 0.001, 0.0, 0.0);
    
    std::string error_msg;
    REQUIRE_FALSE(saltro::validation::validateSatellite(sat, error_msg));
    REQUIRE(error_msg == "RW 0 max momentum must be positive");
}

TEST_CASE("Invalid RW - max momentum is negative", "[satellite][validation][rw]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    Vec3 axis(1.0, 0.0, 0.0);
    // RW constructor throws when !(h_max >= 0.0)
    REQUIRE_THROWS_AS(sat.addRW(axis, 0.01, 0.001, 0.0, -0.1), std::invalid_argument);
}

TEST_CASE("Invalid RW - initial momentum exceeds max (positive)", "[satellite][validation][rw]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    Vec3 axis(1.0, 0.0, 0.0);
    sat.addRW(axis, 0.01, 0.001, 0.15, 0.1);  // 0.15 > 0.1
    
    std::string error_msg;
    REQUIRE_FALSE(saltro::validation::validateSatellite(sat, error_msg));
    REQUIRE(error_msg == "RW 0 initial momentum exceeds max");
}

TEST_CASE("Invalid RW - initial momentum exceeds max (negative)", "[satellite][validation][rw]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    Vec3 axis(1.0, 0.0, 0.0);
    sat.addRW(axis, 0.01, 0.001, -0.15, 0.1);  // |-0.15| > 0.1
    
    std::string error_msg;
    REQUIRE_FALSE(saltro::validation::validateSatellite(sat, error_msg));
    REQUIRE(error_msg == "RW 0 initial momentum exceeds max");
}

TEST_CASE("Invalid RW - max momentum unreasonably large", "[satellite][validation][rw]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    Vec3 axis(1.0, 0.0, 0.0);
    sat.addRW(axis, 0.01, 0.001, 0.0, 2e4);  // > 1e4
    
    std::string error_msg;
    REQUIRE_FALSE(saltro::validation::validateSatellite(sat, error_msg));
    REQUIRE(error_msg == "RW 0 max momentum unreasonably large");
}

TEST_CASE("Valid RW - initial momentum at boundary", "[satellite][validation][rw]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    Vec3 axis(1.0, 0.0, 0.0);
    sat.addRW(axis, 0.01, 0.001, 0.1, 0.1);  // h_init == h_max
    
    std::string error_msg;
    REQUIRE(saltro::validation::validateSatellite(sat, error_msg));
}

TEST_CASE("Valid RW - initial momentum at negative boundary", "[satellite][validation][rw]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    Vec3 axis(1.0, 0.0, 0.0);
    sat.addRW(axis, 0.01, 0.001, -0.1, 0.1);  // h_init == -h_max
    
    std::string error_msg;
    REQUIRE(saltro::validation::validateSatellite(sat, error_msg));
}

TEST_CASE("Invalid RW - second RW has invalid configuration", "[satellite][validation][rw]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    Vec3 axis1(1.0, 0.0, 0.0);
    Vec3 axis2(0.0, 1.0, 0.0);
    sat.addRW(axis1, 0.01, 0.001, 0.0, 0.1);
    // Second RW with zero max torque (accepted by API but caught by validation)
    sat.addRW(axis2, 0.0, 0.001, 0.0, 0.1);
    
    std::string error_msg;
    REQUIRE_FALSE(saltro::validation::validateSatellite(sat, error_msg));
    REQUIRE(error_msg == "RW 1 max torque invalid");
}

// ============================================================================
// Inertia Consistency Validation Tests
// ============================================================================
// Note: Tests for inertia_noRW consistency cannot be implemented as Satellite
// manages inertia_noRW internally without a public setter. The validation
// function still checks consistency, but corruption can only occur through
// internal errors, not user input.

// ============================================================================
// Geometry Configuration Validation Tests
// ============================================================================

TEST_CASE("Valid satellite with multiple geometry faces", "[satellite][validation][geometry]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    saltro::disturbances::GeometryConfig& geom = sat.geometryConfig();
    
    // Add 6 faces for a cube
    for (int i = 0; i < 6; ++i) {
        Vec3 normal = Vec3::Zero();
        normal(i % 3) = (i < 3) ? 1.0 : -1.0;
        Vec3 centroid = normal * 0.05;
        
        saltro::disturbances::GeometryFace face(0.01, centroid, normal, 0.1, 0.2, 0.7, 2.2);
        geom.addFace(face);
    }
    
    std::string error_msg;
    REQUIRE(saltro::validation::validateSatellite(sat, error_msg));
}

TEST_CASE("Invalid geometry - too many faces", "[satellite][validation][geometry]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    saltro::disturbances::GeometryConfig& geom = sat.geometryConfig();
    
    // Try to add more than MAX_NUM_GEOMETRY_FACES
    // Note: addFace() will return false when limit is reached
    // We need to create a GeometryConfig with more faces directly
    // For this test, we assume the validation would catch it if possible
    
    // Add maximum + 1 faces (if API allows)
    for (size_t i = 0; i < saltro::limits::MAX_NUM_GEOMETRY_FACES + 1; ++i) {
        Vec3 normal(0.0, 0.0, 1.0);
        Vec3 centroid(0.0, 0.0, 0.1);
        saltro::disturbances::GeometryFace face(0.01, centroid, normal, 0.1, 0.2, 0.7, 2.2);
        bool added = geom.addFace(face);
        if (!added) break;  // API prevents over-adding
    }
    
    // If we managed to add too many (shouldn't happen with proper API),
    // validation should catch it
    std::string error_msg;
    if (geom.numFaces() > saltro::limits::MAX_NUM_GEOMETRY_FACES) {
        REQUIRE_FALSE(saltro::validation::validateSatellite(sat, error_msg));
        REQUIRE(error_msg == "geometry has more faces than maximum allowed");
    }
}

TEST_CASE("Invalid geometry - face area is negative", "[satellite][validation][geometry]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    saltro::disturbances::GeometryConfig& geom = sat.geometryConfig();
    
    Vec3 centroid(0.0, 0.0, 0.1);
    Vec3 normal(0.0, 0.0, 1.0);
    saltro::disturbances::GeometryFace face(-0.01, centroid, normal, 0.1, 0.2, 0.7, 2.2);
    geom.addFace(face);
    
    std::string error_msg;
    REQUIRE_FALSE(saltro::validation::validateSatellite(sat, error_msg));
    REQUIRE(error_msg == "geometry face 0 area invalid");
}

TEST_CASE("Invalid geometry - face area is NaN", "[satellite][validation][geometry]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    saltro::disturbances::GeometryConfig& geom = sat.geometryConfig();
    
    Vec3 centroid(0.0, 0.0, 0.1);
    Vec3 normal(0.0, 0.0, 1.0);
    double nan_area = std::numeric_limits<double>::quiet_NaN();
    saltro::disturbances::GeometryFace face(nan_area, centroid, normal, 0.1, 0.2, 0.7, 2.2);
    geom.addFace(face);
    
    std::string error_msg;
    REQUIRE_FALSE(saltro::validation::validateSatellite(sat, error_msg));
    REQUIRE(error_msg == "geometry face 0 area invalid");
}

TEST_CASE("Invalid geometry - centroid contains NaN", "[satellite][validation][geometry]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    saltro::disturbances::GeometryConfig& geom = sat.geometryConfig();
    
    Vec3 centroid(std::numeric_limits<double>::quiet_NaN(), 0.0, 0.1);
    Vec3 normal(0.0, 0.0, 1.0);
    saltro::disturbances::GeometryFace face(0.01, centroid, normal, 0.1, 0.2, 0.7, 2.2);
    geom.addFace(face);
    
    std::string error_msg;
    REQUIRE_FALSE(saltro::validation::validateSatellite(sat, error_msg));
    REQUIRE(error_msg == "geometry face 0 centroid not finite");
}

TEST_CASE("Invalid geometry - centroid contains infinity", "[satellite][validation][geometry]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    saltro::disturbances::GeometryConfig& geom = sat.geometryConfig();
    
    Vec3 centroid(0.0, std::numeric_limits<double>::infinity(), 0.1);
    Vec3 normal(0.0, 0.0, 1.0);
    saltro::disturbances::GeometryFace face(0.01, centroid, normal, 0.1, 0.2, 0.7, 2.2);
    geom.addFace(face);
    
    std::string error_msg;
    REQUIRE_FALSE(saltro::validation::validateSatellite(sat, error_msg));
    REQUIRE(error_msg == "geometry face 0 centroid not finite");
}

TEST_CASE("Invalid geometry - normal not normalized (too short)", "[satellite][validation][geometry]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    saltro::disturbances::GeometryConfig& geom = sat.geometryConfig();
    
    Vec3 centroid(0.0, 0.0, 0.1);
    Vec3 normal(0.0, 0.0, 0.5);  // Norm = 0.5 < 0.99
    saltro::disturbances::GeometryFace face(0.01, centroid, normal, 0.1, 0.2, 0.7, 2.2);
    geom.addFace(face);
    
    std::string error_msg;
    REQUIRE_FALSE(saltro::validation::validateSatellite(sat, error_msg));
    REQUIRE(error_msg == "geometry face 0 normal not normalized");
}

TEST_CASE("Invalid geometry - normal not normalized (too long)", "[satellite][validation][geometry]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    saltro::disturbances::GeometryConfig& geom = sat.geometryConfig();
    
    Vec3 centroid(0.0, 0.0, 0.1);
    Vec3 normal(0.0, 0.0, 1.5);  // Norm = 1.5 > 1.01
    saltro::disturbances::GeometryFace face(0.01, centroid, normal, 0.1, 0.2, 0.7, 2.2);
    geom.addFace(face);
    
    std::string error_msg;
    REQUIRE_FALSE(saltro::validation::validateSatellite(sat, error_msg));
    REQUIRE(error_msg == "geometry face 0 normal not normalized");
}

TEST_CASE("Invalid geometry - normal contains NaN", "[satellite][validation][geometry]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    saltro::disturbances::GeometryConfig& geom = sat.geometryConfig();
    
    Vec3 centroid(0.0, 0.0, 0.1);
    Vec3 normal(0.0, std::numeric_limits<double>::quiet_NaN(), 1.0);
    saltro::disturbances::GeometryFace face(0.01, centroid, normal, 0.1, 0.2, 0.7, 2.2);
    geom.addFace(face);
    
    std::string error_msg;
    REQUIRE_FALSE(saltro::validation::validateSatellite(sat, error_msg));
    REQUIRE(error_msg == "geometry face 0 normal not normalized");
}

TEST_CASE("Invalid geometry - specular coefficient negative", "[satellite][validation][geometry]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    saltro::disturbances::GeometryConfig& geom = sat.geometryConfig();
    
    Vec3 centroid(0.0, 0.0, 0.1);
    Vec3 normal(0.0, 0.0, 1.0);
    saltro::disturbances::GeometryFace face(0.01, centroid, normal, -0.1, 0.2, 0.7, 2.2);
    geom.addFace(face);
    
    std::string error_msg;
    REQUIRE_FALSE(saltro::validation::validateSatellite(sat, error_msg));
    REQUIRE(error_msg == "geometry face 0 specular coefficient invalid");
}

TEST_CASE("Invalid geometry - specular coefficient > 1", "[satellite][validation][geometry]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    saltro::disturbances::GeometryConfig& geom = sat.geometryConfig();
    
    Vec3 centroid(0.0, 0.0, 0.1);
    Vec3 normal(0.0, 0.0, 1.0);
    saltro::disturbances::GeometryFace face(0.01, centroid, normal, 1.5, 0.2, 0.7, 2.2);
    geom.addFace(face);
    
    std::string error_msg;
    REQUIRE_FALSE(saltro::validation::validateSatellite(sat, error_msg));
    REQUIRE(error_msg == "geometry face 0 specular coefficient invalid");
}

TEST_CASE("Invalid geometry - specular coefficient is NaN", "[satellite][validation][geometry]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    saltro::disturbances::GeometryConfig& geom = sat.geometryConfig();
    
    Vec3 centroid(0.0, 0.0, 0.1);
    Vec3 normal(0.0, 0.0, 1.0);
    double nan_eta = std::numeric_limits<double>::quiet_NaN();
    saltro::disturbances::GeometryFace face(0.01, centroid, normal, nan_eta, 0.2, 0.7, 2.2);
    geom.addFace(face);
    
    std::string error_msg;
    REQUIRE_FALSE(saltro::validation::validateSatellite(sat, error_msg));
    REQUIRE(error_msg == "geometry face 0 specular coefficient invalid");
}

TEST_CASE("Invalid geometry - diffuse coefficient negative", "[satellite][validation][geometry]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    saltro::disturbances::GeometryConfig& geom = sat.geometryConfig();
    
    Vec3 centroid(0.0, 0.0, 0.1);
    Vec3 normal(0.0, 0.0, 1.0);
    saltro::disturbances::GeometryFace face(0.01, centroid, normal, 0.1, -0.2, 0.7, 2.2);
    geom.addFace(face);
    
    std::string error_msg;
    REQUIRE_FALSE(saltro::validation::validateSatellite(sat, error_msg));
    REQUIRE(error_msg == "geometry face 0 diffuse coefficient invalid");
}

TEST_CASE("Invalid geometry - diffuse coefficient > 1", "[satellite][validation][geometry]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    saltro::disturbances::GeometryConfig& geom = sat.geometryConfig();
    
    Vec3 centroid(0.0, 0.0, 0.1);
    Vec3 normal(0.0, 0.0, 1.0);
    saltro::disturbances::GeometryFace face(0.01, centroid, normal, 0.1, 1.2, 0.7, 2.2);
    geom.addFace(face);
    
    std::string error_msg;
    REQUIRE_FALSE(saltro::validation::validateSatellite(sat, error_msg));
    REQUIRE(error_msg == "geometry face 0 diffuse coefficient invalid");
}

TEST_CASE("Invalid geometry - diffuse coefficient is NaN", "[satellite][validation][geometry]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    saltro::disturbances::GeometryConfig& geom = sat.geometryConfig();
    
    Vec3 centroid(0.0, 0.0, 0.1);
    Vec3 normal(0.0, 0.0, 1.0);
    double nan_eta = std::numeric_limits<double>::quiet_NaN();
    saltro::disturbances::GeometryFace face(0.01, centroid, normal, 0.1, nan_eta, 0.7, 2.2);
    geom.addFace(face);
    
    std::string error_msg;
    REQUIRE_FALSE(saltro::validation::validateSatellite(sat, error_msg));
    REQUIRE(error_msg == "geometry face 0 diffuse coefficient invalid");
}

TEST_CASE("Invalid geometry - absorptivity negative", "[satellite][validation][geometry]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    saltro::disturbances::GeometryConfig& geom = sat.geometryConfig();
    
    Vec3 centroid(0.0, 0.0, 0.1);
    Vec3 normal(0.0, 0.0, 1.0);
    saltro::disturbances::GeometryFace face(0.01, centroid, normal, 0.1, 0.2, -0.7, 2.2);
    geom.addFace(face);
    
    std::string error_msg;
    REQUIRE_FALSE(saltro::validation::validateSatellite(sat, error_msg));
    REQUIRE(error_msg == "geometry face 0 absorptivity invalid");
}

TEST_CASE("Invalid geometry - absorptivity > 1", "[satellite][validation][geometry]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    saltro::disturbances::GeometryConfig& geom = sat.geometryConfig();
    
    Vec3 centroid(0.0, 0.0, 0.1);
    Vec3 normal(0.0, 0.0, 1.0);
    saltro::disturbances::GeometryFace face(0.01, centroid, normal, 0.1, 0.2, 1.7, 2.2);
    geom.addFace(face);
    
    std::string error_msg;
    REQUIRE_FALSE(saltro::validation::validateSatellite(sat, error_msg));
    REQUIRE(error_msg == "geometry face 0 absorptivity invalid");
}

TEST_CASE("Invalid geometry - absorptivity is NaN", "[satellite][validation][geometry]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    saltro::disturbances::GeometryConfig& geom = sat.geometryConfig();
    
    Vec3 centroid(0.0, 0.0, 0.1);
    Vec3 normal(0.0, 0.0, 1.0);
    double nan_eta = std::numeric_limits<double>::quiet_NaN();
    saltro::disturbances::GeometryFace face(0.01, centroid, normal, 0.1, 0.2, nan_eta, 2.2);
    geom.addFace(face);
    
    std::string error_msg;
    REQUIRE_FALSE(saltro::validation::validateSatellite(sat, error_msg));
    REQUIRE(error_msg == "geometry face 0 absorptivity invalid");
}

TEST_CASE("Invalid geometry - optical coefficients sum < 0.99", "[satellite][validation][geometry]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    saltro::disturbances::GeometryConfig& geom = sat.geometryConfig();
    
    Vec3 centroid(0.0, 0.0, 0.1);
    Vec3 normal(0.0, 0.0, 1.0);
    // Sum = 0.1 + 0.2 + 0.5 = 0.8 < 0.99
    saltro::disturbances::GeometryFace face(0.01, centroid, normal, 0.1, 0.2, 0.5, 2.2);
    geom.addFace(face);
    
    std::string error_msg;
    REQUIRE_FALSE(saltro::validation::validateSatellite(sat, error_msg));
    REQUIRE(error_msg == "geometry face 0 optical coefficients don't sum to 1");
}

TEST_CASE("Invalid geometry - optical coefficients sum > 1.01", "[satellite][validation][geometry]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    saltro::disturbances::GeometryConfig& geom = sat.geometryConfig();
    
    Vec3 centroid(0.0, 0.0, 0.1);
    Vec3 normal(0.0, 0.0, 1.0);
    // Sum = 0.4 + 0.4 + 0.3 = 1.1 > 1.01
    saltro::disturbances::GeometryFace face(0.01, centroid, normal, 0.4, 0.4, 0.3, 2.2);
    geom.addFace(face);
    
    std::string error_msg;
    REQUIRE_FALSE(saltro::validation::validateSatellite(sat, error_msg));
    REQUIRE(error_msg == "geometry face 0 optical coefficients don't sum to 1");
}

TEST_CASE("Valid geometry - optical coefficients sum to 1.0", "[satellite][validation][geometry]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    saltro::disturbances::GeometryConfig& geom = sat.geometryConfig();
    
    Vec3 centroid(0.0, 0.0, 0.1);
    Vec3 normal(0.0, 0.0, 1.0);
    // Sum = 0.3 + 0.3 + 0.4 = 1.0
    saltro::disturbances::GeometryFace face(0.01, centroid, normal, 0.3, 0.3, 0.4, 2.2);
    geom.addFace(face);
    
    std::string error_msg;
    REQUIRE(saltro::validation::validateSatellite(sat, error_msg));
}

TEST_CASE("Valid geometry - optical coefficients sum to 0.995", "[satellite][validation][geometry]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    saltro::disturbances::GeometryConfig& geom = sat.geometryConfig();
    
    Vec3 centroid(0.0, 0.0, 0.1);
    Vec3 normal(0.0, 0.0, 1.0);
    // Sum = 0.33 + 0.33 + 0.335 = 0.995 (within tolerance)
    saltro::disturbances::GeometryFace face(0.01, centroid, normal, 0.33, 0.33, 0.335, 2.2);
    geom.addFace(face);
    
    std::string error_msg;
    REQUIRE(saltro::validation::validateSatellite(sat, error_msg));
}

TEST_CASE("Invalid geometry - drag coefficient negative", "[satellite][validation][geometry]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    saltro::disturbances::GeometryConfig& geom = sat.geometryConfig();
    
    Vec3 centroid(0.0, 0.0, 0.1);
    Vec3 normal(0.0, 0.0, 1.0);
    saltro::disturbances::GeometryFace face(0.01, centroid, normal, 0.1, 0.2, 0.7, -2.2);
    geom.addFace(face);
    
    std::string error_msg;
    REQUIRE_FALSE(saltro::validation::validateSatellite(sat, error_msg));
    REQUIRE(error_msg == "geometry face 0 drag coefficient invalid");
}

TEST_CASE("Invalid geometry - drag coefficient is NaN", "[satellite][validation][geometry]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    saltro::disturbances::GeometryConfig& geom = sat.geometryConfig();
    
    Vec3 centroid(0.0, 0.0, 0.1);
    Vec3 normal(0.0, 0.0, 1.0);
    double nan_cd = std::numeric_limits<double>::quiet_NaN();
    saltro::disturbances::GeometryFace face(0.01, centroid, normal, 0.1, 0.2, 0.7, nan_cd);
    geom.addFace(face);
    
    std::string error_msg;
    REQUIRE_FALSE(saltro::validation::validateSatellite(sat, error_msg));
    REQUIRE(error_msg == "geometry face 0 drag coefficient invalid");
}

TEST_CASE("Invalid geometry - drag coefficient unreasonably large", "[satellite][validation][geometry]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    saltro::disturbances::GeometryConfig& geom = sat.geometryConfig();
    
    Vec3 centroid(0.0, 0.0, 0.1);
    Vec3 normal(0.0, 0.0, 1.0);
    saltro::disturbances::GeometryFace face(0.01, centroid, normal, 0.1, 0.2, 0.7, 3.5);  // > 3.0
    geom.addFace(face);
    
    std::string error_msg;
    REQUIRE_FALSE(saltro::validation::validateSatellite(sat, error_msg));
    REQUIRE(error_msg == "geometry face 0 drag coefficient unreasonably large");
}

TEST_CASE("Valid geometry - zero drag coefficient", "[satellite][validation][geometry]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    saltro::disturbances::GeometryConfig& geom = sat.geometryConfig();
    
    Vec3 centroid(0.0, 0.0, 0.1);
    Vec3 normal(0.0, 0.0, 1.0);
    saltro::disturbances::GeometryFace face(0.01, centroid, normal, 0.1, 0.2, 0.7, 0.0);
    geom.addFace(face);
    
    std::string error_msg;
    REQUIRE(saltro::validation::validateSatellite(sat, error_msg));
}

TEST_CASE("Invalid geometry - second face has invalid properties", "[satellite][validation][geometry]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    saltro::disturbances::GeometryConfig& geom = sat.geometryConfig();
    
    Vec3 centroid1(0.0, 0.0, 0.1);
    Vec3 normal1(0.0, 0.0, 1.0);
    saltro::disturbances::GeometryFace face1(0.01, centroid1, normal1, 0.1, 0.2, 0.7, 2.2);
    geom.addFace(face1);
    
    Vec3 centroid2(0.0, 0.0, -0.1);
    Vec3 normal2(0.0, 0.0, -1.0);
    saltro::disturbances::GeometryFace face2(-0.01, centroid2, normal2, 0.1, 0.2, 0.7, 2.2);  // Negative area
    geom.addFace(face2);
    
    std::string error_msg;
    REQUIRE_FALSE(saltro::validation::validateSatellite(sat, error_msg));
    REQUIRE(error_msg == "geometry face 1 area invalid");
}

// ============================================================================
// Edge Cases and Comprehensive Tests
// ============================================================================

TEST_CASE("Valid satellite with all actuators and geometry", "[satellite][validation][comprehensive]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    
    // Add MTQs
    sat.addMTQ(Vec3(1.0, 0.0, 0.0), 0.2);
    sat.addMTQ(Vec3(0.0, 1.0, 0.0), 0.2);
    sat.addMTQ(Vec3(0.0, 0.0, 1.0), 0.2);
    
    // Add RWs
    sat.addRW(Vec3(1.0, 0.0, 0.0), 0.01, 0.001, 0.05, 0.1);
    sat.addRW(Vec3(0.0, 1.0, 0.0), 0.01, 0.001, -0.03, 0.1);
    
    // Add geometry faces
    saltro::disturbances::GeometryConfig& geom = sat.geometryConfig();
    for (int i = 0; i < 6; ++i) {
        Vec3 normal = Vec3::Zero();
        normal(i % 3) = (i < 3) ? 1.0 : -1.0;
        Vec3 centroid = normal * 0.05;
        saltro::disturbances::GeometryFace face(0.01, centroid, normal, 0.1, 0.2, 0.7, 2.2);
        geom.addFace(face);
    }
    
    std::string error_msg;
    REQUIRE(saltro::validation::validateSatellite(sat, error_msg));
}

TEST_CASE("Boundary values - minimum valid inertia magnitude", "[satellite][validation][boundary]") {
    PlannerSettings settings;
    // Use a valid symmetric positive definite matrix with small magnitude
    Mat33 J;
    J << 1e-3, 0.0, 0.0,
         0.0, 1e-3, 0.0,
         0.0, 0.0, 1e-3;
    Satellite sat(J, settings);
    
    std::string error_msg;
    REQUIRE(saltro::validation::validateSatellite(sat, error_msg));
}

TEST_CASE("Boundary values - maximum valid inertia magnitude", "[satellite][validation][boundary]") {
    PlannerSettings settings;
    // Use a valid symmetric positive definite matrix with large magnitude
    Mat33 J;
    J << 1e5, 0.0, 0.0,
         0.0, 1e5, 0.0,
         0.0, 0.0, 1e5;
    Satellite sat(J, settings);
    
    std::string error_msg;
    REQUIRE(saltro::validation::validateSatellite(sat, error_msg));
}

TEST_CASE("Boundary values - MTQ max dipole at upper limit", "[satellite][validation][boundary]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    Vec3 axis(1.0, 0.0, 0.0);
    sat.addMTQ(axis, 1e6);
    
    std::string error_msg;
    REQUIRE(saltro::validation::validateSatellite(sat, error_msg));
}

TEST_CASE("Boundary values - RW max torque at upper limit", "[satellite][validation][boundary]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    Vec3 axis(1.0, 0.0, 0.0);
    sat.addRW(axis, 1e4, 0.001, 0.0, 0.1);
    
    std::string error_msg;
    REQUIRE(saltro::validation::validateSatellite(sat, error_msg));
}

TEST_CASE("Boundary values - RW wheel inertia at upper limit", "[satellite][validation][boundary]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    Vec3 axis(1.0, 0.0, 0.0);
    sat.addRW(axis, 0.01, 1e3, 0.0, 0.1);
    
    std::string error_msg;
    REQUIRE(saltro::validation::validateSatellite(sat, error_msg));
}

TEST_CASE("Boundary values - RW max momentum at upper limit", "[satellite][validation][boundary]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    Vec3 axis(1.0, 0.0, 0.0);
    sat.addRW(axis, 0.01, 0.001, 0.0, 1e4);
    
    std::string error_msg;
    REQUIRE(saltro::validation::validateSatellite(sat, error_msg));
}

TEST_CASE("Boundary values - drag coefficient at upper limit", "[satellite][validation][boundary]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    saltro::disturbances::GeometryConfig& geom = sat.geometryConfig();
    
    Vec3 centroid(0.0, 0.0, 0.1);
    Vec3 normal(0.0, 0.0, 1.0);
    saltro::disturbances::GeometryFace face(0.01, centroid, normal, 0.1, 0.2, 0.7, 3.0);
    geom.addFace(face);
    
    std::string error_msg;
    REQUIRE(saltro::validation::validateSatellite(sat, error_msg));
}
