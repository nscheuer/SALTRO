#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Dense>
#include <saltro/pybind/satellite.h>
#include <saltro/limits.h>
#include <stdexcept>
#include <cmath>

using Mat33 = Eigen::Matrix3d;
using Vec3 = Eigen::Vector3d;

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

static Mat33 singularInertiaMatrix() {
    Mat33 J;
    J << 1.0, 2.0, 3.0,
         2.0, 4.0, 6.0,
         3.0, 6.0, 9.0;
    return J;
}

static Mat33 infiniteInertiaMatrix() {
    Mat33 J = validInertiaMatrix();
    J(0, 0) = std::numeric_limits<double>::infinity();
    return J;
}

// ============================================================================
// Constructor Tests
// ============================================================================

TEST_CASE("Satellite default constructor", "[satellite][constructor]") {
    REQUIRE_NOTHROW(Satellite());
    
    Satellite sat;
    REQUIRE(sat.numMTQ() == 0);
    REQUIRE(sat.numRW() == 0);
    REQUIRE(sat.stateDim() == 7);
    REQUIRE(sat.reducedStateDim() == 6);
    REQUIRE(sat.controlDim() == 0);
    
    // Inertia should be identity
    REQUIRE(sat.inertia().isApprox(Mat33::Identity()));
    REQUIRE(sat.invInertia().isApprox(Mat33::Identity()));
}

TEST_CASE("Satellite constructor with valid inertia", "[satellite][constructor]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    
    REQUIRE_NOTHROW(Satellite(J, settings));
    
    Satellite sat(J, settings);
    REQUIRE(sat.inertia().isApprox(J));
    REQUIRE(sat.invInertia().isApprox(J.inverse()));
    REQUIRE(sat.numMTQ() == 0);
    REQUIRE(sat.numRW() == 0);
}

TEST_CASE("Satellite constructor with singular inertia", "[satellite][constructor]") {
    Mat33 J = singularInertiaMatrix();
    PlannerSettings settings;
    
    REQUIRE_THROWS_AS(Satellite(J, settings), std::invalid_argument);
}

TEST_CASE("Satellite constructor with infinite inertia", "[satellite][constructor]") {
    Mat33 J = infiniteInertiaMatrix();
    PlannerSettings settings;
    
    REQUIRE_THROWS_AS(Satellite(J, settings), std::invalid_argument);
}

// ============================================================================
// Inertia Management Tests
// ============================================================================

TEST_CASE("Satellite setInertia with valid matrix", "[satellite][inertia]") {
    Satellite sat;
    Mat33 J = validInertiaMatrix();
    
    REQUIRE_NOTHROW(sat.setInertia(J));
    REQUIRE(sat.inertia().isApprox(J));
    REQUIRE(sat.invInertia().isApprox(J.inverse()));
}

TEST_CASE("Satellite setInertia with singular matrix", "[satellite][inertia]") {
    Satellite sat;
    Mat33 J = singularInertiaMatrix();
    
    REQUIRE_THROWS_AS(sat.setInertia(J), std::invalid_argument);
}

TEST_CASE("Satellite setInertia with non-finite entries", "[satellite][inertia]") {
    Satellite sat;
    Mat33 J = infiniteInertiaMatrix();
    
    REQUIRE_THROWS_AS(sat.setInertia(J), std::invalid_argument);
}

TEST_CASE("Satellite inertiaNoRW matches inertia when no RWs", "[satellite][inertia]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    
    REQUIRE(sat.inertiaNoRW().isApprox(J));
    REQUIRE(sat.invInertiaNoRW().isApprox(J.inverse()));
}

TEST_CASE("Satellite inertiaNoRW updates when RW is added", "[satellite][inertia]") {
    Mat33 J = validInertiaMatrix();
    PlannerSettings settings;
    Satellite sat(J, settings);
    
    Vec3 axis(1.0, 0.0, 0.0);
    double max_torque = 0.001;
    double J_rw = 1e-5;
    double h0 = 0.0;
    double h_max = 0.01;
    
    sat.addRW(axis, max_torque, J_rw, h0, h_max);
    
    // J_noRW should be J - J_rw * axis * axis^T
    Mat33 expected_J_noRW = J - J_rw * axis * axis.transpose();
    
    REQUIRE(sat.inertiaNoRW().isApprox(expected_J_noRW));
    REQUIRE(sat.invInertiaNoRW().isApprox(expected_J_noRW.inverse()));
}

// ============================================================================
// Adding Actuators Tests
// ============================================================================

TEST_CASE("Add single MTQ", "[satellite][actuators][mtq]") {
    Satellite sat;
    Vec3 axis(1.0, 0.0, 0.0);
    double max_dipole = 0.2;
    
    REQUIRE_NOTHROW(sat.addMTQ(axis, max_dipole));
    REQUIRE(sat.numMTQ() == 1);
    REQUIRE(sat.controlDim() == 1);
}

TEST_CASE("Add multiple MTQs", "[satellite][actuators][mtq]") {
    Satellite sat;
    
    sat.addMTQ(Vec3(1.0, 0.0, 0.0), 0.2);
    REQUIRE(sat.numMTQ() == 1);
    
    sat.addMTQ(Vec3(0.0, 1.0, 0.0), 0.2);
    REQUIRE(sat.numMTQ() == 2);
    
    sat.addMTQ(Vec3(0.0, 0.0, 1.0), 0.2);
    REQUIRE(sat.numMTQ() == 3);
    
    REQUIRE(sat.controlDim() == 3);
}

TEST_CASE("Add MTQ up to maximum limit", "[satellite][actuators][mtq][limits]") {
    Satellite sat;
    
    // Add MAX_NUM_MTQ (4) MTQs
    for (int i = 0; i < saltro::limits::MAX_NUM_MTQ; ++i) {
        Vec3 axis = Vec3::Zero();
        axis(i % 3) = 1.0;
        REQUIRE_NOTHROW(sat.addMTQ(axis, 0.2));
    }
    
    REQUIRE(sat.numMTQ() == saltro::limits::MAX_NUM_MTQ);
}

TEST_CASE("Add MTQ beyond maximum limit", "[satellite][actuators][mtq][limits]") {
    Satellite sat;
    
    // Add MAX_NUM_MTQ (4) MTQs
    for (int i = 0; i < saltro::limits::MAX_NUM_MTQ; ++i) {
        Vec3 axis = Vec3::Zero();
        axis(i % 3) = 1.0;
        sat.addMTQ(axis, 0.2);
    }
    
    // Try to add one more - should throw
    REQUIRE_THROWS_AS(sat.addMTQ(Vec3(1.0, 0.0, 0.0), 0.2), std::out_of_range);
}

TEST_CASE("Add single RW", "[satellite][actuators][rw]") {
    Satellite sat;
    Vec3 axis(1.0, 0.0, 0.0);
    double max_torque = 0.001;
    double J_rw = 1e-5;
    double h0 = 0.0;
    double h_max = 0.01;
    
    REQUIRE_NOTHROW(sat.addRW(axis, max_torque, J_rw, h0, h_max));
    REQUIRE(sat.numRW() == 1);
    REQUIRE(sat.stateDim() == 8);  // 7 + 1 RW
    REQUIRE(sat.reducedStateDim() == 7);  // 6 + 1 RW
    REQUIRE(sat.controlDim() == 1);
}

TEST_CASE("Add multiple RWs", "[satellite][actuators][rw]") {
    Satellite sat;
    double max_torque = 0.001;
    double J_rw = 1e-5;
    double h0 = 0.0;
    double h_max = 0.01;
    
    sat.addRW(Vec3(1.0, 0.0, 0.0), max_torque, J_rw, h0, h_max);
    REQUIRE(sat.numRW() == 1);
    REQUIRE(sat.stateDim() == 8);
    
    sat.addRW(Vec3(0.0, 1.0, 0.0), max_torque, J_rw, h0, h_max);
    REQUIRE(sat.numRW() == 2);
    REQUIRE(sat.stateDim() == 9);
    
    sat.addRW(Vec3(0.0, 0.0, 1.0), max_torque, J_rw, h0, h_max);
    REQUIRE(sat.numRW() == 3);
    REQUIRE(sat.stateDim() == 10);
}

TEST_CASE("Add RW up to maximum limit", "[satellite][actuators][rw][limits]") {
    Satellite sat;
    double max_torque = 0.001;
    double J_rw = 1e-5;
    double h0 = 0.0;
    double h_max = 0.01;
    
    // Add MAX_NUM_RW (4) RWs
    for (int i = 0; i < saltro::limits::MAX_NUM_RW; ++i) {
        Vec3 axis = Vec3::Zero();
        axis(i % 3) = 1.0;
        REQUIRE_NOTHROW(sat.addRW(axis, max_torque, J_rw, h0, h_max));
    }
    
    REQUIRE(sat.numRW() == saltro::limits::MAX_NUM_RW);
    REQUIRE(sat.stateDim() == 7 + saltro::limits::MAX_NUM_RW);
}

TEST_CASE("Add RW beyond maximum limit", "[satellite][actuators][rw][limits]") {
    Satellite sat;
    double max_torque = 0.001;
    double J_rw = 1e-5;
    double h0 = 0.0;
    double h_max = 0.01;
    
    // Add MAX_NUM_RW (4) RWs
    for (int i = 0; i < saltro::limits::MAX_NUM_RW; ++i) {
        Vec3 axis = Vec3::Zero();
        axis(i % 3) = 1.0;
        sat.addRW(axis, max_torque, J_rw, h0, h_max);
    }
    
    // Try to add one more - should throw
    REQUIRE_THROWS_AS(sat.addRW(Vec3(1.0, 0.0, 0.0), max_torque, J_rw, h0, h_max), 
                      std::out_of_range);
}

TEST_CASE("Add both MTQs and RWs", "[satellite][actuators]") {
    Satellite sat;
    
    sat.addMTQ(Vec3(1.0, 0.0, 0.0), 0.2);
    sat.addMTQ(Vec3(0.0, 1.0, 0.0), 0.2);
    sat.addRW(Vec3(0.0, 0.0, 1.0), 0.001, 1e-5, 0.0, 0.01);
    
    REQUIRE(sat.numMTQ() == 2);
    REQUIRE(sat.numRW() == 1);
    REQUIRE(sat.controlDim() == 3);  // 2 MTQs + 1 RW
    REQUIRE(sat.stateDim() == 8);    // 7 + 1 RW
}

// ============================================================================
// Getting Actuators Tests
// ============================================================================

TEST_CASE("Get MTQ with valid index", "[satellite][actuators][mtq]") {
    Satellite sat;
    Vec3 axis(1.0, 0.0, 0.0);
    double max_dipole = 0.2;
    
    sat.addMTQ(axis, max_dipole);
    
    REQUIRE_NOTHROW(sat.getMTQ(0));
    const MTQ& mtq = sat.getMTQ(0);
    REQUIRE(mtq.axis().isApprox(axis.normalized()));
    REQUIRE(mtq.u_max() == max_dipole);
}

TEST_CASE("Get MTQ with negative index", "[satellite][actuators][mtq]") {
    Satellite sat;
    sat.addMTQ(Vec3(1.0, 0.0, 0.0), 0.2);
    
    REQUIRE_THROWS_AS(sat.getMTQ(-1), std::out_of_range);
}

TEST_CASE("Get MTQ with index too large", "[satellite][actuators][mtq]") {
    Satellite sat;
    sat.addMTQ(Vec3(1.0, 0.0, 0.0), 0.2);
    
    REQUIRE_THROWS_AS(sat.getMTQ(1), std::out_of_range);
    REQUIRE_THROWS_AS(sat.getMTQ(5), std::out_of_range);
}

TEST_CASE("Get MTQ when no MTQs added", "[satellite][actuators][mtq]") {
    Satellite sat;
    
    REQUIRE_THROWS_AS(sat.getMTQ(0), std::out_of_range);
}

TEST_CASE("Get multiple MTQs with correct indices", "[satellite][actuators][mtq]") {
    Satellite sat;
    Vec3 axis1(1.0, 0.0, 0.0);
    Vec3 axis2(0.0, 1.0, 0.0);
    Vec3 axis3(0.0, 0.0, 1.0);
    double max_dipole = 0.2;
    
    sat.addMTQ(axis1, max_dipole);
    sat.addMTQ(axis2, max_dipole);
    sat.addMTQ(axis3, max_dipole);
    
    REQUIRE(sat.getMTQ(0).axis().isApprox(axis1.normalized()));
    REQUIRE(sat.getMTQ(1).axis().isApprox(axis2.normalized()));
    REQUIRE(sat.getMTQ(2).axis().isApprox(axis3.normalized()));
}

TEST_CASE("Get RW with valid index", "[satellite][actuators][rw]") {
    Satellite sat;
    Vec3 axis(1.0, 0.0, 0.0);
    double max_torque = 0.001;
    double J_rw = 1e-5;
    double h0 = 0.0;
    double h_max = 0.01;
    
    sat.addRW(axis, max_torque, J_rw, h0, h_max);
    
    REQUIRE_NOTHROW(sat.getRW(0));
    const RW& rw = sat.getRW(0);
    REQUIRE(rw.axis().isApprox(axis.normalized()));
    REQUIRE(rw.u_max() == max_torque);
    REQUIRE(rw.wheelInertia() == J_rw);
    REQUIRE(rw.momentum() == h0);
    REQUIRE(rw.momentumMax() == h_max);
}

TEST_CASE("Get RW with negative index", "[satellite][actuators][rw]") {
    Satellite sat;
    sat.addRW(Vec3(1.0, 0.0, 0.0), 0.001, 1e-5, 0.0, 0.01);
    
    REQUIRE_THROWS_AS(sat.getRW(-1), std::out_of_range);
}

TEST_CASE("Get RW with index too large", "[satellite][actuators][rw]") {
    Satellite sat;
    sat.addRW(Vec3(1.0, 0.0, 0.0), 0.001, 1e-5, 0.0, 0.01);
    
    REQUIRE_THROWS_AS(sat.getRW(1), std::out_of_range);
    REQUIRE_THROWS_AS(sat.getRW(5), std::out_of_range);
}

TEST_CASE("Get RW when no RWs added", "[satellite][actuators][rw]") {
    Satellite sat;
    
    REQUIRE_THROWS_AS(sat.getRW(0), std::out_of_range);
}

TEST_CASE("Get multiple RWs with correct indices", "[satellite][actuators][rw]") {
    Satellite sat;
    Vec3 axis1(1.0, 0.0, 0.0);
    Vec3 axis2(0.0, 1.0, 0.0);
    Vec3 axis3(0.0, 0.0, 1.0);
    double max_torque = 0.001;
    double J_rw = 1e-5;
    double h0 = 0.0;
    double h_max = 0.01;
    
    sat.addRW(axis1, max_torque, J_rw, h0, h_max);
    sat.addRW(axis2, max_torque, J_rw, h0, h_max);
    sat.addRW(axis3, max_torque, J_rw, h0, h_max);
    
    REQUIRE(sat.getRW(0).axis().isApprox(axis1.normalized()));
    REQUIRE(sat.getRW(1).axis().isApprox(axis2.normalized()));
    REQUIRE(sat.getRW(2).axis().isApprox(axis3.normalized()));
}

TEST_CASE("Modify MTQ through non-const reference", "[satellite][actuators][mtq]") {
    Satellite sat;
    sat.addMTQ(Vec3(1.0, 0.0, 0.0), 0.2);
    
    MTQ& mtq = sat.getMTQ(0);
    // MTQ doesn't have many modifiable properties, but we can verify we got a non-const reference
    REQUIRE(mtq.u_max() == 0.2);
}

TEST_CASE("Modify RW through non-const reference", "[satellite][actuators][rw]") {
    Satellite sat;
    sat.addRW(Vec3(1.0, 0.0, 0.0), 0.001, 1e-5, 0.0, 0.01);
    
    RW& rw = sat.getRW(0);
    REQUIRE(rw.momentum() == 0.0);
    
    rw.setMomentum(0.005);
    REQUIRE(rw.momentum() == 0.005);
    REQUIRE(sat.getRW(0).momentum() == 0.005);
}

// ============================================================================
// Dimension Calculations Tests
// ============================================================================

TEST_CASE("State dimension with no actuators", "[satellite][dimensions]") {
    Satellite sat;
    REQUIRE(sat.stateDim() == 7);
    REQUIRE(sat.reducedStateDim() == 6);
    REQUIRE(sat.controlDim() == 0);
}

TEST_CASE("State dimension with only MTQs", "[satellite][dimensions]") {
    Satellite sat;
    sat.addMTQ(Vec3(1.0, 0.0, 0.0), 0.2);
    sat.addMTQ(Vec3(0.0, 1.0, 0.0), 0.2);
    
    REQUIRE(sat.stateDim() == 7);  // MTQs don't add state
    REQUIRE(sat.reducedStateDim() == 6);
    REQUIRE(sat.controlDim() == 2);
}

TEST_CASE("State dimension with only RWs", "[satellite][dimensions]") {
    Satellite sat;
    sat.addRW(Vec3(1.0, 0.0, 0.0), 0.001, 1e-5, 0.0, 0.01);
    sat.addRW(Vec3(0.0, 1.0, 0.0), 0.001, 1e-5, 0.0, 0.01);
    
    REQUIRE(sat.stateDim() == 9);  // 7 + 2 RWs
    REQUIRE(sat.reducedStateDim() == 8);  // 6 + 2 RWs
    REQUIRE(sat.controlDim() == 2);
}

TEST_CASE("State dimension with mixed actuators", "[satellite][dimensions]") {
    Satellite sat;
    sat.addMTQ(Vec3(1.0, 0.0, 0.0), 0.2);
    sat.addMTQ(Vec3(0.0, 1.0, 0.0), 0.2);
    sat.addMTQ(Vec3(0.0, 0.0, 1.0), 0.2);
    sat.addRW(Vec3(1.0, 0.0, 0.0), 0.001, 1e-5, 0.0, 0.01);
    sat.addRW(Vec3(0.0, 1.0, 0.0), 0.001, 1e-5, 0.0, 0.01);
    
    REQUIRE(sat.stateDim() == 9);  // 7 + 2 RWs
    REQUIRE(sat.reducedStateDim() == 8);  // 6 + 2 RWs
    REQUIRE(sat.controlDim() == 5);  // 3 MTQs + 2 RWs
}

TEST_CASE("State dimension at maximum actuators", "[satellite][dimensions]") {
    Satellite sat;
    
    // Add max MTQs
    for (int i = 0; i < saltro::limits::MAX_NUM_MTQ; ++i) {
        Vec3 axis = Vec3::Zero();
        axis(i % 3) = 1.0;
        sat.addMTQ(axis, 0.2);
    }
    
    // Add max RWs
    for (int i = 0; i < saltro::limits::MAX_NUM_RW; ++i) {
        Vec3 axis = Vec3::Zero();
        axis(i % 3) = 1.0;
        sat.addRW(axis, 0.001, 1e-5, 0.0, 0.01);
    }
    
    REQUIRE(sat.stateDim() == 7 + saltro::limits::MAX_NUM_RW);
    REQUIRE(sat.reducedStateDim() == 6 + saltro::limits::MAX_NUM_RW);
    REQUIRE(sat.controlDim() == saltro::limits::MAX_NUM_MTQ + saltro::limits::MAX_NUM_RW);
}

// ============================================================================
// Settings and Geometry Config Tests
// ============================================================================

TEST_CASE("Set and get settings", "[satellite][settings]") {
    Satellite sat;
    PlannerSettings settings;
    
    REQUIRE_NOTHROW(sat.setSettings(settings));
    // Just verify we can get settings back (content depends on PlannerSettings implementation)
    REQUIRE_NOTHROW(sat.settings());
}

TEST_CASE("Set and get geometry config", "[satellite][geometry]") {
    Satellite sat;
    saltro::disturbances::GeometryConfig config;
    
    REQUIRE_NOTHROW(sat.setGeometryConfig(config));
    // Verify we can get it back
    REQUIRE_NOTHROW(sat.geometryConfig());
}

TEST_CASE("Modify geometry config through non-const reference", "[satellite][geometry]") {
    Satellite sat;
    saltro::disturbances::GeometryConfig config;
    sat.setGeometryConfig(config);
    
    // Get non-const reference and verify it's accessible
    REQUIRE_NOTHROW(sat.geometryConfig());
}

// ============================================================================
// State Indices Tests
// ============================================================================

TEST_CASE("State index constants", "[satellite][indices]") {
    REQUIRE(Satellite::AV_INDEX == 0);
    REQUIRE(Satellite::QUAT_INDEX == 3);
    REQUIRE(Satellite::RW_MOMENTUM_INDEX == 7);
}

// ============================================================================
// Edge Cases and Complex Scenarios
// ============================================================================

TEST_CASE("Add actuators with non-normalized axes", "[satellite][actuators]") {
    Satellite sat;
    Vec3 axis_unnormalized(2.0, 0.0, 0.0);  // Not unit length
    
    // Should work - actuator should normalize internally
    REQUIRE_NOTHROW(sat.addMTQ(axis_unnormalized, 0.2));
    REQUIRE_NOTHROW(sat.addRW(axis_unnormalized, 0.001, 1e-5, 0.0, 0.01));
    
    // Verify axes are normalized
    REQUIRE(std::abs(sat.getMTQ(0).axis().norm() - 1.0) < 1e-10);
    REQUIRE(std::abs(sat.getRW(0).axis().norm() - 1.0) < 1e-10);
}

TEST_CASE("Inertia update with multiple RWs", "[satellite][inertia]") {
    Mat33 J = validInertiaMatrix();
    Satellite sat(J, PlannerSettings());
    
    double J_rw = 1e-5;
    sat.addRW(Vec3(1.0, 0.0, 0.0), 0.001, J_rw, 0.0, 0.01);
    sat.addRW(Vec3(0.0, 1.0, 0.0), 0.001, J_rw, 0.0, 0.01);
    sat.addRW(Vec3(0.0, 0.0, 1.0), 0.001, J_rw, 0.0, 0.01);
    
    // J_noRW should be J - sum of J_rw * axis * axis^T for all RWs
    Mat33 expected_J_noRW = J;
    expected_J_noRW(0, 0) -= J_rw;  // RW along x-axis
    expected_J_noRW(1, 1) -= J_rw;  // RW along y-axis
    expected_J_noRW(2, 2) -= J_rw;  // RW along z-axis
    
    REQUIRE(sat.inertiaNoRW().isApprox(expected_J_noRW, 1e-10));
}

TEST_CASE("Different actuator configurations", "[satellite][actuators]") {
    Satellite sat;
    
    // Configuration 1: 3 MTQs, 1 RW
    sat.addMTQ(Vec3(1.0, 0.0, 0.0), 0.2);
    sat.addMTQ(Vec3(0.0, 1.0, 0.0), 0.2);
    sat.addMTQ(Vec3(0.0, 0.0, 1.0), 0.2);
    sat.addRW(Vec3(1.0, 0.0, 0.0), 0.001, 1e-5, 0.0, 0.01);
    
    REQUIRE(sat.numMTQ() == 3);
    REQUIRE(sat.numRW() == 1);
    REQUIRE(sat.controlDim() == 4);
    REQUIRE(sat.stateDim() == 8);
}

TEST_CASE("Zero maximum dipole MTQ", "[satellite][actuators][mtq]") {
    Satellite sat;
    
    // Should be allowed - actuator decides validity
    REQUIRE_NOTHROW(sat.addMTQ(Vec3(1.0, 0.0, 0.0), 0.0));
    REQUIRE(sat.getMTQ(0).u_max() == 0.0);
}

TEST_CASE("Zero maximum torque RW", "[satellite][actuators][rw]") {
    Satellite sat;
    
    // Should be allowed - actuator decides validity
    REQUIRE_NOTHROW(sat.addRW(Vec3(1.0, 0.0, 0.0), 0.0, 1e-5, 0.0, 0.01));
    REQUIRE(sat.getRW(0).u_max() == 0.0);
}

TEST_CASE("RW with different initial momentum values", "[satellite][actuators][rw]") {
    Satellite sat;
    Vec3 axis(1.0, 0.0, 0.0);
    double max_torque = 0.001;
    double J_rw = 1e-5;
    double h_max = 0.01;
    
    sat.addRW(axis, max_torque, J_rw, 0.0, h_max);
    sat.addRW(axis, max_torque, J_rw, 0.005, h_max);
    sat.addRW(axis, max_torque, J_rw, -0.003, h_max);
    
    REQUIRE(sat.getRW(0).momentum() == 0.0);
    REQUIRE(sat.getRW(1).momentum() == 0.005);
    REQUIRE(sat.getRW(2).momentum() == -0.003);
}
