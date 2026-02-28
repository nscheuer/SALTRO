#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <Eigen/Dense>
#include <saltro/validation/validate_initialstate.h>
#include <cmath>
#include <string>
#include <limits>

using VecX = Eigen::VectorXd;
using Vec3 = Eigen::Vector3d;
using Vec4 = Eigen::Vector4d;

// ============================================================================
// NOTE ON TEST DESIGN
// ============================================================================
// The validateInitialState function checks:
//   1. State vector has minimum size 7 (3 angular velocity + 4 quaternion)
//   2. Angular velocity components are finite and magnitude < 10 rad/s
//   3. Quaternion components are finite and normalized (|norm - 1.0| <= 1e-6)
//   4. Reaction wheel momenta (if present) are finite
// ============================================================================

// ============================================================================
// Helper Functions
// ============================================================================

static VecX validInitialState() {
    VecX x0(7);
    x0 << 0.1, 0.05, -0.02,  // angular velocity
          1.0, 0.0, 0.0, 0.0; // quaternion (normalized)
    return x0;
}

static VecX validInitialStateWithRW() {
    VecX x0(9);
    x0 << 0.1, 0.05, -0.02,  // angular velocity
          1.0, 0.0, 0.0, 0.0, // quaternion
          0.05, -0.03;         // RW momenta
    return x0;
}

// ============================================================================
// Basic Validation Tests
// ============================================================================

TEST_CASE("Valid initial state passes validation", "[initialstate][validation]") {
    VecX x0 = validInitialState();
    std::string error_msg;
    
    bool is_valid = saltro::validation::validateInitialState(x0, error_msg);
    
    REQUIRE(is_valid);
    REQUIRE(error_msg.empty());
}

TEST_CASE("Valid initial state with RW passes validation", "[initialstate][validation]") {
    VecX x0 = validInitialStateWithRW();
    std::string error_msg;
    
    bool is_valid = saltro::validation::validateInitialState(x0, error_msg);
    
    REQUIRE(is_valid);
    REQUIRE(error_msg.empty());
}

TEST_CASE("Valid state with zero angular velocity", "[initialstate][validation]") {
    VecX x0(7);
    x0 << 0.0, 0.0, 0.0,     // zero angular velocity (valid)
          1.0, 0.0, 0.0, 0.0; // quaternion
    std::string error_msg;
    
    bool is_valid = saltro::validation::validateInitialState(x0, error_msg);
    
    REQUIRE(is_valid);
}

TEST_CASE("Valid state with normalized quaternion", "[initialstate][validation]") {
    VecX x0(7);
    // Unnormalized, then normalize it
    Vec4 q_unnorm;
    q_unnorm << 2.0, 1.0, 0.5, -0.3;
    Vec4 q_norm = q_unnorm.normalized();
    x0 << 0.1, 0.05, -0.02,    // angular velocity
          q_norm(0), q_norm(1), q_norm(2), q_norm(3); // normalized quaternion
    std::string error_msg;
    
    bool is_valid = saltro::validation::validateInitialState(x0, error_msg);
    
    REQUIRE(is_valid);
}

TEST_CASE("Invalid state with unnormalized quaternion", "[initialstate][validation]") {
    VecX x0(7);
    x0 << 0.1, 0.05, -0.02,    // angular velocity
          2.0, 1.0, 0.5, -0.3; // unnormalized quaternion (should fail)
    std::string error_msg;
    
    bool is_valid = saltro::validation::validateInitialState(x0, error_msg);
    
    REQUIRE_FALSE(is_valid);
    REQUIRE(error_msg == "quaternion not normalized");
}

// ============================================================================
// State Dimension Validation Tests
// ============================================================================

TEST_CASE("State too small fails validation", "[initialstate][validation]") {
    VecX x0(6);
    x0 << 0.1, 0.05, -0.02,  // angular velocity
          1.0, 0.0, 0.0;      // incomplete quaternion
    std::string error_msg;
    
    bool is_valid = saltro::validation::validateInitialState(x0, error_msg);
    
    REQUIRE_FALSE(is_valid);
    REQUIRE(error_msg == "initial state x0 is too small");
}

TEST_CASE("Empty state fails validation", "[initialstate][validation]") {
    VecX x0(0);
    std::string error_msg;
    
    bool is_valid = saltro::validation::validateInitialState(x0, error_msg);
    
    REQUIRE_FALSE(is_valid);
    REQUIRE(error_msg == "initial state x0 is too small");
}

// ============================================================================
// Angular Velocity Validation Tests
// ============================================================================

TEST_CASE("Angular velocity with NaN in first component", "[initialstate][validation]") {
    VecX x0 = validInitialState();
    x0(0) = std::numeric_limits<double>::quiet_NaN();
    std::string error_msg;
    
    bool is_valid = saltro::validation::validateInitialState(x0, error_msg);
    
    REQUIRE_FALSE(is_valid);
    REQUIRE(error_msg == "angular velocity component 0 is not finite");
}

TEST_CASE("Angular velocity with NaN in second component", "[initialstate][validation]") {
    VecX x0 = validInitialState();
    x0(1) = std::numeric_limits<double>::quiet_NaN();
    std::string error_msg;
    
    bool is_valid = saltro::validation::validateInitialState(x0, error_msg);
    
    REQUIRE_FALSE(is_valid);
    REQUIRE(error_msg == "angular velocity component 1 is not finite");
}

TEST_CASE("Angular velocity with NaN in third component", "[initialstate][validation]") {
    VecX x0 = validInitialState();
    x0(2) = std::numeric_limits<double>::quiet_NaN();
    std::string error_msg;
    
    bool is_valid = saltro::validation::validateInitialState(x0, error_msg);
    
    REQUIRE_FALSE(is_valid);
    REQUIRE(error_msg == "angular velocity component 2 is not finite");
}

TEST_CASE("Angular velocity with positive infinity", "[initialstate][validation]") {
    VecX x0 = validInitialState();
    x0(0) = std::numeric_limits<double>::infinity();
    std::string error_msg;
    
    bool is_valid = saltro::validation::validateInitialState(x0, error_msg);
    
    REQUIRE_FALSE(is_valid);
    REQUIRE(error_msg == "angular velocity component 0 is not finite");
}

TEST_CASE("Angular velocity with negative infinity", "[initialstate][validation]") {
    VecX x0 = validInitialState();
    x0(1) = -std::numeric_limits<double>::infinity();
    std::string error_msg;
    
    bool is_valid = saltro::validation::validateInitialState(x0, error_msg);
    
    REQUIRE_FALSE(is_valid);
    REQUIRE(error_msg == "angular velocity component 1 is not finite");
}

TEST_CASE("Angular velocity magnitude unreasonably large", "[initialstate][validation]") {
    VecX x0(7);
    x0 << 8.0, 7.0, 6.0,     // magnitude = sqrt(149) > 10
          1.0, 0.0, 0.0, 0.0;
    std::string error_msg;
    
    bool is_valid = saltro::validation::validateInitialState(x0, error_msg);
    
    REQUIRE_FALSE(is_valid);
    REQUIRE(error_msg == "angular velocity magnitude unreasonably large");
}

TEST_CASE("Angular velocity magnitude exactly at limit", "[initialstate][validation][boundary]") {
    VecX x0(7);
    x0 << 10.0, 0.0, 0.0,    // magnitude = 10.0 (should fail: >= not >)
          1.0, 0.0, 0.0, 0.0;
    std::string error_msg;
    
    bool is_valid = saltro::validation::validateInitialState(x0, error_msg);
    
    REQUIRE_FALSE(is_valid);
    REQUIRE(error_msg == "angular velocity magnitude unreasonably large");
}

TEST_CASE("Angular velocity magnitude just below limit", "[initialstate][validation][boundary]") {
    VecX x0(7);
    x0 << 9.99, 0.0, 0.0,    // magnitude = 9.99 < 10.0
          1.0, 0.0, 0.0, 0.0;
    std::string error_msg;
    
    bool is_valid = saltro::validation::validateInitialState(x0, error_msg);
    
    REQUIRE(is_valid);
}

TEST_CASE("Large but reasonable angular velocity", "[initialstate][validation]") {
    VecX x0(7);
    x0 << 5.0, 5.0, 5.0,     // magnitude = sqrt(75) ≈ 8.66 < 10
          1.0, 0.0, 0.0, 0.0;
    std::string error_msg;
    
    bool is_valid = saltro::validation::validateInitialState(x0, error_msg);
    
    REQUIRE(is_valid);
}

// ============================================================================
// Quaternion Validation Tests
// ============================================================================

TEST_CASE("Quaternion with NaN in first component", "[initialstate][validation]") {
    VecX x0 = validInitialState();
    x0(3) = std::numeric_limits<double>::quiet_NaN();
    std::string error_msg;
    
    bool is_valid = saltro::validation::validateInitialState(x0, error_msg);
    
    REQUIRE_FALSE(is_valid);
    REQUIRE(error_msg == "quaternion component 0 is not finite");
}

TEST_CASE("Quaternion with NaN in second component", "[initialstate][validation]") {
    VecX x0 = validInitialState();
    x0(4) = std::numeric_limits<double>::quiet_NaN();
    std::string error_msg;
    
    bool is_valid = saltro::validation::validateInitialState(x0, error_msg);
    
    REQUIRE_FALSE(is_valid);
    REQUIRE(error_msg == "quaternion component 1 is not finite");
}

TEST_CASE("Quaternion with NaN in third component", "[initialstate][validation]") {
    VecX x0 = validInitialState();
    x0(5) = std::numeric_limits<double>::quiet_NaN();
    std::string error_msg;
    
    bool is_valid = saltro::validation::validateInitialState(x0, error_msg);
    
    REQUIRE_FALSE(is_valid);
    REQUIRE(error_msg == "quaternion component 2 is not finite");
}

TEST_CASE("Quaternion with NaN in fourth component", "[initialstate][validation]") {
    VecX x0 = validInitialState();
    x0(6) = std::numeric_limits<double>::quiet_NaN();
    std::string error_msg;
    
    bool is_valid = saltro::validation::validateInitialState(x0, error_msg);
    
    REQUIRE_FALSE(is_valid);
    REQUIRE(error_msg == "quaternion component 3 is not finite");
}

TEST_CASE("Quaternion with positive infinity", "[initialstate][validation]") {
    VecX x0 = validInitialState();
    x0(3) = std::numeric_limits<double>::infinity();
    std::string error_msg;
    
    bool is_valid = saltro::validation::validateInitialState(x0, error_msg);
    
    REQUIRE_FALSE(is_valid);
    REQUIRE(error_msg == "quaternion component 0 is not finite");
}

TEST_CASE("Quaternion with negative infinity", "[initialstate][validation]") {
    VecX x0 = validInitialState();
    x0(5) = -std::numeric_limits<double>::infinity();
    std::string error_msg;
    
    bool is_valid = saltro::validation::validateInitialState(x0, error_msg);
    
    REQUIRE_FALSE(is_valid);
    REQUIRE(error_msg == "quaternion component 2 is not finite");
}

TEST_CASE("Quaternion all zeros", "[initialstate][validation]") {
    VecX x0(7);
    x0 << 0.1, 0.05, -0.02,  // angular velocity
          0.0, 0.0, 0.0, 0.0; // all zero quaternion
    std::string error_msg;
    
    bool is_valid = saltro::validation::validateInitialState(x0, error_msg);
    
    REQUIRE_FALSE(is_valid);
    REQUIRE(error_msg == "quaternion not normalized");
}

TEST_CASE("Quaternion normalized within positive tolerance", "[initialstate][validation][boundary]") {
    VecX x0(7);
    x0 << 0.1, 0.05, -0.02,  // angular velocity
          1.0 + 9e-7, 0.0, 0.0, 0.0; // norm = 1.0 + 9e-7 (within 1e-6 tolerance)
    std::string error_msg;
    
    bool is_valid = saltro::validation::validateInitialState(x0, error_msg);
    
    REQUIRE(is_valid);
}

TEST_CASE("Quaternion normalized within negative tolerance", "[initialstate][validation][boundary]") {
    VecX x0(7);
    x0 << 0.1, 0.05, -0.02,  // angular velocity
          1.0 - 9e-7, 0.0, 0.0, 0.0; // norm = 1.0 - 9e-7 (within 1e-6 tolerance)
    std::string error_msg;
    
    bool is_valid = saltro::validation::validateInitialState(x0, error_msg);
    
    REQUIRE(is_valid);
}

TEST_CASE("Quaternion norm slightly above tolerance", "[initialstate][validation][boundary]") {
    VecX x0(7);
    x0 << 0.1, 0.05, -0.02,     // angular velocity
          1.0 + 2e-6, 0.0, 0.0, 0.0; // norm = 1.0 + 2e-6 (outside 1e-6 tolerance)
    std::string error_msg;
    
    bool is_valid = saltro::validation::validateInitialState(x0, error_msg);
    
    REQUIRE_FALSE(is_valid);
    REQUIRE(error_msg == "quaternion not normalized");
}

TEST_CASE("Quaternion norm slightly below tolerance", "[initialstate][validation][boundary]") {
    VecX x0(7);
    x0 << 0.1, 0.05, -0.02,     // angular velocity
          1.0 - 2e-6, 0.0, 0.0, 0.0; // norm = 1.0 - 2e-6 (outside 1e-6 tolerance)
    std::string error_msg;
    
    bool is_valid = saltro::validation::validateInitialState(x0, error_msg);
    
    REQUIRE_FALSE(is_valid);
    REQUIRE(error_msg == "quaternion not normalized");
}

TEST_CASE("Quaternion with very small norm", "[initialstate][validation]") {
    VecX x0(7);
    x0 << 0.1, 0.05, -0.02,  // angular velocity
          1e-100, 0.0, 0.0, 0.0; // very small quaternion (should fail)
    std::string error_msg;
    
    bool is_valid = saltro::validation::validateInitialState(x0, error_msg);
    
    REQUIRE_FALSE(is_valid);
    REQUIRE(error_msg == "quaternion not normalized");
}

TEST_CASE("Quaternion with large norm", "[initialstate][validation]") {
    VecX x0(7);
    x0 << 0.1, 0.05, -0.02,     // angular velocity
          100.0, 50.0, -30.0, 20.0; // large unnormalized quaternion (should fail)
    std::string error_msg;
    
    bool is_valid = saltro::validation::validateInitialState(x0, error_msg);
    
    REQUIRE_FALSE(is_valid);
    REQUIRE(error_msg == "quaternion not normalized");
}

// ============================================================================
// Reaction Wheel Momentum Validation Tests
// ============================================================================

TEST_CASE("RW momentum with NaN in first wheel", "[initialstate][validation]") {
    VecX x0 = validInitialStateWithRW();
    x0(7) = std::numeric_limits<double>::quiet_NaN();
    std::string error_msg;
    
    bool is_valid = saltro::validation::validateInitialState(x0, error_msg);
    
    REQUIRE_FALSE(is_valid);
    REQUIRE(error_msg == "reaction wheel momentum 0 is not finite");
}

TEST_CASE("RW momentum with NaN in second wheel", "[initialstate][validation]") {
    VecX x0 = validInitialStateWithRW();
    x0(8) = std::numeric_limits<double>::quiet_NaN();
    std::string error_msg;
    
    bool is_valid = saltro::validation::validateInitialState(x0, error_msg);
    
    REQUIRE_FALSE(is_valid);
    REQUIRE(error_msg == "reaction wheel momentum 1 is not finite");
}

TEST_CASE("RW momentum with positive infinity", "[initialstate][validation]") {
    VecX x0 = validInitialStateWithRW();
    x0(7) = std::numeric_limits<double>::infinity();
    std::string error_msg;
    
    bool is_valid = saltro::validation::validateInitialState(x0, error_msg);
    
    REQUIRE_FALSE(is_valid);
    REQUIRE(error_msg == "reaction wheel momentum 0 is not finite");
}

TEST_CASE("RW momentum with negative infinity", "[initialstate][validation]") {
    VecX x0 = validInitialStateWithRW();
    x0(8) = -std::numeric_limits<double>::infinity();
    std::string error_msg;
    
    bool is_valid = saltro::validation::validateInitialState(x0, error_msg);
    
    REQUIRE_FALSE(is_valid);
    REQUIRE(error_msg == "reaction wheel momentum 1 is not finite");
}

TEST_CASE("Valid state with multiple RWs", "[initialstate][validation]") {
    VecX x0(11);  // 7 + 4 RWs
    x0 << 0.1, 0.05, -0.02,     // angular velocity
          1.0, 0.0, 0.0, 0.0,    // quaternion
          0.05, -0.03, 0.02, -0.01; // 4 RW momenta
    std::string error_msg;
    
    bool is_valid = saltro::validation::validateInitialState(x0, error_msg);
    
    REQUIRE(is_valid);
}

TEST_CASE("Valid state with zero RW momentum", "[initialstate][validation]") {
    VecX x0(9);
    x0 << 0.1, 0.05, -0.02,  // angular velocity
          1.0, 0.0, 0.0, 0.0, // quaternion
          0.0, 0.0;            // zero RW momenta (valid)
    std::string error_msg;
    
    bool is_valid = saltro::validation::validateInitialState(x0, error_msg);
    
    REQUIRE(is_valid);
}

TEST_CASE("Valid state with large RW momentum", "[initialstate][validation]") {
    VecX x0(9);
    x0 << 0.1, 0.05, -0.02,  // angular velocity
          1.0, 0.0, 0.0, 0.0, // quaternion
          100.0, -50.0;        // large RW momenta (no upper limit in validation)
    std::string error_msg;
    
    bool is_valid = saltro::validation::validateInitialState(x0, error_msg);
    
    REQUIRE(is_valid);
}

// ============================================================================
// Comprehensive Edge Cases
// ============================================================================

TEST_CASE("All components at extreme valid values", "[initialstate][validation][comprehensive]") {
    VecX x0(11);
    Vec4 q_unnorm;
    q_unnorm << 1e6, -1e6, 1e6, -1e6;
    Vec4 q_norm = q_unnorm.normalized();
    x0 << 9.9, 0.0, 0.0,           // angular velocity near limit
          q_norm(0), q_norm(1), q_norm(2), q_norm(3),  // normalized quaternion
          1000.0, -1000.0, 500.0, -500.0; // large RW momenta
    std::string error_msg;
    
    bool is_valid = saltro::validation::validateInitialState(x0, error_msg);
    
    REQUIRE(is_valid);
}

TEST_CASE("Minimum valid state size", "[initialstate][validation]") {
    VecX x0(7);  // Exactly 7 elements
    x0 << 0.0, 0.0, 0.0,     // zero angular velocity
          1.0, 0.0, 0.0, 0.0; // identity quaternion
    std::string error_msg;
    
    bool is_valid = saltro::validation::validateInitialState(x0, error_msg);
    
    REQUIRE(is_valid);
}

TEST_CASE("Very large state vector", "[initialstate][validation]") {
    VecX x0(100);  // Unnecessarily large
    x0.setZero();
    x0.segment<3>(0) << 0.1, 0.05, -0.02;
    x0.segment<4>(3) << 1.0, 0.0, 0.0, 0.0;
    // All RW momenta are zero (valid)
    std::string error_msg;
    
    bool is_valid = saltro::validation::validateInitialState(x0, error_msg);
    
    REQUIRE(is_valid);
}

TEST_CASE("Mixed valid and invalid - NaN in RW but valid elsewhere", "[initialstate][validation]") {
    VecX x0(9);
    x0 << 0.1, 0.05, -0.02,  // valid angular velocity
          1.0, 0.0, 0.0, 0.0, // valid quaternion
          0.05, std::numeric_limits<double>::quiet_NaN(); // NaN in RW
    std::string error_msg;
    
    bool is_valid = saltro::validation::validateInitialState(x0, error_msg);
    
    REQUIRE_FALSE(is_valid);
    REQUIRE(error_msg == "reaction wheel momentum 1 is not finite");
}

TEST_CASE("Negative angular velocities are valid", "[initialstate][validation]") {
    VecX x0(7);
    x0 << -5.0, -3.0, -2.0,  // negative angular velocities (valid)
          1.0, 0.0, 0.0, 0.0;
    std::string error_msg;
    
    bool is_valid = saltro::validation::validateInitialState(x0, error_msg);
    
    REQUIRE(is_valid);
}

TEST_CASE("Negative quaternion components are valid", "[initialstate][validation]") {
    VecX x0(7);
    Vec4 q_unnorm;
    q_unnorm << -1.0, -0.5, -0.3, -0.2;
    Vec4 q_norm = q_unnorm.normalized();
    x0 << 0.1, 0.05, -0.02,
          q_norm(0), q_norm(1), q_norm(2), q_norm(3); // negative quaternion normalized (valid)
    std::string error_msg;
    
    bool is_valid = saltro::validation::validateInitialState(x0, error_msg);
    
    REQUIRE(is_valid);
}
