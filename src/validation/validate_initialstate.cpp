#include <saltro/validation/validate_initialstate.h>

#include <cmath>
#include <limits>

namespace saltro::validation {

bool validateInitialState(
    const Eigen::Ref<const Eigen::VectorXd>& x0,
    std::string& error_msg
) {
    // Check minimum state dimension (3 angular velocity + 4 quaternion)
    if (x0.size() < 7) {
        error_msg = "initial state x0 is too small";
        return false;
    }

    // ========================================================================
    // Angular Velocity Validation (first 3 components)
    // ========================================================================
    
    // Check that angular velocity components are finite
    for (int i = 0; i < 3; ++i) {
        if (!std::isfinite(x0(i))) {
            error_msg = "angular velocity component " + std::to_string(i) + " is not finite";
            return false;
        }
    }

    // Check that angular velocity magnitude is reasonable (<= 10 rad/s)
    double omega_mag = x0.segment<3>(0).norm();
    if (omega_mag >= 10.0) {
        error_msg = "angular velocity magnitude unreasonably large";
        return false;
    }

    // ========================================================================
    // Quaternion Validation (components 3-6)
    // ========================================================================
    
    // Check that quaternion components are finite
    for (int i = 3; i < 7; ++i) {
        if (!std::isfinite(x0(i))) {
            error_msg = "quaternion component " + std::to_string(i - 3) + " is not finite";
            return false;
        }
    }

    // Check that quaternion is normalized (within tolerance)
    double quat_norm = x0.segment<4>(3).norm();
    constexpr double TOL = 1e-6;
    if (std::abs(quat_norm - 1.0) > TOL) {
        error_msg = "quaternion not normalized";
        return false;
    }

    // ========================================================================
    // Reaction Wheel Momentum Validation (if present)
    // ========================================================================
    // Optional: validate RW momenta if they exist (size > 7)
    for (int i = 7; i < x0.size(); ++i) {
        if (!std::isfinite(x0(i))) {
            error_msg = "reaction wheel momentum " + std::to_string(i - 7) + " is not finite";
            return false;
        }
    }

    return true;
}

} // namespace saltro::validation
