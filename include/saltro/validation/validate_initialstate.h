#pragma once

#include <Eigen/Dense>
#include <string>

namespace saltro::validation {

/**
 * @brief Validate the initial state vector for trajectory optimization.
 * 
 * The state vector x0 contains:
 * - [0:3] Angular velocity (rad/s) in body frame
 * - [3:7] Quaternion (q0, q1, q2, q3) representing attitude
 * - [7:] (optional) Reaction wheel angular momenta
 * 
 * This function validates:
 * 1. Angular velocity components:
 *    - Are finite (not NaN or infinity)
 *    - Are within reasonable limits (< 10 rad/s)
 * 
 * 2. Quaternion components:
 *    - Are finite (not NaN)
 *    - Are normalized: |norm - 1.0| <= 1e-6
 * 
 * @param x0 Initial state vector (minimum size 7)
 * @param error_msg Output string describing validation failure (if any)
 * @return true if state is valid, false otherwise
 * 
 * @throws std::invalid_argument if x0 has fewer than 7 elements
 */
bool validateInitialState(
    const Eigen::Ref<const Eigen::VectorXd>& x0,
    std::string& error_msg
);

} // namespace saltro::validation
