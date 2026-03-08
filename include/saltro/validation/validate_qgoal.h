#pragma once

#include <Eigen/Dense>
#include <string>

namespace saltro::validation {

/**
 * @brief Validate attitude goal matrix q_goal.
 *
 * Expected shape is (4, N), one goal per column.
 * For each column, two formats are allowed:
 * - Quaternion goal: [q0, qx, qy, qz] (all finite, norm ~= 1)
 * - ECI vector goal: [NaN, x, y, z] (x,y,z finite, norm ~= 1)
 *
 * Mixed column types are allowed.
 *
 * Validation checks include:
 * - matrix has 4 rows and at least 1 column
 * - rows 2..4 never contain NaN/Inf
 * - quaternion goals are finite and normalized
 * - ECI vector goals are finite and normalized
 *
 * @param q_goal Goal matrix of shape (4, N).
 * @param error_msg Output error message on failure.
 * @return true if valid, false otherwise.
 */
bool validateQGoal(
    const Eigen::Ref<const Eigen::MatrixXd>& q_goal,
    std::string& error_msg
);

} // namespace saltro::validation
