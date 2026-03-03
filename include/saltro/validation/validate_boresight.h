#pragma once

#include <Eigen/Dense>
#include <string>

namespace saltro::validation {

/**
 * @brief Validate boresight history matrix.
 *
 * Expected shape is (3, N), one boresight direction per column.
 * Each column must be a finite unit vector (norm ~= 1).
 *
 * Validation checks include:
 * - matrix has 3 rows and at least 1 column
 * - all elements are finite (no NaN/Inf)
 * - each column has norm ~= 1 (within tolerance)
 *
 * @param boresight Boresight history matrix of shape (3, N).
 * @param error_msg Output error message on failure.
 * @return true if valid, false otherwise.
 */
bool validateBoresight(
    const Eigen::Ref<const Eigen::MatrixXd>& boresight,
    std::string& error_msg
);

} // namespace saltro::validation
