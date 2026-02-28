#pragma once

#include <Eigen/Dense>
#include <string>

namespace saltro::validation {

/**
 * @brief Validate a mission time vector expressed in Julian centuries.
 *
 * The mission timeline is expected to be within [0.20, 0.40] Julian centuries
 * and strictly increasing.
 *
 * Validation checks:
 * - vector is not empty
 * - each value is finite (not NaN or infinity)
 * - no value is zero
 * - each value lies within mission bounds [0.20, 0.40]
 * - values are strictly increasing
 *
 * @param jtime Time vector in Julian centuries.
 * @param error_msg Output error message when validation fails.
 * @return true if valid, false otherwise.
 */
bool validateJulianTime(
    const Eigen::Ref<const Eigen::VectorXd>& jtime,
    std::string& error_msg
);

} // namespace saltro::validation
