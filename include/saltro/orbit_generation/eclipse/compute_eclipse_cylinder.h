#pragma once

#include <Eigen/Dense>
#include <saltro/limits.h>

namespace saltro::orbits {

/**
 * @brief Compute Earth eclipse shadow using cylindrical shadow model.
 *
 * Implements a cylindrical shadow model where Earth's shadow is approximated
 * as a cylinder perpendicular to the Sun direction. The satellite is considered
 * in eclipse (umbra) if:
 *
 * 1. The satellite is on the night side of Earth (dot product with sun direction < 0)
 * 2. The perpendicular distance from the satellite position to the sun direction
 *    is less than Earth's mean radius
 *
 * When a satellite is in eclipse, the Sun vector is set to zero. This model is
 * computationally efficient and accurate for LEO altitudes.
 *
 * @param R Spacecraft position vectors (meters). Each column corresponds
 *          to one trajectory sample.
 * @param jtime_length Number of valid trajectory samples.
 * @param S Spacecraft-to-Sun vectors (meters), column-wise. Modified in-place;
 *          set to zero where spacecraft is in eclipse.
 *
 * @return True if the computation succeeds for all samples, false otherwise.
 */
bool compute_eclipse_cylinder(
    const Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& R,
    const int jtime_length,

    Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& S
);

}
