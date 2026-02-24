#pragma once

#include <Eigen/Dense>
#include <saltro/limits.h>

namespace saltro::orbits {

/**
 * @brief Compute Earth eclipse shadow using analytical penumbra/umbra model.
 *
 * Uses an analytical shadow cone model based on apparent angular sizes of
 * Earth and Sun as seen from the spacecraft. The spacecraft is considered
 * in eclipse (penumbra or umbra) if the angular separation between the
 * Earth and Sun centers is smaller than the sum of their apparent angular
 * radii. When in eclipse, the Sun vector is set to zero.
 *
 * @param R Spacecraft position vectors (meters). Each column corresponds
 *          to one trajectory sample.
 * @param jtime_length Number of valid trajectory samples.
 * @param S Spacecraft-to-Sun vectors (meters), column-wise. Modified in-place;
 *          set to zero where spacecraft is in eclipse.
 *
 * @return True if the computation succeeds for all samples, false otherwise.
 */
bool compute_eclipse_penumbra(
    const Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& R,
    const int jtime_length,

    Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& S
);

}
