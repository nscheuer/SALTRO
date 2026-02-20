#pragma once
#include <Eigen/Dense>
#include <saltro/limits.h>

namespace saltro::orbits {

/**
 * @brief Compute geomagnetic field using the IGRF-13 model.
 *
 * Evaluates the Earth magnetic field along a trajectory using the
 * International Geomagnetic Reference Field (IGRF-13) spherical harmonic
 * model.
 *
 * For each trajectory sample \(k\):
 * \f[
 * \mathbf{B}_k =
 * \mathbf{B}_{\text{IGRF13}}\!\left(\mathbf{R}_k, t_k\right)
 * \f]
 *
 * where:
 * - \f$\mathbf{R}_k\f$ is the position vector,
 * - \f$t_k\f$ is the Julian time.
 *
 * The IGRF-13 model includes updated Gauss coefficients and secular
 * variation terms to represent the geomagnetic field evolution.
 *
 * @param R Position vectors of the trajectory (meters). Columns correspond
 *          to time samples.
 * @param jtime Julian time values associated with each sample.
 * @param jtime_length Number of valid trajectory samples.
 * @param B Output magnetic field vectors (Tesla).
 *
 * @return True if the computation succeeds, false otherwise.
 */
bool compute_magnetic_igrf13(
    const Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& R,
    const Eigen::Matrix<double, 1, saltro::limits::MAX_LENGTH_TRAJ>& jtime,
    const int jtime_length,

    Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& B
);

}