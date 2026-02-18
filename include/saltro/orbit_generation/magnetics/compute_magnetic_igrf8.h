#pragma once
#include <Eigen/Dense>
#include <saltro/limits.h>

namespace saltro::orbit {

/**
 * @brief Compute geomagnetic field using the IGRF-8 model.
 *
 * Evaluates the Earth magnetic field along a trajectory using the
 * International Geomagnetic Reference Field (IGRF-8) spherical harmonic
 * model.
 *
 * For each trajectory sample \(k\):
 * \f[
 * \mathbf{B}_k =
 * \mathbf{B}_{\text{IGRF8}}\!\left(\mathbf{R}_k, t_k\right)
 * \f]
 *
 * where \f$\mathbf{R}_k\f$ is the spacecraft position and \f$t_k\f$ is the
 * Julian time. The model accounts for the temporal evolution of Earth’s
 * magnetic field according to IGRF-8 coefficients.
 *
 * @param R Position vectors of the trajectory (meters). Each column is a
 *          sample position.
 * @param jtime Julian time values for each trajectory sample.
 * @param jtime_length Number of valid samples in the trajectory.
 * @param B Output magnetic field vectors (Tesla).
 *
 * @return True if computation completes successfully, false otherwise.
 */
bool compute_magnetic_igrf8(
    const Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& R,
    const Eigen::Matrix<double, 1, saltro::limits::MAX_LENGTH_TRAJ>& jtime,
    const int jtime_length,

    Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& B
);

}