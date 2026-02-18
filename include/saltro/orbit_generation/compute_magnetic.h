#pragma once

#include <Eigen/Dense>
#include <saltro/limits.h>

namespace saltro::orbit {

/**
 * @brief Compute geomagnetic field along a trajectory using a selected model.
 *
 * Evaluates the Earth magnetic field for each trajectory sample using one
 * of several available geomagnetic models.
 *
 * For each sample \(k\):
 * \f[
 * \mathbf{B}_k =
 * \mathbf{B}_{\text{model}}\!\left(\mathbf{R}_k, t_k\right)
 * \f]
 *
 * where \f$\mathbf{R}_k\f$ is the position vector and \f$t_k\f$ is the
 * Julian time. The model used is determined by the @p magnetic_model flag.
 *
 * Supported models:
 * - 0: Tilted dipole approximation  
 * - 1: IGRF-8 spherical harmonic model  
 * - 2: IGRF-13 spherical harmonic model  
 *
 * The resulting magnetic field matrix is filled column-wise:
 * \f[
 * \mathbf{B} =
 * \begin{bmatrix}
 * \mathbf{B}_0 & \mathbf{B}_1 & \cdots & \mathbf{B}_{N-1}
 * \end{bmatrix}
 * \f]
 *
 * @param R Position vectors of the trajectory (meters). Each column is a
 *          sample position.
 * @param jtime Julian time values associated with each sample.
 * @param jtime_length Number of valid trajectory samples.
 * @param magnetic_model Integer identifier selecting the magnetic model:
 *        0 = tilted dipole, 1 = IGRF8, 2 = IGRF13.
 * @param B Output magnetic field vectors (Tesla), column-wise.
 *
 * @return True if the computation succeeds, false otherwise.
 */
bool compute_magnetic(
    const Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& R,
    const Eigen::Matrix<double, 1, saltro::limits::MAX_LENGTH_TRAJ>& jtime,
    const int jtime_length,
    const int magnetic_model,

    Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& B
);

}