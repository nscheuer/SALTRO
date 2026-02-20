#pragma once

#include <Eigen/Dense>
#include <saltro/limits.h>

namespace saltro::orbits {

/**
 * @brief Compute geomagnetic field along a trajectory using a selected model.
 *
 * Dispatches magnetic field computation to one of the available geomagnetic
 * models based on the integer selector \p magnetic_model.
 *
 * For each trajectory sample \f$k\f$, the magnetic field is evaluated as:
 * \f[
 * \mathbf{B}_k =
 * \mathbf{B}_{\text{model}}\!\left(\mathbf{R}_k, t_k\right)
 * \f]
 *
 * where:
 * - \f$\mathbf{R}_k\f$ is the spacecraft position vector (meters),
 * - \f$t_k\f$ is the corresponding Julian time,
 * - \f$\mathbf{B}_k\f$ is the magnetic field vector (Tesla).
 *
 * Supported models:
 * - \f$0\f$ → Tilted dipole approximation  
 * - \f$1\f$ → IGRF-8 spherical harmonic model  
 * - \f$2\f$ → IGRF-13 spherical harmonic model  
 *
 * The output magnetic field is written column-wise:
 * \f[
 * \mathbf{B} =
 * \begin{bmatrix}
 * \mathbf{B}_0 & \mathbf{B}_1 & \cdots & \mathbf{B}_{N-1}
 * \end{bmatrix}
 * \f]
 *
 * This function performs model selection only; the physical computation is
 * delegated to the corresponding model implementation.
 *
 * @param R Spacecraft position vectors (meters). Each column corresponds
 *          to one trajectory sample.
 * @param jtime Julian time values associated with each trajectory sample.
 * @param jtime_length Number of valid trajectory samples.
 * @param magnetic_model Integer identifier selecting the magnetic model:
 *        0 = tilted dipole, 1 = IGRF-8, 2 = IGRF-13.
 * @param B Output magnetic field vectors (Tesla), column-wise.
 *
 * @return True if the selected model computation succeeds for all samples,
 *         false if the model identifier is invalid or a computation fails.
 */
bool compute_magnetic(
    const Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& R,
    const Eigen::Matrix<double, 1, saltro::limits::MAX_LENGTH_TRAJ>& jtime,
    const int jtime_length,
    const int magnetic_model,

    Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& B
);

}