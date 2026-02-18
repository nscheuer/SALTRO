#pragma once

#include <Eigen/Dense>
#include <saltro/limits.h>

namespace saltro::orbits {

/**
 * @brief Compute geomagnetic field using a tilted dipole model.
 *
 * Computes the Earth magnetic field vector along a trajectory using a
 * simplified tilted dipole approximation of the geomagnetic field.
 *
 * For each trajectory sample \(k\), the magnetic field is evaluated as:
 * \f[
 * \mathbf{B}_k = \mathbf{B}_{\text{dipole}}\!\left(\mathbf{R}_k, t_k\right)
 * \f]
 *
 * where \f$\mathbf{R}_k\f$ is the position in an Earth-centered frame and
 * \f$t_k\f$ is the corresponding Julian time. The dipole model assumes a
 * centered magnetic dipole that is tilted with respect to the Earth’s
 * rotation axis.
 *
 * The output field is written column-wise:
 * \f[
 * \mathbf{B} =
 * \begin{bmatrix}
 * \mathbf{B}_0 & \mathbf{B}_1 & \cdots & \mathbf{B}_{N-1}
 * \end{bmatrix}
 * \f]
 *
 * @param R Position vectors of the trajectory in ECEF/ECI frame (meters).
 *          Each column \f$\mathbf{R}_k\f$ corresponds to one time sample.
 * @param jtime Julian time values associated with each trajectory sample.
 * @param jtime_length Number of valid trajectory samples.
 * @param B Output magnetic field vectors (Tesla). Each column corresponds
 *          to the field at the matching trajectory sample.
 *
 * @return True if the computation succeeds for all samples, false otherwise.
 */
bool compute_magnetic_dipole(
    const Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& R,
    const Eigen::Matrix<double, 1, saltro::limits::MAX_LENGTH_TRAJ>& jtime,
    const int jtime_length,

    Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& B
);

}