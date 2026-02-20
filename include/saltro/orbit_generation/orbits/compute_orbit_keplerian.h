#pragma once

#include <Eigen/Dense>
#include <saltro/limits.h>

namespace saltro::orbits {

/**
 * @brief Propagate an orbit using Keplerian two-body dynamics.
 *
 * Computes the spacecraft position and velocity along a trajectory by
 * propagating an initial Cartesian state under the Keplerian two-body
 * assumption (central gravitational field, no perturbations).
 *
 * Given initial state \f$(\mathbf{r}_0,\mathbf{v}_0)\f$ at epoch
 * \f$t_0\f$, the state at each time sample \f$t_k\f$ is obtained from
 * Kepler’s equations:
 * \f[
 * \mathbf{r}_k = \mathbf{r}(t_k;\mathbf{r}_0,\mathbf{v}_0), \qquad
 * \mathbf{v}_k = \mathbf{v}(t_k;\mathbf{r}_0,\mathbf{v}_0)
 * \f]
 *
 * The resulting trajectory matrices are filled column-wise:
 * \f[
 * \mathbf{R} =
 * \begin{bmatrix}
 * \mathbf{r}_0 & \mathbf{r}_1 & \cdots & \mathbf{r}_{N-1}
 * \end{bmatrix}, \qquad
 * \mathbf{V} =
 * \begin{bmatrix}
 * \mathbf{v}_0 & \mathbf{v}_1 & \cdots & \mathbf{v}_{N-1}
 * \end{bmatrix}
 * \f]
 *
 * @param r0 Initial position vector in an inertial frame (meters).
 * @param v0 Initial velocity vector in an inertial frame (m/s).
 * @param jtime Julian time values at which the orbit should be evaluated.
 * @param jtime_length Number of valid time samples.
 * @param R Output position vectors (meters), column-wise.
 * @param V Output velocity vectors (m/s), column-wise.
 *
 * @return True if propagation succeeds for all samples, false otherwise.
 */
bool compute_orbit_keplerian(
    const Eigen::Vector3d& r0,
    const Eigen::Vector3d& v0,
    const Eigen::Matrix<double, 1, saltro::limits::MAX_LENGTH_TRAJ>& jtime,
    const int jtime_length,

    Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& R,
    Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& V
);

}