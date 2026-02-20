#pragma once

#include <Eigen/Dense>
#include <saltro/limits.h>

namespace saltro::orbits {

/**
 * @brief Propagate an orbit using a selected dynamical model.
 *
 * Dispatches orbital state propagation to one of the available orbit
 * models based on the integer selector \p orbit_model.
 *
 * Given initial position \f$\mathbf{r}_0\f$ and velocity
 * \f$\mathbf{v}_0\f$, the state is propagated to each requested time
 * sample \f$t_k\f$:
 * \f[
 * \bigl(\mathbf{R}_k, \mathbf{V}_k\bigr)
 * =
 * \Phi_{\text{model}}\!\left(\mathbf{r}_0, \mathbf{v}_0, t_k\right)
 * \f]
 *
 * where:
 * - \f$\mathbf{R}_k\f$ is the propagated position vector (meters),
 * - \f$\mathbf{V}_k\f$ is the propagated velocity vector (m/s),
 * - \f$\Phi_{\text{model}}\f$ is the state transition defined by the
 *   selected orbit model.
 *
 * Supported models:
 * - \f$0\f$ → Two-body Keplerian propagation
 * - \f$1\f$ → J2 perturbation with RK4 integration
 *
 * The output state matrices are written column-wise:
 * \f[
 * \mathbf{R} =
 * \begin{bmatrix}
 * \mathbf{R}_0 & \mathbf{R}_1 & \cdots & \mathbf{R}_{N-1}
 * \end{bmatrix},
 * \qquad
 * \mathbf{V} =
 * \begin{bmatrix}
 * \mathbf{V}_0 & \mathbf{V}_1 & \cdots & \mathbf{V}_{N-1}
 * \end{bmatrix}
 * \f]
 *
 * This function performs model selection only; the dynamical propagation
 * is delegated to the corresponding orbit model implementation.
 *
 * @param r0 Initial position vector in an inertial frame (meters).
 * @param v0 Initial velocity vector in an inertial frame (m/s).
 * @param jtime Julian centuries corresponding to each trajectory sample.
 * @param jtime_length Number of valid trajectory samples.
 * @param orbit_model Integer identifier selecting the orbit model:
 *        0 = Keplerian two-body propagation.
 * @param R Output propagated position vectors (meters), column-wise.
 * @param V Output propagated velocity vectors (m/s), column-wise.
 *
 * @return True if the selected model propagation succeeds for all samples,
 *         false if the model identifier is invalid or a computation fails.
 */
bool compute_orbit(
    const Eigen::Vector3d& r0,
    const Eigen::Vector3d& v0,
    const Eigen::Matrix<double, 1, saltro::limits::MAX_LENGTH_TRAJ>& jtime,
    const int jtime_length,
    const int orbit_model,

    Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& R,
    Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& V
);

}