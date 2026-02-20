# pragma once

#include <Eigen/Dense>
#include <saltro/limits.h>

namespace saltro::orbits {

/**
 * @brief Compute Sun position vectors along a trajectory using a selected model.
 *
 * Dispatches Sun position computation to one of the available solar ephemeris
 * models based on the integer selector \p sun_model.
 *
 * For each trajectory sample \f$k\f$, the spacecraft-to-Sun vector is evaluated as:
 * \f[
 * \mathbf{S}_k =
 * \mathbf{r}_{\odot}(t_k) - \mathbf{R}_k
 * \f]
 *
 * where:
 * - \f$\mathbf{r}_{\odot}(t_k)\f$ is the Sun position in an Earth-centered
 *   inertial frame at time \f$t_k\f$,
 * - \f$\mathbf{R}_k\f$ is the spacecraft position vector (meters),
 * - \f$\mathbf{S}_k\f$ is the resulting spacecraft-to-Sun vector (meters).
 *
 * Supported models:
 * - \f$0\f$ → NOAA solar position algorithm  
 * - \f$1\f$ → NREL Solar Position Algorithm (SPA)  
 *
 * The output Sun vector matrix is written column-wise:
 * \f[
 * \mathbf{S} =
 * \begin{bmatrix}
 * \mathbf{S}_0 & \mathbf{S}_1 & \cdots & \mathbf{S}_{N-1}
 * \end{bmatrix}
 * \f]
 *
 * This function performs model selection only; the physical solar ephemeris
 * computation is delegated to the corresponding model implementation.
 *
 * @param R Spacecraft position vectors (meters). Each column corresponds to
 *          one trajectory sample.
 * @param jtime Julian centuries associated with each trajectory sample.
 * @param jtime_length Number of valid trajectory samples.
 * @param sun_model Integer identifier selecting the solar model:
 *        0 = NOAA, 1 = NREL SPA.
 * @param S Output spacecraft-to-Sun vectors (meters), column-wise.
 *
 * @return True if the selected model computation succeeds for all samples,
 *         false if the model identifier is invalid or a computation fails.
 */
bool compute_sun(
    Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& R,
    const Eigen::Matrix<double, 1, saltro::limits::MAX_LENGTH_TRAJ>& jtime,
    const int jtime_length,
    const int sun_model,
    Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& S
);

}