#pragma once

#include <Eigen/Dense>
#include <saltro/limits.h>

namespace saltro::orbits {

/**
 * @brief Compute Sun position vectors using the NREL Solar Position Algorithm (SPA).
 *
 * Evaluates the Sun direction and distance relative to the spacecraft for
 * each trajectory sample using the NREL Solar Position Algorithm (SPA). This
 * model computes the apparent solar position with higher fidelity than simple
 * analytical approximations, accounting for Earth rotation, orbital geometry,
 * and solar ephemerides.
 *
 * For each trajectory sample \f$k\f$, the spacecraft-to-Sun vector is computed as:
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
 * The NREL SPA computes the Sun’s apparent right ascension and declination
 * from Julian centuries \f$T\f$, including corrections for nutation,
 * aberration, and Earth orbital motion. The resulting vectors include the
 * Sun–Earth distance (on the order of 1 AU).
 *
 * The output Sun vector matrix is written column-wise:
 * \f[
 * \mathbf{S} =
 * \begin{bmatrix}
 * \mathbf{S}_0 & \mathbf{S}_1 & \cdots & \mathbf{S}_{N-1}
 * \end{bmatrix}
 * \f]
 *
 * @param R Spacecraft position vectors (meters). Each column corresponds to
 *          one trajectory sample.
 * @param jtime Julian centuries associated with each trajectory sample.
 * @param jtime_length Number of valid trajectory samples.
 * @param S Output spacecraft-to-Sun vectors (meters), column-wise.
 *
 * @return True if the computation succeeds for all samples, false otherwise.
 */
bool compute_sun_nrelspa(
    Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& R,
    const Eigen::Matrix<double, 1, saltro::limits::MAX_LENGTH_TRAJ>& jtime,
    const int jtime_length,
    Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& S
);

}