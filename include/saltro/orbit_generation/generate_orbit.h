#pragma once

#include <Eigen/Dense>

#include <saltro/limits.h>

namespace saltro::orbits {

/**
 * @brief Generate orbital state and environment data along a trajectory.
 *
 * High-level driver that propagates a spacecraft orbit and evaluates
 * environmental quantities (magnetic field, Sun vector, and atmospheric
 * density) at each requested time sample. The function dispatches to the
 * selected orbit, magnetic, solar, and density models based on the integer
 * selector flags.
 *
 * Given initial position \f$\mathbf{r}_0\f$ and velocity \f$\mathbf{v}_0\f$,
 * the spacecraft state is propagated to each time sample \f$t_k\f$:
 * \f[
 * \bigl(\mathbf{R}_k, \mathbf{V}_k\bigr)
 * =
 * \Phi_{\text{orbit}}\!\left(\mathbf{r}_0, \mathbf{v}_0, t_k\right)
 * \f]
 *
 * At each propagated state, the following quantities are evaluated:
 *
 * - Magnetic field:
 * \f[
 * \mathbf{B}_k =
 * \mathbf{B}_{\text{model}}\!\left(\mathbf{R}_k, t_k\right)
 * \f]
 *
 * - Spacecraft-to-Sun vector:
 * \f[
 * \mathbf{S}_k =
 * \mathbf{r}_{\odot}(t_k) - \mathbf{R}_k
 * \f]
 *
 * - Atmospheric density:
 * \f[
 * \rho_k =
 * \rho_{\text{model}}\!\left(\mathbf{R}_k, \mathbf{S}_k\right)
 * \f]
 *
 * The results are written column-wise:
 * \f[
 * \mathbf{R} =
 * \begin{bmatrix}
 * \mathbf{R}_0 & \cdots & \mathbf{R}_{N-1}
 * \end{bmatrix},\quad
 * \mathbf{V} =
 * \begin{bmatrix}
 * \mathbf{V}_0 & \cdots & \mathbf{V}_{N-1}
 * \end{bmatrix},\quad
 * \mathbf{B} =
 * \begin{bmatrix}
 * \mathbf{B}_0 & \cdots & \mathbf{B}_{N-1}
 * \end{bmatrix},\quad
 * \mathbf{S} =
 * \begin{bmatrix}
 * \mathbf{S}_0 & \cdots & \mathbf{S}_{N-1}
 * \end{bmatrix}
 * \f]
 *
 * and density as:
 * \f[
 * \boldsymbol{\rho} =
 * \begin{bmatrix}
 * \rho_0 & \cdots & \rho_{N-1}
 * \end{bmatrix}
 * \f]
 *
 * Supported model identifiers:
 * - Orbit model:
 *   - 0 → Keplerian two-body propagation
 *   - 1 → J2 perturbation with RK4 integration
 * - Magnetic model:
 *   - 0 → Tilted dipole
 *   - 1 → IGRF-8
 *   - 2 → IGRF-13
 * - Sun model:
 *   - 0 → NOAA
 *   - 1 → NREL SPA
 * - Density model:
 *   - 0 → Harris–Priester
 *
 * This function performs input validation and orchestrates the individual
 * model calls. Each model computation is applied sequentially along the
 * trajectory samples.
 *
 * @param r0 Initial position vector in an inertial frame (meters).
 * @param v0 Initial velocity vector in an inertial frame (m/s).
 * @param jtime Julian centuries corresponding to each trajectory sample.
 * @param jtime_length Number of valid trajectory samples.
 * @param orbit_model Orbit model selector.
 * @param magnetic_model Magnetic field model selector.
 * @param sun_model Solar position model selector.
 * @param density_model Atmospheric density model selector.
 * @param R Output propagated position vectors (meters), column-wise.
 * @param V Output propagated velocity vectors (m/s), column-wise.
 * @param B Output magnetic field vectors (Tesla), column-wise.
 * @param S Output spacecraft-to-Sun vectors (meters), column-wise.
 * @param rho Output atmospheric density values (kg/m³).
 *
 * @return True if all model computations succeed, false if any stage fails.
 */
bool generate_orbit(
    const Eigen::Vector3d& r0,
    const Eigen::Vector3d& v0,
    const Eigen::Matrix<double, 1, saltro::limits::MAX_LENGTH_TRAJ>& jtime,
    const int jtime_length,
    const int orbit_model,
    const int magnetic_model,
    const int sun_model,
    const int density_model,

    Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& R,
    Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& V,
    Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& B,
    Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& S,
    Eigen::Matrix<double, 1, saltro::limits::MAX_LENGTH_TRAJ>& rho
);

}