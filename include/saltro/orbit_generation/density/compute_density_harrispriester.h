#pragma once

#include <Eigen/Dense>
#include <saltro/limits.h>

namespace saltro::orbits {

/**
 * @brief Compute atmospheric density using the Harris–Priester model.
 *
 * Evaluates the atmospheric mass density along a spacecraft trajectory using
 * the Harris–Priester empirical thermospheric density model. The model provides
 * a simple representation of upper-atmosphere density as a function of altitude
 * and solar position relative to the spacecraft.
 *
 * For each trajectory sample \f$k\f$, the density is computed as:
 * \f[
 * \rho_k = \rho_{\mathrm{HP}}\!\left(\mathbf{R}_k, \mathbf{S}_k\right)
 * \f]
 *
 * where:
 * - \f$\mathbf{R}_k\f$ is the spacecraft position vector (meters),
 * - \f$\mathbf{S}_k\f$ is the Sun direction vector relative to the spacecraft,
 * - \f$\rho_k\f$ is the resulting atmospheric density (kg/m³).
 *
 * The Harris–Priester model accounts for the diurnal bulge in the atmosphere by
 * interpolating between minimum and maximum density profiles based on the angle
 * between the spacecraft position and the Sun direction:
 * \f[
 * \rho_k = \rho_{\min}(h_k) +
 * \bigl(\rho_{\max}(h_k) - \rho_{\min}(h_k)\bigr)\,f(\psi_k)
 * \f]
 *
 * where \f$h_k\f$ is altitude and \f$\psi_k\f$ is the solar zenith angle. The
 * weighting function \f$f(\psi_k)\f$ models the day–night variation of density.
 *
 * The output density is written as a row vector:
 * \f[
 * \boldsymbol{\rho} =
 * \begin{bmatrix}
 * \rho_0 & \rho_1 & \cdots & \rho_{N-1}
 * \end{bmatrix}
 * \f]
 *
 * @param R Spacecraft position vectors (meters). Each column corresponds to one
 *          trajectory sample in an Earth-centered frame.
 * @param S Sun direction vectors (meters or unit vectors). Each column corresponds
 *          to the Sun direction at the matching trajectory sample.
 * @param jtime_length Number of valid trajectory samples.
 * @param rho Output atmospheric density values (kg/m³) for each trajectory sample.
 *
 * @return True if density computation succeeds for all samples, false otherwise.
 */
bool compute_density_harrispriester(
    const Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& R,
    const Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& S,
    const int jtime_length,
    Eigen::Matrix<double, 1, saltro::limits::MAX_LENGTH_TRAJ>& rho
);

}