#pragma once

#include <Eigen/Dense>
#include <saltro/limits.h>

namespace saltro::orbits {

/**
 * @brief Compute atmospheric density along a trajectory using a selected model.
 *
 * Dispatches atmospheric density computation to one of the available
 * density models based on the integer selector \p density_model.
 * Currently supported:
 *
 * - \f$0\f$ → Harris–Priester model
 *
 * For each trajectory sample \f$k\f$, the density is evaluated as:
 * \f[
 * \rho_k = \rho_{\text{model}}\!\left(\mathbf{R}_k, \mathbf{S}_k\right)
 * \f]
 *
 * where:
 * - \f$\mathbf{R}_k\f$ is the spacecraft position vector (meters),
 * - \f$\mathbf{S}_k\f$ is the Sun direction vector at the same sample,
 * - \f$\rho_k\f$ is the atmospheric density (kg/m³).
 *
 * The output is written as a row vector:
 * \f[
 * \boldsymbol{\rho} =
 * \begin{bmatrix}
 * \rho_0 & \rho_1 & \cdots & \rho_{N-1}
 * \end{bmatrix}
 * \f]
 *
 * This function performs model selection only; the physical computation
 * is delegated to the corresponding model implementation.
 *
 * @param R Spacecraft position vectors (meters). Each column corresponds
 *          to one trajectory sample.
 * @param S Sun direction vectors corresponding to each trajectory sample.
 * @param jtime_length Number of valid trajectory samples.
 * @param density_model Integer identifier selecting the density model.
 *                      Currently: 0 = Harris–Priester.
 * @param rho Output atmospheric density values (kg/m³).
 *
 * @return True if the selected model computation succeeds for all samples,
 *         false if the model identifier is invalid or a computation fails.
 */
bool compute_density(
    const Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& R,
    const Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& S,
    const int jtime_length,
    const int density_model,
    Eigen::Matrix<double, 1, saltro::limits::MAX_LENGTH_TRAJ>& rho
);

}