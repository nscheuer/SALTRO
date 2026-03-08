#pragma once

#include <Eigen/Dense>

namespace saltro::math {

/**
 * @brief Compute the quaternion error: q_err = q_goal^{-1} ⊗ q.
 *
 * Returns the relative rotation from q to q_goal. If q == q_goal,
 * the result is [1, 0, 0, 0] (identity quaternion).
 *
 * Convention: scalar-first [q0, q1, q2, q3].
 *
 * @param q_goal Goal quaternion (unit).
 * @param q      Current quaternion (unit).
 * @return Quaternion error (unit), sign-chosen so q0 >= 0.
 */
Eigen::Vector4d quatError(const Eigen::Vector4d& q_goal, const Eigen::Vector4d& q);

/**
 * @brief Convert a small quaternion error to Modified Rodrigues Parameters.
 *
 * Given q_err = [q0, q1, q2, q3], the MRP is:
 * \f[
 *   \boldsymbol{\sigma} = \frac{2 \, \mathbf{q}_{1:3}}{1 + q_0}
 * \f]
 *
 * For small rotations, MRP ≈ rotation vector (angle × axis).
 *
 * @param q_err Quaternion error (unit, q0 >= 0).
 * @return 3D MRP vector.
 */
Eigen::Vector3d quatToMRP(const Eigen::Vector4d& q_err);

/**
 * @brief Build the G projection matrix from full state to reduced state.
 *
 * The full state is [ω(3), q(4), h_rw(nRW)] with dimension 7+nRW.
 * The reduced state is [ω(3), δθ(3), h_rw(nRW)] with dimension 6+nRW.
 *
 * The G matrix maps full-state perturbations to reduced-state perturbations:
 * \f[
 *   G = \begin{bmatrix}
 *     I_3 & 0 & 0 \\
 *     0 & W^T(q) & 0 \\
 *     0 & 0 & I_{nRW}
 *   \end{bmatrix}
 * \f]
 * where W(q) is the 4×3 quaternion kinematics matrix, so W^T is 3×4.
 *
 * G has shape (6+nRW) × (7+nRW).
 *
 * @param q    Unit quaternion at the linearization point.
 * @param nRW  Number of reaction wheels.
 * @return G matrix of size (6+nRW) × (7+nRW).
 */
Eigen::MatrixXd findGMat(const Eigen::Vector4d& q, int nRW);

} // namespace saltro::math
