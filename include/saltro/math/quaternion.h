#pragma once

#include <array>
#include <Eigen/Dense>

namespace saltro::math {

using Vec3 = Eigen::Vector3d;
using Vec4 = Eigen::Vector4d;
using Mat33 = Eigen::Matrix3d;
using Mat43 = Eigen::Matrix<double, 4, 3>;
using Mat44 = Eigen::Matrix<double, 4, 4>;

/**
 * @brief Normalize a quaternion to unit length.
 * 
 * Computes the normalized quaternion:
 * \f[
 * \hat{\mathbf{q}} = \frac{\mathbf{q}}{\|\mathbf{q}\|}
 * \f]
 * 
 * @param q Input quaternion (may be unnormalized).
 * @return Normalized unit quaternion.
 */
Vec4 normalizeQuat(const Vec4& q);

/**
 * @brief Compute the rotation matrix (DCM) from a unit quaternion.
 * 
 * Converts a unit quaternion \f$\mathbf{q} = [q_0, q_1, q_2, q_3]^T\f$ to a
 * 3×3 direction cosine matrix (rotation matrix):
 * \f[
 * \mathbf{C}(\mathbf{q}) = \mathbf{I} + 2q_0[\mathbf{q}_{1:3}]_\times + 
 * 2[\mathbf{q}_{1:3}]_\times^2
 * \f]
 * where \f$[\cdot]_\times\f$ denotes the skew-symmetric cross-product matrix.
 * 
 * @param q Unit quaternion (4D vector).
 * @return 3×3 rotation matrix.
 */
Mat33 rotationMatrix(const Vec4& q);

/**
 * @brief Compute the quaternion conjugate.
 *
 * For a unit quaternion, the conjugate equals the inverse:
 * \f[
 * \mathbf{q}^* = [q_0, -q_1, -q_2, -q_3]^T
 * \f]
 *
 * @param q Input quaternion.
 * @return Conjugate quaternion.
 */
Vec4 quatConj(const Vec4& q);

/**
 * @brief Hamilton product of two quaternions.
 *
 * Computes \f$\mathbf{q}_1 \otimes \mathbf{q}_2\f$ using Hamilton's convention,
 * scalar-first (\f$q = [q_0, \mathbf{q}_v]^T\f$).  Equivalent to the
 * left-multiplication matrix form \f$L(\mathbf{q}_1) \mathbf{q}_2\f$ where
 * \f$L(\mathbf{q}_1) = [\mathbf{q}_1 \mid W(\mathbf{q}_1)]\f$ (4×4).  Implemented
 * via `findWMat` for the non-scalar columns.
 *
 * @param q1 Left-hand quaternion.
 * @param q2 Right-hand quaternion.
 * @return Product \f$\mathbf{q}_1 \otimes \mathbf{q}_2\f$ (4-vector).
 */
Vec4 quatMult(const Vec4& q1, const Vec4& q2);

/**
 * @brief Full SO(3) rotation angle from an error quaternion.
 *
 * Given an error quaternion \f$\mathbf{q}_e = \mathbf{q}_a^{-1} \otimes \mathbf{q}_b\f$,
 * returns the full rotation angle in SO(3):
 * \f[
 * \theta = 2 \arccos(|q_{e,0}|)
 * \f]
 * The absolute value handles the double cover so q and -q give the same angle.
 *
 * NOTE: the attitude cost functions in Satellite::stageCost inline
 * \f$\arccos(|q_{goal} \cdot q|)\f$ — the half-angle — because their analytic
 * gradient needs the raw dot product.  This helper returns the full rotation
 * angle, which is the standard convention for user-facing pointing error.
 *
 * @param q_err Error quaternion.
 * @return Full rotation angle in radians, in [0, π].
 */
double quatAngle(const Vec4& q_err);

/**
 * @brief Compute the W matrix for quaternion kinematics.
 * 
 * Returns the 4×3 matrix that relates angular velocity to quaternion rate:
 * \f[
 * \dot{\mathbf{q}} = \frac{1}{2}\mathbf{W}(\mathbf{q})\boldsymbol{\omega}
 * \f]
 * where \f$\boldsymbol{\omega}\f$ is the angular velocity in rad/s.
 * 
 * @param q Unit quaternion.
 * @return 4×3 W matrix.
 */
Mat43 findWMat(const Vec4& q);

/**
 * @brief Compute the Jacobian of quaternion normalization.
 * 
 * Returns the 4×4 Jacobian matrix of the normalization operator:
 * \f[
 * \mathbf{J}_{\text{norm}} = \frac{\partial \hat{\mathbf{q}}}{\partial \mathbf{q}}
 * \f]
 * 
 * Useful for error propagation when quaternions are normalized.
 * 
 * @param q Input quaternion (typically unnormalized).
 * @return 4×4 Jacobian matrix.
 */
Mat43 quatNormJacobian(const Vec4& q);

/**
 * @brief Compute the skew-symmetric (cross-product) matrix of a vector.
 * 
 * Given a vector \f$\mathbf{v} = [v_1, v_2, v_3]^T\f$, returns the
 * skew-symmetric matrix:
 * \f[
 * [\mathbf{v}]_\times = \begin{bmatrix}
 * 0 & -v_3 & v_2 \\
 * v_3 & 0 & -v_1 \\
 * -v_2 & v_1 & 0
 * \end{bmatrix}
 * \f]
 * 
 * Used for cross-product computation: \f$\mathbf{v} \times \mathbf{w} = [\mathbf{v}]_\times \mathbf{w}\f$.
 * 
 * @param v Input 3D vector.
 * @return 3×3 skew-symmetric matrix.
 */
Mat33 skewSymmetric(const Vec3& v);

/**
 * @brief Compute the derivative of rotated vector with respect to quaternion.
 * 
 * Returns the 4×3 Jacobian of the rotated vector \f$\mathbf{C}(\mathbf{q})\mathbf{v}\f$
 * with respect to the quaternion \f$\mathbf{q}\f$:
 * \f[
 * \frac{\partial \mathbf{C}(\mathbf{q})\mathbf{v}}{\partial \mathbf{q}}
 * \f]
 * 
 * Useful for computing derivatives in attitude dynamics and control.
 * 
 * @param q Unit quaternion.
 * @param v Fixed vector to be rotated.
 * @return 4×3 Jacobian matrix.
 */
Mat43 drotmatTvecdq(const Vec4& q, const Vec3& v);

/**
 * @brief Compute the second derivative of rotated vector (Hessian).
 * 
 * Returns an array of three 4×4 matrices representing the Hessian tensors
 * for the derivative of \f$\mathbf{C}(\mathbf{q})\mathbf{v}\f$ with respect
 * to \f$\mathbf{q}\f$:
 * \f[
 * \frac{\partial^2 [\mathbf{C}(\mathbf{q})\mathbf{v}]_i}{\partial \mathbf{q} \partial \mathbf{q}^T}
 * \f]
 * 
 * Index \f$i = 0,1,2\f$ corresponds to the \f$x,y,z\f$ component.
 * 
 * @param q Unit quaternion.
 * @param v Fixed vector to be rotated.
 * @return Array of three 4×4 Hessian matrices (one per component).
 */
std::array<Mat44, 3> ddrotmatTvecdqdq(const Vec4& q, const Vec3& v);

}
