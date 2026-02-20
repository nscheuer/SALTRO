#pragma once

#include <Eigen/Dense>

namespace saltro::math {

/**
 * @brief Wrap an angle in radians to the interval \f$[0, 2\pi)\f$.
 *
 * Given an input angle \f$\theta\f$ in radians, this function returns an
 * equivalent angle wrapped into the range \f$[0, 2\pi)\f$:
 * \f[
 * \theta_{\text{wrapped}} = \theta \bmod 2\pi
 * \f]
 *
 * This is useful for maintaining angles within a canonical range for
 * orbital or attitude calculations.
 *
 * @param rad Input angle in radians.
 * @return Wrapped angle in radians within \f$[0, 2\pi)\f$.
 */
double wrap_to_2pi(double rad);

/**
 * @brief Wrap an angle in degrees to the interval \f$[0, 360)\f$.
 *
 * Given an input angle \f$\alpha\f$ in degrees, this function returns an
 * equivalent angle wrapped into the range \f$[0, 360)\f$:
 * \f[
 * \alpha_{\text{wrapped}} = \alpha \bmod 360
 * \f]
 *
 * @param deg Input angle in degrees.
 * @return Wrapped angle in degrees within \f$[0, 360)\f$.
 */
double wrap_to_360(double deg);

/**
 * @brief Convert degrees to radians.
 *
 * Converts an angle from degrees to radians using:
 * \f[
 * \theta_{\text{rad}} = \frac{\pi}{180}\,\theta_{\text{deg}}
 * \f]
 *
 * @param deg Angle in degrees.
 * @return Angle in radians.
 */
double deg2rad(double deg);

/**
 * @brief Convert radians to degrees.
 *
 * Converts an angle from radians to degrees using:
 * \f[
 * \theta_{\text{deg}} = \frac{180}{\pi}\,\theta_{\text{rad}}
 * \f]
 *
 * @param rad Angle in radians.
 * @return Angle in degrees.
 */
double rad2deg(double rad);

/**
 * @brief Rotation matrix about the \f$z\f$ axis.
 *
 * Returns the 3×3 rotation matrix representing a right-handed rotation about
 * the \f$z\f$ axis by angle \f$\theta\f$:
 * \f[
 * \mathbf{R}_z(\theta) =
 * \begin{bmatrix}
 * \cos\theta & -\sin\theta & 0 \\
 * \sin\theta & \cos\theta  & 0 \\
 * 0          & 0           & 1
 * \end{bmatrix}
 * \f]
 *
 * @param rad Rotation angle in radians.
 * @return 3×3 rotation matrix.
 */
Eigen::Matrix3d rot_z(double rad);

/**
 * @brief Rotate a vector about the \f$z\f$ axis.
 *
 * Applies a right-handed rotation about the \f$z\f$ axis to a vector:
 * \f[
 * \mathbf{v}_{\text{out}} =
 * \mathbf{R}_z(\theta)\,\mathbf{v}
 * \f]
 *
 * @param v Input 3D vector.
 * @param angle_rad Rotation angle in radians.
 * @return Rotated vector.
 */
Eigen::Vector3d rotate_about_z(const Eigen::Vector3d& v, const double angle_rad);

/**
 * @brief Compute the unit vector of a given vector.
 *
 * Normalizes the input vector:
 * \f[
 * \hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|}
 * \f]
 *
 * If the input vector has zero norm, normalization fails and the function
 * returns false.
 *
 * @param v Input vector.
 * @param out_unit Output unit vector.
 * @return True if normalization succeeds, false if the input vector has zero norm.
 */
bool unit_vector(const Eigen::Vector3d& v, Eigen::Vector3d& out_unit);

/**
 * @brief Clamp a scalar value to a closed interval.
 *
 * Restricts the value \f$x\f$ to lie within the interval \f$[l, h]\f$:
 * \f[
 * \mathrm{clamp}(x) = \min\!\bigl(\max(x,l),h\bigr)
 * \f]
 *
 * @param x Input value.
 * @param lo Lower bound.
 * @param hi Upper bound.
 * @return Clamped value within \f$[lo, hi]\f$.
 */
double clamp(const double x, const double lo, const double hi);

}