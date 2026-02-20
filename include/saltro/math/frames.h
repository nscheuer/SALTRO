#pragma once

#include <Eigen/Dense>

namespace saltro::math {

/**
 * @brief Compute Greenwich Mean Sidereal Time (GMST) angle in radians.
 *
 * Computes the Earth's rotation angle about the inertial \f$z\f$ axis using
 * the Julian centuries \f$T\f$ since J2000:
 * \f[
 * \theta_{\mathrm{GMST}} = \theta_{\mathrm{GMST}}(T)
 * \f]
 *
 * This angle is used to transform vectors between Earth-centered inertial
 * (ECI) and Earth-centered Earth-fixed (ECEF) reference frames.
 *
 * @param jcentury Julian centuries since J2000.
 * @return GMST angle in radians.
 */
double gmst_rad(double jcentury);

/**
 * @brief Direction cosine matrix from ECI to ECEF frame.
 *
 * Computes the rotation matrix that transforms a vector from the Earth-centered
 * inertial (ECI) frame to the Earth-centered Earth-fixed (ECEF) frame using the
 * Greenwich Mean Sidereal Time angle:
 * \f[
 * \mathbf{C}_{\text{ECI}\rightarrow\text{ECEF}} =
 * \mathbf{R}_z\!\bigl(\theta_{\mathrm{GMST}}\bigr)
 * \f]
 *
 * @param jcentury Julian centuries since J2000.
 * @return Direction cosine matrix from ECI to ECEF.
 */
Eigen::Matrix3d eci_to_ecef_dcm(double jcentury);

/**
 * @brief Direction cosine matrix from ECEF to ECI frame.
 *
 * Computes the rotation matrix that transforms a vector from the Earth-centered
 * Earth-fixed (ECEF) frame to the Earth-centered inertial (ECI) frame:
 * \f[
 * \mathbf{C}_{\text{ECEF}\rightarrow\text{ECI}} =
 * \mathbf{R}_z\!\bigl(-\theta_{\mathrm{GMST}}\bigr)
 * \f]
 *
 * @param jcentury Julian centuries since J2000.
 * @return Direction cosine matrix from ECEF to ECI.
 */
Eigen::Matrix3d ecef_to_eci_dcm(double jcentury);
}