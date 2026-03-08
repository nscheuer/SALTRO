#pragma once

#include <Eigen/Dense>
#include <string>

namespace saltro::validation {

/**
 * @brief Validate initial orbital state for Earth LEO trajectory use.
 *
 * Expected input units are:
 * - r0 in meters
 * - v0 in meters per second
 *
 * This function performs practical pre-checks to catch common setup issues:
 * - finite values for position and velocity vectors
 * - likely unit mistakes (km or km/s passed instead of m and m/s)
 * - physically valid bound Earth orbit (negative specific energy)
 * - near-LEO envelope consistency (perigee/apogee altitude bounds)
 * - moderate eccentricity (not highly elliptical)
 *
 * @param r0 Initial inertial position vector (m).
 * @param v0 Initial inertial velocity vector (m/s).
 * @param error_msg Output error message describing first failed check.
 * @return True if state is valid for LEO use, false otherwise.
 */
bool validateOrbitState(
	const Eigen::Vector3d& r0,
	const Eigen::Vector3d& v0,
	std::string& error_msg
);

} // namespace saltro::validation
