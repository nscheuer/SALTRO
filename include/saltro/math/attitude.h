#pragma once

#include <Eigen/Dense>

namespace saltro::math {

/**
 * @brief Pointing error (radians) between current attitude and a target.
 *
 * Dispatches on the attitude-target format convention used across the
 * optimizer pipeline:
 *
 * - **Quaternion mode** (default): `attitude_target` is a unit quaternion
 *   `q_goal = [q0, q1, q2, q3]`.  Returns the full SO(3) rotation angle
 *   between `q` and `q_goal`:
 *   \f[\theta = 2\arccos(|q \cdot q_{goal}|) \in [0, \pi]\f]
 *
 * - **Vector-pointing mode**: signalled by `NaN` in `attitude_target(0)`.
 *   The trailing 3 components are an ECI-frame target direction.  Returns
 *   the angle between that direction and the boresight vector rotated into
 *   ECI by the current attitude `q`:
 *   \f[\theta = \arccos(\hat{b}_{ECI} \cdot \hat{t}_{ECI}) \in [0, \pi]\f]
 *   where \f$\hat{b}_{ECI} = C(q) \cdot \hat{b}_{body}\f$.
 *
 * The `NaN` sentinel matches the convention used throughout the spike
 * removal and visualization code.  The Satellite cost functions use a
 * richer `processAttitudeTarget` dispatch internally.
 *
 * @param q Current attitude quaternion (will be normalized).
 * @param attitude_target Either a unit quaternion or `[NaN, x, y, z]` ECI
 *                        direction.
 * @param boresight_body Boresight vector in body frame (used only in
 *                       vector-pointing mode).
 * @return Pointing error in radians.
 */
double pointingError(
    const Eigen::Vector4d& q,
    const Eigen::Vector4d& attitude_target,
    const Eigen::Vector3d& boresight_body
);

}  // namespace saltro::math
