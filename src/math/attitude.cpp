#include <saltro/math/attitude.h>

#include <saltro/math/mrp.h>
#include <saltro/math/quaternion.h>

#include <algorithm>
#include <cmath>

namespace saltro::math {

double pointingError(
    const Eigen::Vector4d& q,
    const Eigen::Vector4d& attitude_target,
    const Eigen::Vector3d& boresight_body
) {
    // Vector-pointing mode: NaN in attitude_target(0) signals ECI direction
    if (std::isnan(attitude_target(0))) {
        const Eigen::Vector3d target_vec = attitude_target.tail<3>();
        const double target_norm = target_vec.norm();
        if (!std::isfinite(target_norm) || target_norm < 1e-10) {
            return 0.0;
        }
        const Eigen::Vector3d target_unit = target_vec / target_norm;

        const Eigen::Matrix3d C = rotationMatrix(q);
        const Eigen::Vector3d bs_eci_raw = C * boresight_body;
        const double bs_norm = bs_eci_raw.norm();
        if (!std::isfinite(bs_norm) || bs_norm < 1e-10) {
            return 0.0;
        }
        const Eigen::Vector3d bs_eci = bs_eci_raw / bs_norm;

        const double cos_angle = std::clamp(bs_eci.dot(target_unit), -1.0, 1.0);
        return std::acos(cos_angle);
    }

    // Quaternion mode: return full SO(3) rotation angle 2·acos(|q·q_goal|)
    const Eigen::Vector4d q_err = quatError(attitude_target, q);
    return quatAngle(q_err);
}

}  // namespace saltro::math
