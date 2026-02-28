#include <saltro/optimizer/validation/validate_qgoal.h>
#include <cmath>

namespace saltro::optimizer::validation {

bool validateQGoal(
    const Eigen::Matrix<double, 4, saltro::limits::MAX_LENGTH_TRAJ>& q_goal,
    int jtime_length,
    std::string& error_msg
) {
    if (jtime_length <= 0 || jtime_length > saltro::limits::MAX_LENGTH_TRAJ) {
        error_msg = "jtime_length out of range";
        return false;
    }

    for (int k = 0; k < jtime_length; ++k) {
        const Eigen::Vector4d col = q_goal.col(k);
        const bool q0_is_nan = !std::isfinite(col(0));

        if (q0_is_nan) {
            if (!std::isfinite(col(1)) || !std::isfinite(col(2)) || !std::isfinite(col(3))) {
                error_msg = "ECI goal must have finite x,y,z components";
                return false;
            }
            const double eci_norm = col.tail<3>().norm();
            if (eci_norm < 1e-12) {
                error_msg = "ECI direction vector has zero norm";
                return false;
            }
        } else {
            if (!col.allFinite()) {
                error_msg = "Quaternion goal contains non-finite values";
                return false;
            }
            const double quat_norm = col.norm();
            if (quat_norm < 1e-12) {
                error_msg = "Quaternion has zero norm";
                return false;
            }
        }
    }

    return true;
}

}
