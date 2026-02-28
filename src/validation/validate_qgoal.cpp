#include <saltro/validation/validate_qgoal.h>

#include <cmath>

#include <saltro/limits.h>

namespace saltro::validation {

bool validateQGoal(
    const Eigen::Ref<const Eigen::MatrixXd>& q_goal,
    std::string& error_msg
) {
    constexpr double NORM_TOL = 1e-3;

    if (q_goal.rows() != 4) {
        error_msg = "q_goal must have shape (4, N)";
        return false;
    }

    if (q_goal.cols() <= 0) {
        error_msg = "q_goal must have at least one column";
        return false;
    }

    if (q_goal.cols() > saltro::limits::MAX_LENGTH_TRAJ) {
        error_msg = "q_goal exceeds MAX_LENGTH_TRAJ";
        return false;
    }

    for (Eigen::Index col = 0; col < q_goal.cols(); ++col) {
        const double q0 = q_goal(0, col);
        const Eigen::Vector3d tail = q_goal.col(col).segment<3>(1);

        if (!tail.allFinite()) {
            error_msg = "q_goal column " + std::to_string(col) + " has invalid NaN/Inf in rows 2-4";
            return false;
        }

        if (std::isnan(q0)) {
            const double tail_norm = tail.norm();
            if (tail_norm <= 0.0 || !std::isfinite(tail_norm)) {
                error_msg = "q_goal column " + std::to_string(col) + " has zero or non-finite direction norm";
                return false;
            }
            if (std::abs(tail_norm - 1.0) > NORM_TOL) {
                error_msg = "q_goal column " + std::to_string(col) + " ECI direction is not normalized";
                return false;
            }
            continue;
        }

        if (!std::isfinite(q0)) {
            error_msg = "q_goal column " + std::to_string(col) + " has invalid q0 value";
            return false;
        }

        const Eigen::Vector4d quat = q_goal.col(col);
        const double quat_norm = quat.norm();
        if (!std::isfinite(quat_norm) || std::abs(quat_norm - 1.0) > NORM_TOL) {
            error_msg = "q_goal column " + std::to_string(col) + " quaternion is not normalized";
            return false;
        }
    }

    return true;
}

} // namespace saltro::validation
