#include <saltro/validation/validate_boresight.h>

#include <cmath>

#include <saltro/limits.h>

namespace saltro::validation {

bool validateBoresight(
    const Eigen::Ref<const Eigen::MatrixXd>& boresight,
    std::string& error_msg
) {
    constexpr double NORM_TOL = 1e-3;

    if (boresight.rows() != 3) {
        error_msg = "boresight must have shape (3, N)";
        return false;
    }

    if (boresight.cols() <= 0) {
        error_msg = "boresight must have at least one column";
        return false;
    }

    if (boresight.cols() > saltro::limits::MAX_LENGTH_TRAJ) {
        error_msg = "boresight exceeds MAX_LENGTH_TRAJ";
        return false;
    }

    for (Eigen::Index col = 0; col < boresight.cols(); ++col) {
        const Eigen::Vector3d vec = boresight.col(col);

        if (!vec.allFinite()) {
            error_msg = "boresight column " + std::to_string(col) + " contains NaN or Inf";
            return false;
        }

        const double vec_norm = vec.norm();
        if (!std::isfinite(vec_norm) || vec_norm <= 0.0) {
            error_msg = "boresight column " + std::to_string(col) + " has zero or non-finite norm";
            return false;
        }

        if (std::abs(vec_norm - 1.0) > NORM_TOL) {
            error_msg = "boresight column " + std::to_string(col) + " is not normalized (norm = " + std::to_string(vec_norm) + ")";
            return false;
        }
    }

    return true;
}

} // namespace saltro::validation
