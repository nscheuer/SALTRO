# pragma once

#include <Eigen/Dense>
#include <saltro/limits.h>

namespace saltro::orbit {

bool compute_sun(
    Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& R,
    const Eigen::Matrix<double, 1, saltro::limits::MAX_LENGTH_TRAJ>& jtime,
    const int jtime_length,
    const int sun_model,
    Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& S
);

}