#pragma once

#include <Eigen/Dense>
#include <saltro/limits.h>

namespace saltro::orbits {

bool compute_density_harrispriester(
    const Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& R,
    const Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& S,
    const int jtime_length,
    Eigen::Matrix<double, 1, saltro::limits::MAX_LENGTH_TRAJ>& rho
);

}