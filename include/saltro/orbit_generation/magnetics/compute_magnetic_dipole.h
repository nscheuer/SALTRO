#pragma once

#include <Eigen/Dense>
#include <saltro/limits.h>

namespace saltro::orbit {

bool compute_magnetic_dipole(
    const Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& R,
    const Eigen::Matrix<double, 1, saltro::limits::MAX_LENGTH_TRAJ>& jtime,
    const int jtime_length,

    Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& B
);

}