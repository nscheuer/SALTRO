#pragma once

#include <Eigen/Dense>

#include <saltro/limits.h>

namespace saltro::orbit {

bool generate_orbit(
    const Eigen::Vector3d& r0,
    const Eigen::Vector3d& v0,
    const Eigen::Matrix<double, 1, saltro::limits::MAX_LENGTH_TRAJ>& jtime,
    const int jtime_length,
    const int orbit_model,
    const int magnetic_model,
    const int sun_model,
    const int density_model,

    Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& R,
    Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& V,
    Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& B,
    Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& S,
    Eigen::Matrix<double, 1, saltro::limits::MAX_LENGTH_TRAJ>& rho
);

}