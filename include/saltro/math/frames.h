#pragma once

#include <Eigen/Dense>

namespace saltro::math {
    
double wrap_to_2pi(double rad);

double gmst_rad(double jcentury);

Eigen::Matrix3d rot_z(double rad);
Eigen::Matrix3d eci_to_ecef_dcm(double jcentury);
Eigen::Matrix3d ecef_to_eci_dcm(double jcentury);

}