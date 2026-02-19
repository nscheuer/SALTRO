#pragma once

#include <Eigen/Dense>

namespace saltro::math {
    
double wrap_to_2pi(double rad);
double wrap_to_360(double deg);

double deg2rad(double deg);
double rad2deg(double rad);

double gmst_rad(double jcentury);

Eigen::Matrix3d rot_z(double rad);
Eigen::Matrix3d eci_to_ecef_dcm(double jcentury);
Eigen::Matrix3d ecef_to_eci_dcm(double jcentury);

Eigen::Vector3d rotate_about_z(const Eigen::Vector3d& v, const double angle_rad);
bool unit_vector(const Eigen::Vector3d& v, Eigen::Vector3d& out_unit);
double clamp(const double x, const double lo, const double hi);

}