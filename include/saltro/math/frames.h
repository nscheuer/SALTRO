#pragma once

#include <Eigen/Dense>

namespace saltro::math {
    
inline double wrap_to_2pi(double rad);
inline double wrap_to_360(double deg);

inline double deg2rad(double deg);
inline double rad2deg(double rad);

inline double gmst_rad(double jcentury);

Eigen::Matrix3d rot_z(double rad);
Eigen::Matrix3d eci_to_ecef_dcm(double jcentury);
Eigen::Matrix3d ecef_to_eci_dcm(double jcentury);

Eigen::Vector3d rotate_about_z(const Eigen::Vector3d& v, const double angle_rad);
bool unit_vector(const Eigen::Vector3d& v, Eigen::Vector3d& out_unit);
double clamp(const double x, const double lo, const double hi);

}