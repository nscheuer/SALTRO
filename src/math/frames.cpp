#include <saltro/math/frames.h>
#include <cmath>

namespace saltro::math {

inline double wrap_to_2pi(double rad) {
    double x = std::fmod(rad, 2*M_PI);
    if (x < 0.0) x += 2*M_PI;
    return x;
}

inline double wrap_to_360(double deg) {
    double w = std::fmod(deg, 360.0);
    if (w < 0.0) w += 360;
    return w;
}

inline double deg2rad(double deg) {
    return deg * M_PI/180.0;
}

inline double rad2deg(double rad) {
    return rad * 180.0/M_PI;
}

inline double gmst_rad(double jcentury) {
    const double T2 = jcentury * jcentury;
    const double T3 = T2 * jcentury;

    const double gmst_sec = 
        67310.54841 +
        (876600.0 * 3600.0 + 8640184.812866) * jcentury +
        0.093104 * T2 -
        6.2e-6 * T3;
    
    const double gmst_rad = (gmst_sec * (2 * M_PI)) / 86400.0;
    return wrap_to_2pi(gmst_rad);
}

Eigen::Matrix3d rot_z(double rad) {
    const double c = std::cos(rad);
    const double s = std::sin(rad);

    Eigen::Matrix3d R;
    R << c, -s, 0,
         s,  c, 0,
         0,  0, 1;
    return R;
}

Eigen::Matrix3d eci_to_ecef_dcm(double jcentury) {
    return rot_z(gmst_rad(jcentury));
}

Eigen::Matrix3d ecef_to_eci_dcm(double jcentury) {
    return eci_to_ecef_dcm(jcentury).transpose();
}

}