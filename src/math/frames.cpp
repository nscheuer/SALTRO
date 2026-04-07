#include <saltro/math/frames.h>
#include <saltro/math/angles.h>
#include <cmath>


namespace saltro::math {

double gmst_rad(double jcentury) {
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

Eigen::Matrix3d eci_to_ecef_dcm(double jcentury) {
    return rot_z(-gmst_rad(jcentury));
}

Eigen::Matrix3d ecef_to_eci_dcm(double jcentury) {
    return eci_to_ecef_dcm(jcentury).transpose();
}

}