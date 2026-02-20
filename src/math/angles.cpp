#include <saltro/math/angles.h>
#include <cmath>


namespace saltro::math {

double wrap_to_2pi(double rad) {
    double x = std::fmod(rad, 2*M_PI);
    if (x < 0.0) x += 2*M_PI;
    return x;
}

double wrap_to_360(double deg) {
    double w = std::fmod(deg, 360.0);
    if (w < 0.0) w += 360;
    return w;
}

double deg2rad(double deg) {
    return deg * M_PI/180.0;
}

double rad2deg(double rad) {
    return rad * 180.0/M_PI;
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

Eigen::Vector3d rotate_about_z(const Eigen::Vector3d& v, const double angle_rad) {
    const double c = std::cos(angle_rad);
    const double s = std::sin(angle_rad);
    return Eigen::Vector3d(
        c * v.x() - s * v.y(),
        s * v.x() + c * v.y(),
        v.z()
    );
}

bool unit_vector(const Eigen::Vector3d& v, Eigen::Vector3d& out_unit) {
    const double n = v.norm();
    if (!(n > 0.0)) {
        return false;
    }
    out_unit = v / n;
    return true;
}

double clamp(const double x, const double lo, const double hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

}