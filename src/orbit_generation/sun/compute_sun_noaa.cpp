#include <cmath>

#include <saltro/constants/constants.h>
#include <saltro/math/angles.h>
#include <saltro/limits.h>

namespace saltro::orbits {

bool compute_sun_noaa(
    Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& R,
    const Eigen::Matrix<double, 1, saltro::limits::MAX_LENGTH_TRAJ>& jtime,
    const int jtime_length,
    Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& S
) {
    for (int k = 0; k < jtime_length; ++k) {
        const double T = jtime(k);

        const double L0 = saltro::math::wrap_to_360(
            280.46646 + 36000.76983 * T + 0.0003032 * T * T
        );

        const double M = 357.52911 + 35999.05029 * T - 0.0001537 * T * T;
        const double e = 0.016708634 - 0.000042037 * T - 0.0000001267 * T * T;

        const double Mrad = saltro::math::deg2rad(M);
        const double C =
            (1.914602 - 0.004817 * T - 0.000014 * T * T) * std::sin(Mrad)
          + (0.019993 - 0.000101 * T) * std::sin(2.0 * Mrad)
          +  0.000289 * std::sin(3.0 * Mrad);

        const double true_long = L0 + C;
        const double true_anom = M + C;

        const double nu = saltro::math::deg2rad(true_anom);
        const double r_au = (1.000001018 * (1.0 - e * e)) / (1.0 + e * std::cos(nu));

        const double omega = 125.04 - 1934.136 * T;
        const double lambda = true_long - 0.00569 - 0.00478 * std::sin(saltro::math::deg2rad(omega));

        const double seconds = 21.448 - T * (46.815 + T * (0.00059 - T * 0.001813));
        const double eps0 = 23.0 + (26.0 + (seconds / 60.0)) / 60.0;

        const double eps = eps0 + 0.00256 * std::cos(saltro::math::deg2rad(omega));

        const double lam = saltro::math::deg2rad(lambda);
        const double epsr = saltro::math::deg2rad(eps);

        const double sin_lam = std::sin(lam);
        const double cos_lam = std::cos(lam);
        const double cos_eps = std::cos(epsr);
        const double sin_eps = std::sin(epsr);

        const double alpha = std::atan2(cos_eps * sin_lam, cos_lam);
        const double delta = std::asin(sin_eps * sin_lam);

        const double r_m = r_au * saltro::constants::AU_M;
        const double cos_delta = std::cos(delta);

        Eigen::Vector3d r_sun_eci;
        r_sun_eci.x() = r_m * cos_delta * std::cos(alpha);
        r_sun_eci.y() = r_m * cos_delta * std::sin(alpha);
        r_sun_eci.z() = r_m * std::sin(delta);

        S.col(k) = r_sun_eci - R.col(k);
    }

    return true;
}

}