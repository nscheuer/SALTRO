#include <saltro/orbit_generation/eclipse/compute_eclipse_penumbra.h>
#include <saltro/constants/constants.h>

#include <cmath>

namespace saltro::orbits {

bool compute_eclipse_penumbra(
    const Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& R,
    const int jtime_length,

    Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& S
) {
    const double R_earth = saltro::constants::R_EARTH;
    const double R_sun = saltro::constants::R_SUN;
    const double eps = 1e-12;

    for (int k = 0; k < jtime_length; ++k) {
        const Eigen::Vector3d sun_vec = S.col(k);
        const Eigen::Vector3d r_sat = R.col(k);

        const double sun_norm2 = sun_vec.squaredNorm();
        const double r_norm2 = r_sat.squaredNorm();

        if (sun_norm2 < eps || r_norm2 < eps) continue;

        const double sun_norm = std::sqrt(sun_norm2);
        const double r_norm = std::sqrt(r_norm2);
        const double inv_sun = 1.0 / sun_norm;
        const double inv_r = 1.0 / r_norm;

        // Angle between Earth center and Sun as seen from spacecraft.
        double cos_psi = (-r_sat).dot(sun_vec) * inv_r * inv_sun;
        if (cos_psi <= 0.0) continue;
        if (cos_psi > 1.0) cos_psi = 1.0;

        double sin_alpha_e = R_earth * inv_r;
        double sin_alpha_s = R_sun * inv_sun;

        if (sin_alpha_e > 1.0) sin_alpha_e = 1.0;
        if (sin_alpha_s > 1.0) sin_alpha_s = 1.0;

        const double cos_alpha_e = std::sqrt(std::max(0.0, 1.0 - sin_alpha_e * sin_alpha_e));
        const double cos_alpha_s = std::sqrt(std::max(0.0, 1.0 - sin_alpha_s * sin_alpha_s));

        // Compare cos(psi) to cos(alpha_e + alpha_s) without calling acos.
        const double cos_sum = (cos_alpha_e * cos_alpha_s) - (sin_alpha_e * sin_alpha_s);

        if (cos_psi >= cos_sum) {
            S.col(k) = Eigen::Vector3d::Zero();
        }
    }

    return true;
}

}
