#include <saltro/orbit_generation/density/compute_density_harrispriester.h>

#include <saltro/constants/harrispriester.h>
#include <saltro/math/frames.h>

#include <cmath>
#include <cstddef>

namespace saltro::orbits {

static inline double height_above_wgs84_approx(const Eigen::Vector3d& position_ecef_like) {
    const double a = saltro::constants::WGS84_A_M;
    const double f = saltro::constants::WGS84_F;
    const double e2 = f * (2.0 - f);

    const double r = position_ecef_like.norm();
    if (!(r > 0.0)) {
        return 0.0;
    }

    const double sl = position_ecef_like.z() / r;
    const double cl2 = 1.0 - sl * sl;
    const double denom = 1.0 - e2 * cl2;
    const double coef = std::sqrt((1.0 - e2) / denom);

    return r - a * coef;
}

static inline double density_harris_priester_single(const Eigen::Vector3d& r_eci_m, const Eigen::Vector3d& sun_eci_m) {
    const auto& tab = saltro::constants::HARRIS_PRIESTER_TABLE;

    const double alt_m = height_above_wgs84_approx(r_eci_m);
    if (alt_m < tab.front().alt_m) {
        return 0.0;
    }
    if (alt_m > tab.back().alt_m) {
        return 0.0;
    }

    Eigen::Vector3d r_hat;
    Eigen::Vector3d s_hat;
    if (!saltro::math::unit_vector(r_eci_m, r_hat)) {
        return 0.0;
    }
    if (!saltro::math::unit_vector(sun_eci_m, s_hat)) {
        return 0.0;
    }

    const Eigen::Vector3d bul_dir = saltro::math::rotate_about_z(s_hat, saltro::constants::HARRIS_PRIESTER_LAG_RAD);

    Eigen::Vector3d bul_hat;
    if (!saltro::math::unit_vector(bul_dir, bul_hat)) {
        return 0.0;
    }

    const double cos_psi = bul_hat.dot(r_hat);
    const double c2_psi_2 = saltro::math::clamp(0.5 * (1.0 + cos_psi), 0.0, 1.0);
    const double c_psi_2 = std::sqrt(c2_psi_2);

    double cos_pow = 0.0;
    if (c_psi_2 > saltro::constants::HARRIS_PRIESTER_MIN_COS) {
        const double n = saltro::constants::HARRIS_PRIESTER_COS_EXPONENT;
        cos_pow = c2_psi_2 * std::pow(c_psi_2, n - 2.0);
    }

    std::size_t ia = 0;
    while (ia + 2 < tab.size() && alt_m > tab[ia + 1].alt_m) {
        ia++;
    }

    const double h0 = tab[ia].alt_m;
    const double h1 = tab[ia + 1].alt_m;
    const double d_h = (h0 - alt_m) / (h0 - h1);

    const double rmin0 = tab[ia].rho_min_kg_m3;
    const double rmin1 = tab[ia + 1].rho_min_kg_m3;
    const double rho_min = rmin0 * std::pow(rmin1 / rmin0, d_h);

    if (!(cos_pow > 0.0)) {
        return rho_min;
    }

    const double rmax0 = tab[ia].rho_max_kg_m3;
    const double rmax1 = tab[ia + 1].rho_max_kg_m3;
    const double rho_max = rmax0 * std::pow(rmax1 / rmax0, d_h);

    return rho_min + (rho_max - rho_min) * cos_pow;
}

bool compute_denisty_harrispriester(
    const Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& R,
    const Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& S,
    const int jtime_length,
    Eigen::Matrix<double, 1, saltro::limits::MAX_LENGTH_TRAJ>& rho
) {
    if (jtime_length <= 0 || jtime_length > saltro::limits::MAX_LENGTH_TRAJ) {
        return false;
    }

    bool ok = true;

    for (int i = 0; i < jtime_length; ++i) {
        const Eigen::Vector3d r_eci = R.col(i);
        const Eigen::Vector3d s_rel = S.col(i);
        const Eigen::Vector3d sun_eci = r_eci + s_rel;

        const double alt_m = height_above_wgs84_approx(r_eci);
        if (alt_m < saltro::constants::HARRIS_PRIESTER_TABLE.front().alt_m) {
            rho(0, i) = 0.0;
            ok = false;
            continue;
        }

        rho(0, i) = density_harris_priester_single(r_eci, sun_eci);
    }

    return ok;
}

}