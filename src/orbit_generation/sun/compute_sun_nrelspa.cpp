#include <cmath>
#include <saltro/constants/constants.h>
#include <saltro/constants/spa.h>
#include <saltro/math/frames.h>
#include <saltro/limits.h>

namespace saltro::orbits {

// --- Internal Math Helpers ---

double sum_periodic_terms(const double table[][3], int count, double jme) {
    double sum = 0.0;
    for (int i = 0; i < count; ++i) {
        sum += table[i][0] * std::cos(table[i][1] + table[i][2] * jme);
    }
    return sum;
}

double NRELSPA_compute_heliocentric_L(double T) {
    const double jme = T / 10.0; // Julian Millennium
    using namespace saltro::constants::NRELSPA;
    
    // Summing terms for L0 through L5
    double L = sum_periodic_terms(L0_TABLE, L0_COUNT, jme) +
               sum_periodic_terms(L1_TABLE, L1_COUNT, jme) * jme +
               sum_periodic_terms(L2_TABLE, L2_COUNT, jme) * std::pow(jme, 2) +
               sum_periodic_terms(L3_TABLE, L3_COUNT, jme) * std::pow(jme, 3) +
               sum_periodic_terms(L4_TABLE, L4_COUNT, jme) * std::pow(jme, 4) +
               sum_periodic_terms(L5_TABLE, L5_COUNT, jme) * std::pow(jme, 5);
    
    double L_deg = L * 1e-8;
    return saltro::math::deg2rad(saltro::math::wrap_to_360(L_deg));
}

double NRELSPA_compute_heliocentric_B(double T) {
    const double jme = T / 10.0;
    using namespace saltro::constants::NRELSPA;
    
    // Summing terms for B0 and B1
    double B = sum_periodic_terms(B0_TABLE, B0_COUNT, jme) +
               sum_periodic_terms(B1_TABLE, B1_COUNT, jme) * jme;
    
    double B_deg = B * 1e-8;
    return saltro::math::deg2rad(B_deg);
}

double NRELSPA_compute_heliocentric_R(double T) {
    const double jme = T / 10.0;
    using namespace saltro::constants::NRELSPA;
    
    // Summing terms for R0 through R4
    double R = sum_periodic_terms(R0_TABLE, R0_COUNT, jme) +
               sum_periodic_terms(R1_TABLE, R1_COUNT, jme) * jme +
               sum_periodic_terms(R2_TABLE, R2_COUNT, jme) * std::pow(jme, 2) +
               sum_periodic_terms(R3_TABLE, R3_COUNT, jme) * std::pow(jme, 3) +
               sum_periodic_terms(R4_TABLE, R4_COUNT, jme) * std::pow(jme, 4);
    
    return R * 1e-8;
}

void NRELSPA_compute_nutation_obliquity(double T, double& d_psi, double& eps) {
    using namespace saltro::math;
    using namespace saltro::constants::NRELSPA;

    // Fundamental arguments in radians
    double X[5];
    X[0] = deg2rad(297.85036 + 445267.111480 * T); // Mean elongation of the Moon
    X[1] = deg2rad(357.52772 + 35999.050340 * T);  // Mean anomaly of the Sun
    X[2] = deg2rad(134.96298 + 477198.867398 * T); // Mean anomaly of the Moon
    X[3] = deg2rad(93.27191 + 483202.017538 * T);  // Moon's argument of latitude
    X[4] = deg2rad(125.04452 - 1934.136261 * T);   // Longitude of the ascending node

    double dp_sum = 0.0;
    double de_sum = 0.0;

    for (int i = 0; i < Y_COUNT; ++i) {
        double arg = 0.0;
        for (int j = 0; j < 5; ++j) {
            arg += Y_TABLE[i][j] * X[j];
        }
        // Nutation coefficients are in 0.0001 arcseconds
        dp_sum += (Y_TABLE[i][5] + Y_TABLE[i][6] * T) * std::sin(arg);
        de_sum += (Y_TABLE[i][7] + Y_TABLE[i][8] * T) * std::cos(arg);
    }

    d_psi = deg2rad(dp_sum * 1e-4 / 3600.0);
    const double d_eps = deg2rad(de_sum * 1e-4 / 3600.0);

    // Mean obliquity of the ecliptic (IAU 1980)
    const double U = T / 100.0;
    const double eps0_deg = (84381.448 - 4680.93 * U) / 3600.0;
    eps = deg2rad(eps0_deg) + d_eps;
}


bool compute_sun_nrelspa(
    Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& R,
    const Eigen::Matrix<double, 1, saltro::limits::MAX_LENGTH_TRAJ>& jtime,
    const int jtime_length,
    Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& S
) {
    using namespace saltro::math;

    for (int k = 0; k < jtime_length; ++k) {
        const double T = jtime(k);

        const double L_rad = NRELSPA_compute_heliocentric_L(T);
        const double B_rad = NRELSPA_compute_heliocentric_B(T);
        const double R_earth_au = NRELSPA_compute_heliocentric_R(T);

        // Transformation to Geocentric Longitude (theta) and Latitude (beta)
        const double theta = L_rad + M_PI;
        const double beta = -B_rad;

        double d_psi = 0.0;
        double eps = 0.0;
        NRELSPA_compute_nutation_obliquity(T, d_psi, eps);

        // Aberration correction (~20.4898 arcseconds)
        const double aberration_corr = deg2rad(-20.4898 / 3600.0);
        const double lambda = theta + d_psi + aberration_corr;

        // Trignonometric terms for coordinate conversion
        const double sl = std::sin(lambda);
        const double cl = std::cos(lambda);
        const double se = std::sin(eps);
        const double ce = std::cos(eps);
        const double tb = std::tan(beta);
        const double sb = std::sin(beta);
        const double cb = std::cos(beta);

        // Convert to Right Ascension (alpha) and Declination (delta)
        const double alpha = std::atan2(sl * ce - tb * se, cl);
        const double delta = std::asin(sb * ce + cb * se * sl);

        // Convert spherical geocentric coordinates to Cartesian ECI
        const double r_sun_m = R_earth_au * saltro::constants::AU_M;
        const double cd = std::cos(delta);

        Eigen::Vector3d r_sun_eci;
        r_sun_eci.x() = r_sun_m * cd * std::cos(alpha);
        r_sun_eci.y() = r_sun_m * cd * std::sin(alpha);
        r_sun_eci.z() = r_sun_m * std::sin(delta);

        // Relative vector from satellite to Sun
        S.col(k) = r_sun_eci - R.col(k);
    }

    return true;
}

}