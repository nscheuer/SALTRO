#include <catch2/catch_test_macros.hpp>
#include <Eigen/Dense>
#include <cmath>
#include <saltro/orbit_generation/orbits/compute_orbit_J2_RK4.h>
#include <saltro/orbit_generation/orbits/compute_orbit_keplerian.h>
#include <saltro/limits.h>

using namespace saltro;

static constexpr double MU = 3.986004418e14;  // Earth gravitational parameter
static constexpr double RE = 6378136.3;       // Earth equatorial radius
static constexpr double J2 = 1.08262668e-3;   // Earth J2 coefficient
static constexpr double SEC_PER_CENTURY = 3155760000.0;  // Seconds in Julian century

double specific_energy(const Eigen::Vector3d& r, const Eigen::Vector3d& v)
{
    return 0.5 * v.squaredNorm() - MU / r.norm();
}

Eigen::Vector3d angular_momentum(const Eigen::Vector3d& r, const Eigen::Vector3d& v)
{
    return r.cross(v);
}

Eigen::Vector3d apsidal_vector(const Eigen::Vector3d& r, const Eigen::Vector3d& v)
{
    double r_norm = r.norm();
    double v_norm = v.norm();
    
    return ((v_norm * v_norm / MU - 1.0 / r_norm) * r - 
            (r.dot(v) / MU) * v);
}

double longitude_of_perigee(const Eigen::Vector3d& r, const Eigen::Vector3d& v)
{
    Eigen::Vector3d h = angular_momentum(r, v);
    Eigen::Vector3d e_vec = apsidal_vector(r, v);
    
    Eigen::Vector3d n_vec(-h.y(), h.x(), 0.0);
    double n_norm = n_vec.norm();
    
    if (n_norm < 1e-8) {
        return std::nan("");  // Equatorial orbit
    }
    
    double Omega = std::atan2(n_vec.y(), n_vec.x());
    if (Omega < 0) {
        Omega += 2.0 * M_PI;
    }
    
    double e_norm = e_vec.norm();
    if (e_norm < 1e-8) {
        return std::nan("");  // Circular orbit
    }
    
    double dot_ne = n_vec.x() * e_vec.x() + n_vec.y() * e_vec.y();
    double cos_omega = dot_ne / (n_norm * e_norm);
    double sin_omega = e_vec.z() / e_norm;
    
    cos_omega = std::clamp(cos_omega, -1.0, 1.0);
    
    double omega = std::atan2(sin_omega, cos_omega);
    if (omega < 0) {
        omega += 2.0 * M_PI;
    }
    
    return Omega + omega;
}

double right_ascension_ascending_node(const Eigen::Vector3d& r, const Eigen::Vector3d& v)
{
    Eigen::Vector3d h = angular_momentum(r, v);
    Eigen::Vector3d n_vec(-h.y(), h.x(), 0.0);
    double n_norm = n_vec.norm();
    
    if (n_norm < 1e-8) {
        return 0.0;  // Equatorial orbit
    }
    
    double Omega = std::atan2(n_vec.y(), n_vec.x());
    if (Omega < 0) {
        Omega += 2.0 * M_PI;
    }
    return Omega;
}

double inclination(const Eigen::Vector3d& r, const Eigen::Vector3d& v)
{
    Eigen::Vector3d h = angular_momentum(r, v);
    double h_norm = h.norm();
    
    return std::acos(h.z() / h_norm);
}

TEST_CASE("J2 RK4 dimensions and validity", "[orbit][j2][rk4]")
{
    constexpr int N = 10;
    
    Eigen::Vector3d r0(7000e3, 0.0, 0.0);
    Eigen::Vector3d v0(0.0, 7.5e3, 0.0);

    Eigen::Matrix<double,1,limits::MAX_LENGTH_TRAJ> jtime;
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> R;
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> V;

    for (int i = 0; i < N; ++i)
        jtime(i) = (2451545.0 + i * 1e-5) / 36525.0;

    bool ok = orbits::compute_orbit_J2_RK4(r0, v0, jtime, N, R, V);
    REQUIRE(ok);

    for (int i = 0; i < N; ++i)
    {
        REQUIRE(std::isfinite(R(0,i)));
        REQUIRE(std::isfinite(R(1,i)));
        REQUIRE(std::isfinite(R(2,i)));
        REQUIRE(std::isfinite(V(0,i)));
        REQUIRE(std::isfinite(V(1,i)));
        REQUIRE(std::isfinite(V(2,i)));
    }
}

TEST_CASE("J2 RK4 initial conditions", "[orbit][j2][rk4]")
{
    constexpr int N = 5;
    
    Eigen::Vector3d r0(7000e3, 1000e3, 500e3);
    Eigen::Vector3d v0(100.0, 7500.0, 50.0);

    Eigen::Matrix<double,1,limits::MAX_LENGTH_TRAJ> jtime;
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> R;
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> V;

    for (int i = 0; i < N; ++i)
        jtime(i) = (2451545.0 + i * 1e-6) / 36525.0;

    bool ok = orbits::compute_orbit_J2_RK4(r0, v0, jtime, N, R, V);
    REQUIRE(ok);

    REQUIRE(std::abs(R(0, 0) - r0.x()) < 1e-9);
    REQUIRE(std::abs(R(1, 0) - r0.y()) < 1e-9);
    REQUIRE(std::abs(R(2, 0) - r0.z()) < 1e-9);
    REQUIRE(std::abs(V(0, 0) - v0.x()) < 1e-9);
    REQUIRE(std::abs(V(1, 0) - v0.y()) < 1e-9);
    REQUIRE(std::abs(V(2, 0) - v0.z()) < 1e-9);
}

TEST_CASE("J2 RK4 inclination conservation", "[orbit][j2][rk4]")
{
    constexpr int N = 50;
    
    double alt = 400e3;
    double rmag = RE + alt;
    double inc = M_PI / 4.0;  // 45 degrees

    Eigen::Vector3d r0(rmag, 0.0, 0.0);
    Eigen::Vector3d v0(0.0, 
                      std::sqrt(MU / rmag) * std::cos(inc), 
                      std::sqrt(MU / rmag) * std::sin(inc));

    Eigen::Matrix<double,1,limits::MAX_LENGTH_TRAJ> jtime;
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> R;
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> V;

    for (int i = 0; i < N; ++i)
        jtime(i) = (2451545.0 + i * 0.05 / (N - 1)) / 36525.0;

    bool ok = orbits::compute_orbit_J2_RK4(r0, v0, jtime, N, R, V);
    REQUIRE(ok);

    double inc_initial = inclination(r0, v0);
    
    for (int i = 0; i < N; ++i)
    {
        Eigen::Vector3d r_i = R.col(i);
        Eigen::Vector3d v_i = V.col(i);
        double inc_current = inclination(r_i, v_i);
        REQUIRE(std::abs(inc_current - inc_initial) < 1e-3);
    }
}

TEST_CASE("J2 RK4 nodal regression", "[orbit][j2][rk4]")
{
    constexpr int N = 200;
    
    double alt = 400e3;
    double rmag = RE + alt;
    double inc = M_PI / 4.0;

    Eigen::Vector3d r0(rmag, 0.0, 0.0);
    Eigen::Vector3d v0(0.0,
                      std::sqrt(MU / rmag) * std::cos(inc),
                      std::sqrt(MU / rmag) * std::sin(inc));

    Eigen::Matrix<double,1,limits::MAX_LENGTH_TRAJ> jtime;
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> R;
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> V;

    for (int i = 0; i < N; ++i)
        jtime(i) = (2451545.0 + i * 1.0 / (N - 1)) / 36525.0;

    bool ok = orbits::compute_orbit_J2_RK4(r0, v0, jtime, N, R, V);
    REQUIRE(ok);

    double raan_initial = right_ascension_ascending_node(R.col(0), V.col(0));
    double raan_final = right_ascension_ascending_node(R.col(N-1), V.col(N-1));

    double delta_raan = raan_final - raan_initial;
    if (delta_raan > M_PI) {
        delta_raan -= 2.0 * M_PI;
    }
    if (delta_raan < -M_PI) {
        delta_raan += 2.0 * M_PI;
    }

    REQUIRE(delta_raan < 0);
}

TEST_CASE("J2 RK4 apsidal precession", "[orbit][j2][rk4]")
{
    constexpr int N = 200;
    
    double alt = 400e3;
    double rmag = RE + alt;
    double e = 0.3;
    double a = rmag / (1.0 - e);
    
    double r0_mag = rmag;
    double v_circ = std::sqrt(MU / rmag);
    double v0_mag = v_circ * std::sqrt(2.0 - r0_mag / a);

    Eigen::Vector3d r0(r0_mag, 0.0, 0.0);
    Eigen::Vector3d v0(0.0, v0_mag, 0.0);

    Eigen::Matrix<double,1,limits::MAX_LENGTH_TRAJ> jtime;
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> R;
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> V;

    for (int i = 0; i < N; ++i)
        jtime(i) = (2451545.0 + i * 1.0 / (N - 1)) / 36525.0;

    bool ok = orbits::compute_orbit_J2_RK4(r0, v0, jtime, N, R, V);
    REQUIRE(ok);

    double omega_initial = longitude_of_perigee(R.col(0), V.col(0));
    double omega_final = longitude_of_perigee(R.col(N-1), V.col(N-1));

    if (std::isnan(omega_initial) || std::isnan(omega_final)) {
        SKIP("Cannot compute argument of perigee for nearly circular orbit");
    }

    double delta_omega = omega_final - omega_initial;
    if (delta_omega > M_PI) {
        delta_omega -= 2.0 * M_PI;
    }
    if (delta_omega < -M_PI) {
        delta_omega += 2.0 * M_PI;
    }

    REQUIRE(std::abs(delta_omega) > 1e-6);
}

TEST_CASE("J2 RK4 energy conservation", "[orbit][j2][rk4]")
{
    constexpr int N = 50;
    
    Eigen::Vector3d r0(7000e3, 0.0, 0.0);
    Eigen::Vector3d v0(0.0, 7.546e3, 0.0);

    Eigen::Matrix<double,1,limits::MAX_LENGTH_TRAJ> jtime;
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> R;
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> V;

    for (int i = 0; i < N; ++i)
        jtime(i) = (2451545.0 + i * 0.01 / (N - 1)) / 36525.0;

    bool ok = orbits::compute_orbit_J2_RK4(r0, v0, jtime, N, R, V);
    REQUIRE(ok);

    double e0 = specific_energy(R.col(0), V.col(0));
    double max_relative_change = 0.0;
    
    for (int i = 0; i < N; ++i)
    {
        Eigen::Vector3d r_i = R.col(i);
        Eigen::Vector3d v_i = V.col(i);
        double e = specific_energy(r_i, v_i);
        double rel_change = std::abs((e - e0) / e0);
        max_relative_change = std::max(max_relative_change, rel_change);
    }

    REQUIRE(max_relative_change < 1e-4);
}

TEST_CASE("J2 RK4 vs Keplerian deviation", "[orbit][j2][rk4]")
{
    constexpr int N = 100;
    
    Eigen::Vector3d r0(7000e3, 0.0, 0.0);
    Eigen::Vector3d v0(0.0, 7.5e3, 0.0);

    Eigen::Matrix<double,1,limits::MAX_LENGTH_TRAJ> jtime;
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> R_kepl, R_j2;
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> V_kepl, V_j2;

    for (int i = 0; i < N; ++i)
        jtime(i) = (2451545.0 + i * 1e-4) / 36525.0;

    bool ok_kepl = orbits::compute_orbit_keplerian(r0, v0, jtime, N, R_kepl, V_kepl);
    bool ok_j2 = orbits::compute_orbit_J2_RK4(r0, v0, jtime, N, R_j2, V_j2);

    REQUIRE(ok_kepl);
    REQUIRE(ok_j2);

    double pos_diff_initial = (R_j2.col(0) - R_kepl.col(0)).norm();
    double pos_diff_final = (R_j2.col(N-1) - R_kepl.col(N-1)).norm();

    REQUIRE(pos_diff_initial < 1.0);
    REQUIRE(pos_diff_final > 100.0);
}

TEST_CASE("J2 RK4 circular orbit altitude variation", "[orbit][j2][rk4]")
{
    constexpr int N = 100;
    
    double alt = 500e3;
    double rmag = RE + alt;

    Eigen::Vector3d r0(rmag, 0.0, 0.0);
    Eigen::Vector3d v0(0.0, std::sqrt(MU / rmag), 0.0);

    Eigen::Matrix<double,1,limits::MAX_LENGTH_TRAJ> jtime;
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> R;
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> V;

    for (int i = 0; i < N; ++i)
        jtime(i) = (2451545.0 + i * 0.1 / (N - 1)) / 36525.0;

    bool ok = orbits::compute_orbit_J2_RK4(r0, v0, jtime, N, R, V);
    REQUIRE(ok);

    double alt_min = 1e99;
    double alt_max = 0.0;

    for (int i = 0; i < N; ++i)
    {
        double r_mag = R.col(i).norm();
        double alt_i = r_mag - RE;
        alt_min = std::min(alt_min, alt_i);
        alt_max = std::max(alt_max, alt_i);
    }

    double alt_variation = alt_max - alt_min;
    REQUIRE(alt_variation > 1e3);
    REQUIRE(alt_variation < 100e3);
}

TEST_CASE("J2 RK4 single time point", "[orbit][j2][rk4]")
{
    constexpr int N = 1;
    
    Eigen::Vector3d r0(7000e3, 0.0, 0.0);
    Eigen::Vector3d v0(0.0, 7.5e3, 0.0);

    Eigen::Matrix<double,1,limits::MAX_LENGTH_TRAJ> jtime;
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> R;
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> V;

    jtime(0) = 2451545.0 / 36525.0;

    bool ok = orbits::compute_orbit_J2_RK4(r0, v0, jtime, N, R, V);
    REQUIRE(ok);

    REQUIRE(std::abs(R(0, 0) - r0.x()) < 1e-9);
    REQUIRE(std::abs(R(1, 0) - r0.y()) < 1e-9);
    REQUIRE(std::abs(R(2, 0) - r0.z()) < 1e-9);
    REQUIRE(std::abs(V(0, 0) - v0.x()) < 1e-9);
    REQUIRE(std::abs(V(1, 0) - v0.y()) < 1e-9);
    REQUIRE(std::abs(V(2, 0) - v0.z()) < 1e-9);
}

TEST_CASE("J2 RK4 equatorial orbit", "[orbit][j2][rk4]")
{
    constexpr int N = 100;
    
    double alt = 400e3;
    double rmag = RE + alt;

    Eigen::Vector3d r0(rmag, 0.0, 0.0);
    Eigen::Vector3d v0(0.0, std::sqrt(MU / rmag), 0.0);

    Eigen::Matrix<double,1,limits::MAX_LENGTH_TRAJ> jtime;
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> R;
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> V;

    for (int i = 0; i < N; ++i)
        jtime(i) = (2451545.0 + i * 0.05 / (N - 1)) / 36525.0;

    bool ok = orbits::compute_orbit_J2_RK4(r0, v0, jtime, N, R, V);
    REQUIRE(ok);

    for (int i = 0; i < N; ++i)
    {
        Eigen::Vector3d r_i = R.col(i);
        Eigen::Vector3d v_i = V.col(i);
        double inc = inclination(r_i, v_i);
        REQUIRE(std::abs(inc) < 1e-6);
    }
}

TEST_CASE("J2 RK4 polar orbit", "[orbit][j2][rk4]")
{
    constexpr int N = 100;
    
    double alt = 400e3;
    double rmag = RE + alt;

    Eigen::Vector3d r0(rmag, 0.0, 0.0);
    Eigen::Vector3d v0(0.0, 0.0, std::sqrt(MU / rmag));

    Eigen::Matrix<double,1,limits::MAX_LENGTH_TRAJ> jtime;
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> R;
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> V;

    for (int i = 0; i < N; ++i)
        jtime(i) = (2451545.0 + i * 0.05 / (N - 1)) / 36525.0;

    bool ok = orbits::compute_orbit_J2_RK4(r0, v0, jtime, N, R, V);
    REQUIRE(ok);

    for (int i = 0; i < N; ++i)
    {
        Eigen::Vector3d r_i = R.col(i);
        Eigen::Vector3d v_i = V.col(i);
        double inc = inclination(r_i, v_i);
        REQUIRE(std::abs(inc - M_PI / 2.0) < 1e-6);
    }
}

TEST_CASE("J2 RK4 backwards propagation", "[orbit][j2][rk4]")
{
    constexpr int N = 20;
    
    Eigen::Vector3d r0(7000e3, 0.0, 0.0);
    Eigen::Vector3d v0(0.0, 7.5e3, 0.0);

    Eigen::Matrix<double,1,limits::MAX_LENGTH_TRAJ> jtime;
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> R;
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> V;

    for (int i = 0; i < N; ++i)
        jtime(i) = (2451545.0 + 0.01 - i * 0.01 / (N - 1)) / 36525.0;

    bool ok = orbits::compute_orbit_J2_RK4(r0, v0, jtime, N, R, V);
    REQUIRE(ok);

    for (int i = 0; i < N; ++i)
    {
        REQUIRE(std::isfinite(R(0,i)));
        REQUIRE(std::isfinite(R(1,i)));
        REQUIRE(std::isfinite(R(2,i)));
        REQUIRE(std::isfinite(V(0,i)));
        REQUIRE(std::isfinite(V(1,i)));
        REQUIRE(std::isfinite(V(2,i)));
    }
}

TEST_CASE("J2 RK4 high eccentricity", "[orbit][j2][rk4]")
{
    constexpr int N = 100;
    
    double a = 26500e3;
    double e = 0.7;
    double r_peri = a * (1.0 - e);
    double v_peri = std::sqrt(MU * (2.0 / r_peri - 1.0 / a));

    Eigen::Vector3d r0(r_peri, 0.0, 0.0);
    Eigen::Vector3d v0(0.0, v_peri, 0.0);

    Eigen::Matrix<double,1,limits::MAX_LENGTH_TRAJ> jtime;
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> R;
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> V;

    for (int i = 0; i < N; ++i)
        jtime(i) = (2451545.0 + i * 0.05 / (N - 1)) / 36525.0;

    bool ok = orbits::compute_orbit_J2_RK4(r0, v0, jtime, N, R, V);
    REQUIRE(ok);

    double r_min = 1e99;
    double r_max = 0.0;

    for (int i = 0; i < N; ++i)
    {
        double r_mag = R.col(i).norm();
        r_min = std::min(r_min, r_mag);
        r_max = std::max(r_max, r_mag);
        REQUIRE(std::isfinite(r_mag));
    }

    REQUIRE(r_max / r_min > 1.5);
}
