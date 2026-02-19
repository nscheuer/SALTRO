#include <catch2/catch_test_macros.hpp>
#include <Eigen/Dense>
#include <saltro/orbit_generation/orbits/compute_orbit_keplerian.h>
#include <saltro/limits.h>
#include <cmath>

using namespace saltro;

static constexpr double MU = 3.986004418e14;

double specific_energy(const Eigen::Vector3d& r, const Eigen::Vector3d& v)
{
    return 0.5 * v.squaredNorm() - MU / r.norm();
}


TEST_CASE("Keplerian dimensions and validity", "[orbit][kepler]")
{
    constexpr int N = 10;

    Eigen::Vector3d r0(7000e3, 0.0, 0.0);
    Eigen::Vector3d v0(0.0, 7.5e3, 0.0);

    Eigen::Matrix<double,1,limits::MAX_LENGTH_TRAJ> jtime;
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> R;
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> V;

    for (int i = 0; i < N; ++i)
        jtime(i) = 2451545.0 + i * 1e-3;

    bool ok = orbits::compute_orbit_keplerian(r0, v0, jtime, N, R, V);
    REQUIRE(ok);

    for (int i = 0; i < N; ++i)
    {
        REQUIRE(std::isfinite(R(0,i)));
        REQUIRE(std::isfinite(V(0,i)));
    }
}


TEST_CASE("Keplerian energy conservation", "[orbit][kepler]")
{
    constexpr int N = 20;

    Eigen::Vector3d r0(7000e3, 0.0, 0.0);
    Eigen::Vector3d v0(0.0, 7546.0, 0.0);

    Eigen::Matrix<double,1,limits::MAX_LENGTH_TRAJ> jtime;
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> R;
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> V;

    for (int i = 0; i < N; ++i)
        jtime(i) = 2451545.0 + i * 1e-3;

    bool ok = orbits::compute_orbit_keplerian(r0, v0, jtime, N, R, V);
    REQUIRE(ok);

    double e0 = specific_energy(R.col(0), V.col(0));

    for (int i = 1; i < N; ++i)
    {
        double e = specific_energy(R.col(i), V.col(i));
        REQUIRE(std::abs(e - e0) / std::abs(e0) < 1e-6);
    }
}


TEST_CASE("Keplerian circular orbit radius constant", "[orbit][kepler]")
{
    constexpr int N = 60;

    const double Re = 6378136.3;
    const double alt = 400e3;
    const double rmag = Re + alt;

    Eigen::Vector3d r0(rmag, 0.0, 0.0);
    Eigen::Vector3d v0(0.0, std::sqrt(MU / rmag), 0.0);

    Eigen::Matrix<double,1,limits::MAX_LENGTH_TRAJ> jtime;
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> R;
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> V;

    for (int i = 0; i < N; ++i)
        jtime(i) = 2451545.0 + i * 1e-3;

    bool ok = orbits::compute_orbit_keplerian(r0, v0, jtime, N, R, V);
    REQUIRE(ok);

    double rmin = 1e99;
    double rmax = 0.0;

    for (int i = 0; i < N; ++i)
    {
        double r = R.col(i).norm();
        rmin = std::min(rmin, r);
        rmax = std::max(rmax, r);
    }

    REQUIRE(rmax - rmin < 1e3); // ~1 km tolerance
}


TEST_CASE("Keplerian orbital periodicity", "[orbit][kepler]")
{
    const double Re = 6378136.3;
    const double alt = 500e3;
    const double rmag = Re + alt;

    Eigen::Vector3d r0(rmag, 0.0, 0.0);
    Eigen::Vector3d v0(0.0, std::sqrt(MU / rmag), 0.0);

    double period = 2.0 * M_PI * std::sqrt(std::pow(rmag,3) / MU);

    constexpr int N = 50;

    Eigen::Matrix<double,1,limits::MAX_LENGTH_TRAJ> jtime;
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> R;
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> V;

    for (int i = 0; i < N; ++i)
        jtime(i) = 2451545.0 + (period / 86400.0) * i / (N - 1);

    bool ok = orbits::compute_orbit_keplerian(r0, v0, jtime, N, R, V);
    REQUIRE(ok);

    Eigen::Vector3d r_final = R.col(N-1);
    Eigen::Vector3d v_final = V.col(N-1);

    REQUIRE((r_final - r0).norm() < 5e3);
    REQUIRE((v_final - v0).norm() < 5.0);
}