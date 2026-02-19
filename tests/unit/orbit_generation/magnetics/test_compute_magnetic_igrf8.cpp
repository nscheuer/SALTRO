#include <catch2/catch_test_macros.hpp>
#include <Eigen/Dense>
#include <saltro/orbit_generation/magnetics/compute_magnetic_igrf8.h>
#include <saltro/limits.h>
#include <cmath>

using namespace saltro;

TEST_CASE("IGRF8 dimensions and validity", "[magnetic][igrf8]")
{
    constexpr int N = 5;

    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> R;
    Eigen::Matrix<double,1,limits::MAX_LENGTH_TRAJ> jtime;
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> B;

    const double Re = 6378136.3;

    for (int i = 0; i < N; ++i)
    {
        R.col(i) << Re + 400e3, 0.0, 0.0;
        jtime(i) = 2451545.0 + i * 1e-3;
    }

    bool ok = orbits::compute_magnetic_igrf8(R, jtime, N, B);
    REQUIRE(ok);

    for (int i = 0; i < N; ++i)
    {
        double mag = B.col(i).norm();
        REQUIRE(std::isfinite(mag));
        REQUIRE(mag > 0.0);
    }
}


TEST_CASE("IGRF8 magnitude sanity in LEO", "[magnetic][igrf8]")
{
    constexpr int N = 3;

    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> R;
    Eigen::Matrix<double,1,limits::MAX_LENGTH_TRAJ> jtime;
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> B;

    const double Re = 6378136.3;
    double alts[N] = {200e3, 400e3, 700e3};

    for (int i = 0; i < N; ++i)
    {
        R.col(i) << Re + alts[i], 0.0, 0.0;
        jtime(i) = 2451545.0;
    }

    bool ok = orbits::compute_magnetic_igrf8(R, jtime, N, B);
    REQUIRE(ok);

    double B0 = B.col(0).norm();
    double B1 = B.col(1).norm();
    double B2 = B.col(2).norm();

    REQUIRE(B0 > 1e-7);
    REQUIRE(B0 < 1e-3);

    REQUIRE(B1 > 1e-8);
    REQUIRE(B1 < 1e-3);

    REQUIRE(B2 > 1e-9);
    REQUIRE(B2 < 1e-3);

    REQUIRE(B0 > B1);
    REQUIRE(B1 > B2);
}


TEST_CASE("IGRF8 circular orbit smoothness", "[magnetic][igrf8]")
{
    constexpr int N = 60;

    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> R;
    Eigen::Matrix<double,1,limits::MAX_LENGTH_TRAJ> jtime;
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> B;

    const double Re = 6378136.3;
    const double alt = 500e3;
    const double r = Re + alt;

    for (int i = 0; i < N; ++i)
    {
        double theta = 2.0 * M_PI * i / N;

        R.col(i) << r * std::cos(theta),
                    r * std::sin(theta),
                    0.0;

        jtime(i) = 2451545.0 + i * 1e-3;
    }

    bool ok = orbits::compute_magnetic_igrf8(R, jtime, N, B);
    REQUIRE(ok);

    double Bmin = 1e99;
    double Bmax = 0.0;

    for (int i = 0; i < N; ++i)
    {
        double mag = B.col(i).norm();
        REQUIRE(std::isfinite(mag));
        REQUIRE(mag > 0.0);

        Bmin = std::min(Bmin, mag);
        Bmax = std::max(Bmax, mag);
    }

    REQUIRE(Bmax / Bmin < 100.0);
}