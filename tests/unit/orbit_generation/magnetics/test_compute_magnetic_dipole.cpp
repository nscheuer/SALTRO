#include <catch2/catch_test_macros.hpp>
#include <Eigen/Dense>
#include <saltro/orbit_generation/magnetics/compute_magnetic_dipole.h>
#include <saltro/limits.h>
#include <cmath>

using namespace saltro;

TEST_CASE("MagneticDipole dimensions and validity", "[magnetic][dipole]")
{
    constexpr int N = 5;

    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> R;
    Eigen::Matrix<double,1,limits::MAX_LENGTH_TRAJ> jtime;
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> B;

    const double Re = 6378136.3;

    for (int i = 0; i < N; ++i)
    {
        double alt = 400e3;
        double r = Re + alt;

        R.col(i) << r, 0.0, 0.0;
        jtime(i) = 2451545.0 + i * 1e-3;
    }

    bool ok = orbits::compute_magnetic_dipole(R, jtime, N, B);
    REQUIRE(ok);

    for (int i = 0; i < N; ++i)
    {
        REQUIRE(std::isfinite(B(0,i)));
        REQUIRE(std::isfinite(B(1,i)));
        REQUIRE(std::isfinite(B(2,i)));

        double mag = B.col(i).norm();
        REQUIRE(mag > 0.0);
    }
}


TEST_CASE("MagneticDipole order of magnitude sanity", "[magnetic][dipole]")
{
    constexpr int N = 3;

    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> R;
    Eigen::Matrix<double,1,limits::MAX_LENGTH_TRAJ> jtime;
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> B;

    const double Re = 6378136.3;
    double alts[N] = {200e3, 400e3, 700e3};

    for (int i = 0; i < N; ++i)
    {
        double r = Re + alts[i];
        R.col(i) << r, 0.0, 0.0;
        jtime(i) = 2451545.0;
    }

    bool ok = orbits::compute_magnetic_dipole(R, jtime, N, B);
    REQUIRE(ok);

    double Bmag0 = B.col(0).norm();
    double Bmag1 = B.col(1).norm();
    double Bmag2 = B.col(2).norm();

    REQUIRE(Bmag0 > 1e-7);
    REQUIRE(Bmag0 < 1e-3);

    REQUIRE(Bmag1 > 1e-8);
    REQUIRE(Bmag1 < 1e-3);

    REQUIRE(Bmag2 > 1e-9);
    REQUIRE(Bmag2 < 1e-3);

    REQUIRE(Bmag0 > Bmag1);
    REQUIRE(Bmag1 > Bmag2);
}


TEST_CASE("MagneticDipole circular orbit smoothness", "[magnetic][dipole]")
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

    bool ok = orbits::compute_magnetic_dipole(R, jtime, N, B);
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

    REQUIRE(Bmax / Bmin < 50.0); // dipole shouldn't explode
}