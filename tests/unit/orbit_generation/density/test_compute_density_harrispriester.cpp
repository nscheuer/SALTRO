#include <catch2/catch_test_macros.hpp>
#include <Eigen/Dense>
#include <saltro/orbit_generation/density/compute_density_harrispriester.h>
#include <saltro/limits.h>
#include <cmath>

using namespace saltro;

TEST_CASE("HarrisPriester dimensions and basic validity", "[density][hp]")
{
    constexpr int N = 5;

    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> R;
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> S;
    Eigen::Matrix<double,1,limits::MAX_LENGTH_TRAJ> rho;

    const double Re = 6378136.3;

    for (int i = 0; i < N; ++i)
    {
        double alt = 200e3 + i * 100e3;
        double r = Re + alt;

        R.col(i) << r, 0.0, 0.0;
        S.col(i) << 1.0, 0.0, 0.0;
    }

    bool ok = orbits::compute_density_harrispriester(R, S, N, rho);

    REQUIRE(ok);

    for (int i = 0; i < N; ++i)
    {
        REQUIRE(std::isfinite(rho(i)));
        REQUIRE(rho(i) > 0.0);
    }
}

TEST_CASE("HarrisPriester order of magnitude sanity", "[density][hp]")
{
    constexpr int N = 3;

    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> R;
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> S;
    Eigen::Matrix<double,1,limits::MAX_LENGTH_TRAJ> rho;

    const double Re = 6378136.3;

    double alts[N] = {200e3, 400e3, 700e3};

    for (int i = 0; i < N; ++i)
    {
        double r = Re + alts[i];
        R.col(i) << r, 0.0, 0.0;
        S.col(i) << 1.0, 0.0, 0.0;
    }

    bool ok = orbits::compute_density_harrispriester(R, S, N, rho);
    REQUIRE(ok);

    REQUIRE(rho(0) > 1e-12);       // 200 km should be around 1e-9 to 1e-10
    REQUIRE(rho(0) < 1e-6);

    REQUIRE(rho(1) > 1e-14);       // 400 km typically 1e-12 to 1e-11
    REQUIRE(rho(1) < 1e-8);

    REQUIRE(rho(2) > 1e-16);       // 700 km typically 1e-14 to 1e-13
    REQUIRE(rho(2) < 1e-9);

    REQUIRE(rho(0) > rho(1));
    REQUIRE(rho(1) > rho(2));
}

TEST_CASE("HarrisPriester circular orbit smoothness", "[density][hp]")
{
    constexpr int N = 50;

    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> R;
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> S;
    Eigen::Matrix<double,1,limits::MAX_LENGTH_TRAJ> rho;

    const double Re = 6378136.3;
    const double alt = 400e3;
    const double r = Re + alt;

    for (int i = 0; i < N; ++i)
    {
        double theta = 2.0 * M_PI * i / N;

        R.col(i) << r * std::cos(theta),
                    r * std::sin(theta),
                    0.0;

        S.col(i) << 1.0, 0.0, 0.0;
    }

    bool ok = orbits::compute_density_harrispriester(R, S, N, rho);
    REQUIRE(ok);

    for (int i = 0; i < N; ++i)
    {
        REQUIRE(std::isfinite(rho(i)));
        REQUIRE(rho(i) > 0.0);
    }

    double rho_min = rho.head(N).minCoeff();
    double rho_max = rho.head(N).maxCoeff();

    REQUIRE(rho_max / rho_min < 20.0); // HP bulge shouldn't explode
}