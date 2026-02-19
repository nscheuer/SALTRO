#include <catch2/catch_test_macros.hpp>
#include <Eigen/Dense>
#include <saltro/orbit_generation/sun/compute_sun_noaa.h>
#include <saltro/limits.h>
#include <cmath>

using namespace saltro;

static Eigen::VectorXd make_jcenturies(int N, double span_days, double T0 = 0.22)
{
    Eigen::VectorXd T(N);
    for (int i = 0; i < N; ++i)
        T(i) = T0 + (span_days * i / (N - 1)) / 36525.0;
    return T;
}

static Eigen::MatrixXd unitize(const Eigen::MatrixXd& V)
{
    Eigen::MatrixXd U = V;
    for (int i = 0; i < V.cols(); ++i)
        U.col(i) /= V.col(i).norm();
    return U;
}

TEST_CASE("Sun NOAA dimensions and validity", "[sun][noaa]")
{
    constexpr int N = 5;
    const double Re = 6378136.3;

    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> R{};
    Eigen::Matrix<double,1,limits::MAX_LENGTH_TRAJ> jtime{};
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> S{};

    auto T = make_jcenturies(N, 0.01);

    for (int i = 0; i < N; ++i)
    {
        jtime(i) = T(i);
        R.col(i) << Re + 400e3, 0.0, 0.0;
    }

    bool ok = orbits::compute_sun_noaa(R, jtime, N, S);
    REQUIRE(ok);

    for (int i = 0; i < N; ++i)
    {
        REQUIRE(std::isfinite(S(0,i)));
        REQUIRE(std::isfinite(S(1,i)));
        REQUIRE(std::isfinite(S(2,i)));
    }
}

TEST_CASE("Sun NOAA unit direction vectors", "[sun][noaa]")
{
    constexpr int N = 10;
    const double Re = 6378136.3;

    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> R{};
    Eigen::Matrix<double,1,limits::MAX_LENGTH_TRAJ> jtime{};
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> S{};

    auto T = make_jcenturies(N, 0.02);

    for (int i = 0; i < N; ++i)
    {
        jtime(i) = T(i);
        R.col(i) << Re + 500e3, 0.0, 0.0;
    }

    bool ok = orbits::compute_sun_noaa(R, jtime, N, S);
    REQUIRE(ok);

    for (int i = 0; i < N; ++i)
    {
        double norm_m = S.col(i).norm();

        // Distance sanity (~1 AU)
        REQUIRE(norm_m > 1e11);
        REQUIRE(norm_m < 2e11);

        Eigen::Vector3d u = S.col(i) / norm_m;
        double unorm = u.norm();

        REQUIRE(unorm > 0.9);
        REQUIRE(unorm < 1.1);
    }
}

TEST_CASE("Sun NOAA time variation smooth", "[sun][noaa]")
{
    constexpr int N = 50;
    const double Re = 6378136.3;

    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> R{};
    Eigen::Matrix<double,1,limits::MAX_LENGTH_TRAJ> jtime{};
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> S{};

    auto T = make_jcenturies(N, 1.0);

    for (int i = 0; i < N; ++i)
    {
        jtime(i) = T(i);
        R.col(i) << Re + 400e3, 0.0, 0.0;
    }

    bool ok = orbits::compute_sun_noaa(R, jtime, N, S);
    REQUIRE(ok);

    // compare unit directions
    for (int i = 1; i < N; ++i)
    {
        Eigen::Vector3d u0 = S.col(i-1).normalized();
        Eigen::Vector3d u1 = S.col(i).normalized();

        double diff = (u1 - u0).norm();
        REQUIRE(diff < 0.1);
    }
}

TEST_CASE("Sun NOAA circular orbit consistency", "[sun][noaa]")
{
    constexpr int N = 60;
    const double Re = 6378136.3;
    const double alt = 400e3;
    const double r = Re + alt;

    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> R{};
    Eigen::Matrix<double,1,limits::MAX_LENGTH_TRAJ> jtime{};
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> S{};

    auto T = make_jcenturies(N, 0.02);

    for (int i = 0; i < N; ++i)
    {
        jtime(i) = T(i);
        double theta = 2.0 * M_PI * i / N;
        R.col(i) << r * std::cos(theta), r * std::sin(theta), 0.0;
    }

    bool ok = orbits::compute_sun_noaa(R, jtime, N, S);
    REQUIRE(ok);

    for (int i = 0; i < N; ++i)
    {
        double norm_m = S.col(i).norm();

        REQUIRE(norm_m > 1e11);
        REQUIRE(norm_m < 2e11);

        Eigen::Vector3d u = S.col(i) / norm_m;
        double unorm = u.norm();

        REQUIRE(unorm > 0.9);
        REQUIRE(unorm < 1.1);
    }
}