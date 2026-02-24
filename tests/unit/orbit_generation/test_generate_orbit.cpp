#include <catch2/catch_test_macros.hpp>
#include <Eigen/Dense>
#include <saltro/orbit_generation/generate_orbit.h>
#include <saltro/limits.h>
#include <cmath>

using namespace saltro;

static Eigen::VectorXd make_jcenturies(int N, double span_days, double T0 = 0.22)
{
    Eigen::VectorXd T(N);

    if (N == 1)
    {
        T(0) = T0;
        return T;
    }

    for (int i = 0; i < N; ++i)
        T(i) = T0 + (span_days * i / (N - 1)) / 36525.0;

    return T;
}

static void valid_initial_conditions(Eigen::Vector3d& r0,
                                     Eigen::Vector3d& v0)
{
    const double Re = 6378136.3;
    r0 << Re + 400e3, 0.0, 0.0;
    v0 << 0.0, 7660.0, 0.0;
}

TEST_CASE("Generate orbit basic dimensions", "[orbit][generate]")
{
    constexpr int N = 8;

    Eigen::Vector3d r0, v0;
    valid_initial_conditions(r0, v0);

    Eigen::Matrix<double,1,limits::MAX_LENGTH_TRAJ> jtime{};
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> R{}, V{}, B{}, S{};
    Eigen::Matrix<double,1,limits::MAX_LENGTH_TRAJ> rho{};

    auto T = make_jcenturies(N, 0.01);
    for (int i = 0; i < N; ++i)
        jtime(i) = T(i);

    bool ok = orbits::generate_orbit(
        r0, v0, jtime, N,
        0, 0, 0, 0, 0,
        R, V, B, S, rho
    );

    REQUIRE(ok);

    for (int i = 0; i < N; ++i)
    {
        REQUIRE(std::isfinite(R(0,i)));
        REQUIRE(std::isfinite(V(0,i)));
        REQUIRE(std::isfinite(B(0,i)));
        REQUIRE(std::isfinite(S(0,i)));
        REQUIRE(std::isfinite(rho(i)));
    }
}

TEST_CASE("Generate orbit single point", "[orbit][generate]")
{
    constexpr int N = 1;

    Eigen::Vector3d r0, v0;
    valid_initial_conditions(r0, v0);

    Eigen::Matrix<double,1,limits::MAX_LENGTH_TRAJ> jtime{};
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> R{}, V{}, B{}, S{};
    Eigen::Matrix<double,1,limits::MAX_LENGTH_TRAJ> rho{};

    jtime(0) = 0.22;

    bool ok = orbits::generate_orbit(
        r0, v0, jtime, N,
        0, 0, 0, 0, 0,
        R, V, B, S, rho
    );

    REQUIRE(ok);
}

TEST_CASE("Generate orbit non-increasing time", "[orbit][generate]")
{
    constexpr int N = 3;

    Eigen::Vector3d r0, v0;
    valid_initial_conditions(r0, v0);

    Eigen::Matrix<double,1,limits::MAX_LENGTH_TRAJ> jtime{};
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> R{}, V{}, B{}, S{};
    Eigen::Matrix<double,1,limits::MAX_LENGTH_TRAJ> rho{};

    jtime(0) = 0.22;
    jtime(1) = 0.22;   // invalid
    jtime(2) = 0.221;

    bool ok = orbits::generate_orbit(
        r0, v0, jtime, N,
        0, 0, 0, 0, 0,
        R, V, B, S, rho
    );

    REQUIRE_FALSE(ok);
}

TEST_CASE("Generate orbit non-finite time", "[orbit][generate]")
{
    constexpr int N = 3;

    Eigen::Vector3d r0, v0;
    valid_initial_conditions(r0, v0);

    Eigen::Matrix<double,1,limits::MAX_LENGTH_TRAJ> jtime{};
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> R{}, V{}, B{}, S{};
    Eigen::Matrix<double,1,limits::MAX_LENGTH_TRAJ> rho{};

    jtime(0) = 0.22;
    jtime(1) = std::numeric_limits<double>::quiet_NaN();
    jtime(2) = 0.23;

    bool ok = orbits::generate_orbit(
        r0, v0, jtime, N,
        0, 0, 0, 0, 0,
        R, V, B, S, rho
    );

    REQUIRE_FALSE(ok);
}

TEST_CASE("Generate orbit rejects Julian Date", "[orbit][generate]")
{
    constexpr int N = 2;

    Eigen::Vector3d r0, v0;
    valid_initial_conditions(r0, v0);

    Eigen::Matrix<double,1,limits::MAX_LENGTH_TRAJ> jtime{};
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> R{}, V{}, B{}, S{};
    Eigen::Matrix<double,1,limits::MAX_LENGTH_TRAJ> rho{};

    jtime(0) = 2459000.0;
    jtime(1) = 2459000.0001;

    bool ok = orbits::generate_orbit(
        r0, v0, jtime, N,
        0, 0, 0, 0, 0,
        R, V, B, S, rho
    );

    REQUIRE_FALSE(ok);
}

TEST_CASE("Generate orbit invalid r0 magnitude", "[orbit][generate]")
{
    constexpr int N = 5;

    Eigen::Vector3d r0, v0;
    valid_initial_conditions(r0, v0);

    r0 << 1e5, 0.0, 0.0;  // too small

    Eigen::Matrix<double,1,limits::MAX_LENGTH_TRAJ> jtime{};
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> R{}, V{}, B{}, S{};
    Eigen::Matrix<double,1,limits::MAX_LENGTH_TRAJ> rho{};

    auto T = make_jcenturies(N, 0.01);
    for (int i = 0; i < N; ++i)
        jtime(i) = T(i);

    bool ok = orbits::generate_orbit(
        r0, v0, jtime, N,
        0, 0, 0, 0, 0,
        R, V, B, S, rho
    );

    REQUIRE_FALSE(ok);
}

TEST_CASE("Generate orbit invalid v0 magnitude", "[orbit][generate]")
{
    constexpr int N = 5;

    Eigen::Vector3d r0, v0;
    valid_initial_conditions(r0, v0);

    v0 << 0.0, 100.0, 0.0;  // too small

    Eigen::Matrix<double,1,limits::MAX_LENGTH_TRAJ> jtime{};
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> R{}, V{}, B{}, S{};
    Eigen::Matrix<double,1,limits::MAX_LENGTH_TRAJ> rho{};

    auto T = make_jcenturies(N, 0.01);
    for (int i = 0; i < N; ++i)
        jtime(i) = T(i);

    bool ok = orbits::generate_orbit(
        r0, v0, jtime, N,
        0, 0, 0, 0, 0,
        R, V, B, S, rho
    );

    REQUIRE_FALSE(ok);
}

TEST_CASE("Generate orbit non-finite state", "[orbit][generate]")
{
    constexpr int N = 5;

    Eigen::Vector3d r0, v0;
    valid_initial_conditions(r0, v0);

    r0(0) = std::numeric_limits<double>::quiet_NaN();

    Eigen::Matrix<double,1,limits::MAX_LENGTH_TRAJ> jtime{};
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> R{}, V{}, B{}, S{};
    Eigen::Matrix<double,1,limits::MAX_LENGTH_TRAJ> rho{};

    auto T = make_jcenturies(N, 0.01);
    for (int i = 0; i < N; ++i)
        jtime(i) = T(i);

    bool ok = orbits::generate_orbit(
        r0, v0, jtime, N,
        0, 0, 0, 0, 0,
        R, V, B, S, rho
    );

    REQUIRE_FALSE(ok);
}

TEST_CASE("Generate orbit exceeds MAX_LENGTH_TRAJ", "[orbit][generate]")
{
    constexpr int N = limits::MAX_LENGTH_TRAJ + 1;

    Eigen::Vector3d r0, v0;
    valid_initial_conditions(r0, v0);

    Eigen::Matrix<double,1,limits::MAX_LENGTH_TRAJ> jtime{};
    Eigen::Matrix<double,3,limits::MAX_LENGTH_TRAJ> R{}, V{}, B{}, S{};
    Eigen::Matrix<double,1,limits::MAX_LENGTH_TRAJ> rho{};

    bool ok = orbits::generate_orbit(
        r0, v0, jtime, N,
        0, 0, 0, 0, 0,
        R, V, B, S, rho
    );

    REQUIRE_FALSE(ok);
}