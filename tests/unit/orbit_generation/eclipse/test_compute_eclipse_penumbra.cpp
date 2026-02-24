#include <catch2/catch_test_macros.hpp>
#include <Eigen/Dense>
#include <saltro/orbit_generation/eclipse/compute_eclipse_penumbra.h>
#include <saltro/limits.h>
#include <saltro/constants/constants.h>
#include <cmath>

using namespace saltro;

namespace {

bool expected_penumbra_eclipse(const Eigen::Vector3d& r_sat, const Eigen::Vector3d& sun_vec)
{
    const double R_earth = saltro::constants::R_EARTH;
    const double R_sun = saltro::constants::R_SUN;
    const double eps = 1e-12;

    const double sun_norm2 = sun_vec.squaredNorm();
    const double r_norm2 = r_sat.squaredNorm();

    if (sun_norm2 < eps || r_norm2 < eps) return false;

    const double sun_norm = std::sqrt(sun_norm2);
    const double r_norm = std::sqrt(r_norm2);
    const double inv_sun = 1.0 / sun_norm;
    const double inv_r = 1.0 / r_norm;

    double cos_psi = (-r_sat).dot(sun_vec) * inv_r * inv_sun;
    if (cos_psi <= 0.0) return false;
    if (cos_psi > 1.0) cos_psi = 1.0;

    double sin_alpha_e = R_earth * inv_r;
    double sin_alpha_s = R_sun * inv_sun;

    if (sin_alpha_e > 1.0) sin_alpha_e = 1.0;
    if (sin_alpha_s > 1.0) sin_alpha_s = 1.0;

    const double cos_alpha_e = std::sqrt(std::max(0.0, 1.0 - sin_alpha_e * sin_alpha_e));
    const double cos_alpha_s = std::sqrt(std::max(0.0, 1.0 - sin_alpha_s * sin_alpha_s));

    const double cos_sum = (cos_alpha_e * cos_alpha_s) - (sin_alpha_e * sin_alpha_s);

    return cos_psi >= cos_sum;
}

} // namespace

TEST_CASE("eclipse penumbra basic functionality", "[eclipse][penumbra]")
{
    constexpr int N = 5;

    Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> R, S;
    const double Re = saltro::constants::R_EARTH;

    for (int i = 0; i < N; ++i) {
        R.col(i) << Re + 400e3, 0.0, 0.0;
        S.col(i) << 1.496e11, 0.0, 0.0;
    }

    bool ok = orbits::compute_eclipse_penumbra(R, N, S);
    REQUIRE(ok);

    for (int i = 0; i < N; ++i) {
        double sun_norm = S.col(i).norm();
        REQUIRE(sun_norm > 1e10);
    }
}

TEST_CASE("eclipse penumbra detects shadow", "[eclipse][penumbra]")
{
    constexpr int N = 1;

    Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> R, S;
    const double Re = saltro::constants::R_EARTH;

    R.col(0) << -(Re + 400e3), 0.0, 0.0;
    S.col(0) << 1.496e11, 0.0, 0.0;

    bool ok = orbits::compute_eclipse_penumbra(R, N, S);
    REQUIRE(ok);

    REQUIRE(S.col(0).norm() < 1e-10);
}

TEST_CASE("eclipse penumbra off-axis inside cone", "[eclipse][penumbra]")
{
    constexpr int N = 1;

    Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> R, S;
    const double Re = saltro::constants::R_EARTH;
    const double Rs = saltro::constants::R_SUN;
    const double r = Re + 400e3;
    const double d = 1.496e11;

    const double alpha_e = std::asin(Re / r);
    const double alpha_s = std::asin(Rs / d);
    const double psi = (alpha_e + alpha_s) - 1e-4;

    R.col(0) << -r * std::cos(psi), r * std::sin(psi), 0.0;
    S.col(0) << d, 0.0, 0.0;

    bool ok = orbits::compute_eclipse_penumbra(R, N, S);
    REQUIRE(ok);

    REQUIRE(S.col(0).norm() < 1e-10);
}

TEST_CASE("eclipse penumbra off-axis outside cone", "[eclipse][penumbra]")
{
    constexpr int N = 1;

    Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> R, S;
    const double Re = saltro::constants::R_EARTH;
    const double Rs = saltro::constants::R_SUN;
    const double r = Re + 400e3;
    const double d = 1.496e11;

    const double alpha_e = std::asin(Re / r);
    const double alpha_s = std::asin(Rs / d);
    const double psi = (alpha_e + alpha_s) + 1e-4;

    R.col(0) << -r * std::cos(psi), r * std::sin(psi), 0.0;
    S.col(0) << d, 0.0, 0.0;

    Eigen::Vector3d sun_original = S.col(0);

    bool ok = orbits::compute_eclipse_penumbra(R, N, S);
    REQUIRE(ok);

    REQUIRE((S.col(0) - sun_original).norm() < 1e-6);
}

TEST_CASE("eclipse penumbra arbitrary sun direction", "[eclipse][penumbra]")
{
    constexpr int N = 8;

    Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> R, S;
    const double Re = saltro::constants::R_EARTH;
    const double r = Re + 400e3;

    Eigen::Vector3d sun_dir(1.0, 0.5, 0.3);
    sun_dir.normalize();
    const double sun_dist = 1.496e11;

    for (int i = 0; i < N; ++i) {
        double theta = 2.0 * M_PI * i / N;
        R.col(i) << r * std::cos(theta), r * std::sin(theta), 0.0;
        S.col(i) = sun_dist * sun_dir;
    }

    bool ok = orbits::compute_eclipse_penumbra(R, N, S);
    REQUIRE(ok);

    for (int i = 0; i < N; ++i) {
        bool expected = expected_penumbra_eclipse(R.col(i), sun_dist * sun_dir);
        if (expected) {
            REQUIRE(S.col(i).norm() < 1e-10);
        } else {
            REQUIRE(std::abs(S.col(i).norm() - sun_dist) < 1e4);
        }
    }
}

TEST_CASE("eclipse penumbra zero sun vector handling", "[eclipse][penumbra]")
{
    constexpr int N = 2;

    Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> R, S;
    const double Re = saltro::constants::R_EARTH;

    R.col(0) << Re + 400e3, 0.0, 0.0;
    S.col(0) << 1.496e11, 0.0, 0.0;

    R.col(1) << -(Re + 400e3), 0.0, 0.0;
    S.col(1) << 0.0, 0.0, 0.0;

    bool ok = orbits::compute_eclipse_penumbra(R, N, S);
    REQUIRE(ok);

    REQUIRE(S.col(0).norm() > 1e10);
    REQUIRE(S.col(1).norm() < 1e-10);
}

TEST_CASE("eclipse penumbra orbit expected pattern", "[eclipse][penumbra]")
{
    constexpr int N = 24;

    Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> R, S;
    const double Re = saltro::constants::R_EARTH;
    const double r = Re + 400e3;

    for (int i = 0; i < N; ++i) {
        double theta = 2.0 * M_PI * i / N;
        R.col(i) << r * std::cos(theta), r * std::sin(theta), 0.0;
        S.col(i) << 1.496e11, 0.0, 0.0;
    }

    bool ok = orbits::compute_eclipse_penumbra(R, N, S);
    REQUIRE(ok);

    for (int i = 0; i < N; ++i) {
        bool expected = expected_penumbra_eclipse(R.col(i), Eigen::Vector3d(1.496e11, 0.0, 0.0));
        if (expected) {
            REQUIRE(S.col(i).norm() < 1e-10);
        } else {
            REQUIRE(S.col(i).norm() > 1e10);
        }
    }
}
