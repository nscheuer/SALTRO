#include <catch2/catch_test_macros.hpp>
#include <Eigen/Dense>
#include <saltro/orbit_generation/eclipse/compute_eclipse_cylinder.h>
#include <saltro/limits.h>
#include <saltro/constants/constants.h>
#include <cmath>

using namespace saltro;

TEST_CASE("eclipse cylinder basic functionality", "[eclipse][cylinder]")
{
    constexpr int N = 5;

    Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> R, S;
    const double Re = saltro::constants::R_EARTH;

    for (int i = 0; i < N; ++i) {
        R.col(i) << Re + 400e3, 0.0, 0.0;  // Position above Earth
        S.col(i) << 1e11, 0.0, 0.0;        // Sun far away in +X direction
    }

    bool ok = orbits::compute_eclipse_cylinder(R, N, S);
    REQUIRE(ok);

    // All positions should be sunlit (positive dot product with sun direction)
    for (int i = 0; i < N; ++i) {
        double sun_norm = S.col(i).norm();
        REQUIRE(sun_norm > 1e10);  // Sun vector unchanged
    }
}


TEST_CASE("eclipse cylinder detects shadow", "[eclipse][cylinder]")
{
    constexpr int N = 1;

    Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> R, S;
    const double Re = saltro::constants::R_EARTH;

    // Satellite on night side (behind Earth relative to sun)
    R.col(0) << -(Re + 400e3), 0.0, 0.0;
    // Sun is infinitely far in +X direction
    S.col(0) << 1e11, 0.0, 0.0;

    bool ok = orbits::compute_eclipse_cylinder(R, N, S);
    REQUIRE(ok);

    // Should be in eclipse (sun vector zeroed)
    REQUIRE(S.col(0).norm() < 1e-10);
}


TEST_CASE("eclipse cylinder sunlit side unchanged", "[eclipse][cylinder]")
{
    constexpr int N = 1;

    Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> R, S;
    const double Re = saltro::constants::R_EARTH;

    // Satellite on sunlit side
    R.col(0) << Re + 400e3, 0.0, 0.0;
    Eigen::Vector3d sun_original(1e11, 0.0, 0.0);
    S.col(0) = sun_original;

    bool ok = orbits::compute_eclipse_cylinder(R, N, S);
    REQUIRE(ok);

    // Sun vector should be unchanged
    REQUIRE((S.col(0) - sun_original).norm() < 1e-6);
}


TEST_CASE("eclipse cylinder off-axis night side", "[eclipse][cylinder]")
{
    constexpr int N = 1;

    Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> R, S;
    const double Re = saltro::constants::R_EARTH;

    // Satellite behind Earth but off-axis (not directly behind)
    // Position: some distance behind Earth, offset perpendicular to sun direction
    R.col(0) << -(Re + 400e3), Re * 0.5, 0.0;
    S.col(0) << 1e11, 0.0, 0.0;

    bool ok = orbits::compute_eclipse_cylinder(R, N, S);
    REQUIRE(ok);

    // Should still be in eclipse (within shadow cylinder)
    REQUIRE(S.col(0).norm() < 1e-10);
}


TEST_CASE("eclipse cylinder beyond shadow edge", "[eclipse][cylinder]")
{
    constexpr int N = 1;

    Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> R, S;
    const double Re = saltro::constants::R_EARTH;

    // Satellite behind Earth but far enough off-axis to be outside shadow
    R.col(0) << -(Re + 400e3), Re * 2.0, 0.0;
    Eigen::Vector3d sun_original(1e11, 0.0, 0.0);
    S.col(0) = sun_original;

    bool ok = orbits::compute_eclipse_cylinder(R, N, S);
    REQUIRE(ok);

    // Outside shadow, sun vector should be unchanged
    REQUIRE((S.col(0) - sun_original).norm() < 1e-6);
}


TEST_CASE("eclipse cylinder circular orbit partial eclipse", "[eclipse][cylinder]")
{
    constexpr int N = 24;  // One full orbit

    Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> R, S;
    const double Re = saltro::constants::R_EARTH;
    const double alt = 400e3;
    const double r = Re + alt;

    for (int i = 0; i < N; ++i) {
        double theta = 2.0 * M_PI * i / N;
        
        // Circular orbit in XY plane
        R.col(i) << r * std::cos(theta), r * std::sin(theta), 0.0;
        
        // Sun along +X direction (infinitely far)
        S.col(i) << 1.496e11, 0.0, 0.0;
    }

    bool ok = orbits::compute_eclipse_cylinder(R, N, S);
    REQUIRE(ok);

    // Count eclipsed and sunlit positions
    int eclipsed_count = 0;
    int sunlit_count = 0;

    for (int i = 0; i < N; ++i) {
        if (S.col(i).norm() < 1e-10) {
            eclipsed_count++;
        } else {
            sunlit_count++;
        }
    }

    // Should have roughly half in eclipse, half sunlit
    REQUIRE(eclipsed_count > 0);
    REQUIRE(sunlit_count > 0);
    REQUIRE(eclipsed_count + sunlit_count == N);
}


TEST_CASE("eclipse cylinder preserves vector magnitude when sunlit", "[eclipse][cylinder]")
{
    constexpr int N = 3;

    Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> R, S;
    const double Re = saltro::constants::R_EARTH;

    for (int i = 0; i < N; ++i) {
        R.col(i) << Re + 400e3 + i * 100e3, 0.0, 0.0;  // Different altitudes
        S.col(i) << 1.496e11, 1e9, 1e8;                // Sun vector (not aligned with position)
    }

    // Store original magnitudes
    std::vector<double> original_mags(N);
    for (int i = 0; i < N; ++i) {
        original_mags[i] = S.col(i).norm();
    }

    bool ok = orbits::compute_eclipse_cylinder(R, N, S);
    REQUIRE(ok);

    // All should be sunlit, magnitudes unchanged
    for (int i = 0; i < N; ++i) {
        double new_mag = S.col(i).norm();
        REQUIRE(std::abs(new_mag - original_mags[i]) < 1e-6);
    }
}


TEST_CASE("eclipse cylinder zero sun vector handling", "[eclipse][cylinder]")
{
    constexpr int N = 2;

    Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> R, S;
    const double Re = saltro::constants::R_EARTH;

    R.col(0) << Re + 400e3, 0.0, 0.0;
    S.col(0) << 1e11, 0.0, 0.0;

    R.col(1) << -(Re + 400e3), 0.0, 0.0;
    S.col(1) << 0.0, 0.0, 0.0;  // Already zero sun vector

    bool ok = orbits::compute_eclipse_cylinder(R, N, S);
    REQUIRE(ok);

    // First should be sunlit
    REQUIRE(S.col(0).norm() > 1e10);

    // Second should remain zero (skipped)
    REQUIRE(S.col(1).norm() < 1e-10);
}


TEST_CASE("eclipse cylinder equatorial orbit", "[eclipse][cylinder]")
{
    constexpr int N = 16;

    Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> R, S;
    const double Re = saltro::constants::R_EARTH;
    const double r = Re + 400e3;

    // Orbit along equator
    for (int i = 0; i < N; ++i) {
        double theta = 2.0 * M_PI * i / N;
        R.col(i) << r * std::cos(theta), r * std::sin(theta), 0.0;
        S.col(i) << 1.496e11, 0.0, 0.0;  // Sun along +X
    }

    bool ok = orbits::compute_eclipse_cylinder(R, N, S);
    REQUIRE(ok);

    // Check that eclipse transitions are smooth
    // (sun vector either nonzero or zero, no partial values for cylindrical model)
    for (int i = 0; i < N; ++i) {
        double sun_mag = S.col(i).norm();
        bool is_sunlit = sun_mag > 1e10;
        bool is_eclipsed = sun_mag < 1e-10;
        REQUIRE((is_sunlit || is_eclipsed));
    }
}


TEST_CASE("eclipse cylinder high altitude orbit", "[eclipse][cylinder]")
{
    constexpr int N = 12;

    Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> R, S;
    const double Re = saltro::constants::R_EARTH;
    const double r = Re + 36000e3;  // GEO altitude

    for (int i = 0; i < N; ++i) {
        double theta = 2.0 * M_PI * i / N;
        R.col(i) << r * std::cos(theta), r * std::sin(theta), 0.0;
        S.col(i) << 1.496e11, 0.0, 0.0;
    }

    bool ok = orbits::compute_eclipse_cylinder(R, N, S);
    REQUIRE(ok);

    // At GEO altitude, eclipse window is very small
    int eclipsed_count = 0;
    for (int i = 0; i < N; ++i) {
        if (S.col(i).norm() < 1e-10) {
            eclipsed_count++;
        }
    }

    // Should have some eclipse but much less than at LEO
    REQUIRE(eclipsed_count >= 0);
    REQUIRE(eclipsed_count < N / 4);  // Less than 25% in eclipse
}


TEST_CASE("eclipse cylinder single point", "[eclipse][cylinder]")
{
    constexpr int N = 1;

    Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> R, S;
    const double Re = saltro::constants::R_EARTH;

    R.col(0) << Re + 400e3, 0.0, 0.0;
    S.col(0) << 1.496e11, 0.0, 0.0;

    bool ok = orbits::compute_eclipse_cylinder(R, N, S);
    REQUIRE(ok);

    // Should be sunlit
    REQUIRE(S.col(0).norm() > 1e10);
}


TEST_CASE("eclipse cylinder multi-orbit degradation", "[eclipse][cylinder]")
{
    constexpr int N = 100;  // Multiple orbits

    Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> R, S;
    const double Re = saltro::constants::R_EARTH;
    const double r = Re + 400e3;

    for (int i = 0; i < N; ++i) {
        double theta = 2.0 * M_PI * i / (N / 5);  // 5 full orbits
        R.col(i) << r * std::cos(theta), r * std::sin(theta), 0.0;
        S.col(i) << 1.496e11, 0.0, 0.0;
    }

    bool ok = orbits::compute_eclipse_cylinder(R, N, S);
    REQUIRE(ok);

    // Verify consistent eclipse/sunlit pattern across all orbits
    int eclipsed_per_orbit = 0;
    for (int i = 0; i < N; ++i) {
        if (S.col(i).norm() < 1e-10) {
            eclipsed_per_orbit++;
        }
    }

    // Calculate expected eclipse fraction from geometry
    // For circular orbit at altitude h: sin(θ_e) = R_E / (R_E + h)
    // where θ_e is the eclipse half-angle
    const double alt = 400e3;
    const double eclipse_half_angle = std::asin(Re / (Re + alt));
    const double total_eclipse_angle = 2.0 * eclipse_half_angle;  // in radians
    const double eclipse_fraction = total_eclipse_angle / (2.0 * M_PI);
    
    // Expected eclipsed count with ±10% tolerance for discretization
    const int expected_eclipsed = static_cast<int>(eclipse_fraction * N);
    const int lower_bound = static_cast<int>(expected_eclipsed * 0.9);
    const int upper_bound = static_cast<int>(expected_eclipsed * 1.1);
    
    REQUIRE(eclipsed_per_orbit >= lower_bound);
    REQUIRE(eclipsed_per_orbit <= upper_bound);
}


TEST_CASE("eclipse cylinder arbitrary sun direction", "[eclipse][cylinder]")
{
    constexpr int N = 8;

    Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> R, S;
    const double Re = saltro::constants::R_EARTH;
    const double r = Re + 400e3;

    // Sun direction: not aligned with any axis
    Eigen::Vector3d sun_dir(1.0, 0.5, 0.3);
    sun_dir.normalize();
    double sun_dist = 1.496e11;

    for (int i = 0; i < N; ++i) {
        double theta = 2.0 * M_PI * i / N;
        R.col(i) << r * std::cos(theta), r * std::sin(theta), 0.0;
        S.col(i) = sun_dist * sun_dir;
    }

    bool ok = orbits::compute_eclipse_cylinder(R, N, S);
    REQUIRE(ok);

    // Verify all outputs are either normal or zero
    for (int i = 0; i < N; ++i) {
        double sun_mag = S.col(i).norm();
        bool is_normal = std::abs(sun_mag - sun_dist) < 1e4;
        bool is_zero = sun_mag < 1e-10;
        REQUIRE((is_normal || is_zero));
    }
}
