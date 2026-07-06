#include <catch2/catch_test_macros.hpp>
#include <Eigen/Dense>

#include <saltro/orbit_generation/orbits/compute_orbit.h>
#include <saltro/orbit_generation/sun/compute_sun.h>
#include <saltro/orbit_generation/magnetics/compute_magnetic.h>
#include <saltro/orbit_generation/eclipse/compute_eclipse.h>
#include <saltro/orbit_generation/density/compute_density.h>
#include <saltro/limits.h>

// The model-dispatcher functions (compute_orbit/sun/magnetic/eclipse/density)
// route an integer model id to a concrete implementation and return false on an
// unknown id (default case). Only the concrete implementations were tested; the
// dispatchers themselves had no coverage, so an out-of-range model id silently
// returning false (rather than e.g. routing to a wrong model) was unverified.

using namespace saltro;

namespace {
template <int Rows>
using Traj = Eigen::Matrix<double, Rows, limits::MAX_LENGTH_TRAJ>;

constexpr int LEN = 4;

Eigen::Matrix<double, 1, limits::MAX_LENGTH_TRAJ> makeJtime() {
    Eigen::Matrix<double, 1, limits::MAX_LENGTH_TRAJ> jtime;
    jtime.setZero();
    for (int i = 0; i < LEN; ++i) jtime(i) = 2451545.0 + i * 1e-3;  // JD, ~86 s steps
    return jtime;
}
}  // namespace

TEST_CASE("compute_orbit dispatcher: valid model routes, unknown model returns false",
          "[orbit_generation][dispatcher]") {
    const Eigen::Vector3d r0(7000e3, 0.0, 0.0);
    const Eigen::Vector3d v0(0.0, 7546.0, 0.0);  // ~circular LEO
    const auto jtime = makeJtime();
    Traj<3> R, V;
    R.setZero(); V.setZero();

    REQUIRE(orbits::compute_orbit(r0, v0, jtime, LEN, /*KEPLERIAN*/ 0, R, V));
    REQUIRE(R.leftCols(LEN).allFinite());
    REQUIRE_FALSE(orbits::compute_orbit(r0, v0, jtime, LEN, /*unknown*/ 99, R, V));
    REQUIRE_FALSE(orbits::compute_orbit(r0, v0, jtime, LEN, -1, R, V));
}

TEST_CASE("compute_sun dispatcher: valid model routes, unknown model returns false",
          "[orbit_generation][dispatcher]") {
    const auto jtime = makeJtime();
    Traj<3> R, S;
    R.setZero(); S.setZero();
    REQUIRE(orbits::compute_sun(R, jtime, LEN, /*NOAA*/ 0, S));
    REQUIRE(S.leftCols(LEN).allFinite());
    REQUIRE_FALSE(orbits::compute_sun(R, jtime, LEN, /*unknown*/ 99, S));
}

TEST_CASE("compute_magnetic dispatcher: valid model routes, unknown model returns false",
          "[orbit_generation][dispatcher]") {
    const Eigen::Vector3d r0(7000e3, 0.0, 0.0), v0(0.0, 7546.0, 0.0);
    const auto jtime = makeJtime();
    Traj<3> R, V, B;
    R.setZero(); V.setZero(); B.setZero();
    REQUIRE(orbits::compute_orbit(r0, v0, jtime, LEN, 0, R, V));
    REQUIRE(orbits::compute_magnetic(R, jtime, LEN, /*DIPOLE*/ 0, B));
    REQUIRE(B.leftCols(LEN).allFinite());
    REQUIRE_FALSE(orbits::compute_magnetic(R, jtime, LEN, /*unknown*/ 99, B));
}

TEST_CASE("compute_eclipse dispatcher: unknown model returns false",
          "[orbit_generation][dispatcher]") {
    const Eigen::Vector3d r0(7000e3, 0.0, 0.0), v0(0.0, 7546.0, 0.0);
    const auto jtime = makeJtime();
    Traj<3> R, V, S, ecl;
    R.setZero(); V.setZero(); S.setZero(); ecl.setZero();
    REQUIRE(orbits::compute_orbit(r0, v0, jtime, LEN, 0, R, V));
    REQUIRE(orbits::compute_sun(R, jtime, LEN, 0, S));
    REQUIRE_FALSE(orbits::compute_eclipse(R, jtime, LEN, /*unknown*/ 99, ecl));
}

TEST_CASE("compute_density dispatcher: unknown model returns false",
          "[orbit_generation][dispatcher]") {
    const Eigen::Vector3d r0(7000e3, 0.0, 0.0), v0(0.0, 7546.0, 0.0);
    const auto jtime = makeJtime();
    Traj<3> R, V, S;
    R.setZero(); V.setZero(); S.setZero();
    Eigen::Matrix<double, 1, limits::MAX_LENGTH_TRAJ> rho;
    rho.setZero();
    REQUIRE(orbits::compute_orbit(r0, v0, jtime, LEN, 0, R, V));
    REQUIRE(orbits::compute_sun(R, jtime, LEN, 0, S));
    REQUIRE_FALSE(orbits::compute_density(R, S, LEN, /*unknown*/ 99, rho));
}
