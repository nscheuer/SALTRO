#include <saltro/orbit_generation/generate_orbit.h>
#include <saltro/orbit_generation/orbits/compute_orbit.h>
#include <saltro/orbit_generation/magnetics/compute_magnetic.h>
#include <saltro/orbit_generation/sun/compute_sun.h>
#include <saltro/orbit_generation/density/compute_density.h>

#include <cmath>
#include <limits>

namespace saltro::orbits {

static bool validate_generate_orbit(
    const Eigen::Vector3d& r0,
    const Eigen::Vector3d& v0,
    const Eigen::Matrix<double, 1, saltro::limits::MAX_LENGTH_TRAJ>& jtime,
    const int jtime_length
) {
    using limits::MAX_LENGTH_TRAJ;

    if (jtime_length <= 0) return false;
    if (jtime_length > MAX_LENGTH_TRAJ) return false;

    // ---- Finite checks ----
    if (!r0.allFinite()) return false;
    if (!v0.allFinite()) return false;

    // ---- Magnitude sanity (LEO-scale) ----
    const double rnorm = r0.norm();
    const double vnorm = v0.norm();

    // Rough LEO bounds (not strict physics enforcement)
    if (rnorm < 6.0e6 || rnorm > 8.0e6) return false;     // 6000–8000 km
    if (vnorm < 6.0e3 || vnorm > 9.0e3) return false;     // 6–9 km/s

    // ---- jtime structure checks ----
    for (int i = 0; i < jtime_length; ++i)
    {
        if (!std::isfinite(jtime(i)))
            return false;
    }

    // Must be strictly increasing
    for (int i = 1; i < jtime_length; ++i)
    {
        if (jtime(i) <= jtime(i-1))
            return false;
    }

    // ---- Julian centuries sanity ----
    // If user accidentally passes Julian Date (~2.4e6), reject it.
    const double T0 = jtime(0);
    if (std::abs(T0) > 10.0) return false;   // centuries should be small (~0–1)

    return true;
}

bool generate_orbit(
    const Eigen::Vector3d& r0,
    const Eigen::Vector3d& v0,
    const Eigen::Matrix<double, 1, saltro::limits::MAX_LENGTH_TRAJ>& jtime,
    const int jtime_length,
    const int orbit_model,
    const int magnetic_model,
    const int sun_model,
    const int density_model,

    Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& R,
    Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& V,
    Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& B,
    Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& S,
    Eigen::Matrix<double, 1, saltro::limits::MAX_LENGTH_TRAJ>& rho
) {
    if (!validate_generate_orbit(r0, v0, jtime, jtime_length)) return false;
    if (!compute_orbit(r0, v0, jtime, jtime_length, orbit_model, R, V)) return false;
    if (!compute_magnetic(R, jtime, jtime_length, magnetic_model, B)) return false;
    if (!compute_sun(R, jtime, jtime_length, sun_model, S)) return false;
    if (!compute_density(R, S, jtime_length, density_model, rho)) return false;
    return true;
}

}