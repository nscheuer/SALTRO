#include <saltro/orbit_generation/generate_orbit.h>
#include <saltro/orbit_generation/orbits/compute_orbit.h>
#include <saltro/orbit_generation/magnetics/compute_magnetic.h>
#include <saltro/orbit_generation/sun/compute_sun.h>

namespace saltro::orbit {

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
    if (!compute_orbit(r0, v0, jtime, jtime_length, orbit_model, R, V)) return false;
    if (!compute_magnetic(R, jtime, jtime_length, magnetic_model, B)) return false;
    if (!compute_sun(R, jtime, jtime_length, sun_model, S)) return false;
    // if (!compute_density(R, jtime_length, density_model, rho)) return false;
    return true;
}

}