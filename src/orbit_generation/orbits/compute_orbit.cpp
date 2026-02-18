#include <saltro/orbit_generation/orbits/compute_orbit.h>
#include <saltro/orbit_generation/orbits/compute_orbit_keplerian.h>

namespace saltro::orbits {

bool compute_orbit(
    const Eigen::Vector3d& r0,
    const Eigen::Vector3d& v0,
    const Eigen::Matrix<double, 1, saltro::limits::MAX_LENGTH_TRAJ>& jtime,
    const int jtime_length,
    const int orbit_model,

    Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& R,
    Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& V
) {
    switch (orbit_model) {
        case 0: // KEPLERIAN
            return compute_orbit_keplerian(r0, v0, jtime, jtime_length, R, V);
        default:
            return false;
    }
}

}