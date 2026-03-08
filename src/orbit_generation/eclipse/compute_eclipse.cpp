#include <saltro/orbit_generation/eclipse/compute_eclipse.h>
#include <saltro/orbit_generation/eclipse/compute_eclipse_cylinder.h>
#include <saltro/orbit_generation/eclipse/compute_eclipse_penumbra.h>

namespace saltro::orbits {

bool compute_eclipse(
    const Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& R,
    const Eigen::Matrix<double, 1, saltro::limits::MAX_LENGTH_TRAJ>& jtime,
    const int jtime_length,
    const int eclipse_model,

    Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& S
) {
    (void)jtime;

    switch (eclipse_model) {
        case 0: // Cylindrical shadow model
            return compute_eclipse_cylinder(R, jtime_length, S);
        case 1: // Analytical penumbra/umbra shadow cone model
            return compute_eclipse_penumbra(R, jtime_length, S);
        default:
            return false;
    }
}

}
