#include <saltro/orbit_generation/density/compute_density.h>
#include <saltro/orbit_generation/density/compute_density_harrispriester.h>

namespace saltro::orbits {

bool compute_density(
    const Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& R,
    const Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& S,
    const int jtime_length,
    const int density_model,
    Eigen::Matrix<double, 1, saltro::limits::MAX_LENGTH_TRAJ>& rho
) {
    switch (density_model) {
        case 0:
            return compute_denisty_harrispriester(R, S, jtime_length, rho);
        default:
            return false;
    }
}

}