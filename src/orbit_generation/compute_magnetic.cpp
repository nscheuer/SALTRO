#include <saltro/orbit_generation/magnetics/compute_magnetic_dipole.h>

namespace saltro::orbit {

bool compute_magnetic(
    const Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& R,
    const Eigen::Matrix<double, 1, saltro::limits::MAX_LENGTH_TRAJ>& jtime,
    const int jtime_length,
    const int magnetic_model,

    Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& B
) {
    switch (magnetic_model) {
        case 0: // DIPOLE
            return compute_magnetic_dipole(R, jtime, jtime_length, B);
        default:
            return false;
    }
}

}