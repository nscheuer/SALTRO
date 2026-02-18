#include <saltro/orbit_generation/magnetics/compute_magnetic.h>
#include <saltro/orbit_generation/magnetics/compute_magnetic_dipole.h>
#include <saltro/orbit_generation/magnetics/compute_magnetic_igrf8.h>
#include <saltro/orbit_generation/magnetics/compute_magnetic_igrf13.h>

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
        case 1: // IGRF8
            return compute_magnetic_igrf8(R, jtime, jtime_length, B);
        case 2: // IGRF13
            return compute_magnetic_igrf13(R, jtime, jtime_length, B);
        default:
            return false;
    }
}

}