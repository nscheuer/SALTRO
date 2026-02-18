#include <saltro/orbit_generation/sun/compute_sun.h>
#include <saltro/orbit_generation/sun/compute_sun_noaa.h>

namespace saltro::orbit {

bool compute_sun(
    Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& R,
    const Eigen::Matrix<double, 1, saltro::limits::MAX_LENGTH_TRAJ>& jtime,
    const int jtime_length,
    const int sun_model,
    Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& S
) {
    switch (sun_model) {
        case 0: // NOAA
            return compute_sun_noaa(R, jtime, jtime_length, S);
        case 1: // NRELSPA

        case 2: // VSOP87

        default:
            return false;
    }
}


}