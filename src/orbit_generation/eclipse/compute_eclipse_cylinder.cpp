#include <saltro/orbit_generation/eclipse/compute_eclipse_cylinder.h>
#include <saltro/constants/constants.h>

#include <cmath>

namespace saltro::orbits {

bool compute_eclipse_cylinder(
    const Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& R,
    const int jtime_length,

    Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& S
) {
    // Earth's mean radius in meters
    const double R_earth = saltro::constants::R_EARTH;

    for (int k = 0; k < jtime_length; ++k) {
        // Get sun vector (pos_sun - pos_sat)
        Eigen::Vector3d sun_vec = S.col(k);
        
        // Check if sun vector is zero (already eclipsed or invalid)
        if (sun_vec.norm() < 1e-10) continue;

        // Normalize sun direction
        Eigen::Vector3d sun_dir = sun_vec.normalized();

        // Satellite position
        Eigen::Vector3d r_sat = R.col(k);

        // Check if satellite is on night side
        // (negative dot product means behind Earth relative to sun)
        double dot_product = r_sat.dot(sun_dir);
        if (dot_product >= 0.0) continue;  // Sunlit side, not in eclipse

        // Compute perpendicular distance from satellite to sun direction
        // Project satellite position onto sun direction, then get perpendicular distance
        double proj_length = r_sat.dot(sun_dir);
        Eigen::Vector3d proj_point = proj_length * sun_dir;
        Eigen::Vector3d perp_vec = r_sat - proj_point;
        double perp_distance = perp_vec.norm();

        // Check if perpendicular distance is within Earth's radius
        if (perp_distance < R_earth) {
            // Satellite is in eclipse, zero out the sun vector
            S.col(k) = Eigen::Vector3d::Zero();
        }
    }

    return true;
}

}
