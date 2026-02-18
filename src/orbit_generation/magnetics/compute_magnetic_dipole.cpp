#include <saltro/orbit_generation/magnetics/compute_magnetic_dipole.h>
#include <saltro/math/frames.h>
#include <saltro/constants/constants.h>

namespace saltro::orbit {

bool compute_magnetic_dipole(
    const Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& R,
    const Eigen::Matrix<double, 1, saltro::limits::MAX_LENGTH_TRAJ>& jtime,
    const int jtime_length,

    Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& B
) {
    const double sT = std::sin(saltro::constants::TILT_RAD);
    const double cT = std::cos(saltro::constants::TILT_RAD);
    const double cL = std::cos(saltro::constants::LON_RAD);
    const double sL = std::sin(saltro::constants::LON_RAD);

    const Eigen::Vector3d mhat_ecef(-(sT * cL), -(sT * sL), -cT);

    for (int k = 0; k < jtime_length; ++k) {
        const Eigen::Vector3d r_eci = R.col(k);
        const double rmag = r_eci.norm();

        if (rmag <= 0.0) {
            B.col(k).setZero();
            continue;
        }

        const Eigen::Matrix3d C_ecef_eci = saltro::math::eci_to_ecef_dcm(jtime(k));
        const Eigen::Vector3d r_ecef = C_ecef_eci * r_eci;

        const double rmag_ecef = r_ecef.norm();
        if (rmag_ecef <= 0.0) {
            B.col(k).setZero();
            continue;
        }

        const Eigen::Vector3d rhat_ecef = r_ecef / rmag_ecef;

        const double mdotr = mhat_ecef.dot(rhat_ecef);
        const Eigen::Vector3d term = (3.0 * mdotr) * rhat_ecef - mhat_ecef;
        
        const double inv_r3 = 1.0 / (rmag_ecef * rmag_ecef * rmag_ecef);
        const Eigen::Vector3d B_ecef = (saltro::constants::K * inv_r3) * term;

        const Eigen::Matrix3d C_eci_ecef = C_ecef_eci.transpose();
        const Eigen::Vector3d B_eci = C_eci_ecef * B_ecef;

        B.col(k) = B_eci;
    }

    return true;
}

}