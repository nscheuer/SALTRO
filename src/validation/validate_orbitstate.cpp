#include <saltro/validation/validate_orbitstate.h>

#include <saltro/constants/constants.h>

#include <cmath>

namespace saltro::validation {

bool validateOrbitState(
    const Eigen::Vector3d& r0,
    const Eigen::Vector3d& v0,
    std::string& error_msg
) {
    constexpr double MIN_RADIUS_UNIT_HINT = 1e5;         // m
    constexpr double MIN_SPEED_UNIT_HINT = 100.0;        // m/s
    constexpr double MAX_REASONABLE_SPEED = 2e4;         // m/s
    constexpr double MIN_LEO_ALTITUDE = 120e3;           // m
    constexpr double MAX_LEO_ALTITUDE = 2000e3;          // m
    constexpr double MAX_LEO_ECCENTRICITY = 0.2;

    const double mu = saltro::constants::MU_EARTH;
    const double re = saltro::constants::R_EARTH;

    if (!r0.allFinite()) {
        error_msg = "r0 contains non-finite values";
        return false;
    }

    if (!v0.allFinite()) {
        error_msg = "v0 contains non-finite values";
        return false;
    }

    const double rmag = r0.norm();
    const double vmag = v0.norm();

    if (rmag < MIN_RADIUS_UNIT_HINT) {
        error_msg = "r0 magnitude too small; expected meters (did you provide kilometers?)";
        return false;
    }

    if (vmag < MIN_SPEED_UNIT_HINT) {
        error_msg = "v0 magnitude too small; expected m/s (did you provide km/s?)";
        return false;
    }

    if (vmag > MAX_REASONABLE_SPEED) {
        error_msg = "v0 magnitude unreasonably large";
        return false;
    }

    if (rmag <= re) {
        error_msg = "initial position is inside Earth";
        return false;
    }

    const Eigen::Vector3d h = r0.cross(v0);
    const double hmag = h.norm();
    if (!std::isfinite(hmag) || hmag <= 1e-6) {
        error_msg = "angular momentum is too small (degenerate orbit state)";
        return false;
    }

    const double specific_energy = 0.5 * v0.squaredNorm() - mu / rmag;
    if (!std::isfinite(specific_energy)) {
        error_msg = "specific orbital energy is not finite";
        return false;
    }

    if (specific_energy >= 0.0) {
        error_msg = "specific orbital energy is non-negative (orbit not bound to Earth)";
        return false;
    }

    const double a = -mu / (2.0 * specific_energy);
    if (!std::isfinite(a) || a <= 0.0) {
        error_msg = "semi-major axis is invalid";
        return false;
    }

    const Eigen::Vector3d e_vec = (v0.cross(h) / mu) - (r0 / rmag);
    const double e = e_vec.norm();
    if (!std::isfinite(e)) {
        error_msg = "eccentricity is not finite";
        return false;
    }

    if (e >= 1.0) {
        error_msg = "orbit is not closed (eccentricity >= 1)";
        return false;
    }

    if (e > MAX_LEO_ECCENTRICITY) {
        error_msg = "orbit is too elliptical for LEO use (eccentricity > 0.2)";
        return false;
    }

    const double rp = a * (1.0 - e);
    const double ra = a * (1.0 + e);
    if (!std::isfinite(rp) || !std::isfinite(ra)) {
        error_msg = "perigee/apogee radii are not finite";
        return false;
    }

    if (rp <= re) {
        error_msg = "perigee radius is inside Earth";
        return false;
    }

    const double perigee_alt = rp - re;
    const double apogee_alt = ra - re;

    if (perigee_alt < MIN_LEO_ALTITUDE) {
        error_msg = "perigee altitude below LEO bounds";
        return false;
    }

    if (apogee_alt > MAX_LEO_ALTITUDE) {
        error_msg = "apogee altitude above LEO bounds";
        return false;
    }

    return true;
}

} // namespace saltro::validation