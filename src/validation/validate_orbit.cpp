#include <saltro/optimizer/validation/validate_orbit.h>
#include <cmath>

namespace saltro::optimizer::validation {

bool validateOrbitInitialConditions(const Eigen::Vector3d& r0, const Eigen::Vector3d& v0, std::string& error_msg) {
    if (!r0.allFinite()) {
        error_msg = "r0 contains non-finite values";
        return false;
    }

    if (!v0.allFinite()) {
        error_msg = "v0 contains non-finite values";
        return false;
    }

    const double r_norm = r0.norm();
    const double v_norm = v0.norm();

    if (r_norm < 6400.0 || r_norm > 8000.0) {
        error_msg = "r0 magnitude out of LEO range (6400-8000 km)";
        return false;
    }

    if (v_norm < 6.0 || v_norm > 9.0) {
        error_msg = "v0 magnitude out of LEO range (6-9 km/s)";
        return false;
    }

    return true;
}

}
