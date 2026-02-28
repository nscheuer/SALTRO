#include <saltro/optimizer/validation/validate_satellite.h>
#include <cmath>

namespace saltro::optimizer::validation {

bool validateSatellite(const Satellite& satellite, std::string& error_msg) {
    const auto& J = satellite.inertia();
    
    if (!J.allFinite()) {
        error_msg = "Inertia matrix contains non-finite values";
        return false;
    }

    for (int i = 0; i < 3; ++i) {
        if (J(i, i) <= 0.0) {
            error_msg = "Inertia diagonal elements must be positive";
            return false;
        }
    }

    if (std::abs(J.determinant()) < 1e-12) {
        error_msg = "Inertia matrix is singular";
        return false;
    }

    if (satellite.numMTQ() == 0 && satellite.numRW() == 0) {
        error_msg = "Satellite must have at least one actuator";
        return false;
    }

    for (int i = 0; i < satellite.numMTQ(); ++i) {
        const auto& mtq = satellite.getMTQ(i);
        if (std::abs(mtq.u_max()) < 1e-12) {
            error_msg = "MTQ control limit is zero";
            return false;
        }
        if (!mtq.axis().allFinite() || mtq.axis().norm() < 1e-12) {
            error_msg = "MTQ axis invalid";
            return false;
        }
    }

    for (int i = 0; i < satellite.numRW(); ++i) {
        const auto& rw = satellite.getRW(i);
        if (std::abs(rw.u_max()) < 1e-12) {
            error_msg = "RW torque limit is zero";
            return false;
        }
        if (rw.wheelInertia() <= 0.0 || !std::isfinite(rw.wheelInertia())) {
            error_msg = "RW inertia invalid";
            return false;
        }
        if (rw.momentumMax() <= 0.0 || !std::isfinite(rw.momentumMax())) {
            error_msg = "RW momentum limit invalid";
            return false;
        }
        if (!rw.axis().allFinite() || rw.axis().norm() < 1e-12) {
            error_msg = "RW axis invalid";
            return false;
        }
    }

    return true;
}

}
