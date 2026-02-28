#include <saltro/validation/validate_juliantime.h>

#include <cmath>

namespace saltro::validation {

bool validateJulianTime(
    const Eigen::Ref<const Eigen::VectorXd>& jtime,
    std::string& error_msg
) {
    constexpr double MIN_JTIME = 0.20;
    constexpr double MAX_JTIME = 0.40;

    if (jtime.size() <= 0) {
        error_msg = "jtime is empty";
        return false;
    }

    for (Eigen::Index i = 0; i < jtime.size(); ++i) {
        const double t = jtime(i);

        if (!std::isfinite(t)) {
            error_msg = "jtime contains non-finite values";
            return false;
        }

        if (t == 0.0) {
            error_msg = "jtime contains zero values";
            return false;
        }

        if (t < MIN_JTIME || t > MAX_JTIME) {
            error_msg = "jtime is outside mission bounds [0.20, 0.40] Julian centuries";
            return false;
        }

        if (i > 0 && t <= jtime(i - 1)) {
            error_msg = "jtime must be strictly increasing";
            return false;
        }
    }

    return true;
}

} // namespace saltro::validation
