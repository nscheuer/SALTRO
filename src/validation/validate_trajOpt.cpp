#include <saltro/validation/validate_trajOpt.h>

#include <cmath>

#include <saltro/limits.h>
#include <saltro/validation/validate_plannersettings.h>
#include <saltro/validation/validate_satellite.h>

namespace saltro::validation {

bool validatetrajOpt(
    const PlannerSettings& settings,
    const Satellite& satellite,
    const Eigen::Ref<const Eigen::VectorXd>& x0,
    const Eigen::Vector3d& r0,
    const Eigen::Vector3d& v0,
    const Eigen::Ref<const Eigen::VectorXd>& jtime,
    const Eigen::Ref<const Eigen::MatrixXd>& q_goal,

    int state_dim,
    int input_dim,
    int N,

    std::string& error_msg
) {
    std::string nested_error;

    if (!validatePlannerSettings(settings, nested_error)) {
        error_msg = "PlannerSettings validation failed: " + nested_error;
        return false;
    }

    if (!validateSatellite(satellite, nested_error)) {
        error_msg = "Satellite validation failed: " + nested_error;
        return false;
    }

    return true;
}

}
