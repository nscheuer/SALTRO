#include <saltro/validation/validate_trajOpt.h>

#include <cmath>

#include <saltro/limits.h>
#include <saltro/validation/validate_plannersettings.h>
#include <saltro/validation/validate_satellite.h>
#include <saltro/validation/validate_initialstate.h>
#include <saltro/validation/validate_orbitstate.h>
#include <saltro/validation/validate_juliantime.h>
#include <saltro/validation/validate_qgoal.h>

namespace saltro::validation {

namespace {

bool validateTrajOptCrossContext(
    const Satellite& satellite,
    const Eigen::Ref<const Eigen::VectorXd>& x0,
    const Eigen::Ref<const Eigen::VectorXd>& jtime,
    const Eigen::Ref<const Eigen::MatrixXd>& q_goal,
    const int state_dim,
    const int input_dim,
    const int N,
    std::string& error_msg
) {
    if (N <= 0) {
        error_msg = "N must be > 0";
        return false;
    }

    if (N > saltro::limits::MAX_LENGTH_TRAJ) {
        error_msg = "N exceeds MAX_LENGTH_TRAJ";
        return false;
    }

    if (state_dim <= 0 || state_dim > saltro::limits::MAX_STATE_DIM) {
        error_msg = "state_dim is outside [1, MAX_STATE_DIM]";
        return false;
    }

    if (input_dim < 0 || input_dim > saltro::limits::MAX_CTRL_DIM) {
        error_msg = "input_dim is outside [0, MAX_CTRL_DIM]";
        return false;
    }

    if (state_dim != satellite.stateDim()) {
        error_msg = "state_dim does not match satellite.stateDim()";
        return false;
    }

    if (input_dim != satellite.controlDim()) {
        error_msg = "input_dim does not match satellite.controlDim()";
        return false;
    }

    if (x0.size() != state_dim) {
        error_msg = "x0 size does not match state_dim";
        return false;
    }

    if (jtime.size() != N) {
        error_msg = "jtime length does not match N";
        return false;
    }

    if (q_goal.cols() != N) {
        error_msg = "q_goal column count does not match N";
        return false;
    }

    if (q_goal.rows() != 4) {
        error_msg = "q_goal must have 4 rows";
        return false;
    }

    return true;
}

}

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

    if (!validateInitialState(x0, nested_error)) {
        error_msg = "Initial state validation failed: " + nested_error;
        return false;
    }

    if (!validateOrbitState(r0, v0, nested_error)) {
        error_msg = "Orbit state validation failed: " + nested_error;
        return false;
    }

    if (!validateJulianTime(jtime, nested_error)) {
        error_msg = "Julian time validation failed: " + nested_error;
        return false;
    }

    if (!validateQGoal(q_goal, nested_error)) {
        error_msg = "q_goal validation failed: " + nested_error;
        return false;
    }

    if (!validateTrajOptCrossContext(satellite, x0, jtime, q_goal, state_dim, input_dim, N, nested_error)) {
        error_msg = "Cross-context validation failed: " + nested_error;
        return false;
    }

    return true;
}

}
