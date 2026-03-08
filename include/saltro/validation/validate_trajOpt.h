/**
 * @file validate_trajOpt.h
 * @brief Input validation for trajectory optimization problem setup.
 */
#pragma once

#include <string>
#include <Eigen/Dense>

#include <saltro/pybind/plannersettings.h>
#include <saltro/pybind/satellite.h>

namespace saltro::validation {

/**
 * @brief Validate all inputs to the trajectory optimization problem.
 *
 * Performs comprehensive validation of all trajectory optimization inputs before
 * invoking the optimizer. Checks include:
 * 
 * **Planner Settings:**
 * - Cost weights are non-negative and finite
 * - Constraint tolerances are positive
 * - Regularization parameters satisfy min ≤ init ≤ max
 * - Line search parameters are in valid ranges (0 < β₁ < β₂ < 1)
 * - Timestep dt > 0 and horizon length N > 1
 * 
 * **Satellite Configuration:**
 * - Inertia tensor is symmetric positive definite
 * - Actuator counts match configuration (nMTQ, nRW ≥ 0)
 * - Actuator limits are positive finite values
 * 
 * **Initial State:**
 * - Quaternion is normalized: ‖q‖ = 1
 * - Angular velocity is finite and within rate limits
 * - Wheel momenta are within saturation limits
 * 
 * **Reference Trajectories:**
 * - q_goal has shape (4 × N) with normalized quaternions
 * - boresight has shape (3 × N) with unit vectors
 * - jtime has length N with monotonically increasing values
 * - Orbital state (r0, v0) is physically feasible for LEO
 * 
 * **Dimension Consistency:**
 * - state_dim = 7 + nRW
 * - input_dim = 3·nMTQ + nRW + 3 (magic torque)
 * - All trajectory dimensions match horizon length N
 *
 * @param settings Planner configuration to validate
 * @param satellite Satellite model to validate
 * @param x0 Initial attitude state vector (7+nRW × 1)
 * @param r0 Initial position in ECI frame (3 × 1), meters
 * @param v0 Initial velocity in ECI frame (3 × 1), m/s
 * @param jtime Mission time vector in Julian centuries (N × 1)
 * @param q_goal Goal quaternion trajectory (4 × N)
 * @param boresight Boresight constraint trajectory (3 × N)
 * @param state_dim Expected state dimension (7 + nRW)
 * @param input_dim Expected control dimension
 * @param N Expected horizon length
 * @param error_msg Output: detailed error message describing first validation failure
 * @return true if all validation checks pass, false otherwise
 */
bool validatetrajOpt(
    const PlannerSettings& settings,
    const Satellite& satellite,
    const Eigen::Ref<const Eigen::VectorXd>& x0,
    const Eigen::Vector3d& r0,
    const Eigen::Vector3d& v0,
    const Eigen::Ref<const Eigen::VectorXd>& jtime,
    const Eigen::Ref<const Eigen::MatrixXd>& q_goal,
    const Eigen::Ref<const Eigen::MatrixXd>& boresight,

    int state_dim,
    int input_dim,
    int N,

    std::string& error_msg
);

}