/**
 * @file forwardpass.h
 * @brief Forward pass (line search) for iLQR trajectory optimization.
 */
#pragma once

#include <Eigen/Dense>
#include <vector>

#include <saltro/pybind/plannersettings.h>
#include <saltro/pybind/satellite.h>

namespace saltro::optimizer {

/**
 * @brief Perform forward pass with line search in iLQR.
 *
 * Executes a backtracking line search to find an improved trajectory using the feedback
 * gains K and feedforward terms d computed by the backward pass. The improved control
 * policy is:
 * \f[
 * \mathbf{u}_k^{\text{new}} = \mathbf{u}_k + \alpha \mathbf{d}_k + 
 * \mathbf{K}_k (\mathbf{x}_k^{\text{new}} - \mathbf{x}_k)
 * \f]
 * where \f$\alpha \in (0, 1]\f$ is the step size found by line search.
 *
 * The state is propagated forward using the satellite dynamics under the new control.
 * Line search terminates when the Armijo sufficient decrease condition is satisfied:
 * \f[
 * J^{\text{new}} \leq J^{\text{prev}} + \beta_1 \alpha \nabla J^T \mathbf{d}
 * \f]
 * where \f$\beta_1\f$ is the line search parameter from settings, and the gradient
 * term is approximated using the predicted cost change coefficients deltaV.
 *
 * If sufficient decrease is not achieved after max_iters backtracking steps, the
 * line search terminates with the best-found trajectory (which may not improve cost).
 *
 * @param satellite Satellite model with dynamics, cost, and constraint functions
 * @param X State trajectory (N × state_dim); updated in-place with new rollout
 * @param U Control trajectory (N-1 × control_dim); updated in-place with new controls
 * @param K Feedback gain matrices from backward pass (N-1 vector of nu × nxr matrices)
 * @param d Feedforward control terms from backward pass (N-1 vector of nu × 1 vectors)
 * @param deltaV Predicted cost change coefficients [Δ_1, Δ_2] from backward pass
 * @param B Magnetic field trajectory in ECI frame (3 × N), Tesla
 * @param R Position trajectory in ECI frame (3 × N), meters
 * @param V Velocity trajectory in ECI frame (3 × N), m/s
 * @param S Sun direction trajectory in ECI frame (3 × N), unit vectors
 * @param rho Atmospheric density trajectory (1 × N), kg/m³
 * @param boresight Desired boresight direction trajectory (3 × N), unit vectors
 * @param attitude_target Desired attitude trajectory (4 × N), unit quaternions
 * @param settings Planner settings containing line search configuration (beta1, beta2, max_iters)
 * @param jtime Mission time vector in Julian centuries (N × 1)
 * @param J_prev Total cost of previous trajectory
 * @param J_new Output: total cost of new trajectory after line search
 * @return true if line search succeeded and found improvement, false otherwise
 */
bool forwardPass(
    const Satellite& satellite,
    Eigen::Ref<Eigen::MatrixXd> X,
    Eigen::Ref<Eigen::MatrixXd> U,
    const std::vector<Eigen::MatrixXd>& K,
    const std::vector<Eigen::VectorXd>& d,
    const Eigen::Ref<const Eigen::Vector2d>& deltaV,
    const Eigen::Ref<const Eigen::MatrixXd>& B,
    const Eigen::Ref<const Eigen::MatrixXd>& R,
    const Eigen::Ref<const Eigen::MatrixXd>& V,
    const Eigen::Ref<const Eigen::MatrixXd>& S,
    const Eigen::Ref<const Eigen::MatrixXd>& rho,
    const Eigen::Ref<const Eigen::MatrixXd>& boresight,
    const Eigen::Ref<const Eigen::MatrixXd>& attitude_target,
    const PlannerSettings& settings,
    const std::vector<Eigen::VectorXd>& lambda_aug,
    const std::vector<Eigen::VectorXd>& mu_aug,
    const Eigen::Ref<const Eigen::VectorXd>& jtime,
    double J_prev,
    double& J_new
);

inline bool forwardPass(
    const Satellite& satellite,
    Eigen::Ref<Eigen::MatrixXd> X,
    Eigen::Ref<Eigen::MatrixXd> U,
    const std::vector<Eigen::MatrixXd>& K,
    const std::vector<Eigen::VectorXd>& d,
    const Eigen::Ref<const Eigen::Vector2d>& deltaV,
    const Eigen::Ref<const Eigen::MatrixXd>& B,
    const Eigen::Ref<const Eigen::MatrixXd>& R,
    const Eigen::Ref<const Eigen::MatrixXd>& V,
    const Eigen::Ref<const Eigen::MatrixXd>& S,
    const Eigen::Ref<const Eigen::MatrixXd>& rho,
    const Eigen::Ref<const Eigen::MatrixXd>& boresight,
    const Eigen::Ref<const Eigen::MatrixXd>& attitude_target,
    const PlannerSettings& settings,
    const Eigen::Ref<const Eigen::VectorXd>& jtime,
    double J_prev,
    double& J_new
) {
    return forwardPass(
        satellite,
        X,
        U,
        K,
        d,
        deltaV,
        B,
        R,
        V,
        S,
        rho,
        boresight,
        attitude_target,
        settings,
        std::vector<Eigen::VectorXd>{},
        std::vector<Eigen::VectorXd>{},
        jtime,
        J_prev,
        J_new
    );
}

}
