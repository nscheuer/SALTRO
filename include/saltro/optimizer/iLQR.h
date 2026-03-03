#pragma once

#include <Eigen/Dense>
#include <saltro/pybind/satellite.h>
#include <saltro/limits.h>

namespace saltro::optimizer {

/**
 * @brief Iterative Linear Quadratic Regulator (iLQR) trajectory optimization.
 *
 * Performs trajectory refinement using the iterative LQR algorithm.
 * Takes an initial trajectory (X, U) and iteratively improves it by solving
 * local linear quadratic approximations of the optimal control problem.
 *
 * @param settings PlannerSettings containing algorithm parameters and cost weights
 * @param satellite Satellite model with dynamics and cost functions
 * @param X Input trajectory (state matrix, N x state_dim). Modified in-place with optimized trajectory.
 * @param U Input control sequence (N-1 x control_dim). Modified in-place with optimized controls.
 * @param B Input magnetic field trajectory (N x 3). Used for dynamics and cost evaluations.
 * @param J Output cost vector (length N), containing stage costs at each time step
 *
 * @return true if optimization succeeded, false if convergence or numerical issues occurred
 *
 * @note Algorithm runs for settings.ilqr_iterations with convergence tolerance settings.ilqr_tol
 * @note Modifies X, U, and J in-place
 */
bool iLQR(
	const PlannerSettings& settings,
	const Satellite& satellite,
	Eigen::Ref<Eigen::MatrixXd> X,
	Eigen::Ref<Eigen::MatrixXd> U,
    Eigen::Ref<Eigen::VectorXd> B,
	Eigen::Ref<Eigen::VectorXd> J
);

} // namespace saltro::optimizer
