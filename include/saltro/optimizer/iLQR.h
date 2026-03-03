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
 * @param B Input magnetic field trajectory (3 x N)
 * @param boresight Input boresight trajectory (3 x N)
 * @param attitude_target Attitude goal in quaternion mode [q0,qx,qy,qz] or ECI mode [nan,x,y,z]
 * @param J Output total trajectory cost for the current rollout.
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
	const Eigen::Ref<const Eigen::MatrixXd>& R,  // position trajectory (3 x N)
	const Eigen::Ref<const Eigen::MatrixXd>& V,  // velocity trajectory (3 x N)
	const Eigen::Ref<const Eigen::MatrixXd>& B,
	const Eigen::Ref<const Eigen::MatrixXd>& S,  // sun direction trajectory (3 x N)
	const Eigen::Ref<const Eigen::MatrixXd>& rho,  // density trajectory (1 x N)
	const Eigen::Ref<const Eigen::VectorXd>& jtime, // julian centuries times (N)
	const Eigen::Ref<const Eigen::MatrixXd>& boresight,
	const Eigen::Ref<const Eigen::Vector4d>& attitude_target,
	double& J
);

} // namespace saltro::optimizer
