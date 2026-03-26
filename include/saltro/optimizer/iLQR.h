#pragma once

#include <Eigen/Dense>
#include <vector>
#include <saltro/pybind/satellite.h>
#include <saltro/limits.h>

namespace saltro::optimizer {

enum class ILQRStatus {
	// Cost reduction satisfied (delta_J <= cost_tol).
	Converged,
	// Iteration budget exhausted before convergence.
	MaxIterations,
	// Regularization retry loop exceeded reg_max.
	RegularizationExceeded,
};

/**
 * @brief Iterative Linear Quadratic Regulator (iLQR) trajectory optimization.
 *
 * Performs trajectory refinement using the iterative LQR algorithm.
 * Takes an initial trajectory (X, U) and iteratively improves it by solving
 * local linear quadratic approximations of the optimal control problem.
 *
 * The algorithm alternates between:
 * 1. **Backward pass**: Compute optimal feedback gains K and feedforward terms d
 * 2. **Forward pass**: Rollout new trajectory using gains with line search
 *
 * Convergence is determined by gradient norm, cost improvement, or stagnation.
 * The algorithm respects settings for maximum iterations, convergence tolerances,
 * and regularization strategies.
 *
 * @param settings PlannerSettings containing algorithm parameters and cost weights
 * @param satellite Satellite model with dynamics and cost functions
 * @param X State trajectory (N × state_dim). Modified in-place with optimized trajectory.
 * @param U Control trajectory (N-1 × control_dim). Modified in-place with optimized controls.
 * @param R Position trajectory in ECI frame (3 × N), meters
 * @param V Velocity trajectory in ECI frame (3 × N), m/s
 * @param B Magnetic field trajectory in ECI frame (3 × N), Tesla
 * @param S Sun direction trajectory in ECI frame (3 × N), unit vectors
 * @param rho Atmospheric density trajectory (1 × N), kg/m³
 * @param jtime Mission time vector in Julian centuries (N × 1)
 * @param boresight Boresight constraint trajectory (3 × N)
 * @param attitude_target Attitude goal trajectory (4 × N): quaternions or [NaN, x, y, z] for ECI vector
 * @param J Output: total trajectory cost for the optimized trajectory
 *
 * @return true only when convergence is reached,
 *         false when the solve terminates without convergence
 *
 * @note Algorithm modifies X, U, and J in-place
 * @note Convergence is currently determined by settings.ilqr.cost_tol
 */
bool iLQR(
	const PlannerSettings& settings,
	const Satellite& satellite,
	Eigen::Ref<Eigen::MatrixXd> X,
	Eigen::Ref<Eigen::MatrixXd> U,
	const Eigen::Ref<const Eigen::MatrixXd>& R,
	const Eigen::Ref<const Eigen::MatrixXd>& V,
	const Eigen::Ref<const Eigen::MatrixXd>& B,
	const Eigen::Ref<const Eigen::MatrixXd>& S,
	const Eigen::Ref<const Eigen::MatrixXd>& rho,
	const Eigen::Ref<const Eigen::VectorXd>& jtime,
	const Eigen::Ref<const Eigen::MatrixXd>& boresight,
	const Eigen::Ref<const Eigen::MatrixXd>& attitude_target,
	int pass_idx,
	const std::vector<Eigen::VectorXd>& lambda_aug,
	const std::vector<Eigen::VectorXd>& mu_aug,
	ILQRStatus& status,
	double& J
);

bool iLQR(
	const PlannerSettings& settings,
	const Satellite& satellite,
	Eigen::Ref<Eigen::MatrixXd> X,
	Eigen::Ref<Eigen::MatrixXd> U,
	const Eigen::Ref<const Eigen::MatrixXd>& R,
	const Eigen::Ref<const Eigen::MatrixXd>& V,
	const Eigen::Ref<const Eigen::MatrixXd>& B,
	const Eigen::Ref<const Eigen::MatrixXd>& S,
	const Eigen::Ref<const Eigen::MatrixXd>& rho,
	const Eigen::Ref<const Eigen::VectorXd>& jtime,
	const Eigen::Ref<const Eigen::MatrixXd>& boresight,
	const Eigen::Ref<const Eigen::MatrixXd>& attitude_target,
	double& J
);

} // namespace saltro::optimizer
