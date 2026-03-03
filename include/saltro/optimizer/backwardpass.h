#pragma once

#include <Eigen/Dense>
#include <saltro/pybind/satellite.h>
#include <saltro/pybind/plannersettings.h>

namespace saltro::optimizer {

/**
 * @brief Backward pass for iLQR.
 *
 * Computes optimal feedback gains K and feedforward terms d by solving 
 * Riccati-like equations from the final time backwards to the initial time.
 * Initializes the value function at terminal time using terminal cost derivatives.
 *
 * For each timestep k, performs a regularization loop: increases the regularization
 * parameter rho until Q_uu + rho*I becomes positive definite, then solves for gains
 * and updates the value function.
 *
 * @param satellite Satellite model with dynamics and cost/constraint functions
 * @param X State trajectory (N x state_dim)
 * @param U Control trajectory (N-1 x control_dim)
 * @param R Position trajectory (3 x N)
 * @param V Velocity trajectory (3 x N)
 * @param B Magnetic field trajectory (3 x N)
 * @param S Sun direction trajectory (3 x N)
 * @param rho Density trajectory (1 x N)
 * @param boresight Boresight vector trajectory (3 x N)
 * @param attitude_target Goal orientation (quaternion or ECI vector format)
 * @param settings Planner settings (contains cost, regularization config)
 * @param K Output feedback gains (std::vector where K[k] is nu × nx for each timestep)
 * @param d Output feedforward terms (std::vector where d[k] is nu × 1 for each timestep)
 * @param deltaV Output expected cost change coefficients (2x1)
 *               deltaV(0) = first-order term (accumulated across all timesteps)
 *               deltaV(1) = second-order term (accumulated across all timesteps)
 * @return true if backward pass succeeded (regularization converged), false if numerical failure
 */
bool backwardPass(
	const Satellite& satellite,
	const Eigen::Ref<const Eigen::MatrixXd>& X,
	const Eigen::Ref<const Eigen::MatrixXd>& U,
	const Eigen::Ref<const Eigen::MatrixXd>& R,  // position trajectory (3 x N)
	const Eigen::Ref<const Eigen::MatrixXd>& V,  // velocity trajectory (3 x N)
	const Eigen::Ref<const Eigen::MatrixXd>& B,
	const Eigen::Ref<const Eigen::MatrixXd>& S,  // sun direction trajectory (3 x N)
	const Eigen::Ref<const Eigen::MatrixXd>& rho,  // density trajectory (1 x N)
	const Eigen::Ref<const Eigen::MatrixXd>& boresight,
	const Eigen::Ref<const Eigen::Vector4d>& attitude_target,
	const PlannerSettings& settings,
	std::vector<Eigen::MatrixXd>& K,
	std::vector<Eigen::VectorXd>& d,
	Eigen::Ref<Eigen::Vector2d> deltaV
);

} // namespace saltro::optimizer
