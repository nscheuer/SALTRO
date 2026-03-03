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
 * @param satellite Satellite model with dynamics and cost/constraint functions
 * @param X State trajectory (N x state_dim)
 * @param U Control trajectory (N-1 x control_dim)
 * @param B Magnetic field trajectory (3 x N)
 * @param boresight Boresight vector trajectory (3 x N)
 * @param attitude_target Goal orientation (quaternion or ECI vector format)
 * @param cost_cfg Cost configuration
 * @param K Output feedback gains (control_dim x state_dim x N-1)
 * @param d Output feedforward terms (control_dim x N-1)
 * @param deltaV Output expected cost change coefficients (2x1)
 *               deltaV(0) = first-order term
 *               deltaV(1) = second-order term
 */
void backwardPass(
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
	const CostConfig& cost_cfg,
	Eigen::Ref<Eigen::MatrixXd> K,
	Eigen::Ref<Eigen::MatrixXd> d,
	Eigen::Ref<Eigen::Vector2d> deltaV
);

} // namespace saltro::optimizer
