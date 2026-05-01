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
 * @param reg Regularization parameter added to Q_uu diagonal for numerical stability
 * @param K Output feedback gains (std::vector where K[k] is nu × nxr for each timestep)
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
	const Eigen::Ref<const Eigen::MatrixXd>& R,
	const Eigen::Ref<const Eigen::MatrixXd>& V,
	const Eigen::Ref<const Eigen::MatrixXd>& B,
	const Eigen::Ref<const Eigen::MatrixXd>& S,
	const Eigen::Ref<const Eigen::MatrixXd>& rho,
	const Eigen::Ref<const Eigen::MatrixXd>& boresight,
	const Eigen::Ref<const Eigen::MatrixXd>& attitude_target,
	const PlannerSettings& settings,
	double reg,
	std::vector<Eigen::MatrixXd>& K,
	std::vector<Eigen::VectorXd>& d,
	Eigen::Ref<Eigen::Vector2d> deltaV,
	const std::vector<Eigen::VectorXd>& lambda_aug,
	const std::vector<Eigen::VectorXd>& mu_aug,
	std::vector<Eigen::MatrixXd>* Q_uu_out = nullptr,
	std::vector<Eigen::MatrixXd>* Quu_ddp_out = nullptr
);

inline bool backwardPass(
	const Satellite& satellite,
	const Eigen::Ref<const Eigen::MatrixXd>& X,
	const Eigen::Ref<const Eigen::MatrixXd>& U,
	const Eigen::Ref<const Eigen::MatrixXd>& R,
	const Eigen::Ref<const Eigen::MatrixXd>& V,
	const Eigen::Ref<const Eigen::MatrixXd>& B,
	const Eigen::Ref<const Eigen::MatrixXd>& S,
	const Eigen::Ref<const Eigen::MatrixXd>& rho,
	const Eigen::Ref<const Eigen::MatrixXd>& boresight,
	const Eigen::Ref<const Eigen::MatrixXd>& attitude_target,
	const PlannerSettings& settings,
	double reg,
	std::vector<Eigen::MatrixXd>& K,
	std::vector<Eigen::VectorXd>& d,
	Eigen::Ref<Eigen::Vector2d> deltaV
) {
	return backwardPass(
		satellite,
		X,
		U,
		R,
		V,
		B,
		S,
		rho,
		boresight,
		attitude_target,
		settings,
		reg,
		K,
		d,
		deltaV,
		std::vector<Eigen::VectorXd>{},
		std::vector<Eigen::VectorXd>{}
	);
}

} // namespace saltro::optimizer
