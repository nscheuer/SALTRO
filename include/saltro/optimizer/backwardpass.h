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
	std::vector<Eigen::MatrixXd>* K_dist = nullptr
);

/**
 * @brief Square-root backward pass for iLQR.
 *
 * Numerically robust variant of backwardPass() that propagates an
 * upper-triangular square-root factor S_k (P_k = S_k^T S_k) of the cost-to-go
 * Hessian instead of P_k itself, following the square-root backward pass of
 * the ALTRO paper (Howell, Jackson, Manchester, "ALTRO: A Fast Solver for
 * Constrained Trajectory Optimization", IROS 2019, Sec. IV-A), itself
 * inspired by the square-root Kalman filter.
 *
 * Augmented Lagrangian penalty Hessians enter the recursion as rows
 * sqrt(mu_i) * [c_x_i c_u_i] of a stacked QR factorization rather than as
 * mu_i * c^T c outer products, halving the effective condition number. This
 * matters when AL penalties grow large, which is exactly where the dense
 * recursion becomes ill-conditioned.
 *
 * Semantics match backwardPass(): gains K and d use the regularized control
 * Hessian (Q_uu + reg*I), while the value function is propagated with the
 * unregularized Q_uu. Indefinite stage-cost Hessians are clamped to PSD
 * (a real square root only exists for PSD matrices), so results match the
 * dense pass exactly whenever the stage-cost Hessians are PSD.
 *
 * Selected at runtime by setting RegularizationConfig::use_sqrt_bp = true,
 * which makes backwardPass() forward to this function.
 *
 * Parameters and return value are identical to backwardPass().
 */
bool backwardPassSqrt(
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
	const std::vector<Eigen::VectorXd>& mu_aug
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
