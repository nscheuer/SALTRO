#pragma once

#include <Eigen/Dense>
#include <limits>
#include <vector>
#include <saltro/pybind/satellite.h>
#include <saltro/limits.h>

namespace saltro::optimizer {

enum class ILQRStatus {
	// Convergence criteria satisfied (see ILQRBreakReason for which).
	Converged,
	// Iteration budget exhausted before convergence.
	MaxIterations,
	// Regularization retry loop exceeded reg_max.
	RegularizationExceeded,
	// Stall counter (z_count_lim) tripped: cost flat in relative terms for
	// too many consecutive accepted steps. The best trajectory found is
	// returned; this is a successful (non-failure) exit, but NOT Converged —
	// the AL outer loop may not declare victory from it.
	Stalled,
};

// Which break criterion terminated the inner solve (telemetry).
enum class ILQRBreakReason {
	None,
	// Loose tier (intermediate, disjunctive)
	GradientIntermediate,   // g <= loose grad tol
	CostIntermediate,       // |dJ| <= loose cost tol (>=1 accepted step)
	RelCostIntermediate,    // |dJ|/max(|J|,1) <= rel_cost_tol (>=1 accepted step)
	// Strict tier (settle, conjunctive)
	GradientStationary,     // g <= grad_tol at zero accepted steps (optimal warm start)
	StrictConjunction,      // grad AND cost AND !ls_failed
	// Non-converged exits
	Stalled,
	MaxIterations,
	RegularizationExceeded,
};

// Per-call inner-solve telemetry.
struct ILQRTelemetry {
	int iterations = 0;       // iLQR iterations entered (incl. terminal partial one)
	int accepted_steps = 0;   // forward passes accepted
	double last_delta_J = -1.0;   // |dJ| of the last accepted step (-1: none)
	double min_delta_J = -1.0;    // smallest |dJ| over accepted steps (-1: none)
	double final_grad = -1.0;     // last computed gradient metric (-1: none)
	double final_cost = -1.0;     // AL-merit cost of the returned trajectory
	bool ls_failed = false;       // last forward-pass attempt failed
	bool settle = false;          // tier this solve ran in (echo of input)
	ILQRBreakReason break_reason = ILQRBreakReason::None;
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
 * @return true when the solve exits Converged or Stalled (usable trajectory),
 *         false on MaxIterations / RegularizationExceeded
 *
 * @note Algorithm modifies X, U, and J in-place
 * @note The convergence check runs after the backward pass and BEFORE the
 *       forward pass: a genuinely-optimal warm start exits Converged with
 *       zero accepted steps (never RegularizationExceeded).
 * @note `settle` selects the tolerance tier: false = loose/intermediate
 *       (disjunctive criteria), true = strict (conjunctive criteria; used by
 *       the AL outer loop once an iterate is feasible).
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
	bool settle,
	ILQRStatus& status,
	double& J,
	ILQRTelemetry& telemetry
);

// Back-compat overload: loose tier, telemetry discarded.
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
	int pass_idx,
	const std::vector<Eigen::VectorXd>& lambda_aug,
	const std::vector<Eigen::VectorXd>& mu_aug,
	ILQRStatus& status,
	double& J,
	ILQRTelemetry& telemetry
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
