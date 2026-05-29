#include <saltro/optimizer/iLQR.h>
#include <saltro/optimizer/backwardpass.h>
#include <saltro/optimizer/forwardpass.h>
#include <saltro/optimizer/spike_removal.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace saltro::optimizer {

namespace {

Eigen::VectorXd control_at_k(
	const Eigen::Ref<const Eigen::MatrixXd>& U,
	int k,
	int N,
	int control_dim
)
{
	if (U.cols() == N - 1 && k < N - 1) {
		return U.col(k);
	}
	if (U.cols() == N && k < N) {
		return U.col(k);
	}
	return Eigen::VectorXd::Zero(control_dim);
}

double augmented_penalty_total(
	const Satellite& satellite,
	const ConstraintConfig& cnst_cfg,
	const Eigen::Ref<const Eigen::MatrixXd>& X,
	const Eigen::Ref<const Eigen::MatrixXd>& U,
	const Eigen::Ref<const Eigen::MatrixXd>& S,
	const std::vector<Eigen::VectorXd>& lambda_aug,
	const std::vector<Eigen::VectorXd>& mu_aug
)
{
	if (lambda_aug.empty() || mu_aug.empty()) {
		return 0.0;
	}

	const int N = static_cast<int>(X.cols());
	const int n_steps = std::min(N, std::min(static_cast<int>(lambda_aug.size()), static_cast<int>(mu_aug.size())));
	const int nu = satellite.controlDim();

	double total = 0.0;
	for (int k = 0; k < n_steps; ++k) {
		const Eigen::VectorXd c_k = satellite.constraints(
			k,
			N,
			X.col(k),
			control_at_k(U, k, N, nu),
			S.col(k),
			cnst_cfg
		);

		const int c_size = static_cast<int>(c_k.size());
		const int lam_size = static_cast<int>(lambda_aug[k].size());
		const int mu_size = static_cast<int>(mu_aug[k].size());
		const int n_c = std::min(c_size, std::min(lam_size, mu_size));
		for (int i = 0; i < n_c; ++i) {
			const double c_pos = std::max(0.0, c_k(i));
			if (c_pos <= 0.0) {
				continue;
			}
			total += lambda_aug[k](i) * c_pos + 0.5 * mu_aug[k](i) * c_pos * c_pos;
		}
	}

	return total;
}

} // namespace

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
) {
	telemetry = ILQRTelemetry{};
	const CostConfig& cost_cfg = settings.passes[pass_idx].cost;
	const ILQRConfig& ilqr_cfg = settings.passes[pass_idx].ilqr;
	const RegularizationConfig& reg_cfg = settings.passes[pass_idx].reg;
	const ConstraintConfig& cnst_cfg = settings.constraints;

	PlannerSettings pass_settings = settings;
	pass_settings.num_passes = 1;
	pass_settings.passes[0] = settings.passes[pass_idx];
	
	// Preallocate gain and feedforward term vectors
	const int N = static_cast<int>(X.cols());   // Number of timesteps
	const int nu = static_cast<int>(U.rows());  // Control dimension
	const int nxr = satellite.reducedStateDim(); // Reduced state dimension (6 + nRW)
	std::vector<Eigen::MatrixXd> K(N - 1);
	std::vector<Eigen::VectorXd> d(N - 1);
	for (int k = 0; k < N - 1; ++k) {
		K[k] = Eigen::MatrixXd::Zero(nu, nxr);
		d[k] = Eigen::VectorXd::Zero(nu);
	}
	Eigen::Vector2d deltaV = Eigen::Vector2d::Zero();

	// Main iLQR iteration loop.
	// Two-variable regularization scheme (rho, drho) from original ALTRO:
	//   drho is a scaling factor that itself accelerates on repeated failures,
	//   producing super-exponential growth when the optimizer is struggling.
	//   On success, drho decreases, allowing rho to shrink smoothly.
	double reg = reg_cfg.reg_init;
	double dreg = 0.0;

	// Stagnation counter (PhD dlaZcount). Incremented each iteration where the
	// cost-convergence test passes but normal convergence hasn't fired (e.g.,
	// conjunctive_convergence=true and grad_tol unmet). If the counter exceeds
	// z_count_lim, we break as Converged because further iterations cannot
	// improve cost. Reset whenever we see meaningful cost progress.
	int stagnation_count = 0;

	auto increaseReg = [&]() {
		dreg = std::max(dreg * reg_cfg.reg_scale, reg_cfg.reg_scale);
		reg = std::max(reg * dreg, reg_cfg.reg_min);
	};

	auto decreaseReg = [&]() {
		dreg = std::min(dreg / reg_cfg.reg_scale, 1.0 / reg_cfg.reg_scale);
		reg = std::max(dreg * reg, reg_cfg.reg_min);
		// Clamp both ways at reg_min (2026-04-24). Previously decrease snapped
		// `reg` to 0 when it fell below reg_min — that created a sharp
		// discontinuity in Q_uu_reg as reg transitioned reg_min → 0 on adjacent
		// iterations. Near the threshold the solver chatters between reg=0
		// (BP sometimes fails) and reg=reg_min (BP succeeds), producing
		// nondeterministic basin choices across runs. Clamping at reg_min
		// makes Q_uu_reg continuous and the update trajectory deterministic.
		// reg_min = 1e-8 is small enough that the baseline is negligible
		// compared to typical Q_uu scales.
	};

	for (int iteration = 0; iteration < ilqr_cfg.max_iters; ++iteration) {
		const int N_u = std::max(0, N - 1);

		// Regularization retry loop
		while (reg <= reg_cfg.reg_max) {
			deltaV.setZero();

			bool bp_success = backwardPass(
				satellite, X, U.leftCols(N_u), R, V, B, S, rho,
				boresight, attitude_target, pass_settings, reg,
				K, d, deltaV, lambda_aug, mu_aug
			);

			if (!bp_success) {
				increaseReg();
				continue;
			}

			// Decrease reg after successful backward pass (like original)
			decreaseReg();

			double J_prev = satellite.totalCost(X, U.leftCols(N_u), B, boresight, attitude_target, cost_cfg);
			J_prev += augmented_penalty_total(satellite, cnst_cfg, X, U, S, lambda_aug, mu_aug);

			// Save nominal controls before forward pass (needed for spike removal blend)
			const Eigen::MatrixXd U_bar = U;

			bool fp_success = forwardPass(
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
				pass_settings,
				lambda_aug,
				mu_aug,
				jtime,
				J_prev,
				J
			);

			if (!fp_success) {
				// increaseReg + bump + increaseReg (triple increase on FP failure)
				increaseReg();
				reg += reg_cfg.reg_bump;
				increaseReg();
				continue;
			}

			++telemetry.accepted_steps;

			// Spike removal: detect and replace homotopy artifacts after accepted step
			const auto& spike_cfg = settings.passes[pass_idx].spike_removal;
			if (spike_cfg.enabled) {
				applySpikeRemoval(
					satellite, X, U, U_bar, K,
					settings, pass_idx,
					R, V, B, S, rho, jtime, boresight, attitude_target,
					iteration, spike_cfg
				);
			}

			// Both passes succeeded — check convergence.
			// Two-tier cost tolerance (standard 2-level AL pattern):
			//   - `ilqr_cost_tol` = loose inner tol used by the inner iLQR
			//     break ("good enough for this λ/μ, move on").
			//   - `cost_tol` = strict outer tol used for stagnation counting
			//     and by the AL outer break.
			// If ilqr_cost_tol <= cost_tol we disable the 2-tier behavior
			// and treat them identically.
			const double delta_J = std::abs(J_prev - J);
			telemetry.iterations = iteration + 1;
			telemetry.last_delta_J = delta_J;
			telemetry.final_cost = J;
			const double inner_tol = std::max(ilqr_cfg.ilqr_cost_tol, ilqr_cfg.cost_tol);
			const bool inner_cost_converged = (delta_J <= inner_tol);
			const bool outer_cost_converged = (delta_J <= ilqr_cfg.cost_tol);

			// ALTRO gradient test (Howell et al. 2019): max_k ||d_k||_∞ < ε_grad.
			// Use L∞ (max absolute component) rather than L2 — for a control
			// feedforward d_k with heterogeneous units (MTQ vs RW), L∞ directly
			// bounds the per-channel step size, while L2 averages across them.
			bool grad_converged = false;
			if (ilqr_cfg.grad_tol > 0.0) {
				double max_d_inf = 0.0;
				for (int kk = 0; kk < N - 1; ++kk) {
					const double dinf = d[kk].lpNorm<Eigen::Infinity>();
					if (dinf > max_d_inf) max_d_inf = dinf;
				}
				grad_converged = (max_d_inf <= ilqr_cfg.grad_tol);
			}

			// ALTRO-style relative cost convergence: |ΔJ| / max(|J_prev|, 1) < tol.
			// Robust to cost scale (the absolute `cost_tol` is meaningless when
			// J ~ 1e8 with angle_weight=1e6; we'd never see ΔJ < 1.0 even at
			// genuine optimality). Howell et al. 2019 "ALTRO" eq. 9.
			// rel_cost_tol <= 0 disables this path.
			const double J_scale = std::max(std::abs(J_prev), 1.0);
			const double rel_delta = delta_J / J_scale;
			const bool rel_cost_converged =
				(ilqr_cfg.rel_cost_tol > 0.0) && (rel_delta <= ilqr_cfg.rel_cost_tol);

			if (ilqr_cfg.conjunctive_convergence) {
				// Conjunctive: require ALL conditions to hold (cost AND grad).
				// Uses STRICT cost_tol — if conjunctive is requested the caller
				// wants a fully-settled inner solve. grad_tol=0 disables that
				// requirement (treated as satisfied).
				const bool grad_ok = (ilqr_cfg.grad_tol <= 0.0) || grad_converged;
				if ((outer_cost_converged || rel_cost_converged) && grad_ok) {
					status = ILQRStatus::Converged;
					return true;
				}
			} else {
				// Disjunctive (literature-standard for inner): ANY of
				// {loose cost, relative cost, grad, stagnation} fires break.
				if (inner_cost_converged || rel_cost_converged
					|| (ilqr_cfg.grad_tol > 0.0 && grad_converged)) {
					status = ILQRStatus::Converged;
					return true;
				}
			}

			// Stagnation break: if cost has stopped changing at the STRICT
			// outer tol for z_count_lim consecutive iterations, accept as
			// Converged. Prevents the solver from burning max_iters on a
			// flat plateau when conjunctive grad_tol can't be satisfied.
			if (outer_cost_converged) {
				++stagnation_count;
				if (ilqr_cfg.z_count_lim > 0 && stagnation_count >= ilqr_cfg.z_count_lim) {
					status = ILQRStatus::Converged;
					return true;
				}
			} else {
				stagnation_count = 0;
			}

			break;  // Exit regularization loop, continue to next iteration
		}

		// Check if regularization exceeded maximum
		if (reg > reg_cfg.reg_max) {
			status = ILQRStatus::RegularizationExceeded;
			return false;
		}
	}

	status = ILQRStatus::MaxIterations;
	return false;
}

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
) {
	ILQRTelemetry telemetry;
	return iLQR(
		settings,
		satellite,
		X,
		U,
		R,
		V,
		B,
		S,
		rho,
		jtime,
		boresight,
		attitude_target,
		pass_idx,
		lambda_aug,
		mu_aug,
		status,
		J,
		telemetry
	);
}

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
) {
	ILQRStatus status = ILQRStatus::MaxIterations;
	const bool ok = iLQR(
		settings,
		satellite,
		X,
		U,
		R,
		V,
		B,
		S,
		rho,
		jtime,
		boresight,
		attitude_target,
		0,
		std::vector<Eigen::VectorXd>{},
		std::vector<Eigen::VectorXd>{},
		status,
		J
	);
	(void)status;
	return ok;
}

} // namespace saltro::optimizer
