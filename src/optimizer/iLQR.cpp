#include <saltro/optimizer/iLQR.h>
#include <saltro/optimizer/backwardpass.h>
#include <saltro/optimizer/forwardpass.h>

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
			const double ci = c_k(i);
			const double li = lambda_aug[k](i);
			// Lambda term always active with signed c; mu penalty when
			// c > 0 OR lambda > 0. Must match the forward-pass merit exactly:
			// this value seeds J_prev for the first line-search comparison.
			total += li * ci;
			if (ci > 0.0 || li > 0.0) {
				total += 0.5 * mu_aug[k](i) * ci * ci;
			}
		}
	}

	return total;
}

// Per-channel authority denominators for GradMetric::Authority.
// Prefer the validated settings u_max; fall back to the satellite's actuator
// limits (mirrors trajOpt's u_max auto-fill); channels with no/zero authority
// use 1.0 so the metric stays finite.
Eigen::VectorXd authority_denominators(
	const PlannerSettings& settings,
	const Satellite& satellite,
	int nu
)
{
	Eigen::VectorXd denom = Eigen::VectorXd::Ones(nu);
	const auto& u_max = settings.constraints.u_max;
	if (static_cast<int>(u_max.size()) == nu) {
		for (int i = 0; i < nu; ++i) {
			const double v = u_max(i);
			if (std::isfinite(v) && v > 0.0) {
				denom(i) = v;
			}
		}
		return denom;
	}

	int idx = 0;
	auto set_next = [&](double v) {
		if (idx < nu) {
			const double a = std::abs(v);
			denom(idx++) = (std::isfinite(a) && a > 0.0) ? a : 1.0;
		}
	};
	for (int i = 0; i < satellite.numMTQ(); ++i) {
		set_next(satellite.getMTQ(i).u_max());
	}
	for (int i = 0; i < satellite.numRW(); ++i) {
		set_next(satellite.getRW(i).u_max());
	}
	for (int i = 0; i < satellite.numMagic(); ++i) {
		set_next(satellite.getMagic(i).u_max());
	}
	return denom;
}

// Gradient-surrogate metric over the feedforward terms d_k (see GradMetric).
double gradient_metric(
	const GradMetric metric,
	const std::vector<Eigen::VectorXd>& d,
	const Eigen::Ref<const Eigen::MatrixXd>& U,
	const Eigen::VectorXd& authority,
	int N_u
)
{
	const int n = std::min(N_u, static_cast<int>(d.size()));
	if (n <= 0) {
		return 0.0;
	}

	switch (metric) {
		case GradMetric::GNormTassa: {
			// Tassa iLQG.m g_norm: mean_k max_i |d_k(i)| / (|u_k(i)| + 1).
			double sum = 0.0;
			for (int k = 0; k < n; ++k) {
				double mx = 0.0;
				for (int i = 0; i < d[k].size(); ++i) {
					const double u_ki = (k < U.cols() && i < U.rows()) ? U(i, k) : 0.0;
					mx = std::max(mx, std::abs(d[k](i)) / (std::abs(u_ki) + 1.0));
				}
				sum += mx;
			}
			return sum / static_cast<double>(n);
		}
		case GradMetric::LInf: {
			double mx = 0.0;
			for (int k = 0; k < n; ++k) {
				mx = std::max(mx, d[k].lpNorm<Eigen::Infinity>());
			}
			return mx;
		}
		case GradMetric::Authority:
		default: {
			// max_k max_i |d_k(i)| / u_max_i: fraction of actuator authority
			// the next full step requests.
			double mx = 0.0;
			for (int k = 0; k < n; ++k) {
				for (int i = 0; i < d[k].size(); ++i) {
					const double a = (i < authority.size()) ? authority(i) : 1.0;
					mx = std::max(mx, std::abs(d[k](i)) / a);
				}
			}
			return mx;
		}
	}
}

// Relative-progress threshold for the stall counter reset (OldPlanner
// dlaZcount parity: reset when |dJ|/(|J|+1e-10) > 1e-3).
constexpr double STALL_RESET_REL_PROGRESS = 1e-3;

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
	bool settle,
	ILQRStatus& status,
	double& J,
	ILQRTelemetry& telemetry
) {
	telemetry = ILQRTelemetry{};
	telemetry.settle = settle;
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

	// ---- Tier tolerances ----------------------------------------------------
	// Strict tier (settle=true): conjunctive, used by the AL outer loop for
	// settling solves. Loose tier (settle=false): disjunctive intermediate
	// tolerances (ALTRO `*_intermediate` pattern). Unset (0) intermediate
	// values auto-derive 10x their strict counterpart; explicit values are
	// clamped to be no tighter than strict.
	const double strict_cost_tol = ilqr_cfg.cost_tol;
	const double loose_cost_tol = (ilqr_cfg.cost_tol_intermediate > 0.0)
		? std::max(ilqr_cfg.cost_tol_intermediate, strict_cost_tol)
		: 10.0 * strict_cost_tol;
	const double strict_grad_tol = ilqr_cfg.grad_tol;  // <= 0 disables
	const double loose_grad_tol = (strict_grad_tol > 0.0)
		? ((ilqr_cfg.grad_tol_intermediate > 0.0)
			? std::max(ilqr_cfg.grad_tol_intermediate, strict_grad_tol)
			: 10.0 * strict_grad_tol)
		: 0.0;
	const double active_cost_tol = settle ? strict_cost_tol : loose_cost_tol;
	const double active_grad_tol = settle ? strict_grad_tol : loose_grad_tol;

	const Eigen::VectorXd authority = authority_denominators(settings, satellite, nu);

	// State carried across iterations for the pre-FP convergence check.
	double delta_J = -1.0;        // |dJ| of the last accepted step
	double rel_delta_J = -1.0;    // |dJ| / max(|J_prev_step|, 1)
	double rel_progress = -1.0;   // |dJ| / (|J_prev_step| + 1e-10), stall reset
	bool ls_failed = false;       // last forward-pass attempt failed
	int stall_count = 0;          // dlaZcount

	// Main iLQR iteration loop
	double reg = reg_cfg.reg_init;
	for (int iteration = 0; iteration < ilqr_cfg.max_iters; ++iteration) {
		++telemetry.iterations;
		// Reset regularization each iteration unless persistent (ALTRO-style)
		if (!ilqr_cfg.persistent_reg) {
			reg = reg_cfg.reg_init;
		}
		const int N_u = std::max(0, N - 1);
		bool checked_this_iteration = false;

		// Regularization retry loop
		const double reg_iter_start = reg;
		while (reg <= reg_cfg.reg_max) {
			deltaV.setZero();

			bool bp_success = backwardPass(
				satellite, X, U.leftCols(N_u), R, V, B, S, rho,
				boresight, attitude_target, pass_settings, reg,
				K, d, deltaV, lambda_aug, mu_aug
			);

			if (!bp_success) {
				if (ilqr_cfg.persistent_reg) {
					reg = reg * reg_cfg.reg_scale + reg_cfg.reg_bump;
				} else {
					reg *= reg_cfg.reg_scale;  // Legacy: simple multiply
				}
				continue;
			}

			double J_prev = satellite.totalCost(X, U.leftCols(N_u), B, boresight, attitude_target, cost_cfg);
			J_prev += augmented_penalty_total(satellite, cnst_cfg, X, U, S, lambda_aug, mu_aug);

			// ---- Convergence check: after the backward pass, BEFORE the ----
			// forward pass. Placement is load-bearing: a genuinely-optimal
			// warm start has g ~ 0 after the first BP and must exit Converged
			// with zero accepted steps — never die as RegularizationExceeded
			// because no forward pass can improve on an optimum.
			// Only the first successful BP of an iteration is checked, and
			// the gradient branches additionally require that BP to have
			// succeeded WITHOUT regularization escalation: a BP re-run at
			// escalated reg shrinks d ∝ 1/reg, and certifying stationarity
			// from a reg-shrunk gradient is a false convergence (observed on
			// stiff high-mu AL solves, where BP only succeeds at reg >> init
			// and the tiny d would otherwise exit GradientIntermediate at
			// zero accepted steps while badly infeasible).
			if (!checked_this_iteration) {
				checked_this_iteration = true;
				const bool grad_usable = (reg <= reg_iter_start);
				const double g = gradient_metric(
					ilqr_cfg.grad_metric, d, U.leftCols(N_u), authority, N_u);
				telemetry.final_grad = g;
				telemetry.ls_failed = ls_failed;

				const bool have_step = (telemetry.accepted_steps > 0);
				// Zero-accepted-step gradient exits must meet the STRICT
				// grad_tol in both tiers: an optimal warm start has g ~ 0 and
				// sails through, but an infeasible AL-stationary-ish iterate
				// with g in (grad_tol, grad_tol_intermediate] must not be
				// allowed to exit without taking a single step — otherwise
				// the outer loop livelocks (zero-step "converged" exits while
				// the violation never moves). After >= 1 accepted step the
				// intermediate tolerance applies ("good enough for this
				// lambda/mu, move on").
				const bool grad_enabled = (active_grad_tol > 0.0) && grad_usable;
				const bool grad_ok = grad_enabled
					&& (have_step ? (g <= active_grad_tol)
					              : (strict_grad_tol > 0.0 && g <= strict_grad_tol));
				const bool cost_ok = have_step && (delta_J <= active_cost_tol);
				const bool rel_ok = have_step && (ilqr_cfg.rel_cost_tol > 0.0)
					&& (rel_delta_J >= 0.0) && (rel_delta_J <= ilqr_cfg.rel_cost_tol);

				ILQRBreakReason fired = ILQRBreakReason::None;
				if (!settle) {
					// Loose tier: disjunctive (any criterion exits). The cost
					// branches require >= 1 accepted step in THIS call (no dJ
					// exists otherwise); the gradient branch may fire at zero
					// accepted steps (optimal warm start).
					if (grad_ok) {
						fired = ILQRBreakReason::GradientIntermediate;
					} else if (cost_ok) {
						fired = ILQRBreakReason::CostIntermediate;
					} else if (rel_ok) {
						fired = ILQRBreakReason::RelCostIntermediate;
					}
				} else {
					// Strict tier (settling solve): conjunctive.
					if (!have_step) {
						// Zero accepted steps: the gradient alone is the
						// stationarity certificate (cost-change is vacuous).
						if (grad_usable && strict_grad_tol > 0.0 && g <= strict_grad_tol) {
							fired = ILQRBreakReason::GradientStationary;
						}
					} else {
						const bool g_ok_strict =
							(strict_grad_tol <= 0.0)
							|| (grad_usable && g <= strict_grad_tol);
						const bool cost_ok_strict =
							(delta_J <= strict_cost_tol) || rel_ok;
						if (g_ok_strict && cost_ok_strict && !ls_failed) {
							fired = ILQRBreakReason::StrictConjunction;
						}
					}
				}

				if (fired != ILQRBreakReason::None) {
					J = J_prev;  // cost of the (unchanged) current trajectory
					telemetry.final_cost = J;
					telemetry.break_reason = fired;
					status = ILQRStatus::Converged;
					return true;
				}
			}

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
				ls_failed = true;
				telemetry.ls_failed = true;
				if (ilqr_cfg.persistent_reg) {
					// Triple increase (original ALTRO): increaseReg + bump + increaseReg
					reg = reg * reg_cfg.reg_scale + reg_cfg.reg_bump;
					reg += reg_cfg.reg_bump;
					reg = reg * reg_cfg.reg_scale + reg_cfg.reg_bump;
				} else {
					reg *= reg_cfg.reg_scale;  // Legacy: simple multiply
				}
				continue;
			}

			// Persistent regularization: decrease on success (ALTRO-style)
			if (ilqr_cfg.persistent_reg) {
				reg = reg / reg_cfg.reg_scale;
				// Drop to zero below reg_min (pure Newton when well-conditioned)
				if (reg < reg_cfg.reg_min) {
					reg = 0.0;
				}
			}

			// Both passes succeeded: bookkeeping for the next iteration's
			// pre-FP convergence check.
			ls_failed = false;
			telemetry.ls_failed = false;
			++telemetry.accepted_steps;
			delta_J = std::abs(J_prev - J);
			rel_delta_J = delta_J / std::max(std::abs(J_prev), 1.0);
			rel_progress = delta_J / (std::abs(J_prev) + 1e-10);
			telemetry.last_delta_J = delta_J;
			telemetry.final_cost = J;

			// Stall counter (dlaZcount semantics): counts consecutive accepted
			// steps without relative progress; resets when relative progress
			// is made. Tripping it returns the best trajectory as Stalled —
			// a usable, non-failure exit that is NOT Converged.
			if (rel_progress > STALL_RESET_REL_PROGRESS) {
				stall_count = 0;
			} else {
				++stall_count;
				if (ilqr_cfg.z_count_lim > 0 && stall_count >= ilqr_cfg.z_count_lim) {
					telemetry.break_reason = ILQRBreakReason::Stalled;
					status = ILQRStatus::Stalled;
					return true;
				}
			}

			break;  // Exit regularization loop, continue to next iteration
		}

		// Check if regularization exceeded maximum
		if (reg > reg_cfg.reg_max) {
			telemetry.break_reason = ILQRBreakReason::RegularizationExceeded;
			status = ILQRStatus::RegularizationExceeded;
			return false;
		}
	}

	telemetry.break_reason = ILQRBreakReason::MaxIterations;
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
		false,
		status,
		J,
		telemetry
	);
}

// Back-compat overload: loose tier, telemetry reported.
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
		false,
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