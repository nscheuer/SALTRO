#include <saltro/optimizer/backwardpass.h>
#include <saltro/math/integrators/rk4.h>
#include <saltro/math/mrp.h>
#include <iostream>
#include <cmath>

#if defined(SALTRO_DEBUG_BUILD)
#define SALTRO_OPT_DLOG(msg) do { std::cout << msg << std::endl; } while (0)
#else
#define SALTRO_OPT_DLOG(msg) do {} while (0)
#endif

namespace saltro::optimizer {

/// Solve a Riccati step using an eigen-factored inverse of Q_uu.  Replaces
/// LLT(Q_uu + ρI) with a unified modification:
///   - Cell 2 (abs):   λ ← |λ|
///   - Cell 1 (floor): λ ← max(λ, ρ)
///   - Cell 5 (cond):  λ ← max(λ, γ · max(λ))
/// Combined: `λ' = max(|λ|, max(ρ, γ · max(|λ|)))`.  Then Q_uu⁻¹ x = V·diag(1/λ')·V^T·x.
/// Uses UNregularized Q_uu for Riccati value-function propagation (same as
/// standard path), only K and d come from the modified inverse.
void solveRiccattiStepEigen(
	const Eigen::Ref<const Eigen::MatrixXd>& V_eig,          // eigenvectors of Q_uu
	const Eigen::Ref<const Eigen::VectorXd>& lambda_mod,     // modified eigenvalues
	const Eigen::Ref<const Eigen::MatrixXd>& Q_uu,           // UNregularized
	const Eigen::Ref<const Eigen::VectorXd>& Q_u,
	const Eigen::Ref<const Eigen::MatrixXd>& Q_ux,
	const Eigen::Ref<const Eigen::MatrixXd>& Q_xx,
	const Eigen::Ref<const Eigen::VectorXd>& Q_x,
	int k,
	std::vector<Eigen::MatrixXd>& K,
	std::vector<Eigen::VectorXd>& d,
	Eigen::Ref<Eigen::Vector2d> deltaV,
	Eigen::Ref<Eigen::VectorXd> p_k,
	Eigen::Ref<Eigen::MatrixXd> P_k
) {
	// Q_uu⁻¹ · b = V · diag(1/λ_mod) · V^T · b
	const Eigen::VectorXd lam_inv = lambda_mod.cwiseInverse();
	const Eigen::MatrixXd Vt_Qux = V_eig.transpose() * Q_ux;
	const Eigen::VectorXd Vt_Qu  = V_eig.transpose() * Q_u;

	Eigen::MatrixXd K_k = -(V_eig * (lam_inv.asDiagonal() * Vt_Qux));
	Eigen::VectorXd d_k = -(V_eig * (lam_inv.asDiagonal() * Vt_Qu));
	K[k] = K_k;
	d[k] = d_k;

	// Riccati propagation uses the ORIGINAL Q_uu (matches the standard path's
	// reasoning: regularization is a solver aid, not part of the value function).
	P_k = Q_xx
	    + K_k.transpose() * Q_uu * K_k
	    + K_k.transpose() * Q_ux
	    + Q_ux.transpose() * K_k;
	P_k = 0.5 * (P_k + Eigen::MatrixXd(P_k.transpose()));

	p_k = Q_x
	    + K_k.transpose() * Q_uu * d_k
	    + K_k.transpose() * Q_u
	    + Q_ux.transpose() * d_k;

	// CRITICAL (2026-05-27): expected-cost-reduction uses the MODIFIED quadratic
	// model that produced d_k, not the original Q_uu. Using original Q_uu here
	// is inconsistent because along negative-eigenvalue directions of Q_uu the
	// prediction is unboundedly negative (predicts a "free reduction"), causing
	// the line search z-test (z = (J_prev - J_new) / -ΔV) to reject every step
	// (z ≪ 1 because actual reduction is bounded by higher-order terms while
	// predicted is huge). Reg then ramps to reg_max and BP fails entirely.
	//
	// Math: in the eigenbasis with d̂_i = -g_i/|λ_i| (where g_i = (V^T Q_u)_i),
	//   d_k^T·Q_uu·d_k     = Σ_i g_i²/λ_i        (mixed signs, can be negative)
	//   d_k^T·H_mod·d_k    = Σ_i g_i²/|λ_i|      (always positive, bounded)
	// Using H_mod gives the model-consistent prediction -½·Σ_i g_i²/|λ_i| at α=1.
	const Eigen::VectorXd Vt_dk = V_eig.transpose() * d_k;
	const double quad_mod = (lambda_mod.array() * Vt_dk.array().square()).sum();
	deltaV(0) += d_k.dot(Q_u);
	deltaV(1) += 0.5 * quad_mod;
}

/**
 * @brief Solve a single Riccati step in reduced state space.
 *
 * Given the regularized control cost Hessian Q_uu_reg and related Q-matrices
 * (all in reduced state space), computes the feedback gain K and feedforward
 * term d, then propagates the value function (P, p) backward one timestep.
 *
 * CRITICAL: P_k/p_k use UNREGULARIZED Q_uu to prevent regularization from
 * inflating the value function and creating positive feedback in the Riccati
 * recursion. Only K and d use the regularized version.
 */
void solveRiccattiStep(
	const Eigen::Ref<const Eigen::MatrixXd>& Q_uu_reg,
	const Eigen::Ref<const Eigen::MatrixXd>& Q_uu,
	const Eigen::Ref<const Eigen::VectorXd>& Q_u,
	const Eigen::Ref<const Eigen::MatrixXd>& Q_ux,
	const Eigen::Ref<const Eigen::MatrixXd>& Q_xx,
	const Eigen::Ref<const Eigen::VectorXd>& Q_x,
	int k,
	std::vector<Eigen::MatrixXd>& K,
	std::vector<Eigen::VectorXd>& d,
	Eigen::Ref<Eigen::Vector2d> deltaV,
	Eigen::Ref<Eigen::VectorXd> p_k,
	Eigen::Ref<Eigen::MatrixXd> P_k,
	bool equilibrate
) {
	// Compute K_k = -Q_uu_reg^{-1}·Q_ux and d_k = -Q_uu_reg^{-1}·Q_u.
	Eigen::MatrixXd K_k;
	Eigen::VectorXd d_k;
	if (equilibrate) {
		// Symmetric Jacobi (diagonal) equilibration: with
		// D = diag(1/sqrt(Q_uu_reg_ii)), factor Â = D·Q_uu_reg·D (unit
		// diagonal) and recover Q_uu_reg^{-1} = D·Â^{-1}·D.  Solution-
		// preserving (same K, d in exact arithmetic); only collapses the
		// condition number from heterogeneous actuator scales so the accurate
		// step survives finite precision.  Q_uu_reg is PD (caller verified via
		// LLT) and D is real-invertible, so Â is PD by congruence.
		const int nu_local = static_cast<int>(Q_uu_reg.rows());
		Eigen::VectorXd s = Q_uu_reg.diagonal().cwiseAbs().cwiseSqrt();
		for (int j = 0; j < nu_local; ++j) s(j) = (s(j) > 1e-150) ? 1.0 / s(j) : 1.0;
		Eigen::MatrixXd A_hat = s.asDiagonal() * Q_uu_reg * s.asDiagonal();
		A_hat = 0.5 * (A_hat + Eigen::MatrixXd(A_hat.transpose()));
		Eigen::LLT<Eigen::MatrixXd> llt(A_hat);
		Eigen::MatrixXd K_scaled = llt.solve(Eigen::MatrixXd(s.asDiagonal() * Q_ux));
		Eigen::VectorXd d_scaled = llt.solve(Eigen::VectorXd(s.asDiagonal() * Q_u));
		K_k = -(s.asDiagonal() * K_scaled);
		d_k = -(s.asDiagonal() * d_scaled);
	} else {
		// Compute Q_uu_reg^{-1} using LLT
		Eigen::LLT<Eigen::MatrixXd> llt(Q_uu_reg);
		// Solve for feedback gain K_k = -(Q_uu + ρI)^{-1} * Q_ux
		K_k = -llt.solve(Q_ux);
		// Solve for feedforward term d_k = -(Q_uu + ρI)^{-1} * Q_u
		d_k = -llt.solve(Q_u);
	}
	K[k] = K_k;
	d[k] = d_k;
	
	// Use UNREGULARIZED Q_uu for Riccati value function propagation.
	// Using Q_uu_reg here inflates P_k by the regularization amount at each
	// backward step, causing exponential growth that makes backward pass fail.
	P_k = Q_xx 
	  + K_k.transpose() * Q_uu * K_k 
	  + K_k.transpose() * Q_ux 
	  + Q_ux.transpose() * K_k;
	
	// Enforce exact symmetry to prevent numerical asymmetry from accumulating
	// across many backward steps (critical for long horizons).
	P_k = 0.5 * (P_k + Eigen::MatrixXd(P_k.transpose()));
	
	p_k = Q_x 
	  + K_k.transpose() * Q_uu * d_k 
	  + K_k.transpose() * Q_u 
	  + Q_ux.transpose() * d_k;
	
	// Accumulate expected cost reduction
	deltaV(0) += d_k.dot(Q_u);
	deltaV(1) += 0.5 * d_k.dot(Q_uu * d_k);
}

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
	std::vector<Eigen::MatrixXd>* Q_uu_out,
	std::vector<Eigen::MatrixXd>* Quu_ddp_out
) {
	(void)rho;  // Suppress unused parameter warning
	const CostConfig& cost_cfg = settings.passes[0].cost;
	const ConstraintConfig& cnst_cfg = settings.constraints;
	const double dt = (settings.num_passes > 0 && std::isfinite(settings.passes[0].dt) && settings.passes[0].dt > 0.0)
		? settings.passes[0].dt
		: 1.0;

	int N = static_cast<int>(X.cols());   // Number of timesteps
	int nx = static_cast<int>(X.rows());  // Full state dimension (7 + nRW)
	int nu = static_cast<int>(U.rows()); // Control dimension
	int nRW = satellite.numRW();

	// Optional Q_uu / Quu_ddp histories (sized once if requested).
	if (Q_uu_out)   Q_uu_out->assign(static_cast<std::size_t>(std::max(0, N - 1)), Eigen::MatrixXd::Zero(nu, nu));
	if (Quu_ddp_out) Quu_ddp_out->assign(static_cast<std::size_t>(std::max(0, N - 1)), Eigen::MatrixXd::Zero(nu, nu));

	// Initialize terminal cost-to-go: p_N and P_N (in full state space)
	Eigen::VectorXd x_final = X.col(N - 1);
	Eigen::Vector3d boresight_final = boresight.col(N - 1);
	Eigen::Vector3d B_final = B.col(N - 1);
	const Eigen::Vector4d attitude_target_final = attitude_target.col(N - 1);

	auto [p_N_full, p_unused1, p_unused2] = satellite.terminalCostJacobians(x_final, boresight_final, attitude_target_final, B_final, cost_cfg);
	auto [P_N_full, P_unused1, P_unused2] = satellite.terminalCostHessians(x_final, boresight_final, attitude_target_final, B_final, cost_cfg);

	// Project terminal cost Jacobian/Hessian to reduced state using G_N
	Eigen::Vector4d q_final = x_final.segment<4>(3);
	Eigen::MatrixXd G_N = saltro::math::findGMat(q_final, nRW);
	
	Eigen::VectorXd p_k = G_N * p_N_full;
	Eigen::MatrixXd P_k = G_N * P_N_full * G_N.transpose();
	
	// Minimal disturbance config for linearization
	DisturbanceConfig dist_config;

	// Diagnostic: track Q_uu indefiniteness + feedforward magnitude across knots.
	const bool bp_eig_dbg = (std::getenv("SALTRO_BP_EIG") != nullptr);
	double bp_min_eig = 1e300; int bp_min_eig_k = -1, bp_min_eig_ctrl = -1, bp_n_indef = 0;
	double bp_max_dn = 0.0; int bp_max_dk = -1, bp_max_dctrl = -1; Eigen::VectorXd bp_worst_d;

	// Backward loop: k from N-2 down to 0
	for (int k = N - 2; k >= 0; --k) {
		// Extract trajectory data at time k
		const Eigen::VectorXd x_k = X.col(k);
		const Eigen::VectorXd u_k = U.col(k);
		const Eigen::Vector3d B_k = B.col(k);
		const Eigen::Vector3d boresight_k = boresight.col(k);
		const Eigen::Vector3d R_k = R.col(k);
		const Eigen::Vector3d V_k = V.col(k);
		const Eigen::Vector3d S_k = S.col(k);
		const Eigen::Vector4d attitude_target_k = attitude_target.col(k);
		
		// Step 1: Compute stage cost Jacobians and Hessians (full state)
		auto [lx_full, lu_mat, lux_grad] = satellite.stageCostJacobians(k, N, x_k, u_k, boresight_k, attitude_target_k, B_k, cost_cfg);
		auto [lxx_full, luu, lux_hess_full] = satellite.stageCostHessians(k, N, x_k, u_k, boresight_k, attitude_target_k, B_k, cost_cfg);
		
		// Reshape lu from 1×nu matrix to nu vector
		Eigen::VectorXd lu = lu_mat.row(0);
		
		// Scale stage cost derivatives by dt
		lx_full = dt * lx_full;
		lu = dt * lu;
		lxx_full = dt * lxx_full;
		luu = dt * luu;
		lux_hess_full = dt * lux_hess_full;
		
		// Step 2: Build G matrices for projection
		Eigen::Vector4d q_k = x_k.segment<4>(3);
		Eigen::Vector4d q_kp1 = X.col(k + 1).segment<4>(3);
		Eigen::MatrixXd G_k = saltro::math::findGMat(q_k, nRW);
		Eigen::MatrixXd G_kp1 = saltro::math::findGMat(q_kp1, nRW);
		
		// Project stage cost to reduced state
		Eigen::VectorXd lx = G_k * lx_full;
		Eigen::MatrixXd lxx = G_k * lxx_full * G_k.transpose();
		Eigen::MatrixXd lux_hess = lux_hess_full * G_k.transpose();

		// Augmented Lagrangian terms: l += lambda^T c+ + 0.5 * c+^T diag(mu) c+
		if (!lambda_aug.empty() && !mu_aug.empty() && k < static_cast<int>(lambda_aug.size()) && k < static_cast<int>(mu_aug.size())) {
			const Eigen::VectorXd c_k = satellite.constraints(k, N, x_k, u_k, S_k, cnst_cfg);
			auto [c_u, c_x_full] = satellite.constraintJacobians(k, N, x_k, u_k, S_k, cnst_cfg);
			if (lambda_aug[k].size() == c_k.size() && mu_aug[k].size() == c_k.size()) {
				const Eigen::MatrixXd c_x = c_x_full * G_k.transpose();

				// Augmented Lagrangian gradient: w_i = lambda_i + mu_i * c_i
				// Lambda term always active; mu penalty active when c > 0 OR lambda > 0
				Eigen::VectorXd w = Eigen::VectorXd::Zero(c_k.size());
				for (int i = 0; i < c_k.size(); ++i) {
					const double li = lambda_aug[k](i);
					const double mi = mu_aug[k](i);
					const double ci = c_k(i);
					// Lambda gradient always contributes
					w(i) = li;
					// Mu penalty gradient when constraint active or lambda positive
					if (ci > 0.0 || li > 0.0) {
						w(i) += mi * ci;
					}
				}

				lx.noalias() += c_x.transpose() * w;
				lu.noalias() += c_u.transpose() * w;

				// Hessian (outer product) terms: active when c > 0 OR lambda > 0
				for (int i = 0; i < c_k.size(); ++i) {
					const double li = lambda_aug[k](i);
					const double ci = c_k(i);
					if (ci <= 0.0 && li <= 0.0) {
						continue;
					}
					const double mu_i = mu_aug[k](i);
					if (!std::isfinite(mu_i) || mu_i <= 0.0) {
						continue;
					}
					const Eigen::VectorXd cx_i = c_x.row(i).transpose();
					const Eigen::VectorXd cu_i = c_u.row(i).transpose();
					lxx.noalias() += mu_i * (cx_i * cx_i.transpose());
					luu.noalias() += mu_i * (cu_i * cu_i.transpose());
					lux_hess.noalias() += mu_i * (cu_i * cx_i.transpose());
				}
			}
		}
		
		// Clamp lxx to PSD: non-convex cost functions (e.g. ang_cost_func_type=4)
		// can produce indefinite Hessians whose negative eigenvalues compound
		// through the Riccati recursion, making P_k and then Q_uu indefinite.
		// {
		// 	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(lxx);
		// 	Eigen::VectorXd eigvals = eig.eigenvalues();
		// 	if (eigvals(0) < 0.0) {
		// 		eigvals = eigvals.cwiseMax(0.0);
		// 		lxx = eig.eigenvectors() * eigvals.asDiagonal() * eig.eigenvectors().transpose();
		// 	}
		// }
		
		// Step 3: Compute exact discrete-time dynamics Jacobians using RK4 (full state)
		Eigen::MatrixXd A_k_full = Eigen::MatrixXd::Zero(nx, nx);
		Eigen::MatrixXd B_k_dyn_full = Eigen::MatrixXd::Zero(nx, nu);
		
		auto dynamics_jac_wrapper = [&](double t_local, const Eigen::Ref<const Eigen::VectorXd>& x_local,
		                                  const Eigen::Ref<const Eigen::VectorXd>& u_local,
		                                  Eigen::Ref<Eigen::MatrixXd> A_c_out,
		                                  Eigen::Ref<Eigen::MatrixXd> B_c_out,
		                                  Eigen::Ref<Eigen::VectorXd> k_out) {
			(void)t_local;
			auto [A_c, B_c, C_unused] = satellite.dynamicsJacobians(x_local, u_local, dist_config, R_k, B_k, S_k, V_k);
			A_c_out = A_c;
			B_c_out = B_c;
			k_out = satellite.dynamics(x_local, u_local, dist_config, R_k, B_k, S_k, V_k, 0);
		};
		
		saltro::math::rk4_jacobians(dynamics_jac_wrapper, x_k, u_k, 0.0, dt, A_k_full, B_k_dyn_full);

		// Step 4: Project dynamics Jacobians to reduced state space
		// A_reduced = G_{k+1} * A_full * G_k^T
		// B_reduced = G_{k+1} * B_full
		Eigen::MatrixXd A_k = G_kp1 * A_k_full * G_k.transpose();
		Eigen::MatrixXd B_k_dyn = G_kp1 * B_k_dyn_full;

		// Step 4b (optional DDP): compute dynamics Hessian contributions
		// to the Q matrices. Standard DDP:
		//   Q_xx += V_x · f_xx,  Q_uu += V_x · f_uu,  Q_ux += V_x · f_ux
		// where V_x is the cost-to-go gradient (here `p_k`, in REDUCED state)
		// and f_xx etc. are the DISCRETE dynamics Hessians (in FULL state).
		// Projection:
		//   1) Lift reduced V_x to full: coeffs = G_{k+1}^T · p_k  (size nx)
		//   2) Contract: H_full[i,j] = Σ_l coeffs[l] · f_xx_full[l, i, j]
		//   3) Project x-inputs to reduced via G_k: H_reduced = G_k · H_full · G_k^T
		// Control inputs don't need projection (u has no reduced form).
		//
		// Gated by reg_cfg.use_dynamics_hess (flag existed unread prior to 2026-04-24).
		const int nxr_local = static_cast<int>(G_k.rows());
		Eigen::MatrixXd Qxx_ddp = Eigen::MatrixXd::Zero(nxr_local, nxr_local);
		Eigen::MatrixXd Qux_ddp = Eigen::MatrixXd::Zero(nu, nxr_local);
		Eigen::MatrixXd Quu_ddp = Eigen::MatrixXd::Zero(nu, nu);

		const auto& reg_cfg_for_ddp = settings.passes[0].reg;
		if (reg_cfg_for_ddp.use_dynamics_hess) {
			std::vector<Eigen::MatrixXd> F_xx_full, F_ux_full, F_uu_full;

			// Callback wraps satellite's continuous Jacobians + Hessians for RK4 composition.
			auto dyn_hess_wrapper = [&](double /*t_local*/,
			                             const Eigen::Ref<const Eigen::VectorXd>& x_local,
			                             const Eigen::Ref<const Eigen::VectorXd>& u_local,
			                             Eigen::Ref<Eigen::MatrixXd> A_out,
			                             Eigen::Ref<Eigen::MatrixXd> B_out,
			                             Eigen::Ref<Eigen::VectorXd> k_out,
			                             std::vector<Eigen::MatrixXd>& fxx_out,
			                             std::vector<Eigen::MatrixXd>& fux_out,
			                             std::vector<Eigen::MatrixXd>& fuu_out) {
				auto [A_c, B_c, C_unused] = satellite.dynamicsJacobians(
					x_local, u_local, dist_config, R_k, B_k, S_k, V_k);
				A_out = A_c;
				B_out = B_c;
				k_out = satellite.dynamics(x_local, u_local, dist_config, R_k, B_k, S_k, V_k, 0);

				auto [hxx, hux, huu] = satellite.dynamicsHessians(
					x_local, u_local, dist_config, R_k, B_k, S_k, V_k);
				const int nx_l = static_cast<int>(x_local.size());
				const int nu_l = static_cast<int>(u_local.size());
				fxx_out.assign(static_cast<std::size_t>(nx_l),
				               Eigen::MatrixXd::Zero(nx_l, nx_l));
				fux_out.assign(static_cast<std::size_t>(nx_l),
				               Eigen::MatrixXd::Zero(nu_l, nx_l));
				fuu_out.assign(static_cast<std::size_t>(nx_l),
				               Eigen::MatrixXd::Zero(nu_l, nu_l));
				for (int l = 0; l < nx_l; ++l) {
					fxx_out[static_cast<std::size_t>(l)] =
						hxx.slice(l).topLeftCorner(nx_l, nx_l);
					fux_out[static_cast<std::size_t>(l)] =
						hux.slice(l).topLeftCorner(nu_l, nx_l);
					fuu_out[static_cast<std::size_t>(l)] =
						huu.slice(l).topLeftCorner(nu_l, nu_l);
				}
			};

			saltro::math::rk4_hessians(dyn_hess_wrapper, x_k, u_k, 0.0, dt, F_xx_full, F_ux_full, F_uu_full);

			// Lift reduced V_x (p_k) to full-state coefficients, then contract.
			const Eigen::VectorXd coeffs = G_kp1.transpose() * p_k;  // (nx,)
			for (int l = 0; l < nx; ++l) {
				const double c = coeffs(l);
				if (std::abs(c) < 1e-30) continue;
				Qxx_ddp.noalias() += c * (G_k * F_xx_full[static_cast<std::size_t>(l)] * G_k.transpose());
				Qux_ddp.noalias() += c * (F_ux_full[static_cast<std::size_t>(l)] * G_k.transpose());
				Quu_ddp.noalias() += c * F_uu_full[static_cast<std::size_t>(l)];
			}

			// Attitude-manifold curvature correction for the mrp-mrp block of
			// Qxx_ddp. Planning with Attitude (Jackson 2021) eq 15 gives the
			// reduced-state Hessian of a scalar h(q) as
			//   ∇²h_reduced = G^T · (∂²h/∂q²) · G − I₃ · (∂h/∂q) · q
			// Treating `h(x) = V_x · f(x)` as a scalar:
			//   ∂h/∂q   = V_x_full · (∂f_full/∂q)   (1×4 vector)
			//   ∂²h/∂q² = V_x · f_xx_full (what's above)
			// So the correction for the mrp block is `-I₃ · (V_x · ∂f/∂q · q)`.
			// Missing this term makes the DDP quadratic model inconsistent with
			// the reduced-state parameterization; in practice the solver
			// "converges quickly to wrong answers" because the quadratic model
			// is biased relative to the true reduced cost-to-go.
			//
			// Qux and Quu are mixed / control-only respectively, no manifold
			// correction needed there.
			{
				constexpr int QUAT_START = 3;                       // full state: ω(0-2), q(3-6)
				constexpr int MRP_START  = 3;                       // reduced: [ω(0-2), mrp(3-5), h(6+)]

				// V_x_full · ∂f_full/∂q: extract the 4 quaternion columns of A_k_full.
				// Row vector (1×4) of scalar gradient components wrt q.
				const Eigen::RowVector4d Vq =
					coeffs.transpose() * A_k_full.block(0, QUAT_START, nx, 4);
				const Eigen::Vector4d q_nom = x_k.segment<4>(QUAT_START);
				const double corr = Vq * q_nom;    // scalar (V_x · ∂f/∂q · q)

				Qxx_ddp.block<3, 3>(MRP_START, MRP_START).noalias()
					-= corr * Eigen::Matrix3d::Identity();
			}
		}

		// Diagnostic: copy Quu_ddp before any clip / fold into Q_uu (zero
		// matrix when use_dynamics_hess=false, so always meaningful).
		if (Quu_ddp_out) (*Quu_ddp_out)[static_cast<std::size_t>(k)] = Quu_ddp;

		// PSD-clip Quu_ddp AND Qxx_ddp: drop indefinite "anti-curvature"
		// eigenmodes. RK4 cross-stage interaction can produce a Quu_ddp
		// diagonal severely negative for control axes that are linear-in-u
		// in continuous time (notably RW: ḣ = −u_rw → continuous f_uu = 0;
		// discrete F_uu nonzero, indefinite when contracted with the huge
		// cost-to-go gradient p_l). Same artifact mechanism in Qxx_ddp via
		// the manifold projection and ω-quadratic curvature, with the
		// indefiniteness propagating forward through P_k. Clipping both
		// keeps genuine positive-curvature info while dropping the
		// spurious anti-curvature.
		if (settings.passes[0].reg.psd_clip_quu_ddp) {
			auto psd_clip = [](Eigen::MatrixXd& M) {
				const Eigen::MatrixXd Msym = 0.5 * (M + Eigen::MatrixXd(M.transpose()));
				Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Msym);
				const Eigen::VectorXd lam = es.eigenvalues().cwiseMax(0.0);
				M = es.eigenvectors() * lam.asDiagonal() * es.eigenvectors().transpose();
			};
			psd_clip(Quu_ddp);
			psd_clip(Qxx_ddp);
			// Qux_ddp is the off-diagonal cross block: no standalone PSD
			// concept, but we leave it intact so the combined DDP Hessian
			// keeps its structural relationship between x and u curvatures.
		}

		// Step 5: Assemble Q matrices (all in reduced state space now)
		Eigen::MatrixXd Q_xx = lxx + A_k.transpose() * P_k * A_k + Qxx_ddp;
		Eigen::MatrixXd Q_uu = luu + B_k_dyn.transpose() * P_k * B_k_dyn + Quu_ddp;
		Eigen::MatrixXd Q_ux = lux_hess + B_k_dyn.transpose() * P_k * A_k + Qux_ddp;
		Eigen::VectorXd Q_x = lx + A_k.transpose() * p_k;
		Eigen::VectorXd Q_u = lu + B_k_dyn.transpose() * p_k;

		// Symmetrize Q_uu before regularization (numerical noise can break Cholesky)
		Q_uu = 0.5 * (Q_uu + Eigen::MatrixXd(Q_uu.transpose()));

		// Diagnostic: copy Q_uu (un-regularized, post-symmetrization).
		if (Q_uu_out) (*Q_uu_out)[static_cast<std::size_t>(k)] = Q_uu;

		// Diagnostic: report indefiniteness of the UNREGULARIZED Q_uu (the one
		// used for P_k and ΔV).  A negative eigenvalue here means the iLQR step
		// is an ascent direction along that mode → forward pass cannot descend.
		if (bp_eig_dbg) {
			Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_dbg(Q_uu);
			const double me = es_dbg.eigenvalues().minCoeff();
			if (me < -1e-9) ++bp_n_indef;
			if (me < bp_min_eig) {
				bp_min_eig = me; bp_min_eig_k = k;
				es_dbg.eigenvectors().col(0).cwiseAbs().maxCoeff(&bp_min_eig_ctrl);
			}
		}

		const auto& reg_cfg = settings.passes[0].reg;
		if (reg_cfg.use_eigen_modification) {
			Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Q_uu);

			// Uniform-trigger emulation: mimic `LLT(Q_uu + ρI)` failing
			// when Q_uu isn't PD-after-ρ (λ_min < -ρ).  Without this,
			// the eigen path never signals "bump ρ" for conditioning
			// reasons, only FP failures drive reg growth — too coarse.
			const double lam_min_orig = es.eigenvalues().minCoeff();
			if (reg_cfg.eigen_reg_mimic_uniform_trigger && lam_min_orig < -reg) {
				SALTRO_OPT_DLOG("[BP-E] FAIL-mimic k=" << k << " reg=" << reg
				                 << " λmin=" << lam_min_orig);
				return false;
			}

			// Cells gated by flags.  Cell 1 (absolute floor at ρ) always on.
			Eigen::VectorXd lam_mod = reg_cfg.eigen_reg_use_abs
				? Eigen::VectorXd(es.eigenvalues().cwiseAbs())     // Cell 2
				: Eigen::VectorXd(es.eigenvalues());
			double floor_v = reg;                                   // Cell 1
			if (reg_cfg.eigen_reg_use_relative_floor) {             // Cell 5
				const double rel = reg_cfg.eigen_reg_condition_cap
					* (reg_cfg.eigen_reg_use_abs
						? lam_mod.maxCoeff()
						: lam_mod.cwiseAbs().maxCoeff());
				floor_v = std::max(floor_v, rel);
			}
			if (reg_cfg.eigen_reg_add_mode) {
				// λ' = (abs?|λ|:λ) + floor — uniform-Tikhonov-compatible
				lam_mod = (lam_mod.array() + floor_v).matrix();
			} else {
				// λ' = max(|λ|, floor) — leaves healthy eigenvalues untouched
				lam_mod = lam_mod.cwiseMax(floor_v);
			}

			solveRiccattiStepEigen(es.eigenvectors(), lam_mod, Q_uu, Q_u, Q_ux,
			                       Q_xx, Q_x, k, K, d, deltaV, p_k, P_k);
			SALTRO_OPT_DLOG("[BP-E] k=" << k << " reg=" << reg << " λmin=" << lam_mod.minCoeff()
			                 << " λmax=" << lam_mod.maxCoeff() << " ||d||=" << d[k].norm());
		} else {
			Eigen::MatrixXd Q_uu_reg = Q_uu + reg * Eigen::MatrixXd::Identity(nu, nu);
			Eigen::LLT<Eigen::MatrixXd> llt(Q_uu_reg);

			if (llt.info() != Eigen::Success) {
				SALTRO_OPT_DLOG("[BP] FAIL k=" << k << " reg=" << reg);
				return false;
			}

			solveRiccattiStep(Q_uu_reg, Q_uu, Q_u, Q_ux, Q_xx, Q_x, k, K, d, deltaV, p_k, P_k,
			                  reg_cfg.equilibrate_quu);
			SALTRO_OPT_DLOG("[BP] k=" << k << " reg=" << reg << " ||d||=" << d[k].norm());
		}
		if (bp_eig_dbg) {
			int didx = 0; const double dn = d[k].cwiseAbs().maxCoeff(&didx);
			if (dn > bp_max_dn) { bp_max_dn = dn; bp_max_dk = k; bp_max_dctrl = didx; bp_worst_d = d[k]; }
		}
	}
	if (bp_eig_dbg) {
		std::cout << "[BP-EIG] reg=" << reg << " min_eig(Q_uu)=" << bp_min_eig
		          << " @k=" << bp_min_eig_k << " neg-mode_ctrl=" << bp_min_eig_ctrl
		          << " n_indef=" << bp_n_indef << "/" << (N - 1)
		          << " | max|d|_inf=" << bp_max_dn << " @k=" << bp_max_dk
		          << " ctrl=" << bp_max_dctrl;
		if (bp_worst_d.size() > 0) std::cout << " d=" << bp_worst_d.transpose();
		std::cout << std::endl;
	}
	SALTRO_OPT_DLOG("[BP] success dV1=" << deltaV(0) << " dV2=" << deltaV(1));

	return true;

} 

} // namespace saltro::optimizer
