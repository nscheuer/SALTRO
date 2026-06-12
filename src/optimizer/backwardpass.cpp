#include <saltro/optimizer/backwardpass.h>
#include <saltro/math/integrators/rk4.h>
#include <saltro/math/matrix.h>
#include <saltro/math/mrp.h>
#include <iostream>
#include <cmath>

#if defined(SALTRO_DEBUG_BUILD)
#define SALTRO_OPT_DLOG(msg) do { std::cout << msg << std::endl; } while (0)
#else
#define SALTRO_OPT_DLOG(msg) do {} while (0)
#endif

namespace saltro::optimizer {

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
	Eigen::Ref<Eigen::MatrixXd> P_k
) {
	// Compute Q_uu_reg^{-1} using LLT
	Eigen::LLT<Eigen::MatrixXd> llt(Q_uu_reg);
	
	// Solve for feedback gain K_k = -(Q_uu + ρI)^{-1} * Q_ux
	Eigen::MatrixXd K_k = -llt.solve(Q_ux);
	K[k] = K_k;
	
	// Solve for feedforward term d_k = -(Q_uu + ρI)^{-1} * Q_u
	Eigen::VectorXd d_k = -llt.solve(Q_u);
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
	
	// p_k update: the textbook form has 4 terms,
	//   p_k = Q_x + K^T·Q_uu·d_k + K^T·Q_u + Q_ux^T·d_k,
	// but d_k = -Q_uu^{-1}·Q_u, so K^T·Q_uu·d_k = -K^T·Q_u and the middle
	// two cancel exactly in real arithmetic.  In fp64 they cancel only
	// up to ~eps·||K||·||Q_uu||·||d_k|| of catastrophic-cancellation
	// noise, which accumulates across the Riccati recursion.  We use
	// the algebraically equivalent 2-term form to avoid that noise.
	p_k = Q_x + Q_ux.transpose() * d_k;
	
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
	const std::vector<Eigen::VectorXd>& mu_aug
) {
	if (settings.passes[0].reg.use_sqrt_bp) {
		return backwardPassSqrt(
			satellite, X, U, R, V, B, S, rho, boresight, attitude_target,
			settings, reg, K, d, deltaV, lambda_aug, mu_aug
		);
	}
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

	// Fold the terminal-knot (k = N-1) AL constraint terms into the cost-to-go
	// seed. The forward-pass merit and the alilqr multiplier update both price
	// constraints at the terminal knot, so the quadratic model must see them
	// too; otherwise a terminal violation is an unexplainable merit offset the
	// line search can never optimize away. There is no control at the terminal
	// knot (the merit evaluates it with u = 0 and constraints(N-1, N, ...)
	// drops the actuator rows), so only the state terms enter: c_x^T w into
	// p_k and the active-set Gauss-Newton mu * c_x c_x^T terms into P_k.
	// Semantics must match the in-loop AL block below and the FP merit:
	// lambda gradient always on (signed c); mu terms when c > 0 OR lambda > 0.
	if (!lambda_aug.empty() && !mu_aug.empty()
			&& (N - 1) < static_cast<int>(lambda_aug.size())
			&& (N - 1) < static_cast<int>(mu_aug.size())) {
		const int kT = N - 1;
		const Eigen::VectorXd u_zero = Eigen::VectorXd::Zero(satellite.controlDim());
		const Eigen::Vector3d S_final = S.col(kT);
		const Eigen::VectorXd c_T = satellite.constraints(kT, N, x_final, u_zero, S_final, cnst_cfg);
		if (lambda_aug[kT].size() == c_T.size() && mu_aug[kT].size() == c_T.size()) {
			auto [cT_u, cT_x_full] = satellite.constraintJacobians(kT, N, x_final, u_zero, S_final, cnst_cfg);
			(void)cT_u;  // No control exists at the terminal knot.
			const Eigen::MatrixXd c_x = cT_x_full * G_N.transpose();

			Eigen::VectorXd w = Eigen::VectorXd::Zero(c_T.size());
			for (int i = 0; i < c_T.size(); ++i) {
				const double li = lambda_aug[kT](i);
				const double ci = c_T(i);
				w(i) = li;
				if (ci > 0.0 || li > 0.0) {
					w(i) += mu_aug[kT](i) * ci;
				}
			}
			p_k.noalias() += c_x.transpose() * w;

			for (int i = 0; i < c_T.size(); ++i) {
				if (c_T(i) <= 0.0 && lambda_aug[kT](i) <= 0.0) {
					continue;
				}
				const double mu_i = mu_aug[kT](i);
				if (!std::isfinite(mu_i) || mu_i <= 0.0) {
					continue;
				}
				const Eigen::VectorXd cx_i = c_x.row(i).transpose();
				P_k.noalias() += mu_i * (cx_i * cx_i.transpose());
			}
			// Keep the seed exactly symmetric (rank-1 updates are symmetric in
			// exact arithmetic; this guards against numerical asymmetry).
			P_k = 0.5 * (P_k + Eigen::MatrixXd(P_k.transpose()));
		}
	}

	// Linearize with the same disturbance config the forward pass rolls out
	// (forwardpass.cpp uses settings.disturbances). A default-constructed
	// config here makes A_k/B_k disagree with the actual rollout whenever any
	// plan_for_* flag is on, biasing the expected-decrease prediction.
	const DisturbanceConfig& dist_config = settings.disturbances;
	
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
		
		// NOTE: previously scaled stage-cost derivatives by dt, but
		// Satellite::totalCost SUMS stageCost without any dt factor (see
		// satellite.cpp `J_total += stage_cost`). Multiplying gradients by
		// dt was therefore inconsistent with the cost being optimized — BP's
		// predicted ΔV came out a factor of dt too large in pure-cost regions
		// and mixed in AL regions (since the AL gradient is added below WITHOUT
		// dt scaling, so AL part was un-scaled while stage part was over-scaled).
		// Confirmed via closed-loop FD gradient check: bisect by cost component
		// (angle only, +ω, +control, +AL) all gave ratio = 1/dt = 0.10 pre-fix
		// and ratio = 1.000 ± 0.025 after the dt factors were removed.
		// Smoke test on the chronic 12_omega_5x ict=1e-3 + spike scenario:
		//   pre-fix : 137 iters, 6.26°PE, ls_attempts_exceeded
		//   post-fix: 325 iters, 2.14°PE, converged (better than synced 3.0°)
		
		// Step 2: Build G matrices for projection
		Eigen::Vector4d q_k = x_k.segment<4>(3);
		Eigen::Vector4d q_kp1 = X.col(k + 1).segment<4>(3);
		Eigen::MatrixXd G_k = saltro::math::findGMat(q_k, nRW);
		Eigen::MatrixXd G_kp1 = saltro::math::findGMat(q_kp1, nRW);
		
		// Project stage cost to reduced state
		Eigen::VectorXd lx = G_k * lx_full;
		Eigen::MatrixXd lxx = G_k * lxx_full * G_k.transpose();
		Eigen::MatrixXd lux_hess = lux_hess_full * G_k.transpose();

		const RegularizationConfig& reg_cfg = settings.passes[0].reg;

		// Augmented Lagrangian terms: l += lambda^T c + active-set 0.5 * c^T diag(mu) c
		if (!lambda_aug.empty() && !mu_aug.empty() && k < static_cast<int>(lambda_aug.size()) && k < static_cast<int>(mu_aug.size())) {
			const Eigen::VectorXd c_k = satellite.constraints(k, N, x_k, u_k, S_k, cnst_cfg);
			auto [c_u, c_x_full] = satellite.constraintJacobians(k, N, x_k, u_k, S_k, cnst_cfg);
			if (lambda_aug[k].size() == c_k.size() && mu_aug[k].size() == c_k.size()) {
				const Eigen::MatrixXd c_x = c_x_full * G_k.transpose();

				// ALTRO active-set semantics: the lambda gradient is always on
				// (signed c, so feasibility is rewarded and lambda can pull
				// back); the mu penalty is active when c > 0 OR lambda > 0.
				// Must match the FP/inner merit and the alilqr lambda update.
				Eigen::VectorXd w = Eigen::VectorXd::Zero(c_k.size());
				for (int i = 0; i < c_k.size(); ++i) {
					const double li = lambda_aug[k](i);
					const double ci = c_k(i);
					w(i) = li;
					if (ci > 0.0 || li > 0.0) {
						w(i) += mu_aug[k](i) * ci;
					}
				}

				lx.noalias() += c_x.transpose() * w;
				lu.noalias() += c_u.transpose() * w;

				for (int i = 0; i < c_k.size(); ++i) {
					if (c_k(i) <= 0.0 && lambda_aug[k](i) <= 0.0) {
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

				// G13 (optional DDP): constraint CURVATURE term.
				// The GN Hessian above keeps only mu·c_x c_x^T; the true AL
				// Hessian also has Σ_i w_i · ∂²c_i/∂(·)², where w_i is the SAME
				// active-set merit gradient (lambda_i + active·mu_i·c_i) used for
				// lx/lu just above. Gated on the SAME active set (c>0 OR λ>0).
				// x-blocks are projected to reduced state with G_k (consistent
				// with c_x = c_x_full · G_k^T); u-blocks need no projection.
				// Gated by reg.use_constraint_hess (dead knob before this).
				if (reg_cfg.use_constraint_hess) {
					auto [H_uu, H_ux, H_xx] = satellite.constraintHessians(k, N, x_k, u_k, S_k, cnst_cfg);
					const int nxr = static_cast<int>(G_k.rows());
					Eigen::MatrixXd Cxx_full = Eigen::MatrixXd::Zero(nx, nx);
					Eigen::MatrixXd Cux_full = Eigen::MatrixXd::Zero(nu, nx);
					Eigen::MatrixXd Cuu = Eigen::MatrixXd::Zero(nu, nu);
					for (int i = 0; i < c_k.size(); ++i) {
						const double li = lambda_aug[k](i);
						const double ci = c_k(i);
						if (ci <= 0.0 && li <= 0.0) {
							continue;
						}
						const double wi = w(i);
						if (!std::isfinite(wi) || wi == 0.0) {
							continue;
						}
						Cxx_full.noalias() += wi * H_xx.slice(i).topLeftCorner(nx, nx);
						Cux_full.noalias() += wi * H_ux.slice(i).topLeftCorner(nu, nx);
						Cuu.noalias()      += wi * H_uu.slice(i).topLeftCorner(nu, nu);
					}
					Eigen::MatrixXd Cxx = G_k * Cxx_full * G_k.transpose();   // (nxr × nxr)
					Eigen::MatrixXd Cux = Cux_full * G_k.transpose();         // (nu × nxr)
					if (reg_cfg.psd_clip_quu_ddp) {
						saltro::math::psd_clip(Cxx);
						saltro::math::psd_clip(Cuu);
					}
					(void)nxr;
					lxx.noalias()       += Cxx;
					luu.noalias()       += Cuu;
					lux_hess.noalias()  += Cux;
				}
			}
		}
		
		// Optional PSD clamp on lxx (reg.psd_clamp_lxx): non-convex cost
		// Hessians (historically ang_cost_func_type=2, raw acos, which was
		// concave in d — now removed; but the exact vec-mode angle-cost
		// Hessian with the Gauss-Newton flag off can still be indefinite) can
		// produce indefinite lxx whose negative eigenvalues compound through
		// the Riccati recursion, making P_k and then Q_uu indefinite.
		//
		// TESTING/DIAGNOSTIC aid only -- NOT recommended for production.
		// Turning it on proves that an indefinite cost Hessian is the
		// culprit when a solve fails, but the eigendecomposition per knot is
		// slow and clamping masks model problems rather than fixing them.
		// Default false: the clamp is skipped and behavior is identical to
		// the unflagged backward pass.
		if (settings.passes[0].reg.psd_clamp_lxx) {
			Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(lxx);
			Eigen::VectorXd eigvals = eig.eigenvalues();
			if (eigvals(0) < 0.0) {
				eigvals = eigvals.cwiseMax(0.0);
				lxx = eig.eigenvectors() * eigvals.asDiagonal() * eig.eigenvectors().transpose();
			}
		}
		
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
		
		rk4_jacobians(dynamics_jac_wrapper, x_k, u_k, 0.0, dt, A_k_full, B_k_dyn_full);
		
		// Step 4: Project dynamics Jacobians to reduced state space
		// A_reduced = G_{k+1} * A_full * G_k^T
		// B_reduced = G_{k+1} * B_full
		Eigen::MatrixXd A_k = G_kp1 * A_k_full * G_k.transpose();
		Eigen::MatrixXd B_k_dyn = G_kp1 * B_k_dyn_full;

		// Step 4b (optional DDP, G12): dynamics SECOND-ORDER terms.
		// Gauss-Newton (the default) drops the pᵀ·∂²f term; full DDP adds
		//   Q_xx += Σ_l p_l ∂²f_l/∂x²,  Q_uu += Σ_l p_l ∂²f_l/∂u²,
		//   Q_ux += Σ_l p_l ∂²f_l/∂u∂x,
		// where p (cost-to-go gradient) is in REDUCED state and the discrete
		// dynamics Hessian F_*[l] (from rk4_hessians, consistent with the
		// non-normalizing rk4_jacobians above) is in FULL state. Projection,
		// matching how A_k/B_k are projected with G:
		//   1) lift reduced p to full: coeffs = G_{k+1}^T · p_k  (size nx)
		//   2) contract: H_full = Σ_l coeffs[l] · F_*[l]
		//   3) project x-inputs to reduced with G_k. Controls need no projection.
		// Gated by reg.use_dynamics_hess (dead knob before this).
		const int nxr_local = static_cast<int>(G_k.rows());
		Eigen::MatrixXd Qxx_ddp = Eigen::MatrixXd::Zero(nxr_local, nxr_local);
		Eigen::MatrixXd Qux_ddp = Eigen::MatrixXd::Zero(nu, nxr_local);
		Eigen::MatrixXd Quu_ddp = Eigen::MatrixXd::Zero(nu, nu);

		if (reg_cfg.use_dynamics_hess) {
			std::vector<Eigen::MatrixXd> F_xx_full, F_ux_full, F_uu_full;

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
				fxx_out.assign(static_cast<std::size_t>(nx_l), Eigen::MatrixXd::Zero(nx_l, nx_l));
				fux_out.assign(static_cast<std::size_t>(nx_l), Eigen::MatrixXd::Zero(nu_l, nx_l));
				fuu_out.assign(static_cast<std::size_t>(nx_l), Eigen::MatrixXd::Zero(nu_l, nu_l));
				for (int l = 0; l < nx_l; ++l) {
					fxx_out[static_cast<std::size_t>(l)] = hxx.slice(l).topLeftCorner(nx_l, nx_l);
					fux_out[static_cast<std::size_t>(l)] = hux.slice(l).topLeftCorner(nu_l, nx_l);
					fuu_out[static_cast<std::size_t>(l)] = huu.slice(l).topLeftCorner(nu_l, nu_l);
				}
			};

			rk4_hessians(dyn_hess_wrapper, x_k, u_k, 0.0, dt, F_xx_full, F_ux_full, F_uu_full);

			const Eigen::VectorXd coeffs = G_kp1.transpose() * p_k;  // (nx,)
			for (int l = 0; l < nx; ++l) {
				const double c = coeffs(l);
				if (std::abs(c) < 1e-30) continue;
				Qxx_ddp.noalias() += c * (G_k * F_xx_full[static_cast<std::size_t>(l)] * G_k.transpose());
				Qux_ddp.noalias() += c * (F_ux_full[static_cast<std::size_t>(l)] * G_k.transpose());
				Quu_ddp.noalias() += c * F_uu_full[static_cast<std::size_t>(l)];
			}

			// Attitude-manifold curvature correction for the mrp-mrp block of
			// Qxx_ddp (Jackson 2021, "Planning with Attitude", eq. 15). For a
			// scalar h(x) = p · f(x), the reduced-state Hessian carries an
			// extra -I₃·(∂h/∂q · q) term beyond G^T(∂²h/∂q²)G. Without it the
			// DDP quadratic model is biased relative to the reduced cost-to-go.
			{
				constexpr int QUAT_START = 3;  // full state: ω(0-2), q(3-6)
				constexpr int MRP_START  = 3;  // reduced:    ω(0-2), mrp(3-5), h(6+)
				const Eigen::RowVector4d Vq =
					coeffs.transpose() * A_k_full.block(0, QUAT_START, nx, 4);
				const Eigen::Vector4d q_nom = x_k.segment<4>(QUAT_START);
				const double corr = Vq * q_nom;
				Qxx_ddp.block<3, 3>(MRP_START, MRP_START).noalias()
					-= corr * Eigen::Matrix3d::Identity();
			}

			// PSD-clip the DDP curvature (opt-in): drop indefinite eigenmodes so
			// true second-order terms cannot make Q_uu an ascent direction.
			if (reg_cfg.psd_clip_quu_ddp) {
				saltro::math::psd_clip(Quu_ddp);
				saltro::math::psd_clip(Qxx_ddp);
				// Qux_ddp is the cross block (no standalone PSD concept); left intact.
			}
		}

		// Step 5: Assemble Q matrices (all in reduced state space now)
		Eigen::MatrixXd Q_xx = lxx + A_k.transpose() * P_k * A_k + Qxx_ddp;
		Eigen::MatrixXd Q_uu = luu + B_k_dyn.transpose() * P_k * B_k_dyn + Quu_ddp;
		Eigen::MatrixXd Q_ux = lux_hess + B_k_dyn.transpose() * P_k * A_k + Qux_ddp;
		Eigen::VectorXd Q_x = lx + A_k.transpose() * p_k;
		Eigen::VectorXd Q_u = lu + B_k_dyn.transpose() * p_k;
		
		Eigen::MatrixXd Q_uu_reg = Q_uu + reg * Eigen::MatrixXd::Identity(nu, nu);

		// Eigen's LLT does not reliably flag NaN input (NaN comparisons are
		// all false), so a poisoned Q_uu can factor "successfully" and emit
		// NaN gains that only die later in the rollout, misattributed to the
		// line search. Check explicitly before and after the solve.
		if (!Q_uu_reg.allFinite()) {
			SALTRO_OPT_DLOG("[BP] FAIL k=" << k << " Q_uu_reg not finite");
			return false;
		}

		Eigen::LLT<Eigen::MatrixXd> llt(Q_uu_reg);

		if (llt.info() != Eigen::Success) {
			SALTRO_OPT_DLOG("[BP] FAIL k=" << k << " reg=" << reg);
			return false;
		}

		solveRiccattiStep(Q_uu_reg, Q_uu, Q_u, Q_ux, Q_xx, Q_x, k, K, d, deltaV, p_k, P_k);

		if (!K[k].allFinite() || !d[k].allFinite() || !P_k.allFinite()) {
			SALTRO_OPT_DLOG("[BP] FAIL k=" << k << " non-finite gains or value function");
			return false;
		}
		SALTRO_OPT_DLOG("[BP] k=" << k << " reg=" << reg << " ||d||=" << d[k].norm());
	}
	SALTRO_OPT_DLOG("[BP] success dV1=" << deltaV(0) << " dV2=" << deltaV(1));
	
	return true;

} 

} // namespace saltro::optimizer
