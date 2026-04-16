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
	const std::vector<Eigen::VectorXd>& mu_aug
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
		
		rk4_jacobians(dynamics_jac_wrapper, x_k, u_k, 0.0, dt, A_k_full, B_k_dyn_full);
		
		// Step 4: Project dynamics Jacobians to reduced state space
		// A_reduced = G_{k+1} * A_full * G_k^T
		// B_reduced = G_{k+1} * B_full
		Eigen::MatrixXd A_k = G_kp1 * A_k_full * G_k.transpose();
		Eigen::MatrixXd B_k_dyn = G_kp1 * B_k_dyn_full;
		
		// Step 5: Assemble Q matrices (all in reduced state space now)
		Eigen::MatrixXd Q_xx = lxx + A_k.transpose() * P_k * A_k;
		Eigen::MatrixXd Q_uu = luu + B_k_dyn.transpose() * P_k * B_k_dyn;
		Eigen::MatrixXd Q_ux = lux_hess + B_k_dyn.transpose() * P_k * A_k;
		Eigen::VectorXd Q_x = lx + A_k.transpose() * p_k;
		Eigen::VectorXd Q_u = lu + B_k_dyn.transpose() * p_k;
		
		// Symmetrize Q_uu before regularization (numerical noise can break Cholesky)
		Q_uu = 0.5 * (Q_uu + Eigen::MatrixXd(Q_uu.transpose()));
		Eigen::MatrixXd Q_uu_reg = Q_uu + reg * Eigen::MatrixXd::Identity(nu, nu);
		Eigen::LLT<Eigen::MatrixXd> llt(Q_uu_reg);
		
		if (llt.info() != Eigen::Success) {
			SALTRO_OPT_DLOG("[BP] FAIL k=" << k << " reg=" << reg);
			return false;
		}
		
		solveRiccattiStep(Q_uu_reg, Q_uu, Q_u, Q_ux, Q_xx, Q_x, k, K, d, deltaV, p_k, P_k);
		SALTRO_OPT_DLOG("[BP] k=" << k << " reg=" << reg << " ||d||=" << d[k].norm());
	}
	SALTRO_OPT_DLOG("[BP] success dV1=" << deltaV(0) << " dV2=" << deltaV(1));
	
	return true;

} 

} // namespace saltro::optimizer
