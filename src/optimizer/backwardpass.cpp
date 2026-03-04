#include <saltro/optimizer/backwardpass.h>
#include <iostream>

#if defined(SALTRO_DEBUG_BUILD)
#define SALTRO_OPT_DLOG(msg) do { std::cout << msg << std::endl; } while (0)
#else
#define SALTRO_OPT_DLOG(msg) do {} while (0)
#endif

namespace saltro::optimizer {

/**
 * @brief Solve a single Riccati step.
 *
 * Given the regularized control cost Hessian Q_uu_reg and related Q-matrices,
 * computes the feedback gain K and feedforward term d, then propagates
 * the value function (P, p) backward one timestep using Riccati equations.
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
	
	// Riccati update for value function Hessian
	// P_k = Q_xx + K_k^T * Q_uu * K_k + K_k^T * Q_ux + Q_ux^T * K_k
	P_k = Q_xx 
	  + K_k.transpose() * Q_uu * K_k 
	  + K_k.transpose() * Q_ux 
	  + Q_ux.transpose() * K_k;
	
	// Riccati update for value function gradient
	// p_k = Q_x + K_k^T * Q_uu * d_k + K_k^T * Q_u + Q_ux^T * d_k
	p_k = Q_x 
	  + K_k.transpose() * Q_uu * d_k 
	  + K_k.transpose() * Q_u 
	  + Q_ux.transpose() * d_k;
	
	// Accumulate expected cost reduction
	// V_1k = d_k^T * Q_u
	// V_2k = 1/2 * d_k^T * Q_uu * d_k
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
	std::vector<Eigen::MatrixXd>& K,
	std::vector<Eigen::VectorXd>& d,
	Eigen::Ref<Eigen::Vector2d> deltaV
) {
	(void)rho;  // Suppress unused parameter warning
	const CostConfig& cost_cfg = settings.passes[0].cost;
	const RegularizationConfig& reg_cfg = settings.passes[0].reg;

	int N = static_cast<int>(X.cols());   // Number of timesteps
	int nx = static_cast<int>(X.rows());  // State dimension
	int nu = static_cast<int>(U.rows()); // Control dimension

	// Initialize terminal cost-to-go: p_N and P_N
	Eigen::VectorXd x_final = X.col(N - 1);
	Eigen::Vector3d boresight_final = boresight.col(N - 1);
	Eigen::Vector3d B_final = B.col(N - 1);
	const Eigen::Vector4d attitude_target_final = attitude_target.col(N - 1);

	auto [p_N, p_unused1, p_unused2] = satellite.terminalCostJacobians(x_final, boresight_final, attitude_target_final, B_final, cost_cfg);
	auto [P_N, P_unused1, P_unused2] = satellite.terminalCostHessians(x_final, boresight_final, attitude_target_final, B_final, cost_cfg);

	// Initialize value function at terminal time
	Eigen::VectorXd p_k = p_N;
	Eigen::MatrixXd P_k = P_N;
	
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
		
		// Step 1: Compute stage cost Jacobians and Hessians
		auto [lx, lu_mat, lux_grad] = satellite.stageCostJacobians(k, N, x_k, u_k, boresight_k, attitude_target_k, B_k, cost_cfg);
		auto [lxx, luu, lux_hess] = satellite.stageCostHessians(k, N, x_k, u_k, boresight_k, attitude_target_k, B_k, cost_cfg);
		
		// Reshape lu from 1×nu matrix to nu vector
		Eigen::VectorXd lu = lu_mat.row(0);
		
		// Step 2: Compute dynamics Jacobians
		auto [A_k, B_k_dyn, dyn_unused] = satellite.dynamicsJacobians(x_k, u_k, dist_config, R_k, B_k, S_k, V_k);
		
		// Step 3: Assemble Q matrices
		Eigen::MatrixXd Q_xx = lxx + A_k.transpose() * P_k * A_k;
		Eigen::MatrixXd Q_uu = luu + B_k_dyn.transpose() * P_k * B_k_dyn;
		Eigen::MatrixXd Q_ux = lux_hess + B_k_dyn.transpose() * P_k * A_k;
		Eigen::VectorXd Q_x = lx + A_k.transpose() * p_k;
		Eigen::VectorXd Q_u = lu + B_k_dyn.transpose() * p_k;
		
		// Step 4: Regularization loop - increase rho until Q_uu + rho*I is positive definite
		double rho_reg = reg_cfg.reg_init;
		bool converged = false;
		int reg_tries = 0;
		
		while (rho_reg <= reg_cfg.reg_max) {
			++reg_tries;
			Eigen::MatrixXd Q_uu_reg = Q_uu + rho_reg * Eigen::MatrixXd::Identity(nu, nu);
			Eigen::LLT<Eigen::MatrixXd> llt(Q_uu_reg);
			
			if (llt.info() == Eigen::Success) {
				// Q_uu_reg is positive definite - solve for gains
				solveRiccattiStep(Q_uu_reg, Q_uu, Q_u, Q_ux, Q_xx, Q_x, k, K, d, deltaV, p_k, P_k);
				SALTRO_OPT_DLOG("[BP] k=" << k
					<< " rho=" << rho_reg
					<< " tries=" << reg_tries
					<< " ||Q_u||=" << Q_u.norm()
					<< " ||d||=" << d[k].norm()
					<< " dV1=" << deltaV(0)
					<< " dV2=" << deltaV(1));
				converged = true;
				break;
			}
			// Increase regularization and try again
			rho_reg *= reg_cfg.reg_scale;
		}
		
		if (!converged) {
			// Regularization failed - return false
			SALTRO_OPT_DLOG("[BP] FAIL k=" << k << " reg_init=" << reg_cfg.reg_init << " reg_max=" << reg_cfg.reg_max);
			return false;
		}
	}
	SALTRO_OPT_DLOG("[BP] success dV1=" << deltaV(0) << " dV2=" << deltaV(1));
	
	return true;

} 

} // namespace saltro::optimizer
