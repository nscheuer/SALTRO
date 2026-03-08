#include <saltro/optimizer/backwardpass.h>
#include <saltro/math/integrators/rk4.h>
#include <iostream>
#include <cmath>

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
	
	// CRITICAL FIX: Use Q_uu_reg (regularized) consistently in Riccati updates
	// This ensures P_k remains positive definite even with large regularization
	// Traditional form: P_k = Q_xx - Q_ux^T * (Q_uu + ρI)^{-1} * Q_ux
	// Equivalent form with K_k = -(Q_uu + ρI)^{-1} * Q_ux:
	// P_k = Q_xx + K_k^T * (Q_uu + ρI) * K_k + K_k^T * Q_ux + Q_ux^T * K_k
	P_k = Q_xx 
	  + K_k.transpose() * Q_uu_reg * K_k 
	  + K_k.transpose() * Q_ux 
	  + Q_ux.transpose() * K_k;

	P_k = 0.5 * (P_k + Eigen::MatrixXd(P_k.transpose()));
	
	// Riccati update for value function gradient (also use Q_uu_reg)
	// p_k = Q_x + K_k^T * (Q_uu + ρI) * d_k + K_k^T * Q_u + Q_ux^T * d_k
	p_k = Q_x 
	  + K_k.transpose() * Q_uu_reg * d_k 
	  + K_k.transpose() * Q_u 
	  + Q_ux.transpose() * d_k;
	
	// Accumulate expected cost reduction (use unregularized Q_uu for accurate prediction)
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
	double reg,
	std::vector<Eigen::MatrixXd>& K,
	std::vector<Eigen::VectorXd>& d,
	Eigen::Ref<Eigen::Vector2d> deltaV
) {
	(void)rho;  // Suppress unused parameter warning
	const CostConfig& cost_cfg = settings.passes[0].cost;
	const RegularizationConfig& reg_cfg = settings.passes[0].reg;
	const double dt = (settings.num_passes > 0 && std::isfinite(settings.passes[0].dt) && settings.passes[0].dt > 0.0)
		? settings.passes[0].dt
		: 1.0;

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
		
		// CRITICAL: Scale stage cost derivatives by dt for proper discrete-time formulation
		// Continuous cost: J = ∫[l(x,u)]dt → Discrete: J ≈ Σ(l × dt)
		// This ensures numerical stability across different timestep sizes
		lx = dt * lx;
		lu = dt * lu;
		lxx = dt * lxx;
		luu = dt * luu;
		lux_hess = dt * lux_hess;
		
		// Step 2: Compute exact discrete-time dynamics Jacobians using RK4
		// This matches the forward pass RK4 integration exactly, providing superior
		// accuracy compared to Euler discretization (old approach: A = I + dt*A_c).
		Eigen::MatrixXd A_k = Eigen::MatrixXd::Zero(nx, nx);
		Eigen::MatrixXd B_k_dyn = Eigen::MatrixXd::Zero(nx, nu);
		
		// Lambda wrapper for satellite dynamics Jacobians compatible with rk4_jacobians
		auto dynamics_jac_wrapper = [&](double t_local, const Eigen::Ref<const Eigen::VectorXd>& x_local,
		                                  const Eigen::Ref<const Eigen::VectorXd>& u_local,
		                                  Eigen::Ref<Eigen::MatrixXd> A_c_out,
		                                  Eigen::Ref<Eigen::MatrixXd> B_c_out,
		                                  Eigen::Ref<Eigen::VectorXd> k_out) {
			(void)t_local; // RK4 doesn't need explicit time dependence for our dynamics
			
			// Compute continuous Jacobians
			auto [A_c, B_c, C_unused] = satellite.dynamicsJacobians(x_local, u_local, dist_config, R_k, B_k, S_k, V_k);
			A_c_out = A_c;
			B_c_out = B_c;
			
			// Also compute k = f(x, u) for RK4 stage evaluation
			k_out = satellite.dynamics(x_local, u_local, dist_config, R_k, B_k, S_k, V_k, 0);
		};
		
		// Compute exact RK4 discrete Jacobians
		rk4_jacobians(dynamics_jac_wrapper, x_k, u_k, 0.0, dt, A_k, B_k_dyn);
		
		// Step 3: Assemble Q matrices
		Eigen::MatrixXd Q_xx = lxx + A_k.transpose() * P_k * A_k;
		Eigen::MatrixXd Q_uu = luu + B_k_dyn.transpose() * P_k * B_k_dyn;
		Eigen::MatrixXd Q_ux = lux_hess + B_k_dyn.transpose() * P_k * A_k;
		Eigen::VectorXd Q_x = lx + A_k.transpose() * p_k;
		Eigen::VectorXd Q_u = lu + B_k_dyn.transpose() * p_k;
		
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
