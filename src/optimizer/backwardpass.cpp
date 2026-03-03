#include <saltro/optimizer/backwardpass.h>

namespace saltro::optimizer {

void backwardPass(
	const Satellite& satellite,
	const Eigen::Ref<const Eigen::MatrixXd>& X,
	const Eigen::Ref<const Eigen::MatrixXd>& U,
	const Eigen::Ref<const Eigen::MatrixXd>& R,
	const Eigen::Ref<const Eigen::MatrixXd>& V,
	const Eigen::Ref<const Eigen::MatrixXd>& B,
	const Eigen::Ref<const Eigen::MatrixXd>& S,
	const Eigen::Ref<const Eigen::MatrixXd>& rho,
	const Eigen::Ref<const Eigen::MatrixXd>& boresight,
	const Eigen::Ref<const Eigen::Vector4d>& attitude_target,
	const CostConfig& cost_cfg,
	Eigen::Ref<Eigen::MatrixXd> K,
	Eigen::Ref<Eigen::MatrixXd> d,
	Eigen::Ref<Eigen::Vector2d> deltaV
) {
	const int N = X.rows();
	const int nx = satellite.stateDim();
	const int nu = satellite.controlDim();

	// Initialize terminal cost-to-go: p_N and P_N
	Eigen::VectorXd x_final = X.row(N - 1);
	Eigen::Vector3d boresight_final = boresight.col(N - 1);
	Eigen::Vector3d B_final = B.col(N - 1);

	auto [p_N, _, __] = satellite.terminalCostJacobians(x_final, boresight_final, attitude_target, B_final, cost_cfg);
	auto [P_N, ___, ____] = satellite.terminalCostHessians(x_final, boresight_final, attitude_target, B_final, cost_cfg);

	// Initialize value function at terminal time
	Eigen::VectorXd p_k = p_N;
	Eigen::MatrixXd P_k = P_N;
	
	// Minimal disturbance config for linearization
	DisturbanceConfig dist_config;
	
	// Backward loop: k from N-2 down to 0
	for (int k = N - 2; k >= 0; --k) {
		// Extract trajectory data at time k
		const Eigen::VectorXd x_k = X.row(k);
		const Eigen::VectorXd u_k = U.row(k);
		const Eigen::Vector3d B_k = B.col(k);
		const Eigen::Vector3d boresight_k = boresight.col(k);
		const Eigen::Vector3d R_k = R.col(k);
		const Eigen::Vector3d V_k = V.col(k);
		const Eigen::Vector3d S_k = S.col(k);
		const int rho_k = static_cast<int>(rho(0, k));
		
		// Step 1: Compute stage cost Jacobians and Hessians
		auto [lx, lu_mat, lux_grad] = satellite.stageCostJacobians(k, N, x_k, u_k, boresight_k, attitude_target, B_k, cost_cfg);
		auto [lxx, luu, lux_hess] = satellite.stageCostHessians(k, N, x_k, u_k, boresight_k, attitude_target, B_k, cost_cfg);
		
		// Reshape lu from 1×nu matrix to nu vector
		Eigen::VectorXd lu = lu_mat.row(0);
		
		// Step 2: Compute dynamics Jacobians
		auto [A_k, B_k_dyn, _] = satellite.dynamicsJacobians(x_k, u_k, dist_config, R_k, B_k, S_k, V_k);
		
		// Step 3: Assemble Q matrices
		Eigen::MatrixXd Q_xx = lxx + A_k.transpose() * P_k * A_k;
		Eigen::MatrixXd Q_uu = luu + B_k_dyn.transpose() * P_k * B_k_dyn;
		Eigen::MatrixXd Q_ux = lux_hess + B_k_dyn.transpose() * P_k * A_k;
		Eigen::VectorXd Q_x = lx + A_k.transpose() * p_k;
		Eigen::VectorXd Q_u = lu + B_k_dyn.transpose() * p_k;
		
		// TODO: Check convexity of Q_uu
		// TODO: Compute K_k and d_k via linear solve
		// TODO: Update P_k and p_k using Riccati
		// TODO: Accumulate deltaV
		(void)Q_xx; (void)Q_uu; (void)Q_ux; (void)Q_x; (void)Q_u; (void)rho_k;
	}
	
	K.setZero();
	d.setZero();
	deltaV.setZero();
}

} // namespace saltro::optimizer
