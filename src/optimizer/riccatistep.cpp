#include <saltro/optimizer/riccatistep.h>

namespace saltro::optimizer {

void solveRiccattiStep(
	const Eigen::Ref<const Eigen::MatrixXd>& Q_uu_reg,
	const Eigen::Ref<const Eigen::VectorXd>& Q_u,
	const Eigen::Ref<const Eigen::MatrixXd>& Q_ux,
	const Eigen::Ref<const Eigen::MatrixXd>& Q_xx,
	const Eigen::Ref<const Eigen::VectorXd>& Q_x,
	int k,
	Eigen::Ref<Eigen::MatrixXd> K,
	Eigen::Ref<Eigen::MatrixXd> d,
	Eigen::Ref<Eigen::Vector2d> deltaV,
	Eigen::Ref<Eigen::VectorXd> p_k,
	Eigen::Ref<Eigen::MatrixXd> P_k
) {
	const int nx = Q_xx.rows();
	const int nu = Q_uu_reg.rows();
	
	// Compute Q_uu_reg^{-1} using LLT (already factored, but recompute for safety)
	Eigen::LLT<Eigen::MatrixXd> llt(Q_uu_reg);
	
	// Step 1: Solve for feedback gain K_k = -Q_uu_reg^{-1} * Q_ux
	Eigen::MatrixXd K_k = -llt.solve(Q_ux);
	K.middleCols(k, 1) = K_k;
	
	// Step 2: Solve for feedforward term d_k = -Q_uu_reg^{-1} * Q_u
	Eigen::VectorXd d_k = -llt.solve(Q_u);
	d.col(k) = d_k;
	
	// Step 3: Riccati update for value function Hessian
	// P_k = Q_xx - Q_ux^T * Q_uu_reg^{-1} * Q_ux
	Eigen::MatrixXd Q_uu_inv_Qux = llt.solve(Q_ux);
	P_k = Q_xx - Q_ux.transpose() * Q_uu_inv_Qux;
	
	// Step 4: Riccati update for value function gradient
	// p_k = Q_x - Q_ux^T * Q_uu_reg^{-1} * Q_u
	Eigen::VectorXd Q_uu_inv_Qu = llt.solve(Q_u);
	p_k = Q_x - Q_ux.transpose() * Q_uu_inv_Qu;
	
	// Step 5: Accumulate expected cost reduction
	// ΔV_1 = sum_k (Q_u_k · d_k)
	// ΔV_2 = sum_k (0.5 * d_k^T * Q_uu * d_k)
	deltaV(0) += Q_u.dot(d_k);
	deltaV(1) += 0.5 * d_k.dot(Q_uu_reg * d_k);
}

} // namespace saltro::optimizer
