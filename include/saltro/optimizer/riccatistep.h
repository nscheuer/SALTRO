#pragma once

#include <Eigen/Dense>

namespace saltro::optimizer {

/**
 * @brief Solve a single Riccati step in the backward pass.
 *
 * Given the regularized control cost Hessian Q_uu and related Q-matrices,
 * computes the feedback gain K and feedforward term d, then propagates
 * the value function (P, p) backward one timestep using Riccati equations.
 *
 * @param Q_uu_reg Regularized control cost Hessian (nu x nu, positive definite)
 * @param Q_u Control cost gradient (nu,)
 * @param Q_ux Cross-derivative of control and state cost (nu x nx)
 * @param Q_xx State cost Hessian (nx x nx)
 * @param Q_x State cost gradient (nx,)
 * @param k Current timestep index
 * @param K Output feedback gains (control_dim x state_dim x N-1)
 * @param d Output feedforward terms (control_dim x N-1)
 * @param deltaV Output expected cost reduction (2x1)
 * @param p_k Input/Output value function gradient (will be updated)
 * @param P_k Input/Output value function Hessian (will be updated)
 */
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
);

} // namespace saltro::optimizer
