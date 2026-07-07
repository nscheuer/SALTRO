#include <saltro/optimizer/backwardpass.h>
#include <saltro/math/integrators/rk4.h>
#include <saltro/math/mrp.h>
#include <iostream>
#include <cmath>
#include <limits>

#if defined(SALTRO_DEBUG_BUILD)
#define SALTRO_OPT_DLOG(msg) do { std::cout << msg << std::endl; } while (0)
#else
#define SALTRO_OPT_DLOG(msg) do {} while (0)
#endif

namespace saltro::optimizer {

namespace {

/**
 * @brief Upper-triangular R factor of a QR decomposition of M (cols x cols).
 *
 * R satisfies R^T R = M^T M. Used to compress a tall stacked square-root
 * factor back to a small triangular one without ever forming M^T M, which is
 * the operation that squares the condition number.
 */
Eigen::MatrixXd qrFactor(const Eigen::Ref<const Eigen::MatrixXd>& M) {
	const Eigen::Index n = M.cols();
	if (M.rows() < n) {
		Eigen::MatrixXd padded = Eigen::MatrixXd::Zero(n, n);
		padded.topRows(M.rows()) = M;
		Eigen::HouseholderQR<Eigen::MatrixXd> qr_padded(padded);
		Eigen::MatrixXd R = qr_padded.matrixQR().topRows(n).triangularView<Eigen::Upper>();
		return R;
	}
	Eigen::HouseholderQR<Eigen::MatrixXd> qr(M);
	Eigen::MatrixXd R = qr.matrixQR().topRows(n).triangularView<Eigen::Upper>();
	return R;
}

/**
 * @brief Square-root factor F with F^T F = clamp_psd(M).
 *
 * A real square root only exists for PSD matrices, so negative eigenvalues
 * are clamped to zero. For the stage-cost Hessians this also prevents
 * negative curvature (e.g. from non-convex angle costs) from compounding
 * through the Riccati recursion.
 */
Eigen::MatrixXd psdSqrtFactor(const Eigen::Ref<const Eigen::MatrixXd>& M) {
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(0.5 * (M + M.transpose()));
	const Eigen::VectorXd scale = eig.eigenvalues().cwiseMax(0.0).cwiseSqrt();
	return scale.asDiagonal() * eig.eigenvectors().transpose();
}

/**
 * @brief Square-root analog of solveRiccattiStep.
 *
 * Inputs are the state/control column blocks of a square-root factor of the
 * joint action-value Hessian: [F_x F_u]^T [F_x F_u] = [Q_xx Q_xu; Q_ux Q_uu]
 * (unregularized). Gains use the regularized factor Z_uu_reg obtained by
 * stacking sqrt(reg)*I rows under F_u (ALTRO eq. 21); the value function is
 * propagated with the UNREGULARIZED factor, matching the dense backward pass
 * (regularization must not inflate the cost-to-go).
 *
 * The cost-to-go square root follows ALTRO eq. (26):
 *   P_k = [I; K]^T [Q_xx Q_xu; Q_ux Q_uu] [I; K] = (F_x + F_u K)^T (F_x + F_u K)
 * so S_k = qr(F_x + F_u K). Building the joint factor by row-stacking (rather
 * than reconstructing it from Z_xx and Z_uu with Cholesky downdates, eqs.
 * 27-28) avoids inverting Z_xx and the numerically fragile downdate, and is
 * exact for any K, including the regularized one.
 */
bool solveRiccattiStepSqrt(
	const Eigen::Ref<const Eigen::MatrixXd>& F_x,
	const Eigen::Ref<const Eigen::MatrixXd>& F_u,
	const Eigen::Ref<const Eigen::VectorXd>& Q_x,
	const Eigen::Ref<const Eigen::VectorXd>& Q_u,
	double reg,
	int k,
	std::vector<Eigen::MatrixXd>& K,
	std::vector<Eigen::VectorXd>& d,
	Eigen::Ref<Eigen::Vector2d> deltaV,
	Eigen::VectorXd& p_k,
	Eigen::MatrixXd& S_k,
	Eigen::MatrixXd* Z_uu_reg_out = nullptr
) {
	const Eigen::Index nu = F_u.cols();

	// Z_uu_reg = qr([F_u; sqrt(reg) I]): triangular factor of Q_uu + reg*I.
	Eigen::MatrixXd stacked(F_u.rows() + nu, nu);
	stacked.topRows(F_u.rows()) = F_u;
	stacked.bottomRows(nu) = std::sqrt(std::max(reg, 0.0)) * Eigen::MatrixXd::Identity(nu, nu);
	const Eigen::MatrixXd Z_uu_reg = qrFactor(stacked);

	if (!Z_uu_reg.allFinite() || !(Z_uu_reg.diagonal().cwiseAbs().minCoeff() > 0.0)) {
		return false;
	}
	if (Z_uu_reg_out != nullptr) {
		*Z_uu_reg_out = Z_uu_reg;
	}

	const Eigen::MatrixXd Q_ux = F_u.transpose() * F_x;

	// K = -(Q_uu + reg*I)^{-1} Q_ux and d = -(...)^{-1} Q_u via two
	// triangular solves each (ALTRO eqs. 22-23).
	auto regSolve = [&](const Eigen::MatrixXd& rhs) {
		Eigen::MatrixXd y = Z_uu_reg.transpose().triangularView<Eigen::Lower>().solve(rhs);
		return Eigen::MatrixXd(Z_uu_reg.triangularView<Eigen::Upper>().solve(y));
	};
	const Eigen::MatrixXd K_k = -regSolve(Q_ux);
	const Eigen::VectorXd d_k = -regSolve(Q_u);
	K[k] = K_k;
	d[k] = d_k;

	// Q_uu·d through the factor, used only for the expected-reduction term
	// below: Q_uu d = F_u^T (F_u d).
	const Eigen::VectorXd Fu_d = F_u * d_k;

	// Cost-to-go gradient. K_k and d_k are solved against the REGULARIZED
	// Z_uu, so the K^T·Q_uu·d_k + K^T·Q_u cross terms cancel exactly and p_k
	// collapses to the short Jacobson form -- matching the dense pass
	// (backwardpass.cpp). Carrying the long form with the unregularized Q_uu
	// (= F_u^T F_u) instead leaves a residual reg·K^T d_k and breaks
	// dense/sqrt parity.
	p_k = Q_x + Q_ux.transpose() * d_k;

	// Accumulate expected cost reduction (ALTRO eq. 25).
	deltaV(0) += d_k.dot(Q_u);
	deltaV(1) += 0.5 * Fu_d.squaredNorm();

	// Cost-to-go square root (ALTRO eqs. 26/29). PSD and symmetric by
	// construction, so no explicit symmetrization is needed.
	S_k = qrFactor(F_x + F_u * K_k);

	return S_k.allFinite() && p_k.allFinite() && K_k.allFinite() && d_k.allFinite();
}

} // namespace

bool backwardPassSqrt(
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
	std::vector<Eigen::MatrixXd>* K_dist
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

	// Initialize terminal cost-to-go: p_N and S_N with S_N^T S_N = P_N
	// (in reduced state space).
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
	Eigen::MatrixXd S_k = psdSqrtFactor(G_N * P_N_full * G_N.transpose());

	// Fold the terminal-knot (k = N-1) AL constraint terms into the cost-to-go
	// seed, mirroring the dense backward pass: the FP merit and the alilqr
	// multiplier update both price constraints at the terminal knot, so the
	// quadratic model must see them too. In square-root form this is exactly
	// ALTRO eq. (19): the Gauss-Newton penalty enters as rows
	// sqrt(mu_i) * c_x_i stacked under the terminal cost factor. Only state
	// terms exist (the merit evaluates the terminal knot with u = 0).
	// Semantics match the in-loop AL block below and the FP merit: lambda
	// gradient always on (signed c); mu terms when c > 0 OR lambda > 0.
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
			const int nxr_T = static_cast<int>(c_x.cols());

			Eigen::VectorXd w = Eigen::VectorXd::Zero(c_T.size());
			int n_active_T = 0;
			for (int i = 0; i < c_T.size(); ++i) {
				const double li = lambda_aug[kT](i);
				const double ci = c_T(i);
				w(i) = li;
				if (ci > 0.0 || li > 0.0) {
					w(i) += mu_aug[kT](i) * ci;
					if (std::isfinite(mu_aug[kT](i)) && mu_aug[kT](i) > 0.0) {
						++n_active_T;
					}
				}
			}
			p_k.noalias() += c_x.transpose() * w;

			if (n_active_T > 0) {
				Eigen::MatrixXd stacked(S_k.rows() + n_active_T, nxr_T);
				stacked.topRows(S_k.rows()) = S_k;
				int row = static_cast<int>(S_k.rows());
				for (int i = 0; i < c_T.size(); ++i) {
					if (c_T(i) <= 0.0 && lambda_aug[kT](i) <= 0.0) {
						continue;
					}
					const double mu_i = mu_aug[kT](i);
					if (!std::isfinite(mu_i) || mu_i <= 0.0) {
						continue;
					}
					stacked.row(row) = std::sqrt(mu_i) * c_x.row(i);
					++row;
				}
				S_k = qrFactor(stacked);
			}
		}
	}

	// Linearize with the same disturbance config the forward pass rolls out
	// (forwardpass.cpp uses settings.disturbances), mirroring the dense pass.
	const DisturbanceConfig& dist_config = settings.disturbances;

	// Disturbance-aware (eq. 7.40) TVLQR: when K_dist is requested, also build a
	// feedback gain on the disturbance-torque error δτ. We augment the value
	// function with a constant-δτ channel; Sxd is its δx–δτ coupling block
	// (P̃_xτ), seeded at zero at the terminal knot. K_x, p_k and the cost-to-go
	// factor S_k are left untouched (the δτ channel has no control authority
	// and is unpenalized, so the state recursion is unchanged) — identical
	// semantics to the dense pass; only the P·D and Q products are formed
	// through the square-root factors instead of the dense P_k.
	Eigen::MatrixXd Sxd;
	if (K_dist != nullptr) {
		Sxd = Eigen::MatrixXd::Zero(satellite.reducedStateDim(), 3);
	}

	// Backward loop: k from N-2 down to 0
	for (int k = N - 2; k >= 0; --k) {
		// Extract trajectory data at time k
		const Eigen::VectorXd x_k = X.col(k);
		const Eigen::VectorXd u_k = U.col(k);
		const Eigen::Vector3d B_k = B.col(k);
		const Eigen::Vector3d boresight_k = boresight.col(k);
		const Eigen::Vector3d R_k = R.col(k);
		const Eigen::Vector3d V_k = V.col(k);
		const Eigen::Vector3d S_k_sun = S.col(k);
		const Eigen::Vector4d attitude_target_k = attitude_target.col(k);

		// Step 1: Compute stage cost Jacobians and Hessians (full state)
		auto [lx_full, lu_mat, lux_grad] = satellite.stageCostJacobians(k, N, x_k, u_k, boresight_k, attitude_target_k, B_k, cost_cfg);
		auto [lxx_full, luu, lux_hess_full] = satellite.stageCostHessians(k, N, x_k, u_k, boresight_k, attitude_target_k, B_k, cost_cfg);

		// Reshape lu from 1×nu matrix to nu vector
		Eigen::VectorXd lu = lu_mat.row(0);

		// NB: stage cost derivatives are NOT dt-scaled here -- the dense pass
		// (backwardpass.cpp) dropped that scaling, so the sqrt pass must match
		// to keep dense/sqrt parity.

		// Step 2: Build G matrices for projection
		Eigen::Vector4d q_k = x_k.segment<4>(3);
		Eigen::Vector4d q_kp1 = X.col(k + 1).segment<4>(3);
		Eigen::MatrixXd G_k = saltro::math::findGMat(q_k, nRW);
		Eigen::MatrixXd G_kp1 = saltro::math::findGMat(q_kp1, nRW);

		// Project stage cost to reduced state
		Eigen::VectorXd lx = G_k * lx_full;
		Eigen::MatrixXd lxx = G_k * lxx_full * G_k.transpose();
		Eigen::MatrixXd lux_hess = lux_hess_full * G_k.transpose();

		const int nxr = static_cast<int>(lxx.rows());

		// Augmented Lagrangian terms. Gradient contributions are identical to
		// the dense backward pass; the Gauss-Newton penalty Hessian is kept in
		// square-root form as rows sqrt(mu_i) * [c_x_i c_u_i] (ALTRO eqs.
		// 19-21) instead of forming mu * c^T c outer products, so large
		// penalties only enter the recursion through their square root.
		// ALTRO active-set semantics (matching the dense pass, FP merit, and
		// alilqr lambda update): the lambda gradient is always on (signed c,
		// so feasibility is rewarded and lambda can pull back); the mu
		// penalty is active when c > 0 OR lambda > 0.
		Eigen::MatrixXd Cx_rows(0, nxr);
		Eigen::MatrixXd Cu_rows(0, nu);
		if (!lambda_aug.empty() && !mu_aug.empty() && k < static_cast<int>(lambda_aug.size()) && k < static_cast<int>(mu_aug.size())) {
			const Eigen::VectorXd c_k = satellite.constraints(k, N, x_k, u_k, S_k_sun, cnst_cfg);
			auto [c_u, c_x_full] = satellite.constraintJacobians(k, N, x_k, u_k, S_k_sun, cnst_cfg);
			if (lambda_aug[k].size() == c_k.size() && mu_aug[k].size() == c_k.size()) {
				const Eigen::MatrixXd c_x = c_x_full * G_k.transpose();

				Eigen::VectorXd w = Eigen::VectorXd::Zero(c_k.size());
				int n_active = 0;
				for (int i = 0; i < c_k.size(); ++i) {
					const double li = lambda_aug[k](i);
					const double ci = c_k(i);
					w(i) = li;
					if (ci > 0.0 || li > 0.0) {
						w(i) += mu_aug[k](i) * ci;
						const double mu_i = mu_aug[k](i);
						if (std::isfinite(mu_i) && mu_i > 0.0) {
							++n_active;
						}
					}
				}

				lx.noalias() += c_x.transpose() * w;
				lu.noalias() += c_u.transpose() * w;

				Cx_rows.resize(n_active, nxr);
				Cu_rows.resize(n_active, nu);
				int row = 0;
				for (int i = 0; i < c_k.size(); ++i) {
					if (c_k(i) <= 0.0 && lambda_aug[k](i) <= 0.0) {
						continue;
					}
					const double mu_i = mu_aug[k](i);
					if (!std::isfinite(mu_i) || mu_i <= 0.0) {
						continue;
					}
					const double sqrt_mu = std::sqrt(mu_i);
					Cx_rows.row(row) = sqrt_mu * c_x.row(i);
					Cu_rows.row(row) = sqrt_mu * c_u.row(i);
					++row;
				}
			}
		}
		const int n_active = static_cast<int>(Cx_rows.rows());

		// Step 3: Compute exact discrete-time dynamics Jacobians using RK4 (full state)
		Eigen::MatrixXd A_k_full = Eigen::MatrixXd::Zero(nx, nx);
		Eigen::MatrixXd B_k_dyn_full = Eigen::MatrixXd::Zero(nx, nu);

		auto dynamics_jac_wrapper = [&](double t_local, const Eigen::Ref<const Eigen::VectorXd>& x_local,
		                                  const Eigen::Ref<const Eigen::VectorXd>& u_local,
		                                  Eigen::Ref<Eigen::MatrixXd> A_c_out,
		                                  Eigen::Ref<Eigen::MatrixXd> B_c_out,
		                                  Eigen::Ref<Eigen::VectorXd> k_out) {
			(void)t_local;
			auto [A_c, B_c, C_unused] = satellite.dynamicsJacobians(x_local, u_local, dist_config, R_k, B_k, S_k_sun, V_k);
			A_c_out = A_c;
			B_c_out = B_c;
			k_out = satellite.dynamics(x_local, u_local, dist_config, R_k, B_k, S_k_sun, V_k, 0);
		};

		rk4_jacobians(dynamics_jac_wrapper, x_k, u_k, 0.0, dt, A_k_full, B_k_dyn_full);

		// Step 4: Project dynamics Jacobians to reduced state space
		Eigen::MatrixXd A_k = G_kp1 * A_k_full * G_k.transpose();
		Eigen::MatrixXd B_k_dyn = G_kp1 * B_k_dyn_full;

		// Step 5: Assemble a square-root factor [F_x F_u] of the joint
		// action-value Hessian [Q_xx Q_xu; Q_ux Q_uu] by row-stacking:
		//   - a PSD square root of the joint stage-cost Hessian
		//   - S_{k+1} [A_k B_k]   (the dynamics term [A B]^T P_{k+1} [A B])
		//   - sqrt(mu_i) [c_x_i c_u_i] for each active AL constraint
		Eigen::MatrixXd L_joint(nxr + nu, nxr + nu);
		L_joint.topLeftCorner(nxr, nxr) = lxx;
		L_joint.topRightCorner(nxr, nu) = lux_hess.transpose();
		L_joint.bottomLeftCorner(nu, nxr) = lux_hess;
		L_joint.bottomRightCorner(nu, nu) = luu;
		const Eigen::MatrixXd J_stage = psdSqrtFactor(L_joint);

		const Eigen::Index m = (nxr + nu) + nxr + n_active;
		Eigen::MatrixXd F_x(m, nxr);
		Eigen::MatrixXd F_u(m, nu);
		F_x.topRows(nxr + nu) = J_stage.leftCols(nxr);
		F_u.topRows(nxr + nu) = J_stage.rightCols(nu);
		F_x.middleRows(nxr + nu, nxr).noalias() = S_k * A_k;
		F_u.middleRows(nxr + nu, nxr).noalias() = S_k * B_k_dyn;
		F_x.bottomRows(n_active) = Cx_rows;
		F_u.bottomRows(n_active) = Cu_rows;

		// Gradient terms are vectors and need no square-root treatment.
		Eigen::VectorXd Q_x = lx + A_k.transpose() * p_k;
		Eigen::VectorXd Q_u = lu + B_k_dyn.transpose() * p_k;

		// Capture the incoming cost-to-go factor S_{k+1} before
		// solveRiccattiStepSqrt overwrites S_k with the step-k factor (needed
		// for the δτ recursion: P_{k+1} = S_{k+1}^T S_{k+1}).
		Eigen::MatrixXd S_kp1_dist;
		Eigen::MatrixXd Z_uu_reg_dist;
		if (K_dist != nullptr) {
			S_kp1_dist = S_k;
		}

		if (!solveRiccattiStepSqrt(F_x, F_u, Q_x, Q_u, reg, k, K, d, deltaV, p_k, S_k,
		                           K_dist != nullptr ? &Z_uu_reg_dist : nullptr)) {
			SALTRO_OPT_DLOG("[BP-SQRT] FAIL k=" << k << " reg=" << reg);
			return false;
		}

		if (K_dist != nullptr) {
			// RK4-exact discrete disturbance Jacobian D = ∂x_{k+1}/∂τ, reduced.
			// Identical construction to the dense pass (backwardpass.cpp): the
			// (body-frame) disturbance torque enters the dynamics exactly like
			// an additive control torque, so append 3 dummy "disturbance
			// controls" whose B-column is jac_dist and reuse rk4_jacobians. (A
			// first-order dt·G·jac_dist form omits the attitude coupling, which
			// over a full step is large.)
			const int nu_aug = nu + 3;
			Eigen::VectorXd u_aug = Eigen::VectorXd::Zero(nu_aug);
			u_aug.head(nu) = u_k;
			auto dist_jac_wrapper = [&](double /*t*/,
			                            const Eigen::Ref<const Eigen::VectorXd>& x_local,
			                            const Eigen::Ref<const Eigen::VectorXd>& u_local,
			                            Eigen::Ref<Eigen::MatrixXd> A_c_out,
			                            Eigen::Ref<Eigen::MatrixXd> B_c_out,
			                            Eigen::Ref<Eigen::VectorXd> k_out) {
				const Eigen::VectorXd u_real = u_local.head(nu);
				auto [A_c, B_c, jac_dist_c] =
					satellite.dynamicsJacobians(x_local, u_real, dist_config, R_k, B_k, S_k_sun, V_k);
				A_c_out = A_c;
				B_c_out.leftCols(nu) = B_c;
				B_c_out.rightCols(3) = jac_dist_c;
				k_out = satellite.dynamics(x_local, u_real, dist_config, R_k, B_k, S_k_sun, V_k, 0);
			};
			Eigen::MatrixXd A_aug_full = Eigen::MatrixXd::Zero(nx, nx);
			Eigen::MatrixXd BD_full = Eigen::MatrixXd::Zero(nx, nu_aug);
			rk4_jacobians(dist_jac_wrapper, x_k, u_aug, 0.0, dt, A_aug_full, BD_full);
			const Eigen::MatrixXd D_k = G_kp1 * BD_full.rightCols(3);   // (nxr × 3)

			// Augmented Riccati on [δx; δτ] with δτ_{k+1}=δτ — same recursion
			// as the dense pass, but every product goes through a factor:
			//   P_{k+1} D = S_{k+1}^T (S_{k+1} D)
			//   K_τ = -(Q_uu+ρI)^{-1} Bᵀ(P D + Sxd)  via the Z_uu_reg
			//         triangular solves (same solve path as K_x)
			//   Kᵀ Q_uu K_τ = (F_u K)ᵀ (F_u K_τ),  Q_uxᵀ K_τ = F_xᵀ (F_u K_τ)
			const Eigen::MatrixXd PD_S =
				S_kp1_dist.transpose() * (S_kp1_dist * D_k) + Sxd;      // (nxr × 3)
			const Eigen::MatrixXd Q_utau = B_k_dyn.transpose() * PD_S;  // (nu  × 3)
			Eigen::MatrixXd y_tau =
				Z_uu_reg_dist.transpose().triangularView<Eigen::Lower>().solve(Q_utau);
			const Eigen::MatrixXd K_tau =
				-Z_uu_reg_dist.triangularView<Eigen::Upper>().solve(y_tau);
			if (!K_tau.allFinite()) {
				SALTRO_OPT_DLOG("[BP-SQRT] FAIL k=" << k << " non-finite disturbance gain");
				return false;
			}
			(*K_dist)[static_cast<std::size_t>(k)] = K_tau;
			const Eigen::MatrixXd Fu_Ktau = F_u * K_tau;
			Sxd = A_k.transpose() * PD_S
				+ (F_u * K[k]).transpose() * Fu_Ktau
				+ K[k].transpose() * Q_utau
				+ F_x.transpose() * Fu_Ktau;
		}
		SALTRO_OPT_DLOG("[BP-SQRT] k=" << k << " reg=" << reg << " ||d||=" << d[k].norm());
	}
	SALTRO_OPT_DLOG("[BP-SQRT] success dV1=" << deltaV(0) << " dV2=" << deltaV(1));

	return true;
}

} // namespace saltro::optimizer
