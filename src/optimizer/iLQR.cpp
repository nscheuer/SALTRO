#include <saltro/optimizer/iLQR.h>
#include <saltro/optimizer/backwardpass.h>
#include <saltro/optimizer/forwardpass.h>
#include <saltro/optimizer/spike_removal.h>

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
	ILQRStatus& status,
	double& J
) {
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

	// Main iLQR iteration loop
	for (int iteration = 0; iteration < ilqr_cfg.max_iters; ++iteration) {
		// Reset regularization at the start of each iteration
		double reg = reg_cfg.reg_init;
		const int N_u = std::max(0, N - 1);
		
		// Regularization retry loop
		while (reg <= reg_cfg.reg_max) {
			deltaV.setZero();
			
			bool bp_success = backwardPass(
				satellite, X, U.leftCols(N_u), R, V, B, S, rho, 
				boresight, attitude_target, pass_settings, reg,
				K, d, deltaV, lambda_aug, mu_aug
			);
			
			if (!bp_success) {
				reg *= reg_cfg.reg_scale;
				continue;
			}

			double J_prev = satellite.totalCost(X, U.leftCols(N_u), B, boresight, attitude_target, cost_cfg);
			J_prev += augmented_penalty_total(satellite, cnst_cfg, X, U, S, lambda_aug, mu_aug);

			// Save nominal controls before forward pass (needed for spike removal blend)
			const Eigen::MatrixXd U_bar = U;

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
				reg *= reg_cfg.reg_scale;
				continue;
			}

			// Spike removal: detect and replace homotopy artifacts after accepted step
			const auto& spike_cfg = settings.passes[pass_idx].spike_removal;
			if (spike_cfg.enabled) {
				applySpikeRemoval(
					satellite, X, U, U_bar, K,
					settings, pass_idx,
					R, V, B, S, rho, jtime, boresight, attitude_target,
					iteration, spike_cfg
				);
			}

			// Both passes succeeded
			double delta_J = std::abs(J_prev - J);
			if (delta_J <= ilqr_cfg.cost_tol) {
				status = ILQRStatus::Converged;
				return true;
			}
			
			break;  // Exit regularization loop, continue to next iteration
		}
		
		// Check if regularization exceeded maximum
		if (reg > reg_cfg.reg_max) {
			status = ILQRStatus::RegularizationExceeded;
			return false;
		}
	}
	
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