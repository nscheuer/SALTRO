#include <saltro/optimizer/iLQR.h>
#include <saltro/optimizer/backwardpass.h>
#include <saltro/optimizer/forwardpass.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace saltro::optimizer {

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
		
		// Regularization retry loop
		while (reg <= reg_cfg.reg_max) {
			deltaV.setZero();
			
			bool bp_success = backwardPass(
				satellite, X, U, R, V, B, S, rho, 
				boresight, attitude_target, pass_settings, reg,
				K, d, deltaV, lambda_aug, mu_aug
			);
			
			if (!bp_success) {
				reg *= reg_cfg.reg_scale;
				continue;
			}

			const int N_u = std::max(0, N - 1);
			double J_prev = satellite.totalCost(X, U.leftCols(N_u), B, boresight, attitude_target, cost_cfg);
			
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