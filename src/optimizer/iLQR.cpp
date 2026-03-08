#include <saltro/optimizer/iLQR.h>
#include <saltro/optimizer/backwardpass.h>
#include <saltro/optimizer/forwardpass.h>

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
	double& J
) {
	const CostConfig& cost_cfg = settings.passes[0].cost;
	const ILQRConfig& ilqr_cfg = settings.passes[0].ilqr;
	const RegularizationConfig& reg_cfg = settings.passes[0].reg;
	
	// Preallocate gain and feedforward term vectors
	int N = X.cols();   // Number of timesteps
	int nx = X.rows();  // Full state dimension
	int nu = U.rows();  // Control dimension
	int nxr = satellite.reducedStateDim(); // Reduced state dimension (6 + nRW)
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
				boresight, attitude_target, settings, reg,
				K, d, deltaV
			);
			
			if (!bp_success) {
				reg *= reg_cfg.reg_scale;
				continue;
			}

			double J_prev = satellite.totalCost(X, U, B, boresight, attitude_target, cost_cfg);
			
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
				settings,
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
				return true;  // Converged
			}
			
			break;  // Exit regularization loop, continue to next iteration
		}
		
		// Check if regularization exceeded maximum
		if (reg > reg_cfg.reg_max) {
			return false;  // Regularization exceeded
		}
	}
	
	// Max iterations reached
	return false;
}

} // namespace saltro::optimizer