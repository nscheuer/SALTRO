#include <saltro/optimizer/iLQR.h>
#include <saltro/optimizer/backwardpass.h>

#include <cmath>
#include <vector>

namespace {

void forwardPassStub(
	Eigen::Ref<Eigen::MatrixXd> X,
	Eigen::Ref<Eigen::MatrixXd> U,
	const std::vector<Eigen::MatrixXd>& K,
	const std::vector<Eigen::VectorXd>& d,
	const Eigen::Ref<const Eigen::Vector2d>& deltaV,
	double J_prev,
	double& J_new
) {
	(void)X;
	(void)U;
	(void)K;
	(void)d;
	(void)deltaV;
	J_new = J_prev;
}

}

namespace saltro::optimizer {

bool iLQR(
	const PlannerSettings& settings,
	const Satellite& satellite,
	Eigen::Ref<Eigen::MatrixXd> X,
	Eigen::Ref<Eigen::MatrixXd> U,
	const Eigen::Ref<const Eigen::MatrixXd>& R,
	const Eigen::Ref<const Eigen::MatrixXd>& V,
	Eigen::Ref<Eigen::MatrixXd> B,
	const Eigen::Ref<const Eigen::MatrixXd>& S,
	const Eigen::Ref<const Eigen::MatrixXd>& rho,
	const Eigen::Ref<const Eigen::MatrixXd>& boresight,
	const Eigen::Ref<const Eigen::Vector4d>& attitude_target,
	double& J
) {
	const CostConfig& cost_cfg = settings.passes[0].cost;
	const ILQRConfig& ilqr_cfg = settings.passes[0].ilqr;
	J = satellite.totalCost(X, U, B, boresight, attitude_target, cost_cfg);

	double J_prev = J + 2.0 * ilqr_cfg.cost_tol;
	int iter = 0;
	
	// Preallocate gain and feedforward term vectors
	int N = X.cols();   // Number of timesteps
	int nx = X.rows();  // State dimension
	int nu = U.rows();  // Control dimension
	std::vector<Eigen::MatrixXd> K(N - 1);
	std::vector<Eigen::VectorXd> d(N - 1);
	for (int k = 0; k < N - 1; ++k) {
		K[k] = Eigen::MatrixXd::Zero(nu, nx);
		d[k] = Eigen::VectorXd::Zero(nu);
	}
	Eigen::Vector2d deltaV = Eigen::Vector2d::Zero();

	while (std::abs(J_prev - J) > ilqr_cfg.cost_tol && iter < ilqr_cfg.max_iters) {
		J_prev = J;
		bool bp_success = backwardPass(satellite, X, U, R, V, B, S, rho, boresight, attitude_target, settings, K, d, deltaV);
		
		if (!bp_success) {
			// Backward pass numerical failure
			return false;
		}

		forwardPassStub(X, U, K, d, deltaV, J_prev, J);
		++iter;
	}

	return true;
}

} 
