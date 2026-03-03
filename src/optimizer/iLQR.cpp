#include <saltro/optimizer/iLQR.h>
#include <saltro/optimizer/backwardpass.h>

#include <cmath>

namespace {

void forwardPassStub(
	Eigen::Ref<Eigen::MatrixXd> X,
	Eigen::Ref<Eigen::MatrixXd> U,
	const Eigen::Ref<const Eigen::MatrixXd>& K,
	const Eigen::Ref<const Eigen::MatrixXd>& d,
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
	Eigen::MatrixXd K = Eigen::MatrixXd::Zero(U.rows(), X.cols());
	Eigen::MatrixXd d = Eigen::MatrixXd::Zero(U.rows(), U.cols());
	Eigen::Vector2d deltaV = Eigen::Vector2d::Zero();

	while (std::abs(J_prev - J) > ilqr_cfg.cost_tol && iter < ilqr_cfg.max_iters) {
		J_prev = J;
		backwardPass(satellite, X, U, R, V, B, S, rho, boresight, attitude_target, cost_cfg, K, d, deltaV);

		forwardPassStub(X, U, K, d, deltaV, J_prev, J);
		++iter;
	}

	return true;
}

} 
