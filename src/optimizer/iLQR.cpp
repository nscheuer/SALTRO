#include <saltro/optimizer/iLQR.h>

namespace saltro::optimizer {

bool iLQR(
	const PlannerSettings& settings,
	const Satellite& satellite,
	Eigen::Ref<Eigen::MatrixXd> X,
	Eigen::Ref<Eigen::MatrixXd> U,
    Eigen::Ref<Eigen::VectorXd> B,
	Eigen::Ref<Eigen::VectorXd> J
) {
	// TODO: Implement iLQR algorithm
	return false;
}

} 
