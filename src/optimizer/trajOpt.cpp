#include <saltro/optimizer/trajOpt.h>
#include <saltro/validation/validate_trajOpt.h>

namespace saltro::optimizer {

bool trajOpt(
	const PlannerSettings& settings,
	const Satellite& satellite,
	const Satellite::VecX& x0,
	const Eigen::Vector3d& r0,
	const Eigen::Vector3d& v0,
	const Eigen::Ref<const Eigen::VectorXd>& jtime,
	const Eigen::Ref<const Eigen::MatrixXd>& q_goal,

	Eigen::Ref<Eigen::MatrixXd> X,
	Eigen::Ref<Eigen::MatrixXd> U,
	Eigen::Ref<Eigen::MatrixXd> K,

	int state_dim,
	int input_dim,
	int N
) {
	std::string error_msg;
	if (!validation::validatetrajOpt(settings, satellite, x0, r0, v0, jtime, q_goal, state_dim, input_dim, N, error_msg)) {
		throw std::runtime_error("trajOpt input validation failed: " + error_msg);
	}
	return true;
}

}
