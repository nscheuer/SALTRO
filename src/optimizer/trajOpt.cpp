#include <saltro/optimizer/trajOpt.h>
#include <saltro/optimizer/warm_start.h>
#include <saltro/orbit_generation/generate_orbit.h>
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
	const Eigen::Ref<const Eigen::MatrixXd>& boresight,

	Eigen::Ref<Eigen::MatrixXd> X,
	Eigen::Ref<Eigen::MatrixXd> U,
	Eigen::Ref<Eigen::MatrixXd> K,

	int state_dim,
	int input_dim,
	int N
) {
	std::string error_msg;
	if (!validation::validatetrajOpt(settings, satellite, x0, r0, v0, jtime, q_goal, boresight, state_dim, input_dim, N, error_msg)) {
		throw std::runtime_error("trajOpt input validation failed: " + error_msg);
	}

	Eigen::Matrix<double, 1, saltro::limits::MAX_LENGTH_TRAJ> jtime_fixed = Eigen::Matrix<double, 1, saltro::limits::MAX_LENGTH_TRAJ>::Zero();
	Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ> R;
	Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ> V;
	Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ> B;
	Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ> S;
	Eigen::Matrix<double, 1, saltro::limits::MAX_LENGTH_TRAJ> rho;

	jtime_fixed.leftCols(N) = jtime.transpose();

	const bool orbit_ok = orbits::generate_orbit(r0,v0,jtime_fixed,N,0,0,0,0,0,R,V,B,S,rho);

	if (!orbit_ok) {
		throw std::runtime_error("trajOpt failed to generate orbit");
	}

	const bool warm_start_ok = warm_start(settings, satellite, x0, jtime, q_goal, N, R, V, B, S, rho, X, U);

	if (!warm_start_ok) {
		throw std::runtime_error("trajOpt failed to warm-start trajectory");
	}

	return true;
}

}
