#include <saltro/optimizer/trajOpt.h>
#include <saltro/validation/validate_plannersettings.h>
#include <saltro/validation/validate_satellite.h>

#include <cmath>

namespace saltro::optimizer {

// Static workspace allocated in data segment at program initialization
static Eigen::Matrix<double, saltro::limits::MAX_STATE_DIM, saltro::limits::MAX_LENGTH_TRAJ> X_static;
static Eigen::Matrix<double, saltro::limits::MAX_CTRL_DIM, saltro::limits::MAX_LENGTH_TRAJ> U_static;
static saltro::math::Tensor3<saltro::limits::MAX_CTRL_DIM, saltro::limits::MAX_STATE_DIM, saltro::limits::MAX_LENGTH_TRAJ> K_static;

bool trajOpt(
	const PlannerSettings& settings,
	const Satellite& satellite,
	const Satellite::VecX& x0,
	const Eigen::Vector3d& r0,
	const Eigen::Vector3d& v0,
	const Eigen::Matrix<double, 1, saltro::limits::MAX_LENGTH_TRAJ>& jtime,
	const Eigen::Matrix<double, 4, saltro::limits::MAX_LENGTH_TRAJ>& q_goal,
	int jtime_length,
	Eigen::Matrix<double, saltro::limits::MAX_STATE_DIM, saltro::limits::MAX_LENGTH_TRAJ>& X,
	Eigen::Matrix<double, saltro::limits::MAX_CTRL_DIM, saltro::limits::MAX_LENGTH_TRAJ>& U,
	saltro::math::Tensor3<saltro::limits::MAX_CTRL_DIM,
	                     saltro::limits::MAX_STATE_DIM,
	                     saltro::limits::MAX_LENGTH_TRAJ>& K
) {
	(void)settings;

	X_static.setZero();
	U_static.setZero();
	K_static.setZero();

	std::string error_msg;
	
	if (!validation::validatePlannerSettings(settings, error_msg)) {
		throw std::invalid_argument("Invalid PlannerSettings: " + error_msg);
	}
	
	if (!validation::validateSatellite(satellite, error_msg)) {
		throw std::invalid_argument("Invalid Satellite configuration: " + error_msg);
	}

	// Copy static storage to output references
	X = X_static;
	U = U_static;
	K = K_static;

	return true;
}

}
