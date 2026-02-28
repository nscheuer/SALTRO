#pragma once

#include <Eigen/Dense>

#include <saltro/limits.h>
#include <saltro/math/tensor.h>
#include <saltro/pybind/plannersettings.h>
#include <saltro/pybind/satellite.h>

namespace saltro::optimizer {

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
);

}
