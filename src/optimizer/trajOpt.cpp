#include <saltro/optimizer/trajOpt.h>

#include <cmath>

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
) {
	(void)settings;

	X.setZero();
	U.setZero();
	K.setZero();

	if (jtime_length <= 0 || jtime_length > saltro::limits::MAX_LENGTH_TRAJ) {
		return false;
	}

	if (!r0.allFinite() || !v0.allFinite()) {
		return false;
	}

	if (x0.size() != satellite.stateDim() || !x0.allFinite()) {
		return false;
	}

	for (int i = 0; i < jtime_length; ++i) {
		if (!std::isfinite(jtime(i))) {
			return false;
		}
		if (!q_goal.col(i).allFinite()) {
			return false;
		}
		if (q_goal.col(i).norm() <= 1e-12) {
			return false;
		}
	}

	for (int i = 1; i < jtime_length; ++i) {
		if (jtime(i) <= jtime(i - 1)) {
			return false;
		}
	}

	const int nx = satellite.stateDim();
	for (int k = 0; k < jtime_length; ++k) {
		X.topRows(nx).col(k) = x0;
		if (nx >= Satellite::QUAT_INDEX + 4) {
			const Eigen::Vector4d qk = q_goal.col(k).normalized();
			X.block<4, 1>(Satellite::QUAT_INDEX, k) = qk;
		}
	}

	return true;
}

}
