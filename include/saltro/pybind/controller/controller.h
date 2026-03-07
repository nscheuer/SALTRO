#pragma once

#include <Eigen/Dense>

#include <saltro/pybind/satellite.h>

namespace saltro::controller {

class Controller {
public:
	Controller(const Satellite& satellite);
	virtual ~Controller() = default;

	virtual Satellite::VecX find_u(
		const Satellite::VecX& x,
		const Eigen::Vector3d& B_eci,
		const Eigen::Vector4d& q_goal,
		const Eigen::Vector3d& boresight_body
	) const = 0;

protected:
	virtual void autoTuneGains() = 0;

	const Satellite& satellite_;

	double expected_b_field_leo_ = 35e-6;
};

}

