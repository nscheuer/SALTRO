#pragma once

#include <Eigen/Dense>

#include <saltro/pybind/satellite.h>

namespace saltro::controller {

class Controller {
public:
	Controller(const Satellite& satellite);
	virtual ~Controller() = default;
	virtual void set_dt(double dt);

	virtual Satellite::VecX find_u(
		const Satellite::VecX& x,
		const Eigen::Vector3d& B_eci,
		const Eigen::Vector4d& q_goal,
		const Eigen::Vector3d& boresight_body
	) const = 0;

protected:
	virtual void autoTuneGains() = 0;
	double dt_seconds() const;

	const Satellite& satellite_;

	double expected_b_field_leo_ = 35e-6;

private:
	bool dt_initialized_ = false;
	double dt_seconds_ = 0.2;
};

}

