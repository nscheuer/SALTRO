#pragma once

#include <Eigen/Dense>
#include <string>

namespace saltro::optimizer::validation {

bool validateOrbitInitialConditions(const Eigen::Vector3d& r0, const Eigen::Vector3d& v0, std::string& error_msg);

}
