#pragma once

#include <Eigen/Dense>
#include <saltro/limits.h>
#include <string>

namespace saltro::optimizer::validation {

bool validateQGoal(
    const Eigen::Matrix<double, 4, saltro::limits::MAX_LENGTH_TRAJ>& q_goal,
    int jtime_length,
    std::string& error_msg
);

}
