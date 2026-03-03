#pragma once

#include <Eigen/Dense>

#include <saltro/limits.h>
#include <saltro/pybind/satellite.h>

namespace saltro::optimizer {

bool warm_start(
    const PlannerSettings& settings,
    const Satellite& satellite,
    const Satellite::VecX& x0,
    const Eigen::Ref<const Eigen::VectorXd>& jtime,
    const Eigen::Ref<const Eigen::MatrixXd>& q_goal,
    const Eigen::Ref<const Eigen::MatrixXd>& boresight,
    int N,
    const Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& R,
    const Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& V,
    const Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& B,
    const Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& S,
    const Eigen::Matrix<double, 1, saltro::limits::MAX_LENGTH_TRAJ>& rho,
    Eigen::Ref<Eigen::MatrixXd> X,
    Eigen::Ref<Eigen::MatrixXd> U
);

}
