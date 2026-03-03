// Forward pass interface for iLQR. Implementation to be provided.
#pragma once

#include <Eigen/Dense>
#include <vector>

#include <saltro/pybind/plannersettings.h>
#include <saltro/pybind/satellite.h>

namespace saltro::optimizer {

bool forwardPass(
    const Satellite& satellite,
    Eigen::Ref<Eigen::MatrixXd> X,
    Eigen::Ref<Eigen::MatrixXd> U,
    const std::vector<Eigen::MatrixXd>& K,
    const std::vector<Eigen::VectorXd>& d,
    const Eigen::Ref<const Eigen::Vector2d>& deltaV,
    const Eigen::Ref<const Eigen::MatrixXd>& B,
    const Eigen::Ref<const Eigen::MatrixXd>& R,
    const Eigen::Ref<const Eigen::MatrixXd>& V,
    const Eigen::Ref<const Eigen::MatrixXd>& S,
    const Eigen::Ref<const Eigen::MatrixXd>& rho,
    const Eigen::Ref<const Eigen::MatrixXd>& boresight,
    const Eigen::Ref<const Eigen::Vector4d>& attitude_target,
    const PlannerSettings& settings,
    const Eigen::Ref<const Eigen::VectorXd>& jtime,
    double J_prev,
    double& J_new
);

}
