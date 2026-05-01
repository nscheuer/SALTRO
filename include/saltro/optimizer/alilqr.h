#pragma once

#include <Eigen/Dense>

#include <saltro/pybind/plannersettings.h>
#include <saltro/pybind/satellite.h>

namespace saltro::optimizer {

enum class ALILQRStatus {
    // Constraint violation reached configured tolerance.
    Converged,
    // Outer-iteration budget exhausted before constraint convergence.
    MaxOuterIterations,
    // Inner iLQR failed with a non-recoverable status.
    InnerFailed,
    // Penalty parameter μ saturated at penalty_max on every active constraint
    // yet cmax is still > constraint_tol. Further outer iterations cannot grow
    // the penalty, so continuing is pointless. Mirrors PhD's "penMax" exit.
    PenaltyMaxReached,
};

bool alilqr(
    const PlannerSettings& settings,
    int pass_idx,
    const Satellite& satellite,
    Eigen::Ref<Eigen::MatrixXd> X,
    Eigen::Ref<Eigen::MatrixXd> U,
    const Eigen::Ref<const Eigen::MatrixXd>& R,
    const Eigen::Ref<const Eigen::MatrixXd>& V,
    const Eigen::Ref<const Eigen::MatrixXd>& B,
    const Eigen::Ref<const Eigen::MatrixXd>& S,
    const Eigen::Ref<const Eigen::MatrixXd>& rho,
    const Eigen::Ref<const Eigen::VectorXd>& jtime,
    const Eigen::Ref<const Eigen::MatrixXd>& boresight,
    const Eigen::Ref<const Eigen::MatrixXd>& attitude_target,
    ALILQRStatus& status,
    double& max_constraint_violation
);

}
