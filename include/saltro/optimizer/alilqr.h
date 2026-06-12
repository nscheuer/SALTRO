#pragma once

#include <Eigen/Dense>
#include <vector>

#include <saltro/optimizer/iLQR.h>
#include <saltro/pybind/plannersettings.h>
#include <saltro/pybind/satellite.h>

namespace saltro::optimizer {

enum class ALILQRStatus {
    // Two-sided convergence: a strict-tier (settling) inner solve converged
    // AND max constraint violation <= constraint_tol — or the opt-in
    // constraint_tol_strict fast path fired (with inner progress).
    Converged,
    // Outer-iteration budget exhausted before constraint convergence.
    MaxOuterIterations,
    // Global inner-iteration budget (auglag.max_total_iters) exhausted.
    // The best trajectory found is returned.
    MaxTotalIterations,
    // Every penalty saturated at its cap while still infeasible, and the
    // dual updates stalled past the patience window. The best trajectory
    // found is returned.
    PenaltyMaxReached,
    // Inner iLQR failed with a non-recoverable status.
    InnerFailed,
};

inline const char* alilqrStatusName(ALILQRStatus status)
{
    switch (status) {
        case ALILQRStatus::Converged: return "Converged";
        case ALILQRStatus::MaxOuterIterations: return "MaxOuterIterations";
        case ALILQRStatus::MaxTotalIterations: return "MaxTotalIterations";
        case ALILQRStatus::PenaltyMaxReached: return "PenaltyMaxReached";
        case ALILQRStatus::InnerFailed: return "InnerFailed";
    }
    return "Unknown";
}

// How the Converged declaration fired (telemetry).
enum class ALConvergedVia {
    None,        // did not converge
    FastPath,    // constraint_tol_strict fast path (with inner progress)
    Settled,     // strict-tier inner solve converged while feasible
};

// One outer iteration's record.
struct ALOuterIterRecord {
    bool settle = false;                 // tier requested for this inner solve
    ILQRStatus inner_status = ILQRStatus::MaxIterations;
    ILQRBreakReason inner_break_reason = ILQRBreakReason::None;
    int inner_iterations = 0;
    int accepted_steps = 0;
    double max_c = -1.0;                 // max violation after this solve
};

// Per-call AL outer-loop telemetry.
struct ALILQRTelemetry {
    std::vector<ALOuterIterRecord> outer;      // one record per outer iteration
    std::vector<double> max_c_family;          // final per-family max violation
    double nominal_cost = -1.0;                // satellite cost of returned traj
    double penalty_cost = -1.0;                // AL penalty share at final lambda/mu
    int total_inner_iterations = 0;
    ALConvergedVia converged_via = ALConvergedVia::None;
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
    double& max_constraint_violation,
    ALILQRTelemetry& telemetry
);

// Back-compat overload: telemetry discarded.
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
