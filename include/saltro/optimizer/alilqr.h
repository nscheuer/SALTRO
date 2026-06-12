#pragma once

#include <Eigen/Dense>
#include <limits>
#include <vector>

#include <saltro/optimizer/iLQR.h>
#include <saltro/pybind/plannersettings.h>
#include <saltro/pybind/satellite.h>

namespace saltro::optimizer {

enum class ALILQRStatus {
    // Two-sided convergence: a strict-tier (settling) inner solve settled
    // (Converged, or Stalled at a cost plateau — OldPlanner dlaZcount /
    // antispike stagnation parity) AND max constraint violation <=
    // constraint_tol — or the opt-in constraint_tol_strict fast path fired
    // (with inner progress).
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
    double final_grad = -1.0;            // last gradient metric of the solve
    double last_delta_J = -1.0;          // |dJ| of the solve's last accepted step
    double min_delta_J = -1.0;           // smallest |dJ| over the solve's accepted steps
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

// ---- Pure outer-loop verdict logic (BREAK_GATE_DESIGN.md §8) --------------
//
// Every termination decision of the AL outer loop is folded into a single
// side-effect-free function over named predicates, evaluated exactly once per
// outer iteration. The λ/μ update is a separate pure-compute block in
// alilqr.cpp whose outputs (penalty saturation, dual stall, patience) feed
// this function as inputs; the update is committed only when the verdict is
// not Converged (a Converged exit reports telemetry at the pre-update λ/μ).
// The exhaustive truth table in tests/unit/optimizer/test_break_gates.cpp is
// the executable documentation of this function.

// Named predicates for one outer iteration, all computed by the caller.
struct GateInputs {
    /// ω*-side settlement certificate: the inner solve exited Converged, or
    /// Stalled with either certificate — absolute flatness (some accepted
    /// step had |ΔJ| ≤ the strict cost_tol) or dual settlement (the previous
    /// outer iteration's max relative Δλ was below lambda_stall_tol).
    bool inner_settled = false;
    /// Progress floor: Converged, Stalled, or ≥ 1 accepted step. Guards the
    /// fast path so a literal do-nothing trajectory can never be blessed.
    bool inner_progress = false;
    /// This inner solve ran the strict/conjunctive (settling) tier, i.e. the
    /// previous iterate was already feasible (settle handoff).
    bool ran_strict_tier = false;
    /// ≥ 1 constraint exists and every μ entry sits at its family cap
    /// (post-ramp values of this iteration).
    bool all_mu_saturated = false;
    /// ≥ 2 consecutive saturated, infeasible, non-contracting outer
    /// iterations whose dual updates were negligible (< lambda_stall_tol).
    bool lambda_stalled = false;
    /// penalty_max_patience > 0 and the count of consecutive saturated,
    /// infeasible, non-contracting outer iterations has reached it.
    bool patience_exhausted = false;
    /// Max constraint violation after this inner solve (η side).
    double max_c = std::numeric_limits<double>::infinity();
    /// Completed outer iterations counting the current one (outer_iter + 1).
    int outer_iters_done = 0;
    /// Total inner iLQR iterations accumulated across all outer iterations.
    long total_inner_iters = 0;
};

// The gate-relevant slice of AugLagConfig.
struct GateConfig {
    double constraint_tol = 0.0;
    double constraint_tol_strict = 0.0;  // <= 0 disables the fast path
    int min_outer_iters = 0;
    long max_total_iters = 0;            // <= 0 disables the global budget
};

// MaxOuterIterations is not a decide() verdict: it is the structural
// fallthrough when the outer for-loop ends without a verdict, and InnerFailed
// (non-recoverable inner status) aborts before the gate is reached.
enum class Verdict {
    Converged,
    PenaltyMaxReached,
    MaxTotalIterations,
    Continue,
};

inline const char* verdictName(Verdict v)
{
    switch (v) {
        case Verdict::Converged: return "Converged";
        case Verdict::PenaltyMaxReached: return "PenaltyMaxReached";
        case Verdict::MaxTotalIterations: return "MaxTotalIterations";
        case Verdict::Continue: return "Continue";
    }
    return "Unknown";
}

struct GateDecision {
    Verdict verdict = Verdict::Continue;
    ALConvergedVia via = ALConvergedVia::None;  // set iff verdict == Converged
};

// Pure verdict function: no side effects, total over all inputs.
GateDecision decide(const GateInputs& in, const GateConfig& cfg);

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
