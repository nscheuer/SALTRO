#include <saltro/optimizer/alilqr.h>

#include <algorithm>
#include <limits>
#include <vector>

#include <saltro/optimizer/iLQR.h>

namespace saltro::optimizer {

namespace {

bool isRecoverableInnerFailure(const ILQRStatus status)
{
    // AL outer loop may still reduce constraint violation after a non-converged
    // inner solve, so these statuses do not immediately terminate AL.
    return status == ILQRStatus::MaxIterations || status == ILQRStatus::RegularizationExceeded;
}

// AL penalty share of the merit function at the current lambda/mu. Must match
// the active-set semantics in iLQR.cpp / forwardpass.cpp.
double penalty_share(
    const std::vector<Eigen::VectorXd>& c_list,
    const std::vector<Eigen::VectorXd>& lambda_aug,
    const std::vector<Eigen::VectorXd>& mu_aug
)
{
    double total = 0.0;
    const size_t n_steps = std::min(c_list.size(), std::min(lambda_aug.size(), mu_aug.size()));
    for (size_t k = 0; k < n_steps; ++k) {
        const auto& c_k = c_list[k];
        const int n_c = static_cast<int>(std::min<Eigen::Index>(
            c_k.size(), std::min(lambda_aug[k].size(), mu_aug[k].size())));
        for (int i = 0; i < n_c; ++i) {
            const double ci = c_k(i);
            const double li = lambda_aug[k](i);
            total += li * ci;
            if (ci > 0.0 || li > 0.0) {
                total += 0.5 * mu_aug[k](i) * ci * ci;
            }
        }
    }
    return total;
}

double max_violation_of(const std::vector<Eigen::VectorXd>& c_list)
{
    double max_violation = 0.0;
    for (const auto& c_k : c_list) {
        for (int i = 0; i < c_k.size(); ++i) {
            max_violation = std::max(max_violation, std::max(0.0, c_k(i)));
        }
    }
    return max_violation;
}

Eigen::VectorXd control_at_k(const Eigen::Ref<const Eigen::MatrixXd>& U, int k, int N, int control_dim)
{
    if (U.cols() == N - 1 && k < N - 1) {
        return U.col(k);
    }
    if (U.cols() == N && k < N) {
        return U.col(k);
    }
    return Eigen::VectorXd::Zero(control_dim);
}

std::vector<Eigen::VectorXd> collect_constraints(
    const PlannerSettings& settings,
    const Satellite& satellite,
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const Eigen::Ref<const Eigen::MatrixXd>& U,
    const Eigen::Ref<const Eigen::MatrixXd>& S
)
{
    const int N = static_cast<int>(X.cols());
    std::vector<Eigen::VectorXd> c_list(static_cast<size_t>(N));

    for (int k = 0; k < N; ++k) {
        c_list[static_cast<size_t>(k)] = satellite.constraints(
            k,
            N,
            X.col(k),
            control_at_k(U, k, N, satellite.controlDim()),
            S.col(k),
            settings.constraints
        );
    }

    return c_list;
}

} // namespace

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
    double& max_constraint_violation_out,
    ALILQRTelemetry& telemetry
)
{
    telemetry = ALILQRTelemetry{};
    const auto& aug = settings.passes[pass_idx].auglag;
    const int NUM_FAMILIES = static_cast<int>(ConstraintFamily::NumFamilies);

    auto c0 = collect_constraints(settings, satellite, X, U, S);
    std::vector<Eigen::VectorXd> lambda_aug;
    std::vector<Eigen::VectorXd> mu_aug;
    std::vector<std::vector<int>> family_id;  // family_id[k][i] = family of constraint i at step k
    const int N_steps = static_cast<int>(c0.size());
    lambda_aug.reserve(c0.size());
    mu_aug.reserve(c0.size());
    family_id.reserve(c0.size());

    // Start at penalty_init / penalty_scale so the first ramp lands at penalty_init.
    const double mu_init = aug.penalty_init / aug.penalty_scale;
    for (int k = 0; k < N_steps; ++k) {
        const auto& c_k = c0[static_cast<size_t>(k)];
        const bool is_terminal = (k == N_steps - 1);
        const int nc = static_cast<int>(c_k.size());

        std::vector<int> fams(static_cast<size_t>(nc));
        for (int i = 0; i < nc; ++i) {
            fams[static_cast<size_t>(i)] = satellite.constraintFamily(i, is_terminal);
        }

        lambda_aug.push_back(Eigen::VectorXd::Constant(nc, aug.lag_mult_init));
        mu_aug.push_back(Eigen::VectorXd::Constant(nc, mu_init));
        family_id.push_back(std::move(fams));
    }

    // Finalize helper: telemetry fields valid on every exit path.
    const auto& cost_cfg = settings.passes[pass_idx].cost;
    const int N_u = std::max(0, static_cast<int>(X.cols()) - 1);
    auto finalize = [&](const std::vector<Eigen::VectorXd>& c_list) {
        telemetry.nominal_cost = satellite.totalCost(
            X, U.leftCols(N_u), B, boresight, attitude_target, cost_cfg);
        telemetry.penalty_cost = penalty_share(c_list, lambda_aug, mu_aug);
    };

    // Settle handoff (simplified omega_k/eta_k forcing sequences): loose inner solves
    // while infeasible; once an iterate is feasible, the NEXT inner call runs
    // the strict/conjunctive tier. Converged is declared only off a
    // strict-tier solve that itself converged with max_c <= constraint_tol —
    // i.e., two consecutive feasible iterates, the second solved tight. The
    // warm start counts as the iterate preceding outer_iter 0.
    bool settle = false;
    {
        const auto c_warm = collect_constraints(settings, satellite, X, U, S);
        settle = (max_violation_of(c_warm) <= aug.constraint_tol);
    }

    int lambda_stall_run = 0;   // consecutive saturated iters with stalled duals
    int saturated_iters = 0;    // consecutive saturated-and-infeasible iters

    for (int outer_iter = 0; outer_iter < aug.max_outer_iters; ++outer_iter) {
        double J = 0.0;
        ILQRStatus ilqr_status = ILQRStatus::MaxIterations;
        ILQRTelemetry inner_telemetry;
        const bool ilqr_ok = iLQR(
            settings,
            satellite,
            X,
            U,
            R,
            V,
            B,
            S,
            rho,
            jtime,
            boresight,
            attitude_target,
            pass_idx,
            lambda_aug,
            mu_aug,
            settle,
            ilqr_status,
            J,
            inner_telemetry
        );
        (void)J;

        telemetry.total_inner_iterations += inner_telemetry.iterations;

        const auto c_list = collect_constraints(settings, satellite, X, U, S);
        const double max_c = max_violation_of(c_list);
        max_constraint_violation_out = max_c;

        // Per-family max violation at this outer iter. Used to gate ramping
        // (only ramp families that aren't contracting) and for telemetry.
        std::vector<double> max_c_family(static_cast<size_t>(NUM_FAMILIES), 0.0);
        for (size_t k = 0; k < c_list.size(); ++k) {
            const auto& c_k = c_list[k];
            const auto& fams = family_id[k];
            for (int i = 0; i < c_k.size(); ++i) {
                const double v = std::max(0.0, c_k(i));
                const int f = fams[static_cast<size_t>(i)];
                if (f >= 0 && f < NUM_FAMILIES && v > max_c_family[static_cast<size_t>(f)]) {
                    max_c_family[static_cast<size_t>(f)] = v;
                }
            }
        }
        telemetry.max_c_family = max_c_family;

        ALOuterIterRecord record;
        record.settle = settle;
        record.inner_status = ilqr_status;
        record.inner_break_reason = inner_telemetry.break_reason;
        record.inner_iterations = inner_telemetry.iterations;
        record.accepted_steps = inner_telemetry.accepted_steps;
        record.max_c = max_c;
        record.final_grad = inner_telemetry.final_grad;
        record.last_delta_J = inner_telemetry.last_delta_J;
        telemetry.outer.push_back(record);

        if (!ilqr_ok && !isRecoverableInnerFailure(ilqr_status)) {
            status = ALILQRStatus::InnerFailed;
            finalize(c_list);
            return false;
        }

        const bool inner_converged = (ilqr_status == ILQRStatus::Converged);
        // A Stalled exit is a *plateaued* solve: >= z_count_lim accepted steps
        // with no further relative cost progress at this lambda/mu. Both
        // reference implementations treat it as a settled subproblem
        // (OldPlanner's outer gate accepts the dlaZcount branch of ilqrBreak;
        // PKMN_antispike returns Converged from its stagnation counter), so it
        // qualifies as "solved" for the settled declaration and the
        // on_converged dual bar. It can never be a do-nothing solve.
        const bool inner_settled =
            inner_converged || (ilqr_status == ILQRStatus::Stalled);
        const bool inner_progress = inner_settled || (inner_telemetry.accepted_steps > 0);

        // ---- Convergence declaration (two-sided, omega*/eta*-style) ----
        // Fast path: violation far below tolerance. Bypasses min_outer_iters
        // and the settled-strict-solve requirement, but never the
        // inner-progress floor (a do-nothing trajectory cannot be blessed).
        if (aug.constraint_tol_strict > 0.0
            && max_c <= aug.constraint_tol_strict
            && inner_progress) {
            status = ALILQRStatus::Converged;
            telemetry.converged_via = ALConvergedVia::FastPath;
            finalize(c_list);
            return true;
        }
        // Settled path: this solve ran the strict tier (previous iterate was
        // feasible), converged on its own criteria (omega* side), and the
        // result is feasible (eta* side), after enough dual maturation.
        if (max_c <= aug.constraint_tol
            && settle
            && inner_settled
            && (outer_iter + 1) >= aug.min_outer_iters) {
            status = ALILQRStatus::Converged;
            telemetry.converged_via = ALConvergedVia::Settled;
            finalize(c_list);
            return true;
        }

        // Tier handoff for the next inner call.
        settle = (max_c <= aug.constraint_tol);

        // ---- Dual update gating (dual_update_mode) ----
        // Classical safeguarded-AL rule, softened: lambda may only be updated
        // from an inner solve that met the mode's bar; updating lambda off a
        // garbage iterate poisons the dual sequence.
        bool lambda_bar_met = false;
        switch (aug.dual_update_mode) {
            case DualUpdateMode::OnProgress:
                lambda_bar_met = inner_progress;
                break;
            case DualUpdateMode::OnConverged:
            default:
                // Settled solves only (Converged or Stalled-at-plateau):
                // eliminates the one-step-then-RegExceeded dual-poisoning
                // path while keeping duals alive on plateau exits.
                lambda_bar_met = inner_settled;
                break;
        }

        double max_rel_dlambda = 0.0;
        for (size_t k = 0; k < c_list.size(); ++k) {
            if (lambda_bar_met) {
                for (int i = 0; i < lambda_aug[k].size(); ++i) {
                    // Dual update uses the RAW signed constraint value so
                    // lambda can decrease for satisfied constraints (proper
                    // dual evolution); the max(0) keeps inequality
                    // multipliers non-negative.
                    const double old_l = lambda_aug[k](i);
                    const double new_l = std::min(
                        aug.lag_mult_max,
                        std::max(0.0, old_l + mu_aug[k](i) * c_list[k](i)));
                    max_rel_dlambda = std::max(
                        max_rel_dlambda,
                        std::abs(new_l - old_l) / std::max(std::abs(old_l), 1.0));
                    lambda_aug[k](i) = new_l;
                }
            }

            // Penalty ramp (global schedule; the per-family schedule is
            // parked with #52).
            mu_aug[k] = (mu_aug[k] * aug.penalty_scale).cwiseMin(aug.penalty_max);
        }

        // ---- PenaltyMaxReached: all penalties capped while infeasible ----
        // With mu pegged, lambda updates are still a slow method of
        // multipliers; wait for them to stall (2 consecutive stalled iters)
        // or for the patience window before giving up honestly.
        bool any_constraint = false;
        bool all_saturated = true;
        for (size_t k = 0; k < mu_aug.size() && all_saturated; ++k) {
            for (int i = 0; i < mu_aug[k].size(); ++i) {
                any_constraint = true;
                if (mu_aug[k](i) < aug.penalty_max) {
                    all_saturated = false;
                    break;
                }
            }
        }
        if (any_constraint && all_saturated && max_c > aug.constraint_tol) {
            ++saturated_iters;
            if (max_rel_dlambda < aug.lambda_stall_tol) {
                ++lambda_stall_run;
            } else {
                lambda_stall_run = 0;
            }
            if (lambda_stall_run >= 2
                || (aug.penalty_max_patience > 0 && saturated_iters >= aug.penalty_max_patience)) {
                status = ALILQRStatus::PenaltyMaxReached;
                finalize(c_list);
                return false;
            }
        } else {
            saturated_iters = 0;
            lambda_stall_run = 0;
        }

        // ---- Global inner-iteration budget (never throws) ----
        if (aug.max_total_iters > 0
            && telemetry.total_inner_iterations >= aug.max_total_iters) {
            status = ALILQRStatus::MaxTotalIterations;
            finalize(c_list);
            return false;
        }
    }

    status = ALILQRStatus::MaxOuterIterations;
    {
        const auto c_list = collect_constraints(settings, satellite, X, U, S);
        max_constraint_violation_out = max_violation_of(c_list);
        finalize(c_list);
    }
    return false;
}

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
    double& max_constraint_violation_out
)
{
    ALILQRTelemetry telemetry;
    return alilqr(
        settings,
        pass_idx,
        satellite,
        X,
        U,
        R,
        V,
        B,
        S,
        rho,
        jtime,
        boresight,
        attitude_target,
        status,
        max_constraint_violation_out,
        telemetry
    );
}

}
