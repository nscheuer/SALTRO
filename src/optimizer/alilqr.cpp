#include <saltro/optimizer/alilqr.h>

#include <algorithm>
#include <limits>
#include <vector>

#include <saltro/optimizer/al_slack.h>
#include <saltro/optimizer/iLQR.h>

namespace saltro::optimizer {

namespace {

bool isRecoverableInnerFailure(const ILQRStatus status)
{
    // AL outer loop may still reduce constraint violation after a non-converged
    // inner solve, so these statuses do not immediately terminate AL.
    return status == ILQRStatus::MaxIterations || status == ILQRStatus::RegularizationExceeded;
}

bool hasInnerProgress(const ILQRStatus status, const ILQRTelemetry& telemetry)
{
    return status == ILQRStatus::Converged || telemetry.accepted_steps > 0;
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

double max_constraint_violation(
    const PlannerSettings& settings,
    const Satellite& satellite,
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const Eigen::Ref<const Eigen::MatrixXd>& U,
    const Eigen::Ref<const Eigen::MatrixXd>& S
)
{
    const auto c_list = collect_constraints(settings, satellite, X, U, S);

    double max_violation = 0.0;
    for (const auto& c_k : c_list) {
        for (int i = 0; i < c_k.size(); ++i) {
            max_violation = std::max(max_violation, std::max(0.0, c_k(i)));
        }
    }

    return max_violation;
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
    double& max_constraint_violation_out
)
{
    const auto& aug = settings.passes[pass_idx].auglag;
    const int NUM_FAMILIES = static_cast<int>(ConstraintFamily::NumFamilies);

    // State-slack two-phase control (al_slack.h). Phase 1 (slack): state
    // constraints are relaxed by analytically-eliminated slacks so the inner
    // solve stays conditioned while large violations are traded off at price
    // slack_rho. Phase 2 (polish): once the TRUE max violation reaches
    // slack_off_tol, slacks are dropped and AL continues warm-started from
    // the same trajectory and multipliers. The inner stack reads the flag
    // from the settings copy below; convergence is only declared in the
    // polish phase, so every slack run ends with >= 1 slack-free solve.
    bool slack_phase = aug.use_state_slack;
    const double slack_switch_tol = std::max(aug.slack_off_tol, aug.constraint_tol);
    PlannerSettings phase_settings = settings;

    // Continuation on the slack price (Huber annealing): slack_rho_cur is
    // the live cap used by every AL site this outer iter. With
    // slack_rho_scale > 1 it ramps up across slack-phase iterations so the
    // relaxation anneals toward the exact penalty (al_slack.h / plannersettings.h).
    double slack_rho_cur = aug.slack_rho;

    // Stall fallback bookkeeping: if the slack phase stops contracting the
    // TRUE violation (the slack price is below the binding constraint's
    // multiplier, so the optimizer permanently "buys" the violation), drop
    // the slacks and continue exact rather than burning the outer budget.
    double slack_best_c = std::numeric_limits<double>::infinity();
    int slack_stall_count = 0;
    // Continuation latch: set once the soft cap stalls above slack_off_tol;
    // from then on rho anneals every iteration until the goal or the ceiling.
    // Latching (vs re-deciding each iter) avoids a dribble where each small
    // post-ramp contraction would otherwise veto the next ramp.
    bool slack_continuation_active = false;

    // Per-family AL config: if the user provided vectors of the right size
    // (NumFamilies), use them; otherwise fall back to the scalar values.
    auto per_family_or_default = [&NUM_FAMILIES](
        const std::vector<double>& v, double fallback
    ) -> std::vector<double> {
        if (static_cast<int>(v.size()) == NUM_FAMILIES) return v;
        return std::vector<double>(NUM_FAMILIES, fallback);
    };
    const auto p_init_f  = per_family_or_default(aug.penalty_init_per_family,  aug.penalty_init);
    const auto p_max_f   = per_family_or_default(aug.penalty_max_per_family,   aug.penalty_max);
    const auto p_scale_f = per_family_or_default(aug.penalty_scale_per_family, aug.penalty_scale);

    auto c0 = collect_constraints(settings, satellite, X, U, S);
    std::vector<Eigen::VectorXd> lambda_aug;
    std::vector<Eigen::VectorXd> mu_aug;
    std::vector<std::vector<int>> family_id;  // family_id[k][i] = family of constraint i at step k
    const int N_steps = static_cast<int>(c0.size());
    lambda_aug.reserve(c0.size());
    mu_aug.reserve(c0.size());
    family_id.reserve(c0.size());

    for (int k = 0; k < N_steps; ++k) {
        const auto& c_k = c0[static_cast<size_t>(k)];
        const bool is_terminal = (k == N_steps - 1);
        const int nc = static_cast<int>(c_k.size());

        std::vector<int> fams(static_cast<size_t>(nc));
        Eigen::VectorXd mu_init_vec = Eigen::VectorXd::Zero(nc);
        for (int i = 0; i < nc; ++i) {
            const int f = satellite.constraintFamily(i, is_terminal);
            fams[static_cast<size_t>(i)] = f;
            const double pinit_f =
                (f >= 0 && f < NUM_FAMILIES) ? p_init_f[static_cast<size_t>(f)] : aug.penalty_init;
            const double pscale_f =
                (f >= 0 && f < NUM_FAMILIES) ? p_scale_f[static_cast<size_t>(f)] : aug.penalty_scale;
            // Mirror original behavior: start at pinit / pscale so first ramp lands at pinit.
            mu_init_vec(i) = (pscale_f > 0.0) ? (pinit_f / pscale_f) : pinit_f;
        }

        lambda_aug.push_back(Eigen::VectorXd::Constant(nc, aug.lag_mult_init));
        mu_aug.push_back(std::move(mu_init_vec));
        family_id.push_back(std::move(fams));
    }

    // Per-family violation tracking (for conditional ramping). Initial value
    // ∞ means "first iter always treats it as not-yet-contracting" (i.e. ramp).
    std::vector<double> max_c_family_prev(static_cast<size_t>(NUM_FAMILIES),
                                          std::numeric_limits<double>::infinity());

    for (int outer_iter = 0; outer_iter < aug.max_outer_iters; ++outer_iter) {
        // Publish the live slack price to the inner stack (BP/FP/iLQR read
        // it from phase_settings) so the merit, the BP model and the dual
        // update below all price the slack at the same rho this iteration.
        phase_settings.passes[pass_idx].auglag.slack_rho = slack_rho_cur;

        double J = 0.0;
        ILQRStatus ilqr_status = ILQRStatus::MaxIterations;
        ILQRTelemetry ilqr_telemetry;
        const bool ilqr_ok = iLQR(
            phase_settings,
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
            ilqr_status,
            J,
            ilqr_telemetry
        );

        if (!ilqr_ok) {
            if (!isRecoverableInnerFailure(ilqr_status)) {
                status = ALILQRStatus::InnerFailed;
                max_constraint_violation_out = max_constraint_violation(settings, satellite, X, U, S);
                return false;
            }
        }

        // True (unslacked) violation: slack only changes what the optimizer
        // is penalized for, never what counts as satisfied.
        const double max_c = max_constraint_violation(settings, satellite, X, U, S);
        max_constraint_violation_out = max_c;
        if (!slack_phase && max_c <= aug.constraint_tol) {
            if (hasInnerProgress(ilqr_status, ilqr_telemetry)) {
                status = ALILQRStatus::Converged;
                return true;
            }
            status = ALILQRStatus::InnerFailed;
            return false;
        }

        bool ramp_rho_next = false;
        if (slack_phase) {
            bool drop_slacks = false;

            // Shared stall signal: did the TRUE violation contract >5% vs the
            // best seen this slack phase? Both continuation (raise the cap)
            // and the stall fallback (drop the slacks) react to a stall; a
            // contraction resets the counter. (First iter: best = +inf, so it
            // always counts as contracting — never ramp/drop on iter 0.)
            const bool contracting = (max_c < 0.95 * slack_best_c);
            if (contracting) {
                slack_best_c = max_c;
                slack_stall_count = 0;
            } else {
                ++slack_stall_count;
            }
            const bool can_continue = (aug.slack_rho_scale > 1.0);

            // CONTINUATION latch: the slack phase stalled at an equilibrium
            // above slack_off_tol because the cap is below the constraint's
            // true shadow price. Commit to annealing rho upward. Gated on the
            // stall + still-far-from-goal, so it NEVER engages while the soft
            // cap is productively driving the violation down to slack_off_tol
            // (e.g. RW binding cases) — those wins are preserved for any
            // slack_rho_scale.
            if (can_continue && !slack_continuation_active
                    && !contracting && max_c > slack_switch_tol) {
                slack_continuation_active = true;
            }

            if (max_c <= slack_switch_tol) {
                // "Reasonable satisfaction" reached: drop the slacks and
                // polish the same trajectory with the exact constraints.
                drop_slacks = true;
            } else if (slack_continuation_active && slack_rho_cur >= aug.slack_rho_max) {
                // Continuation exhausted: the cap annealed to its ceiling
                // without reaching slack_off_tol, so hand off to exact AL.
                drop_slacks = true;
            } else if (slack_continuation_active) {
                // Anneal the cap every iteration once latched — a graceful,
                // conditioning-preserving handoff toward exact AL (contrast
                // the abrupt stall-fallback drop). Applied after the dual
                // update so this iter's lambda used the same rho the solve did.
                ramp_rho_next = true;
            } else if (aug.slack_stall_iters > 0
                       && slack_stall_count >= aug.slack_stall_iters) {
                // Stall fallback (opt-in, abrupt): drop the slacks after
                // slack_stall_iters non-contracting iters. Only reached when
                // continuation is off (slack_rho_scale == 1).
                drop_slacks = true;
            }
            if (drop_slacks) {
                // lambda/mu carry over (warm start); the multiplier update
                // below already runs slack-free for this iterate.
                slack_phase = false;
                phase_settings.passes[pass_idx].auglag.use_state_slack = false;
            }
        }

        const auto c_list = collect_constraints(settings, satellite, X, U, S);

        // Per-family max violation at this outer iter. Used to gate ramping
        // (only ramp families that aren't contracting).
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

        // Conditional ramp decision per family. If contraction_ratio ≤ 0,
        // always ramp (original behavior). Otherwise ramp only if family
        // violation didn't contract sufficiently.
        std::vector<bool> ramp_family(static_cast<size_t>(NUM_FAMILIES), true);
        if (aug.family_contraction_ratio > 0.0) {
            for (int f = 0; f < NUM_FAMILIES; ++f) {
                const double prev = max_c_family_prev[static_cast<size_t>(f)];
                ramp_family[static_cast<size_t>(f)] =
                    (max_c_family[static_cast<size_t>(f)] > aug.family_contraction_ratio * prev);
            }
        }
        // Record this iter's violations for the next iter's contraction check.
        for (int f = 0; f < NUM_FAMILIES; ++f) {
            max_c_family_prev[static_cast<size_t>(f)] = max_c_family[static_cast<size_t>(f)];
        }

        for (size_t k = 0; k < c_list.size(); ++k) {
            // Dual update uses the RAW signed constraint value so lambda can
            // decrease for satisfied constraints (proper dual evolution); the
            // cwiseMax(0) keeps inequality multipliers non-negative.
            // In the slack phase, slack families update against the shifted
            // constraint c - s* (matching the BP/FP model), which saturates
            // lambda at the slack price instead of chasing the absorbed
            // violation.
            Eigen::VectorXd c_eff = c_list[k];
            if (slack_phase) {
                const auto& fams_k = family_id[k];
                for (int i = 0; i < c_eff.size(); ++i) {
                    if (isStateSlackFamily(fams_k[static_cast<size_t>(i)])) {
                        c_eff(i) -= optimalSlack(
                            c_eff(i), lambda_aug[k](i), mu_aug[k](i),
                            slack_rho_cur, aug.slack_sigma
                        );
                    }
                }
            }
            lambda_aug[k] = (lambda_aug[k] + mu_aug[k].cwiseProduct(c_eff))
                                .cwiseMin(aug.lag_mult_max)
                                .cwiseMax(0.0);

            // Per-family ramp: each entry of mu_aug[k] uses its family's scale and cap.
            // Skip scaling for families that contracted (ramp_family[f] == false).
            const auto& fams = family_id[k];
            for (int i = 0; i < mu_aug[k].size(); ++i) {
                const int f = fams[static_cast<size_t>(i)];
                const double pmax_f =
                    (f >= 0 && f < NUM_FAMILIES) ? p_max_f[static_cast<size_t>(f)] : aug.penalty_max;
                const double pscale_f =
                    (f >= 0 && f < NUM_FAMILIES) ? p_scale_f[static_cast<size_t>(f)] : aug.penalty_scale;
                const bool do_ramp =
                    (f >= 0 && f < NUM_FAMILIES) ? ramp_family[static_cast<size_t>(f)] : true;
                if (do_ramp) {
                    mu_aug[k](i) = std::min(mu_aug[k](i) * pscale_f, pmax_f);
                }
            }
        }

        // Continuation: anneal the slack cap upward for the next slack-phase
        // iteration when the phase stalled (ramp_rho_next set above). Ramped
        // AFTER the dual update so this iter's lambda used the same rho the
        // inner solve did.
        if (ramp_rho_next) {
            slack_rho_cur = std::min(slack_rho_cur * aug.slack_rho_scale, aug.slack_rho_max);
        }
    }

    status = ALILQRStatus::MaxOuterIterations;
    max_constraint_violation_out = max_constraint_violation(settings, satellite, X, U, S);
    return false;
}

}
