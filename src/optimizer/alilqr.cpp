#include <saltro/optimizer/alilqr.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#include <saltro/optimizer/iLQR.h>
#include <saltro/pybind/plannersettings.h>
#include <saltro/pybind/satellite.h>

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
    lambda_aug.reserve(N_steps);
    mu_aug.reserve(N_steps);
    family_id.reserve(N_steps);

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
    std::vector<double> max_c_family_prev(NUM_FAMILIES,
                                          std::numeric_limits<double>::infinity());

    for (int outer_iter = 0; outer_iter < aug.max_outer_iters; ++outer_iter) {
        double J = 0.0;
        ILQRStatus ilqr_status = ILQRStatus::MaxIterations;
        ILQRTelemetry ilqr_telemetry;
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

        const double max_c = max_constraint_violation(settings, satellite, X, U, S);
        max_constraint_violation_out = max_c;

        // Outer-break gate (PhD-aligned 2026-04-23):
        // Declare converged when constraints are satisfied AND either
        //   (a) the inner iLQR reported ILQRStatus::Converged at this outer
        //       iteration (so the trajectory actually settled to the current
        //       λ/μ), AND we have had at least min_outer_iters iterations of
        //       λ/μ updates (so duals are not arbitrary); OR
        //   (b) max_c is below a stricter tolerance — a fast path that trusts
        //       the constraint-satisfaction signal outright (disabled when
        //       constraint_tol_strict <= 0, the default).
        // Previously saltro exited on (max_c <= constraint_tol) alone, which
        // could declare victory after an inner MaxIterations bailout or before
        // duals had settled.
        if (max_c <= aug.constraint_tol) {
            const bool inner_ok = (ilqr_status == ILQRStatus::Converged);
            const bool outer_matured = (outer_iter + 1) >= aug.min_outer_iters;
            const bool strict_path =
                (aug.constraint_tol_strict > 0.0) && (max_c <= aug.constraint_tol_strict);
            if (strict_path || (inner_ok && outer_matured)) {
                status = ALILQRStatus::Converged;
                return true;
            }
        }

        const auto c_list = collect_constraints(settings, satellite, X, U, S);

        // Per-family max violation at this outer iter. Used to gate ramping
        // (only ramp families that aren't contracting).
        std::vector<double> max_c_family(NUM_FAMILIES, 0.0);
        for (size_t k = 0; k < c_list.size(); ++k) {
            const auto& c_k = c_list[k];
            const auto& fams = family_id[k];
            for (int i = 0; i < c_k.size(); ++i) {
                const double v = std::max(0.0, c_k(i));
                const int f = fams[static_cast<size_t>(i)];
                if (f >= 0 && f < NUM_FAMILIES && v > max_c_family[f]) {
                    max_c_family[f] = v;
                }
            }
        }

        if (std::getenv("SALTRO_AL_DEBUG")) {
            double mu_rwm = -1.0, lam_rwm = 0.0;
            for (size_t k = 0; k < c_list.size() && mu_rwm < 0.0; ++k) {
                const auto& fams = family_id[k];
                for (int i = 0; i < static_cast<int>(fams.size()); ++i)
                    if (fams[static_cast<size_t>(i)] == 4) {
                        mu_rwm = mu_aug[k](i); lam_rwm = lambda_aug[k](i); break;
                    }
            }
            // also report the MAX lambda over all RWMomentum entries (the binding knot)
            double lam_rwm_max = 0.0;
            for (size_t k = 0; k < c_list.size(); ++k) {
                const auto& fams = family_id[k];
                for (int i = 0; i < static_cast<int>(fams.size()); ++i)
                    if (fams[static_cast<size_t>(i)] == 4 && lambda_aug[k](i) > lam_rwm_max)
                        lam_rwm_max = lambda_aug[k](i);
            }
            std::fprintf(stderr,
                "[AL] outer=%d ilqr=%d acc=%d it=%d J=%.3e max_c=%.3e | RWmom=%.4e | mu_RWmom=%.3e lam_RWmom_max=%.3e\n",
                outer_iter, static_cast<int>(ilqr_status),
                ilqr_telemetry.accepted_steps, ilqr_telemetry.iterations,
                ilqr_telemetry.final_cost, max_c,
                max_c_family[4], mu_rwm, lam_rwm_max);
        }

        // Conditional ramp decision per family. If contraction_ratio ≤ 0,
        // always ramp (original behavior). Otherwise ramp only if family
        // violation didn't contract sufficiently.
        std::vector<bool> ramp_family(NUM_FAMILIES, true);
        if (aug.family_contraction_ratio > 0.0) {
            for (int f = 0; f < NUM_FAMILIES; ++f) {
                const double prev = max_c_family_prev[static_cast<size_t>(f)];
                ramp_family[f] = (max_c_family[f] > aug.family_contraction_ratio * prev);
            }
        }
        // Record this iter's violations for the next iter's contraction check.
        for (int f = 0; f < NUM_FAMILIES; ++f) max_c_family_prev[f] = max_c_family[f];

        bool any_mu_below_max = false;
        for (size_t k = 0; k < c_list.size(); ++k) {
            // Lambda update uses RAW constraint value (not clamped to positive).
            // This allows lambda to decrease for satisfied constraints,
            // maintaining proper dual variable evolution.
            lambda_aug[k] = (lambda_aug[k] + mu_aug[k].cwiseProduct(c_list[k]))
                                .cwiseMin(aug.lag_mult_max);
            // For inequality constraints, lambda must stay non-negative
            lambda_aug[k] = lambda_aug[k].cwiseMax(0.0);

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
                if (mu_aug[k](i) < pmax_f) {
                    any_mu_below_max = true;
                }
            }
        }

        // PhD "penMax" exit: if μ has saturated everywhere and we still can't
        // drive cmax below constraint_tol, further outer iterations cannot
        // grow penalty. Break out rather than burning remaining outer budget.
        if (!any_mu_below_max && max_c > aug.constraint_tol) {
            status = ALILQRStatus::PenaltyMaxReached;
            max_constraint_violation_out = max_c;
            return false;
        }
    }

    status = ALILQRStatus::MaxOuterIterations;
    max_constraint_violation_out = max_constraint_violation(settings, satellite, X, U, S);
    return false;
}

}
