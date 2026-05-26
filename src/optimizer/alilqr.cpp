#include <saltro/optimizer/alilqr.h>

#include <algorithm>
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

    auto c0 = collect_constraints(settings, satellite, X, U, S);
    std::vector<Eigen::VectorXd> lambda_aug;
    std::vector<Eigen::VectorXd> mu_aug;
    lambda_aug.reserve(c0.size());
    mu_aug.reserve(c0.size());

    const double mu_init = aug.penalty_init / aug.penalty_scale;
    for (const auto& c_k : c0) {
        lambda_aug.push_back(Eigen::VectorXd::Constant(c_k.size(), aug.lag_mult_init));
        mu_aug.push_back(Eigen::VectorXd::Constant(c_k.size(), mu_init));
    }

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
        bool any_mu_below_max = false;
        for (size_t k = 0; k < c_list.size(); ++k) {
            // Lambda update uses RAW constraint value (not clamped to positive).
            // This allows lambda to decrease for satisfied constraints,
            // maintaining proper dual variable evolution.
            lambda_aug[k] = (lambda_aug[k] + mu_aug[k].cwiseProduct(c_list[k]))
                                .cwiseMin(aug.lag_mult_max);
            // For inequality constraints, lambda must stay non-negative
            lambda_aug[k] = lambda_aug[k].cwiseMax(0.0);
            mu_aug[k] = (mu_aug[k] * aug.penalty_scale).cwiseMin(aug.penalty_max);
            for (int i = 0; i < mu_aug[k].size(); ++i) {
                if (mu_aug[k](i) < aug.penalty_max) {
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
