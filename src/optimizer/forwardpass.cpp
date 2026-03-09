#include <saltro/optimizer/forwardpass.h>

#include <saltro/math/integrators/rk4.h>
#include <saltro/math/mrp.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

#if defined(SALTRO_DEBUG_BUILD)
#define SALTRO_OPT_DLOG(msg) do { std::cout << msg << std::endl; } while (0)
#else
#define SALTRO_OPT_DLOG(msg) do {} while (0)
#endif

namespace saltro::optimizer {

static bool linesearch(
    double J_minus,
    double J_new,
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const Eigen::Ref<const Eigen::MatrixXd>& U,
    const Eigen::Ref<const Eigen::MatrixXd>& X_bar,
    const Eigen::Ref<const Eigen::MatrixXd>& U_bar,
    double alpha,
    const Eigen::Ref<const Eigen::Vector2d>& deltaV,
    const LineSearchConfig& ls_cfg
)
{
    (void)X;
    (void)U;
    (void)X_bar;
    (void)U_bar;

    const double delta_V_alpha = alpha * (deltaV(0) + alpha * deltaV(1));
    if (!std::isfinite(delta_V_alpha) || std::abs(delta_V_alpha) < 1e-16) {
        return false;
    }

    const double z = (J_minus - J_new) / (-delta_V_alpha);
    return (z >= ls_cfg.beta1 && z <= ls_cfg.beta2);
}

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
    const Eigen::Ref<const Eigen::MatrixXd>& attitude_target,
    const PlannerSettings& settings,
    const std::vector<Eigen::VectorXd>& lambda_aug,
    const std::vector<Eigen::VectorXd>& mu_aug,
    const Eigen::Ref<const Eigen::VectorXd>& jtime,
    double J_prev,
    double& J_new
)
{
    const int N = static_cast<int>(X.cols());
    const int nx = static_cast<int>(X.rows());
    const int nu = static_cast<int>(U.rows());

    const auto& dist_cfg = settings.disturbances;
    const CostConfig& cost_cfg = settings.passes[0].cost;
    const ConstraintConfig& cnst_cfg = settings.constraints;
    const LineSearchConfig& ls_cfg = settings.passes[0].linesearch;
    const double J_minus = J_prev;

    Eigen::MatrixXd X_bar = Eigen::MatrixXd::Zero(nx, N);
    Eigen::MatrixXd U_bar = Eigen::MatrixXd::Zero(nu, std::max(0, N - 1));

    for (int iter = 0; iter < ls_cfg.max_iters; ++iter) {
        const double alpha = std::ldexp(1.0, -iter); // 1, 0.5, 0.25, ...
        SALTRO_OPT_DLOG("[FP] trial=" << iter << " alpha=" << alpha << " J_prev=" << J_prev);

        X_bar.setZero();
        U_bar.setZero();
        X_bar.col(0) = X.col(0);

        for (int k = 0; k < N - 1; ++k) {
            const double dt_centuries = jtime(k + 1) - jtime(k);
            const double dt = dt_centuries * 36525.0 * 86400.0;

            // u_bar(k) = u(k) + K_k(δz_k) + alpha * d_k
            // where δz_k is the reduced-state error using MRP for attitude
            Eigen::VectorXd u_bar_k = U.col(k);
            if (k < static_cast<int>(K.size())) {
                // Compute reduced-state error δz = [δω, δθ_MRP, δh_rw]
                const int nRW = satellite.numRW();
                const int nxr = satellite.reducedStateDim(); // 6 + nRW
                Eigen::VectorXd state_error_reduced(nxr);
                
                // Angular velocity error (direct difference)
                state_error_reduced.head<3>() = X_bar.col(k).head<3>() - X.col(k).head<3>();
                
                // Attitude error using MRP: q_err = q_ref^{-1} ⊗ q_bar, then MRP
                const Eigen::Vector4d q_ref = X.col(k).segment<4>(3);
                const Eigen::Vector4d q_bar = X_bar.col(k).segment<4>(3);
                Eigen::Vector4d q_err = saltro::math::quatError(q_ref, q_bar);
                Eigen::Vector3d mrp_err = saltro::math::quatToMRP(q_err);
                state_error_reduced.segment<3>(3) = mrp_err;
                
                // RW momentum error (direct difference)
                for (int i = 0; i < nRW; ++i) {
                    state_error_reduced(6 + i) = X_bar.col(k)(7 + i) - X.col(k)(7 + i);
                }
                
                u_bar_k += K[k] * state_error_reduced;
            }
            if (k < static_cast<int>(d.size())) {
                u_bar_k += alpha * d[k];
            }

            U_bar.col(k) = u_bar_k;

            // Integrate dynamics with RK4 to get x_bar(k+1)
            Eigen::VectorXd x_next;
            rk4_step<Eigen::VectorXd>(
                [&](double, const Eigen::VectorXd& x_state, Eigen::VectorXd& dxdt) {
                    const Eigen::Vector3d R_k = R.col(k);
                    const Eigen::Vector3d V_k = V.col(k);
                    const Eigen::Vector3d S_k = S.col(k);
                    const int rho_k = static_cast<int>(std::max(0.0, std::round(rho(0, k))));
                    dxdt = satellite.dynamics(
                        x_state,
                        u_bar_k,
                        dist_cfg,
                        R_k,
                        B.col(k),
                        S_k,
                        V_k,
                        rho_k
                    );
                },
                X_bar.col(k),
                0.0,
                dt,
                x_next
            );

            Eigen::Vector4d q = x_next.segment<4>(3);
            q.normalize();
            x_next.segment<4>(3) = q;
            X_bar.col(k + 1) = x_next;
        }

        // Compute new trajectory nominal cost with rollout
        J_new = satellite.totalCost(X_bar, U_bar, B, boresight, attitude_target, cost_cfg);

        // Add augmented Lagrangian penalty terms when provided.
        if (!lambda_aug.empty() && !mu_aug.empty()) {
            const int n_lam = static_cast<int>(lambda_aug.size());
            const int n_mu = static_cast<int>(mu_aug.size());
            const int n_steps = std::min(N, std::min(n_lam, n_mu));
            for (int k = 0; k < n_steps; ++k) {
                const Eigen::VectorXd x_k = X_bar.col(k);
                Eigen::VectorXd u_k = Eigen::VectorXd::Zero(nu);
                if (k < U_bar.cols()) {
                    u_k = U_bar.col(k);
                }
                const Eigen::VectorXd c_k = satellite.constraints(k, N, x_k, u_k, S.col(k), cnst_cfg);

                for (int i = 0; i < c_k.size(); ++i) {
                    if (c_k(i) <= 0.0) {
                        continue;
                    }
                    const double ci = c_k(i);
                    const double li = lambda_aug[k](i);
                    const double mi = mu_aug[k](i);
                    J_new += li * ci + 0.5 * mi * ci * ci;
                }
            }
        }

        // Line search check
        const bool ls_ok = linesearch(J_minus, J_new, X, U, X_bar, U_bar, alpha, deltaV, ls_cfg);
        const double delta_V_alpha = alpha * (deltaV(0) + alpha * deltaV(1));
        const double z = (std::isfinite(delta_V_alpha) && std::abs(delta_V_alpha) >= 1e-16)
            ? (J_minus - J_new) / (-delta_V_alpha)
            : std::numeric_limits<double>::quiet_NaN();
        (void)z;  // Used only in debug logging macro
        SALTRO_OPT_DLOG("[FP] alpha=" << alpha << " J_new=" << J_new << " z=" << z << " ls_ok=" << static_cast<int>(ls_ok));
        if (ls_ok) {
            // Overwrite outputs with the new trajectory/control
            X = X_bar;
            if (U_bar.cols() == U.cols()) {
                U = U_bar;
            } else if (U_bar.cols() == U.cols() - 1) {
                U.leftCols(U_bar.cols()) = U_bar;
            }
            SALTRO_OPT_DLOG("[FP] accepted alpha=" << alpha << " J=" << J_new);
            return true;
        }
    }

    SALTRO_OPT_DLOG("[FP] failed all line-search trials; keep J=" << J_prev);
    J_new = J_prev;
    return false;
}

}
