#include <saltro/optimizer/forwardpass.h>

#include <saltro/math/integrators/rk4.h>
#include <algorithm>
#include <cmath>

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
    const Eigen::Ref<const Eigen::Vector4d>& attitude_target,
    const PlannerSettings& settings,
    const Eigen::Ref<const Eigen::VectorXd>& jtime,
    double J_prev,
    double& J_new
)
{
    const int N = static_cast<int>(X.cols());
    const int nx = static_cast<int>(X.rows());
    const int nu = static_cast<int>(U.rows());
    if (N <= 0 || nx <= 0 || nu <= 0) {
        J_new = J_prev;
        return false;
    }

    const auto& dist_cfg = settings.disturbances;
    const CostConfig& cost_cfg = settings.passes[0].cost;
    const LineSearchConfig& ls_cfg = settings.passes[0].linesearch;
    const double J_minus = J_prev;

    Eigen::MatrixXd X_bar = Eigen::MatrixXd::Zero(nx, N);
    Eigen::MatrixXd U_bar = Eigen::MatrixXd::Zero(nu, std::max(0, N - 1));

    for (int iter = 0; iter < ls_cfg.max_iters; ++iter) {
        const double alpha = std::ldexp(1.0, -iter); // 1, 0.5, 0.25, ...

        X_bar.setZero();
        U_bar.setZero();
        X_bar.col(0) = X.col(0);

        bool rollout_ok = true;

        for (int k = 0; k < N - 1; ++k) {
            double dt = 0.0;
            if (jtime.size() > k + 1) {
                const double dt_centuries = jtime(k + 1) - jtime(k);
                if (std::isfinite(dt_centuries) && dt_centuries > 0.0) {
                    dt = dt_centuries * 36525.0 * 86400.0;
                }
            }
            if ((!std::isfinite(dt) || dt <= 0.0) && settings.num_passes > 0 && std::isfinite(settings.passes[0].dt) && settings.passes[0].dt > 0.0) {
                dt = settings.passes[0].dt;
            }
            if (!std::isfinite(dt) || dt <= 0.0) {
                rollout_ok = false;
                break;
            }

            // u_bar(k) = u(k) + K_k(x_bar(k) - x(k)) + alpha * d_k
            Eigen::VectorXd u_bar_k = U.col(k);
            if (k < static_cast<int>(K.size())) {
                u_bar_k += K[k] * (X_bar.col(k) - X.col(k));
            }
            if (k < static_cast<int>(d.size())) {
                u_bar_k += alpha * d[k];
            }

            U_bar.col(k) = u_bar_k;

            // Integrate dynamics with RK4 to get x_bar(k+1)
            Eigen::VectorXd x_next;
            rk4_step<Eigen::VectorXd>(
                [&](double, const Eigen::VectorXd& x_state, Eigen::VectorXd& dxdt) {
                    const Eigen::Vector3d R_k = (R.cols() > k) ? Eigen::Vector3d(R.col(k)) : Eigen::Vector3d::Zero();
                    const Eigen::Vector3d V_k = (V.cols() > k) ? Eigen::Vector3d(V.col(k)) : Eigen::Vector3d::Zero();
                    const Eigen::Vector3d S_k = (S.cols() > k) ? Eigen::Vector3d(S.col(k)) : Eigen::Vector3d::Zero();
                    const int rho_k = (rho.cols() > k) ? static_cast<int>(std::max(0.0, std::round(rho(0, k)))) : 0;
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

            if (x_next.size() == nx) {
                X_bar.col(k + 1) = x_next;
            } else {
                rollout_ok = false;
                break;
            }
        }

        if (!rollout_ok) {
            continue;
        }

        // Compute new trajectory cost with rollout
        J_new = satellite.totalCost(X_bar, U_bar, B, boresight, attitude_target, cost_cfg);

        // Line search check
        const bool ls_ok = linesearch(J_minus, J_new, X, U, X_bar, U_bar, alpha, deltaV, ls_cfg);
        if (ls_ok) {
            // Overwrite outputs with the new trajectory/control
            X = X_bar;
            if (U_bar.cols() == U.cols()) {
                U = U_bar;
            } else if (U_bar.cols() == U.cols() - 1) {
                U.leftCols(U_bar.cols()) = U_bar;
            }
            return true;
        }
    }

    J_new = J_prev;
    return false;
}

}
