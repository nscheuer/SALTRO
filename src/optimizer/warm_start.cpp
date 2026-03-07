#include <saltro/optimizer/warm_start.h>

#include <cmath>
#include <memory>

#include <saltro/math/integrators/rk4.h>
#include <saltro/pybind/controller/controller.h>
#include <saltro/pybind/controller/excitationcontroller.h>
#include <saltro/pybind/controller/integratedbdotcontroller.h>
#include <saltro/pybind/controller/zerocontroller.h>

namespace saltro::optimizer {

bool warm_start(
    const PlannerSettings& settings,
    const Satellite& satellite,
    const Satellite::VecX& x0,
    const Eigen::Ref<const Eigen::VectorXd>& jtime,
    const Eigen::Ref<const Eigen::MatrixXd>& q_goal,
    const Eigen::Ref<const Eigen::MatrixXd>& boresight,
    int N,
    const Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& R,
    const Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& V,
    const Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& B,
    const Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ>& S,
    const Eigen::Matrix<double, 1, saltro::limits::MAX_LENGTH_TRAJ>& rho,
    Eigen::Ref<Eigen::MatrixXd> X,
    Eigen::Ref<Eigen::MatrixXd> U
) {
    const int nx = satellite.stateDim();
    const int nu = satellite.controlDim();

    if (q_goal.rows() != 4 || q_goal.cols() != N) {
        return false;
    }
    if (boresight.rows() != 3 || boresight.cols() != N) {
        return false;
    }

    std::unique_ptr<controller::Controller> active_controller;
    switch (settings.init_traj.initcontroller) {
        case 0:
            active_controller = std::make_unique<controller::ZeroController>(satellite);
            break;
        case 1:
            active_controller = std::make_unique<controller::ExcitationController>(satellite);
            break;
        case 2:
            active_controller = std::make_unique<controller::IntegratedBdotController>(satellite);
            break;
        default:
            return false;
    }

    X.setZero();
    U.setZero();

    Satellite::VecX xk = x0;
    X.col(0) = xk;

    for (int k = 0; k < N; ++k) {
        double dt = 0.0;
        if (k < N - 1) {
            const double dt_centuries = jtime(k + 1) - jtime(k);
            if (std::isfinite(dt_centuries) && dt_centuries > 0.0) {
                dt = dt_centuries * 36525.0 * 86400.0;
            }
        } else if (k > 0) {
            const double dt_centuries = jtime(k) - jtime(k - 1);
            if (std::isfinite(dt_centuries) && dt_centuries > 0.0) {
                dt = dt_centuries * 36525.0 * 86400.0;
            }
        }

        if ((!std::isfinite(dt) || dt <= 0.0) && settings.num_passes > 0 && std::isfinite(settings.passes[0].dt) && settings.passes[0].dt > 0.0) {
            dt = settings.passes[0].dt;
        }

        if (!std::isfinite(dt) || dt <= 0.0) {
            return false;
        }

        const Eigen::Vector4d q_goal_k = q_goal.col(k);
        const Eigen::Vector3d boresight_k = boresight.col(k);
        Satellite::VecX uk = active_controller->find_u(xk, B.col(k), q_goal_k, boresight_k);

        if (uk.size() != nu) {
            return false;
        }

        // Scale control from 1s nominal timestep to actual timestep
        // Controller outputs torque-impulse for 1s application; divide by dt to get torque
        Satellite::VecX uk_scaled = uk / dt;

        // Clamp to actuator limits AFTER scaling
        for (int i = 0; i < satellite.numMTQ(); ++i) {
            const double umax = std::abs(satellite.getMTQ(i).u_max());
            uk_scaled(i) = std::clamp(uk_scaled(i), -umax, umax);
        }
        for (int i = 0; i < satellite.numRW(); ++i) {
            const int ui = satellite.numMTQ() + i;
            const double umax = std::abs(satellite.getRW(i).u_max());
            uk_scaled(ui) = std::clamp(uk_scaled(ui), -umax, umax);
        }

        U.col(k) = uk_scaled;

        if (k == N - 1) {
            break;
        }

        const int rho_k = static_cast<int>(std::max(0.0, std::round(rho(k))));

        Satellite::VecX x_next;
        rk4_step<Satellite::VecX>(
            [&](double, const Satellite::VecX& x_state, Satellite::VecX& dxdt) {
                dxdt = satellite.dynamics(
                    x_state,
                    uk_scaled,
                    settings.disturbances,
                    R.col(k),
                    B.col(k),
                    S.col(k),
                    V.col(k),
                    rho_k
                );
            },
            xk,
            0.0,
            dt,
            x_next
        );

        if (x_next.size() >= Satellite::QUAT_INDEX + 4) {
            Eigen::Vector4d q_next = x_next.segment<4>(Satellite::QUAT_INDEX);
            const double qn = q_next.norm();
            if (!std::isfinite(qn) || qn <= 1e-12) {
                return false;
            }
            x_next.segment<4>(Satellite::QUAT_INDEX) = q_next / qn;
        }

        X.col(k + 1) = x_next;
        xk = x_next;
    }

    return true;
}

}
