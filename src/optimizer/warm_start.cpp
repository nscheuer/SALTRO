#include <saltro/optimizer/warm_start.h>

#include <cmath>
#include <memory>

#include <saltro/pybind/controller/controller.h>
#include <saltro/pybind/controller/excitationcontroller.h>
#include <saltro/pybind/controller/integratedbdotcontroller.h>
#include <saltro/pybind/controller/pdcontroller.h>
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
    const int nu = satellite.controlDim();

    if (q_goal.rows() != 4 || q_goal.cols() != N) {
        return false;
    }
    if (boresight.rows() != 3 || boresight.cols() != N) {
        return false;
    }

    // initcontroller == 4: use the caller-provided X/U as the warm-start
    // trajectory verbatim (do NOT regenerate). Enables coarse-to-fine: the
    // caller resamples a coarse solve onto this grid, writes it into X/U, and
    // calls trajOpt with initcontroller=4 so the fine pass refines it.
    // We validate sizes and quaternion norms; force x0 at k=0; renormalize quats.
    if (settings.init_traj.initcontroller == 4) {
        if (X.rows() < Satellite::QUAT_INDEX + 4 || X.cols() < N || U.cols() < N) {
            return false;
        }
        X.col(0) = x0;  // pin initial state
        for (int k = 0; k < N; ++k) {
            Eigen::Vector4d q = X.col(k).segment<4>(Satellite::QUAT_INDEX);
            const double qn = q.norm();
            if (!std::isfinite(qn) || qn <= 1e-9) return false;
            X.col(k).segment<4>(Satellite::QUAT_INDEX) = q / qn;
        }
        return true;
    }

    std::unique_ptr<controller::Controller> active_controller;
    switch (settings.init_traj.initcontroller) {
        case 0:
            active_controller = std::make_unique<controller::ZeroController>(satellite);
            break;
        case 1: {
            const double exc_dt = (settings.num_passes > 0 && std::isfinite(settings.passes[0].dt)
                                   && settings.passes[0].dt > 0.0)
                                      ? settings.passes[0].dt
                                      : controller::ExcitationController::kDtRefSeconds;
            active_controller = std::make_unique<controller::ExcitationController>(satellite, exc_dt);
            break;
        }
        case 2:
            active_controller = std::make_unique<controller::IntegratedBdotController>(satellite);
            break;
        case 3: {
            auto pd = std::make_unique<controller::PDController>(satellite);
            // Optional gain softening for coarse-dt stability (see InitTrajConfig).
            const double gs = settings.init_traj.pd_gain_scale;
            if (std::isfinite(gs) && gs > 0.0 && gs != 1.0) {
                pd->setGains(gs * pd->kp_q(), gs * pd->kd_w());
            }
            // Pre-cancel the planned body-fixed propulsion torque so the
            // warm-start actuates against it rather than letting ω drift over a
            // coarse horizon. τ_ff = −τ_prop (body frame); allocation clamps it
            // into the actuator envelope.
            if (settings.disturbances.plan_for_prop) {
                pd->setFeedforwardTorque(-settings.disturbances.prop_torque);
            }
            active_controller = std::move(pd);
            break;
        }
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
        for (int i = 0; i < satellite.numMagic(); ++i) {
            const int ui = satellite.numMTQ() + satellite.numRW() + i;
            const double umax = std::abs(satellite.getMagic(i).u_max());
            uk_scaled(ui) = std::clamp(uk_scaled(ui), -umax, umax);
        }

        U.col(k) = uk_scaled;

        if (k == N - 1) {
            break;
        }

        const int rho_k = static_cast<int>(std::max(0.0, std::round(rho(k))));

        // Integrate dynamics + quaternion renormalization in one call.
        Satellite::VecX x_next = satellite.dynamicsStepRK4(
            xk, uk_scaled, dt, settings.disturbances,
            R.col(k), B.col(k), S.col(k), V.col(k), rho_k
        );

        // Quaternion sanity + renormalization. At coarse dt (e.g. 10s) RK4 can
        // leave the quaternion norm deviated by more than the old 1e-6 reject
        // tolerance after a large per-step rotation — that's expected, not a
        // failure. Only reject on a TRULY degenerate quaternion (norm ~0 or
        // non-finite); otherwise renormalize and continue. This is what lets a
        // coarse warm-start succeed (prerequisite for coarse-to-fine).
        if (x_next.size() >= Satellite::QUAT_INDEX + 4) {
            const double qn = x_next.segment<4>(Satellite::QUAT_INDEX).norm();
            if (!std::isfinite(qn) || qn < 1e-6) {
                return false;
            }
            x_next.segment<4>(Satellite::QUAT_INDEX) /= qn;
        }

        X.col(k + 1) = x_next;
        xk = x_next;
    }

    return true;
}

}
