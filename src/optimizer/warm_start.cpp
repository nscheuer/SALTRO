#include <saltro/optimizer/warm_start.h>

#include <algorithm>
#include <cmath>
#include <memory>

#include <saltro/math/integrators/rk4.h>
#include <saltro/math/mrp.h>
#include <saltro/math/quaternion.h>
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

    std::unique_ptr<controller::Controller> active_controller;
    // Non-owning handle to the PD controller (only set for case 3) so we can
    // apply the per-knot goal-rate feedforward without widening the
    // polymorphic Controller interface.
    controller::PDController* pd_controller = nullptr;
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
        case 3: {
            auto pd = std::make_unique<controller::PDController>(satellite);
            pd_controller = pd.get();
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

        // --- Goal-rate feedforward (ported from OldPlanner smartbdot wkdes) --
        // Finite-difference the goal trajectory between this knot and the next
        // to get the body-frame rate ω_des that tracks the moving goal, and
        // feed it to the PD controller so it damps ω toward ω_des rather than
        // toward zero.  Only active for case 3 with the flag enabled.
        if (pd_controller != nullptr && settings.init_traj.pd_goal_rate_ff_enabled) {
            Eigen::Vector3d omega_des = Eigen::Vector3d::Zero();
            const int knext = (k < N - 1) ? (k + 1) : k;
            if (knext != k) {
                Eigen::Vector4d qk_n = xk.segment<4>(Satellite::QUAT_INDEX);
                const double qkn = qk_n.norm();
                const Eigen::Vector4d goal_kp1 = q_goal.col(knext);
                if (std::isfinite(qkn) && qkn > 1e-12) {
                    qk_n /= qkn;
                    const Eigen::Matrix3d R_T =
                        saltro::math::rotationMatrix(qk_n).transpose();
                    if (std::isnan(q_goal_k(0)) && std::isnan(goal_kp1(0))) {
                        // Vector goals: ω = asin(|e_k×e_{k+1}|)·dir / dt (ECI),
                        // then rotated into the body frame.
                        Eigen::Vector3d e0 = q_goal_k.tail<3>();
                        Eigen::Vector3d e1 = goal_kp1.tail<3>();
                        const double n0 = e0.norm();
                        const double n1 = e1.norm();
                        if (n0 > 1e-12 && n1 > 1e-12) {
                            e0 /= n0;
                            e1 /= n1;
                            const Eigen::Vector3d cr = e0.cross(e1);
                            const double s = std::clamp(cr.norm(), 0.0, 1.0);
                            if (s > 1e-12) {
                                omega_des = R_T * ((std::asin(s) / dt) * (cr / s));
                            }
                        }
                    } else if (!std::isnan(q_goal_k(0)) && !std::isnan(goal_kp1(0))) {
                        // Quaternion goals: relative rotation dq from goal_k to
                        // goal_{k+1}, dq = quatError(goal_k, goal_{k+1})
                        //             = goal_k^{-1} ⊗ goal_{k+1}
                        // (NOT the reverse order, which negates ω_des and makes
                        // the FF fight the maneuver).  dq.vec lives in the
                        // goal_k BODY frame, so map goal-body → ECI with
                        // R(goal_k) before the usual ECI → satellite-body R_T.
                        // ω = 2·asin(|dq.vec|)·dir / dt.
                        const double gkn = q_goal_k.norm();
                        const Eigen::Vector4d dq =
                            saltro::math::quatError(q_goal_k, goal_kp1);
                        Eigen::Vector3d v = dq.tail<3>();
                        const double vn = v.norm();
                        if (vn > 1e-12 && std::isfinite(gkn) && gkn > 1e-12) {
                            const Eigen::Matrix3d R_goal =
                                saltro::math::rotationMatrix(q_goal_k / gkn);
                            omega_des =
                                R_T * (R_goal *
                                       ((2.0 * std::asin(std::clamp(vn, 0.0, 1.0)) / dt)
                                        * (v / vn)));
                        }
                    }
                }
            }
            pd_controller->setGoalRate(omega_des.allFinite() ? omega_des
                                                             : Eigen::Vector3d::Zero());
        }

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
