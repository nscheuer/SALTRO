#include <saltro/pybind/controller/excitationcontroller.h>

#include <algorithm>
#include <cmath>

namespace saltro::controller {

ExcitationController::ExcitationController(const Satellite& satellite, double dt)
    : Controller(satellite) {
    dt_seconds_ = (std::isfinite(dt) && dt > 0.0) ? dt : kDtRefSeconds;
    autoTuneGains();
}

Satellite::VecX ExcitationController::find_u(
    const Satellite::VecX& x,
    const Eigen::Vector3d& B_eci,
    const Eigen::Vector4d& q_goal,
    const Eigen::Vector3d& boresight_body
) const {
    Satellite::VecX u(satellite_.controlDim());
    u.setZero();

    if (satellite_.controlDim() == 0) {
        return u;
    }

    Eigen::Vector3d omega = Eigen::Vector3d::Zero();
    if (x.size() >= 3) {
        omega = x.segment<3>(0);
    }

    Eigen::Vector3d goal_dir = Eigen::Vector3d::UnitX();
    if (boresight_body.allFinite() && boresight_body.norm() > 1e-12) {
        goal_dir = boresight_body.normalized();
    } else if (q_goal.tail<3>().allFinite() && q_goal.tail<3>().norm() > 1e-12) {
        goal_dir = q_goal.tail<3>().normalized();
    }

    const double b_norm = B_eci.norm();
    const double b_safe = std::max(5e-6, b_norm);
    const double b_scale = std::clamp(expected_b_field_leo_ / b_safe, 0.5, 2.5);

    const int num_mtq = satellite_.numMTQ();
    const int num_rw = satellite_.numRW();

    for (int i = 0; i < num_mtq; ++i) {
        const MTQ& mtq = satellite_.getMTQ(i);
        const Eigen::Vector3d axis = mtq.axis();
        const double u_max = std::abs(mtq.u_max());
        const double excitation = mtq_excitation_fraction_ * u_max * axis.dot(goal_dir);
        const double damping = -mtq_rate_damping_gain_ * b_scale * omega.dot(axis);
        const double cmd_limit = mtq_command_fraction_limit_ * u_max;
        u(i) = std::clamp(excitation + damping, -cmd_limit, cmd_limit);
    }

    for (int i = 0; i < num_rw; ++i) {
        const int ctrl_idx = num_mtq + i;
        const RW& rw = satellite_.getRW(i);
        const Eigen::Vector3d axis = rw.axis();
        const double u_max = std::abs(rw.u_max());
        const double excitation = rw_excitation_fraction_ * u_max * axis.dot(goal_dir);
        const double damping = -rw_rate_damping_gain_ * omega.dot(axis);
        const double cmd_limit = rw_command_fraction_limit_ * u_max;
        u(ctrl_idx) = std::clamp(excitation + damping, -cmd_limit, cmd_limit);
    }

    return u;
}

void ExcitationController::autoTuneGains() {
    const double J_avg = std::max(1e-6, satellite_.inertia().trace() / 3.0);
    const double base_target_accel = 0.5 * 3.14159265358979323846 / 180.0;
    // Scale target α by (dt_ref / dt) so per-knot ω impulse (α·dt) stays bounded on
    // long-horizon passes; at dt == kDtRefSeconds this is a no-op.
    const double dt_scale = kDtRefSeconds / std::max(1e-6, dt_seconds_);
    const double target_accel = base_target_accel * dt_scale;
    const double target_tau = J_avg * target_accel;

    const int num_mtq = satellite_.numMTQ();
    const int num_rw = satellite_.numRW();

    if (num_rw > 0) {
        double rw_u_max_avg = 0.0;
        for (int i = 0; i < num_rw; ++i) {
            rw_u_max_avg += std::abs(satellite_.getRW(i).u_max());
        }
        rw_u_max_avg /= static_cast<double>(num_rw);

        rw_command_fraction_limit_ = std::clamp(target_tau / std::max(1e-9, rw_u_max_avg), 0.03, 0.18);
        rw_excitation_fraction_ = 0.5 * rw_command_fraction_limit_;
        rw_rate_damping_gain_ = (rw_command_fraction_limit_ * rw_u_max_avg) / std::max(1e-3, omega_ref_rad_s_);
    } else {
        rw_command_fraction_limit_ = 0.0;
        rw_excitation_fraction_ = 0.0;
        rw_rate_damping_gain_ = 0.0;
    }

    if (num_mtq > 0) {
        double mtq_u_max_avg = 0.0;
        for (int i = 0; i < num_mtq; ++i) {
            mtq_u_max_avg += std::abs(satellite_.getMTQ(i).u_max());
        }
        mtq_u_max_avg /= static_cast<double>(num_mtq);

        const double mtq_torque_authority = mtq_u_max_avg * std::max(5e-6, expected_b_field_leo_);
        mtq_command_fraction_limit_ = std::clamp(target_tau / std::max(1e-12, mtq_torque_authority), 0.03, 0.25);
        mtq_excitation_fraction_ = 0.5 * mtq_command_fraction_limit_;
        mtq_rate_damping_gain_ = (mtq_command_fraction_limit_ * mtq_u_max_avg) / std::max(1e-3, omega_ref_rad_s_);
    } else {
        mtq_command_fraction_limit_ = 0.0;
        mtq_excitation_fraction_ = 0.0;
        mtq_rate_damping_gain_ = 0.0;
    }
}

}
