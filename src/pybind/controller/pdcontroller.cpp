#include <saltro/pybind/controller/pdcontroller.h>

#include <algorithm>
#include <cmath>

#include <saltro/limits.h>
#include <saltro/math/quaternion.h>
#include <saltro/math/mrp.h>

namespace saltro::controller {

PDController::PDController(const Satellite& satellite)
    : Controller(satellite) {
    autoTuneGains();
}

void PDController::setGains(double kp_q, double kd_w) {
    kp_q_ = kp_q;
    kd_w_ = kd_w;
}

void PDController::setRWScale(double rw_scale) {
    rw_scale_ = std::clamp(rw_scale, 0.0, 1.0);
}

void PDController::autoTuneGains() {
    const double J_mean = satellite_.inertia().trace() / 3.0;
    const double omega_n = 0.1;
    const double zeta = 0.7;
    kp_q_ = 2.0 * J_mean * omega_n * omega_n;
    kd_w_ = 2.0 * zeta * std::sqrt(J_mean * kp_q_);
    rw_scale_ = 1.0;
}

Satellite::VecX PDController::find_u(
    const Satellite::VecX& x,
    const Eigen::Vector3d& B_eci,
    const Eigen::Vector4d& q_goal,
    const Eigen::Vector3d& boresight_body
) const {
    (void)boresight_body;

    const int n_mtq = satellite_.numMTQ();
    const int n_rw = satellite_.numRW();
    const int nu = n_mtq + n_rw;

    Satellite::VecX u = Satellite::VecX::Zero(std::max(nu, 0));
    if (nu <= 0 || nu > saltro::limits::MAX_CTRL_DIM) return u;
    if (x.size() < Satellite::QUAT_INDEX + 4) return u;

    const Eigen::Vector3d omega = x.segment<3>(Satellite::AV_INDEX);
    const Eigen::Vector4d q = x.segment<4>(Satellite::QUAT_INDEX);
    const double qn = q.norm();
    if (!std::isfinite(qn) || qn <= 1e-12) return u;

    const Eigen::Vector4d q_err = saltro::math::quatError(q_goal, q);
    const Eigen::Vector3d tau_des = -kp_q_ * q_err.tail<3>() - kd_w_ * omega;

    // Numerical actuator Jacobian J = ∂τ/∂u  (3 × nu) via finite differences.
    const Eigen::VectorXd u_zero = Eigen::VectorXd::Zero(nu);
    const Eigen::Vector3d tau_zero = satellite_.actuatorTorque(x, u_zero, B_eci);

    Eigen::MatrixXd J(3, nu);
    Eigen::VectorXd authority(nu);
    for (int i = 0; i < nu; ++i) {
        Eigen::VectorXd u_unit = Eigen::VectorXd::Zero(nu);
        u_unit(i) = 1.0;
        const Eigen::Vector3d tau_i = satellite_.actuatorTorque(x, u_unit, B_eci);
        J.col(i) = tau_i - tau_zero;
        authority(i) = J.col(i).norm();
    }

    // Authority-weighted Tikhonov: high-authority channels penalized less.
    // reg_i = base_reg * (authority_max / authority_i)^2.  Channels with zero
    // authority (e.g., MTQ aligned with B_body) get very high weight so they
    // do not dominate the solution despite being un-actuated.
    const double base_reg = 1e-6;
    const double auth_max = authority.maxCoeff();
    Eigen::VectorXd reg_weights(nu);
    if (auth_max < 1e-12) {
        return u;
    }
    for (int i = 0; i < nu; ++i) {
        if (authority(i) < 1e-9 * auth_max) {
            reg_weights(i) = 1e6;
        } else {
            const double ratio = auth_max / authority(i);
            reg_weights(i) = base_reg * ratio * ratio;
        }
    }

    // RW preference: when rw_scale_ < 1, penalize RW channels.  rw_scale=0
    // gives effectively infinite weight (MTQ-only allocation); rw_scale=1
    // leaves reg_weights untouched (equal priority).
    if (rw_scale_ < 1.0) {
        const double rw_penalty = (rw_scale_ <= 0.0)
            ? 1e6
            : 1.0 / (rw_scale_ * rw_scale_);
        for (int i = 0; i < n_rw; ++i) {
            reg_weights(n_mtq + i) *= rw_penalty;
        }
    }

    Eigen::MatrixXd JtJ = J.transpose() * J;
    for (int i = 0; i < nu; ++i) {
        JtJ(i, i) += reg_weights(i);
    }

    Eigen::VectorXd u_raw;
    try {
        u_raw = JtJ.ldlt().solve(J.transpose() * tau_des);
    } catch (...) {
        return u;
    }
    if (!u_raw.allFinite()) return u;

    // Scale-to-max-saturation: uniform scaling preserves the torque direction.
    // Independent per-channel clipping would distort the direction away from
    // the desired torque.
    double max_ratio = 1.0;
    for (int i = 0; i < n_mtq; ++i) {
        const double u_max = std::abs(satellite_.getMTQ(i).u_max());
        if (u_max > 0.0) {
            const double r = std::abs(u_raw(i)) / u_max;
            if (r > max_ratio) max_ratio = r;
        }
    }
    for (int i = 0; i < n_rw; ++i) {
        const double u_max = std::abs(satellite_.getRW(i).u_max());
        if (u_max > 0.0) {
            const double r = std::abs(u_raw(n_mtq + i)) / u_max;
            if (r > max_ratio) max_ratio = r;
        }
    }
    if (max_ratio > 1.0) {
        u_raw /= max_ratio;
    }

    u = u_raw;
    return u;
}

}
