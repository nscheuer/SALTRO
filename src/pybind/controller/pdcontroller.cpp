#include <saltro/pybind/controller/pdcontroller.h>

#include <algorithm>
#include <cmath>

#include <saltro/limits.h>
#include <saltro/math/quaternion.h>
#include <saltro/math/mrp.h>

namespace saltro::controller {

namespace {

/// Vector-goal sentinel: q_goal[0] = NaN means the remaining three entries are
/// the inertial-frame target direction r̂_eci (see Satellite::stageCost).
inline bool isECIFormat(const Eigen::Vector4d& q_goal) {
    return std::isnan(q_goal(0));
}

}  // namespace

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

void PDController::setGoalRate(const Eigen::Vector3d& omega_des) {
    omega_des_ = omega_des;
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
    const int n_mtq = satellite_.numMTQ();
    const int n_rw = satellite_.numRW();
    const int n_magic = satellite_.numMagic();
    const int nu = satellite_.controlDim();

    Satellite::VecX u = Satellite::VecX::Zero(std::max(nu, 0));
    if (nu <= 0 || nu > saltro::limits::MAX_CTRL_DIM) return u;
    if (x.size() < Satellite::QUAT_INDEX + 4) return u;

    const Eigen::Vector3d omega = x.segment<3>(Satellite::AV_INDEX);
    const Eigen::Vector4d q = x.segment<4>(Satellite::QUAT_INDEX);
    const double qn = q.norm();
    if (!std::isfinite(qn) || qn <= 1e-12) return u;

    // Goal-rate feedforward (OldPlanner smartbdot wkdes): damp ω toward the
    // desired body rate ω_des instead of toward zero.  omega_err = ω - ω_des.
    // When ω_des is unset (NaN, the default), this reduces to ω_err = ω.
    Eigen::Vector3d omega_err = omega;
    if (omega_des_.allFinite()) {
        omega_err = omega - omega_des_;
    }

    // Compute desired body-frame torque.  Two cases:
    //
    // (a) Quaternion goal (`q_goal` = unit quaternion):  classic
    //         τ_des = −k_p · q_err.vec − k_d · ω
    //     where `q_err = quatError(q_goal, q)` and the vector part of the
    //     error quaternion is, for small errors, ≈ ½·θ_err·axis.
    //
    // (b) Vector goal (`q_goal[0] = NaN`, `q_goal[1:4] = r̂_eci`):  pure
    //     boresight-pointing.  Drive `bs_body` → `R(q)^T · r̂_eci = r_body`.
    //     The body-frame torque that rotates `bs` toward `r_body` is the
    //     cross-product `bs × r_body` (magnitude `sin(θ_err)`, direction
    //     perpendicular to both):
    //         τ_des = +k_p · (bs × r_body) − k_d · ω.
    //     Note the sign vs case (a): the vec error vector points TOWARD the
    //     goal, while q_err.vec points AWAY (hence the minus in case (a)).
    //
    // Without this branch, calling PDController with a vector goal feeds
    // `NaN` through `quatError` → `τ_des = NaN` → the `!u_raw.allFinite()`
    // guard returns `u = 0` at every knot, silently degenerating to the
    // ZeroController behavior.
    Eigen::Vector3d tau_des;
    if (isECIFormat(q_goal)) {
        Eigen::Vector3d r_eci = q_goal.tail<3>();
        const double rn = r_eci.norm();
        if (!std::isfinite(rn) || rn <= 1e-12) return u;
        r_eci /= rn;

        Eigen::Vector3d bs = boresight_body;
        const double bsn = bs.norm();
        if (!std::isfinite(bsn) || bsn <= 1e-12) return u;
        bs /= bsn;

        // r_body = R(q)^T · r̂_eci  (target direction expressed in body frame)
        const Eigen::Matrix3d R_T = saltro::math::rotationMatrix(q).transpose();
        const Eigen::Vector3d r_body = R_T * r_eci;

        tau_des = kp_q_ * bs.cross(r_body) - kd_w_ * omega_err;
    } else {
        const Eigen::Vector4d q_err = saltro::math::quatError(q_goal, q);
        tau_des = -kp_q_ * q_err.tail<3>() - kd_w_ * omega_err;
    }

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
    for (int i = 0; i < n_magic; ++i) {
        const int ui = n_mtq + n_rw + i;
        const double u_max = std::abs(satellite_.getMagic(i).u_max());
        if (u_max > 0.0) {
            const double r = std::abs(u_raw(ui)) / u_max;
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
