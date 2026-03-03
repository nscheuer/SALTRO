#include <saltro/pybind/controller/integratedbdotcontroller.h>

#include <algorithm>
#include <cmath>

#include <saltro/math/quaternion.h>

namespace saltro::controller {

IntegratedBdotController::IntegratedBdotController(const Satellite& satellite)
    : Controller(satellite) {
    autoTuneGains();

    mtq_prev_bdot_.assign(satellite_.numMTQ(), 0.0);
    rw_prev_w_.assign(satellite_.numRW(), 0.0);
}

Satellite::VecX IntegratedBdotController::find_u(
    const Satellite::VecX& x,
    const Eigen::Vector3d& B_eci,
    const Eigen::Vector4d& q_goal,
    const Eigen::Vector3d& boresight_body
) const {
    (void)q_goal;
    (void)boresight_body;

    Satellite::VecX u(satellite_.controlDim());
    u.setZero();

    if (satellite_.controlDim() == 0 || x.size() < Satellite::QUAT_INDEX + 4) {
        return u;
    }

    const Eigen::Vector3d w = x.segment<3>(Satellite::AV_INDEX);
    Eigen::Vector4d q = x.segment<4>(Satellite::QUAT_INDEX);
    const double qn = q.norm();
    if (!std::isfinite(qn) || qn <= 1e-12) {
        return u;
    }
    q /= qn;

    const Eigen::Matrix3d R_T = saltro::math::rotationMatrix(q).transpose();
    const Eigen::Vector3d B_body = R_T * B_eci;

    const Eigen::Vector3d bdot_est = -w.cross(B_body);
    const double dt = std::max(1e-6, dt_seconds());

    for (int i = 0; i < satellite_.numMTQ(); ++i) {
        const MTQ& mtq = satellite_.getMTQ(i);
        const Eigen::Vector3d axis = mtq.axis();
        const double bdot_i = axis.dot(bdot_est);
        const double dbdot_i = (bdot_i - mtq_prev_bdot_[i]) / dt;

        const double raw = -mtq_kp_[i] * bdot_i - mtq_kd_[i] * dbdot_i;
        const double umax = std::abs(mtq.u_max());
        u(i) = std::clamp(raw, -umax, umax);

        mtq_prev_bdot_[i] = bdot_i;
    }

    for (int i = 0; i < satellite_.numRW(); ++i) {
        const int ui = satellite_.numMTQ() + i;
        const RW& rw = satellite_.getRW(i);
        const Eigen::Vector3d axis = rw.axis();

        const double w_i = axis.dot(w);
        const double dw_i = (w_i - rw_prev_w_[i]) / dt;

        const double raw = -rw_kp_[i] * w_i - rw_kd_[i] * dw_i;
        const double umax = std::abs(rw.u_max());
        u(ui) = std::clamp(raw, -umax, umax);

        rw_prev_w_[i] = w_i;
    }

    return u;
}

void IntegratedBdotController::autoTuneGains() {
    const double Javg = std::max(1e-6, satellite_.inertia().trace() / 3.0);
    const double wref = std::max(1e-3, max_rate_ref_);
    const double tau_target = Javg * wref / 25.0;
    const double dt_tune = std::max(1e-3, dt_seconds());

    mtq_kp_.assign(satellite_.numMTQ(), 0.0);
    mtq_kd_.assign(satellite_.numMTQ(), 0.0);
    rw_kp_.assign(satellite_.numRW(), 0.0);
    rw_kd_.assign(satellite_.numRW(), 0.0);

    const double Bref = std::max(5e-6, expected_b_field_leo_);
    const double bdot_ref = wref * Bref;

    for (int i = 0; i < satellite_.numMTQ(); ++i) {
        const MTQ& mtq = satellite_.getMTQ(i);
        const double umax = std::max(1e-9, std::abs(mtq.u_max()));

        const double m_demand = tau_target / Bref;
        // Keep command envelope conservative to avoid high-rate limit cycles.
        const double cmd_fraction = std::clamp(m_demand / umax, 0.05, 0.35);
        const double cmd_allow = cmd_fraction * umax;

        const double kp = cmd_allow / std::max(1e-12, bdot_ref);
        mtq_kp_[i] = kp;
        mtq_kd_[i] = 0.15 * kp * dt_tune;
    }

    for (int i = 0; i < satellite_.numRW(); ++i) {
        const RW& rw = satellite_.getRW(i);
        const double umax = std::max(1e-9, std::abs(rw.u_max()));

        const double cmd_fraction = std::clamp(tau_target / umax, 0.05, 0.35);
        const double cmd_allow = cmd_fraction * umax;

        const double kp = cmd_allow / wref;
        rw_kp_[i] = kp;
        rw_kd_[i] = 0.2 * kp * dt_tune;
    }
}

}
