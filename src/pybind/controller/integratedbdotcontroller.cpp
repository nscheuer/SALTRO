#include <saltro/pybind/controller/integratedbdotcontroller.h>

#include <algorithm>
#include <cmath>

#include <saltro/math/quaternion.h>

namespace saltro::controller {

IntegratedBdotController::IntegratedBdotController(const Satellite& satellite)
    : Controller(satellite) {
    autoTuneGains();
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

    for (int i = 0; i < satellite_.numMTQ(); ++i) {
        const MTQ& mtq = satellite_.getMTQ(i);
        const Eigen::Vector3d axis = mtq.axis();
        const double bdot_i = axis.dot(bdot_est);

        u(i) = -mtq_kp_[i] * bdot_i;
    }

    for (int i = 0; i < satellite_.numRW(); ++i) {
        const int ui = satellite_.numMTQ() + i;
        const RW& rw = satellite_.getRW(i);
        const Eigen::Vector3d axis = rw.axis();

        const double w_i = axis.dot(w);

        u(ui) = -rw_kp_[i] * w_i;
    }

    return u;
}

void IntegratedBdotController::autoTuneGains() {
    const Eigen::Matrix3d J = satellite_.inertia();

    mtq_kp_.assign(satellite_.numMTQ(), 0.0);
    mtq_kd_.assign(satellite_.numMTQ(), 0.0);
    rw_kp_.assign(satellite_.numRW(), 0.0);
    rw_kd_.assign(satellite_.numRW(), 0.0);

    // Simple constant gain for MTQ: kp * J_axis
    const double mtq_gain_const = 1e8;
    for (int i = 0; i < satellite_.numMTQ(); ++i) {
        const MTQ& mtq = satellite_.getMTQ(i);
        const Eigen::Vector3d axis = mtq.axis();
        const double J_axis = (J * axis).dot(axis);
        mtq_kp_[i] = mtq_gain_const * J_axis;
    }

    // Simple constant gain for RW: kp * J_axis
    const double rw_gain_const = 1e-1;
    for (int i = 0; i < satellite_.numRW(); ++i) {
        const RW& rw = satellite_.getRW(i);
        const Eigen::Vector3d axis = rw.axis();
        const double J_axis = (J * axis).dot(axis);
        rw_kp_[i] = rw_gain_const * J_axis;
    }
}

}
