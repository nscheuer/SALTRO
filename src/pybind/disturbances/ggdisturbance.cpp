#include <saltro/pybind/disturbances/ggdisturbance.h>

#include <array>
#include <cmath>
#include <saltro/constants/constants.h>
#include <saltro/math/quaternion.h>

namespace saltro::disturbances {

GGDisturbance::GGDisturbance(const Mat33& inertia) : inertia_(inertia) {
}

GGDisturbance::Vec3 GGDisturbance::torque(const BaseState& /*x*/, const DisturbanceConfig& dist_cfg) const {
    // This override is not useful for GG disturbance since position must be provided
    // Use the overload with r_body parameter instead
    if (!active_ || !dist_cfg.plan_for_gg) {
        return Vec3::Zero();
    }
    return Vec3::Zero();
}

GGDisturbance::Vec3 GGDisturbance::torque(const BaseState& /*x*/, const DisturbanceConfig& dist_cfg,
                                         const Vec3& r_body, const Mat33& J) const {
    if (!active_ || !dist_cfg.plan_for_gg) {
        return Vec3::Zero();
    }

    const double r_norm = r_body.norm();
    if (!std::isfinite(r_norm) || r_norm <= 0.0) {
        return Vec3::Zero();
    }

    const Vec3 r_hat = r_body / r_norm;
    const Vec3 nadir = -r_hat;
    
    const double mu_e = saltro::constants::MU_EARTH;
    const double const_term = 3.0 * mu_e / (r_norm * r_norm * r_norm);
    
    return const_term * nadir.cross(J * nadir);
}

GGDisturbance::Mat34 GGDisturbance::dtorque_dq(const BaseState& /*x*/, const DisturbanceConfig& dist_cfg,
                                               const Vec3& r_body, const Mat33& J,
                                               const Mat34& dr_dq) const {
    Mat34 jac = Mat34::Zero();
    if (!active_ || !dist_cfg.plan_for_gg) {
        return jac;
    }

    const double r_norm = r_body.norm();
    if (!std::isfinite(r_norm) || r_norm <= 0.0) {
        return jac;
    }

    const Vec3 r_hat = r_body / r_norm;
    const Vec3 nadir = -r_hat;
    
    const double mu_e = saltro::constants::MU_EARTH;
    const double const_term = 3.0 * mu_e / (r_norm * r_norm * r_norm);

    // Compute derivative of normalized vector: dr_hat/dq
    // dr_hat/dq = (I - r_hat * r_hat^T) / ||r|| * dr/dq
    const Mat33 proj = Mat33::Identity() - r_hat * r_hat.transpose();
    const Mat34 dr_hat_dq = (proj * dr_dq) / r_norm;
    const Mat34 dnadir_dq = -dr_hat_dq;

    // Compute derivative of norm: d||r||/dq = r^T / ||r|| * dr/dq
    const Eigen::RowVector4d dnorm_dq = (r_body.transpose() * dr_dq) / r_norm;

    // Compute derivative of const_term: dc/dq
    // c = 3*mu_e / ||r||^3, so dc/dq = -9*mu_e / ||r||^4 * d||r||/dq
    const Eigen::RowVector4d dc_dq = -9.0 * mu_e * dnorm_dq / (r_norm * r_norm * r_norm * r_norm);

    // Compute derivative of vector term: v = nadir × (J * nadir)
    // dv/dq = dnadir/dq × (J * nadir) + nadir × (J * dnadir/dq)
    const Vec3 J_nadir = J * nadir;
    Mat34 dv_dq = Mat34::Zero();
    for (int j = 0; j < 4; ++j) {
        dv_dq.col(j) = dnadir_dq.col(j).cross(J_nadir) + nadir.cross(J * dnadir_dq.col(j));
    }

    // Torque = c * v, so dT/dq = dc/dq ⊗ v + c * dv/dq
    const Vec3 vec_term = nadir.cross(J_nadir);
    for (int j = 0; j < 4; ++j) {
        jac.col(j) = dc_dq(j) * vec_term + const_term * dv_dq.col(j);
    }

    return jac;
}

GGDisturbance::T443 GGDisturbance::ddtorque_dqdq(const BaseState& /*x*/, const DisturbanceConfig& dist_cfg,
                                                 const Vec3& r_body, const Mat33& J, const Mat34& dr_dq,
                                                 const std::array<Mat44, 3>& d2r_dq2) const {
    // Factored chain rule:
    //   d2tau_l/dq_i dq_j = sum_ab (d2tau_l/dr_a dr_b)(dr_a/dq_i)(dr_b/dq_j)
    //                     + sum_a  (dtau_l/dr_a)(d2r_a/dq_i dq_j)
    // The quaternion dependence is entirely in dr_dq / d2r_dq2 (the rotation
    // Jacobian/Hessian helpers); the torque's dependence on the body vector r is
    // clean 3-D calculus. This replaces the previous hand-expanded form, which
    // was only correct on the qv-qv block at the identity quaternion and wrong
    // in every q0-coupled term off identity (it disagreed with finite
    // differences of the renormalizing dynamics at non-identity attitudes).
    //
    // Gravity-gradient torque as a function of the (un-normalized) body position
    // vector r:  tau = 3*mu*(r x J r) / |r|^5   (the two nadir = -r/|r| signs
    // cancel, and one 1/|r|^2 folds the normalization into the 1/|r|^3 factor).
    T443 H = T443::Zero();
    if (!active_ || !dist_cfg.plan_for_gg) {
        return H;
    }
    const double rn = r_body.norm();
    if (!std::isfinite(rn) || rn <= 0.0) {
        return H;
    }

    const double k = 3.0 * saltro::constants::MU_EARTH;
    const double r5 = std::pow(rn, -5);
    const double r7 = std::pow(rn, -7);
    const double r9 = std::pow(rn, -9);

    const Vec3 g = r_body.cross(J * r_body);                                  // g = r x Jr
    const Mat33 dg = saltro::math::skewSymmetric(r_body) * J
                   - saltro::math::skewSymmetric(J * r_body);                 // dg/dr  (l,a)

    // Body-vector torque Jacobian  dtau_l/dr_a
    const Mat33 dtau_dr = k * (r5 * dg - 5.0 * r7 * g * r_body.transpose());

    // Constant pieces of g_{l,ab} = d2g_l/dr_a dr_b = (skew(e_b)J - skew(J e_b))(l,a)
    std::array<Mat33, 3> gab;  // gab[b](l,a)
    for (int b = 0; b < 3; ++b) {
        Vec3 eb = Vec3::Zero();
        eb(b) = 1.0;
        gab[static_cast<size_t>(b)] =
            saltro::math::skewSymmetric(eb) * J - saltro::math::skewSymmetric(J * eb);
    }

    for (int l = 0; l < 3; ++l) {
        // Body-vector torque Hessian  d2tau_l/dr_a dr_b
        Mat33 d2tau_l;
        for (int a = 0; a < 3; ++a) {
            const double ha = -5.0 * r7 * r_body(a);
            for (int b = 0; b < 3; ++b) {
                const double hb = -5.0 * r7 * r_body(b);
                const double hab = -5.0 * r7 * (a == b ? 1.0 : 0.0)
                                 + 35.0 * r9 * r_body(a) * r_body(b);
                d2tau_l(a, b) = k * (hb * dg(l, a)
                                     + r5 * gab[static_cast<size_t>(b)](l, a)
                                     + dg(l, b) * ha
                                     + g(l) * hab);
            }
        }
        // Contract with the quaternion Jacobian/Hessian of r.
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                double val = 0.0;
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        val += d2tau_l(a, b) * dr_dq(a, i) * dr_dq(b, j);
                    }
                    val += dtau_dr(l, a) * d2r_dq2[static_cast<size_t>(a)](i, j);
                }
                H.slice(l)(i, j) = val;
            }
        }
    }

    return H;
}

}  // namespace saltro::disturbances
