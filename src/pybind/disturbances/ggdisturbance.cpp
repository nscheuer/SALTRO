#include <saltro/pybind/disturbances/ggdisturbance.h>

#include <array>
#include <cmath>
#include <saltro/constants/constants.h>

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
    T443 H = T443::Zero();
    if (!active_ || !dist_cfg.plan_for_gg) {
        return H;
    }

    const double r_norm = r_body.norm();
    if (!std::isfinite(r_norm) || r_norm <= 0.0) {
        return H;
    }

    const Vec3 r_hat = r_body / r_norm;
    const Vec3 nadir = -r_hat;
    
    const double mu_e = saltro::constants::MU_EARTH;
    const double const_term = 3.0 * mu_e / (r_norm * r_norm * r_norm);

    // First derivatives
    const Mat33 proj = Mat33::Identity() - r_hat * r_hat.transpose();
    const Mat34 dr_hat_dq = (proj * dr_dq) / r_norm;
    const Mat34 dnadir_dq = -dr_hat_dq;
    const Eigen::RowVector4d dnorm_dq = (r_body.transpose() * dr_dq) / r_norm;
    const Eigen::RowVector4d dc_dq = -9.0 * mu_e * dnorm_dq / (r_norm * r_norm * r_norm * r_norm);

    // Second derivative of norm: d²||r||/dq²
    Mat44 d2norm_dq2 = Mat44::Zero();
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            const Vec3 d2r_ij(d2r_dq2[0](i, j), d2r_dq2[1](i, j), d2r_dq2[2](i, j));
            d2norm_dq2(i, j) = (r_body.dot(d2r_ij) + dr_dq.col(i).dot(dr_dq.col(j))) / r_norm
                             - (r_body.dot(dr_dq.col(i)) * r_body.dot(dr_dq.col(j))) / (r_norm * r_norm * r_norm);
        }
    }

    // Second derivative of const_term: d²c/dq²
    const Mat44 d2c_dq2 = -9.0 * mu_e * d2norm_dq2 / (r_norm * r_norm * r_norm * r_norm)
                        + 36.0 * mu_e * dnorm_dq.transpose() * dnorm_dq / (r_norm * r_norm * r_norm * r_norm * r_norm);

    // Second derivative of r_hat: d²r_hat/dq²
    std::array<Mat44, 3> d2r_hat_dq2 = {Mat44::Zero(), Mat44::Zero(), Mat44::Zero()};
    for (int k = 0; k < 3; ++k) {
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                const Vec3 d2r_ij(d2r_dq2[0](i, j), d2r_dq2[1](i, j), d2r_dq2[2](i, j));
                d2r_hat_dq2[static_cast<size_t>(k)](i, j) = 
                    (proj(k, 0) * d2r_ij(0) + proj(k, 1) * d2r_ij(1) + proj(k, 2) * d2r_ij(2)) / r_norm
                    - (proj.row(k) * dr_dq.col(i))(0) * dnorm_dq(j) / (r_norm * r_norm)
                    - (proj.row(k) * dr_dq.col(j))(0) * dnorm_dq(i) / (r_norm * r_norm)
                    - r_hat(k) * d2norm_dq2(i, j) / r_norm
                    + 2.0 * r_hat(k) * dnorm_dq(i) * dnorm_dq(j) / (r_norm * r_norm);
            }
        }
    }
    std::array<Mat44, 3> d2nadir_dq2 = {-d2r_hat_dq2[0], -d2r_hat_dq2[1], -d2r_hat_dq2[2]};

    // First derivative of vector term
    const Vec3 J_nadir = J * nadir;
    Mat34 dv_dq = Mat34::Zero();
    for (int j = 0; j < 4; ++j) {
        dv_dq.col(j) = dnadir_dq.col(j).cross(J_nadir) + nadir.cross(J * dnadir_dq.col(j));
    }

    // Second derivative of vector term: d²v/dq²
    std::array<Mat44, 3> d2v_dq2 = {Mat44::Zero(), Mat44::Zero(), Mat44::Zero()};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            const Vec3 d2nadir_ij(d2nadir_dq2[0](i, j), d2nadir_dq2[1](i, j), d2nadir_dq2[2](i, j));
            const Vec3 term = d2nadir_ij.cross(J_nadir)
                            + dnadir_dq.col(i).cross(J * dnadir_dq.col(j))
                            + dnadir_dq.col(j).cross(J * dnadir_dq.col(i))
                            + nadir.cross(J * d2nadir_ij);
            
            d2v_dq2[0](i, j) = term(0);
            d2v_dq2[1](i, j) = term(1);
            d2v_dq2[2](i, j) = term(2);
        }
    }

    // Compute Hessian: d²T/dq² = d²c/dq² ⊗ v + dc/dq ⊗ dv/dq + (dc/dq ⊗ dv/dq)^T + c * d²v/dq²
    const Vec3 vec_term = nadir.cross(J_nadir);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            const Vec3 hess_ij = d2c_dq2(i, j) * vec_term
                               + dc_dq(i) * dv_dq.col(j)
                               + dc_dq(j) * dv_dq.col(i)
                               + const_term * Vec3(d2v_dq2[0](i, j), d2v_dq2[1](i, j), d2v_dq2[2](i, j));
            
            H.slice(0)(i, j) = hess_ij(0);
            H.slice(1)(i, j) = hess_ij(1);
            H.slice(2)(i, j) = hess_ij(2);
        }
    }

    return H;
}

}  // namespace saltro::disturbances
