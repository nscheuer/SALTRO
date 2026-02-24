#include <saltro/pybind/disturbances/srpdisturbance.h>

#include <array>
#include <cmath>
#include <saltro/constants/constants.h>

namespace saltro::disturbances {
namespace {

constexpr double kSunVecEps = 1e-12;

struct NormalizedDerivatives {
    SRPDisturbance::Vec3 u = SRPDisturbance::Vec3::Zero();
    SRPDisturbance::Mat34 du_dq = SRPDisturbance::Mat34::Zero();
    std::array<SRPDisturbance::Mat44, 3> d2u_dq2 = {SRPDisturbance::Mat44::Zero(),
                                                    SRPDisturbance::Mat44::Zero(),
                                                    SRPDisturbance::Mat44::Zero()};
    bool valid = false;
};

bool normalizeVec(const SRPDisturbance::Vec3& v, SRPDisturbance::Vec3& u) {
    const double n = v.norm();
    if (!std::isfinite(n) || n <= kSunVecEps) {
        u.setZero();
        return false;
    }
    u = v / n;
    return true;
}

NormalizedDerivatives normalizeVecWithJacobian(const SRPDisturbance::Vec3& v,
                                                const SRPDisturbance::Mat34& dv_dq) {
    NormalizedDerivatives out;

    const double n = v.norm();
    if (!std::isfinite(n) || n <= kSunVecEps) {
        return out;
    }

    out.u = v / n;
    const SRPDisturbance::Mat33 proj = SRPDisturbance::Mat33::Identity() - out.u * out.u.transpose();
    out.du_dq = (proj * dv_dq) / n;
    out.valid = true;
    return out;
}

NormalizedDerivatives normalizeVecWithDerivatives(const SRPDisturbance::Vec3& v,
                                                  const SRPDisturbance::Mat34& dv_dq,
                                                  const std::array<SRPDisturbance::Mat44, 3>& d2v_dq2) {
    NormalizedDerivatives out;

    const double n = v.norm();
    if (!std::isfinite(n) || n <= kSunVecEps) {
        return out;
    }

    out.u = v / n;

    const SRPDisturbance::Mat33 proj = SRPDisturbance::Mat33::Identity() - out.u * out.u.transpose();
    out.du_dq = (proj * dv_dq) / n;

    Eigen::RowVector4d dn_dq = (v.transpose() * dv_dq) / n;
    SRPDisturbance::Mat44 d2n_dq2 = SRPDisturbance::Mat44::Zero();
    for (int i = 0; i < 4; ++i) {
        const SRPDisturbance::Vec3 dv_i = dv_dq.col(i);
        for (int j = 0; j < 4; ++j) {
            const SRPDisturbance::Vec3 dv_j = dv_dq.col(j);
            const SRPDisturbance::Vec3 d2v_ij(d2v_dq2[0](i, j), d2v_dq2[1](i, j), d2v_dq2[2](i, j));
            const double term1 = (v.dot(d2v_ij) + dv_i.dot(dv_j)) / n;
            const double term2 = (v.dot(dv_i) * v.dot(dv_j)) / (n * n * n);
            d2n_dq2(i, j) = term1 - term2;
        }
    }

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            const SRPDisturbance::Vec3 d2v_ij(d2v_dq2[0](i, j), d2v_dq2[1](i, j), d2v_dq2[2](i, j));
            const SRPDisturbance::Vec3 dv_i = dv_dq.col(i);
            const SRPDisturbance::Vec3 dv_j = dv_dq.col(j);
            const double dn_i = dn_dq(i);
            const double dn_j = dn_dq(j);

            SRPDisturbance::Vec3 term = d2v_ij / n;
            term -= (dv_i * dn_j + dv_j * dn_i) / (n * n);
            term -= v * d2n_dq2(i, j) / (n * n);
            term += 2.0 * v * dn_i * dn_j / (n * n * n);

            out.d2u_dq2[0](i, j) = term(0);
            out.d2u_dq2[1](i, j) = term(1);
            out.d2u_dq2[2](i, j) = term(2);
        }
    }

    out.valid = true;
    return out;
}

}  // namespace

SRPDisturbance::SRPDisturbance(const GeometryConfig& config) : config_(config) {
}

SRPDisturbance::Vec3 SRPDisturbance::torque(const BaseState& /*x*/, const DisturbanceConfig& dist_cfg) const {
    if (!active_ || !dist_cfg.plan_for_srp) {
        return Vec3::Zero();
    }

    return Vec3::Zero();
}

SRPDisturbance::Vec3 SRPDisturbance::torque(const BaseState& /*x*/, const DisturbanceConfig& dist_cfg,
                                            const Vec3& v_body) const {
    if (!active_ || !dist_cfg.plan_for_srp) {
        return Vec3::Zero();
    }

    Vec3 s_hat = Vec3::Zero();
    if (!normalizeVec(v_body, s_hat)) {
        return Vec3::Zero();
    }

    Vec3 torque = Vec3::Zero();
    const size_t n_faces = config_.numFaces();

    for (size_t i = 0; i < n_faces; ++i) {
        const GeometryFace& face = config_.getFace(i);
        if (face.area <= 0.0) {
            continue;
        }

        const double n_norm = face.normal.norm();
        if (!std::isfinite(n_norm) || n_norm <= 0.0) {
            continue;
        }

        const Vec3 n_hat = face.normal / n_norm;
        const double incidence = n_hat.dot(s_hat);
        if (incidence <= 0.0) {
            continue;
        }

        const double proj_area = face.area * incidence;
        const double m_s = proj_area * (face.eta_a + face.eta_d);
        const double m_n = proj_area * (2.0 * face.eta_s * incidence + (2.0 / 3.0) * face.eta_d);

        const Vec3 lever = face.centroid;
        torque += m_s * lever.cross(s_hat) + m_n * lever.cross(n_hat);
    }

    const double P = saltro::constants::SOLAR_CONSTANT / saltro::constants::C_LIGHT;
    return -P * torque;
}

SRPDisturbance::Mat34 SRPDisturbance::dtorque_dq(const BaseState& /*x*/, const DisturbanceConfig& dist_cfg,
                                                 const Vec3& v_body, const Mat34& dV_dq) const {
    Mat34 jac = Mat34::Zero();
    if (!active_ || !dist_cfg.plan_for_srp) {
        return jac;
    }

    const auto normed = normalizeVecWithJacobian(v_body, dV_dq);
    if (!normed.valid) {
        return jac;
    }

    const Vec3 s_hat = normed.u;
    const Mat34 ds_dq = normed.du_dq;

    const size_t n_faces = config_.numFaces();
    for (size_t i = 0; i < n_faces; ++i) {
        const GeometryFace& face = config_.getFace(i);
        if (face.area <= 0.0) {
            continue;
        }

        const double n_norm = face.normal.norm();
        if (!std::isfinite(n_norm) || n_norm <= 0.0) {
            continue;
        }

        const Vec3 n_hat = face.normal / n_norm;
        const double incidence = n_hat.dot(s_hat);
        if (incidence <= 0.0) {
            continue;
        }

        const Eigen::RowVector4d dcos_dq = n_hat.transpose() * ds_dq;

        const double proj_area = face.area * incidence;
        const double m_s = proj_area * (face.eta_a + face.eta_d);
        const double m_n = proj_area * (2.0 * face.eta_s * incidence + (2.0 / 3.0) * face.eta_d);

        const Eigen::RowVector4d dm_s_dq = face.area * (face.eta_a + face.eta_d) * dcos_dq;
        const Eigen::RowVector4d dm_n_dq = face.area * (4.0 * face.eta_s * incidence + (2.0 / 3.0) * face.eta_d) * dcos_dq;

        const Vec3 lever = face.centroid;
        const Vec3 t_s = lever.cross(s_hat);
        const Vec3 t_n = lever.cross(n_hat);

        for (int j = 0; j < 4; ++j) {
            jac.col(j) += dm_s_dq(j) * t_s + m_s * lever.cross(ds_dq.col(j)) + dm_n_dq(j) * t_n;
        }
    }

    const double P = saltro::constants::SOLAR_CONSTANT / saltro::constants::C_LIGHT;
    return -P * jac;
}

SRPDisturbance::T443 SRPDisturbance::ddtorque_dqdq(const BaseState& /*x*/, const DisturbanceConfig& dist_cfg,
                                                   const Vec3& v_body, const Mat34& dV_dq,
                                                   const std::array<Mat44, 3>& d2V_dq2) const {
    T443 H = T443::Zero();
    if (!active_ || !dist_cfg.plan_for_srp) {
        return H;
    }

    const auto normed = normalizeVecWithDerivatives(v_body, dV_dq, d2V_dq2);
    if (!normed.valid) {
        return H;
    }

    const Vec3 s_hat = normed.u;
    const Mat34 ds_dq = normed.du_dq;
    const auto& d2s_dq2 = normed.d2u_dq2;

    const size_t n_faces = config_.numFaces();
    for (size_t i = 0; i < n_faces; ++i) {
        const GeometryFace& face = config_.getFace(i);
        if (face.area <= 0.0) {
            continue;
        }

        const double n_norm = face.normal.norm();
        if (!std::isfinite(n_norm) || n_norm <= 0.0) {
            continue;
        }

        const Vec3 n_hat = face.normal / n_norm;
        const double incidence = n_hat.dot(s_hat);
        if (incidence <= 0.0) {
            continue;
        }

        const Eigen::RowVector4d dcos_dq = n_hat.transpose() * ds_dq;
        Mat44 ddcos_dq2 = Mat44::Zero();
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                const Vec3 d2s_rc(d2s_dq2[0](r, c), d2s_dq2[1](r, c), d2s_dq2[2](r, c));
                ddcos_dq2(r, c) = n_hat.dot(d2s_rc);
            }
        }

        const double proj_area = face.area * incidence;
        const Eigen::RowVector4d dproj_area_dq = face.area * dcos_dq;
        const Mat44 ddproj_area_dq2 = face.area * ddcos_dq2;

        const double m_s = proj_area * (face.eta_a + face.eta_d);
        const Eigen::RowVector4d dm_s_dq = dproj_area_dq * (face.eta_a + face.eta_d);
        const Mat44 ddm_s_dq2 = ddproj_area_dq2 * (face.eta_a + face.eta_d);

        const double m_n = proj_area * (2.0 * face.eta_s * incidence + (2.0 / 3.0) * face.eta_d);
        const Eigen::RowVector4d dm_n_dq = dproj_area_dq * (2.0 * face.eta_s * incidence + (2.0 / 3.0) * face.eta_d)
                                         + proj_area * (2.0 * face.eta_s * dcos_dq);
        const Mat44 tmp = dproj_area_dq.transpose() * (2.0 * face.eta_s * dcos_dq);
        const Mat44 ddm_n_dq2 = ddproj_area_dq2 * (2.0 * face.eta_s * incidence + (2.0 / 3.0) * face.eta_d)
                              + tmp + tmp.transpose()
                              + proj_area * (2.0 * face.eta_s * ddcos_dq2);

        const Vec3 lever = face.centroid;
        const Vec3 t_s = lever.cross(s_hat);
        const Vec3 t_n = lever.cross(n_hat);

        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                const Vec3 d2s_rc(d2s_dq2[0](r, c), d2s_dq2[1](r, c), d2s_dq2[2](r, c));
                Vec3 term = ddm_s_dq2(r, c) * t_s;
                term += dm_s_dq(r) * lever.cross(ds_dq.col(c));
                term += dm_s_dq(c) * lever.cross(ds_dq.col(r));
                term += m_s * lever.cross(d2s_rc);
                term += ddm_n_dq2(r, c) * t_n;

                H.slice(0)(r, c) += term(0);
                H.slice(1)(r, c) += term(1);
                H.slice(2)(r, c) += term(2);
            }
        }
    }

    const double P = saltro::constants::SOLAR_CONSTANT / saltro::constants::C_LIGHT;
    H.slice(0) *= -P;
    H.slice(1) *= -P;
    H.slice(2) *= -P;
    return H;
}

}  // namespace saltro::disturbances
