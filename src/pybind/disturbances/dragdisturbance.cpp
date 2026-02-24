#include <saltro/pybind/disturbances/dragdisturbance.h>

#include <array>
#include <cmath>

namespace saltro::disturbances {

DragDisturbance::DragDisturbance(const GeometryConfig& config) : config_(config) {
}

DragDisturbance::Vec3 DragDisturbance::torque(const BaseState& /*x*/, const DisturbanceConfig& dist_cfg) const {
    if (!active_ || !dist_cfg.plan_for_aero) {
        return Vec3::Zero();
    }

    const Vec3 v_body = dist_cfg.drag_coeff;
    const double v_norm = v_body.norm();
    if (!std::isfinite(v_norm) || v_norm <= 0.0) {
        return Vec3::Zero();
    }

    Vec3 summed_lever = Vec3::Zero();
    const size_t n_faces = config_.numFaces();

    for (size_t i = 0; i < n_faces; ++i) {
        const GeometryFace& face = config_.getFace(i);
        if (face.area <= 0.0 || face.CD <= 0.0) {
            continue;
        }

        const double incidence = face.normal.dot(v_body);
        if (incidence <= 0.0) {
            continue;
        }

        summed_lever += (face.CD * face.area * incidence) * face.centroid;
    }

    return -0.5 * summed_lever.cross(v_body);
}

DragDisturbance::Vec3 DragDisturbance::torque(const BaseState& /*x*/, const DisturbanceConfig& dist_cfg,
                                              const Vec3& v_body) const {
    if (!active_ || !dist_cfg.plan_for_aero) {
        return Vec3::Zero();
    }

    const double v_norm = v_body.norm();
    if (!std::isfinite(v_norm) || v_norm <= 0.0) {
        return Vec3::Zero();
    }

    Vec3 summed_lever = Vec3::Zero();
    const size_t n_faces = config_.numFaces();

    for (size_t i = 0; i < n_faces; ++i) {
        const GeometryFace& face = config_.getFace(i);
        if (face.area <= 0.0 || face.CD <= 0.0) {
            continue;
        }

        const double incidence = face.normal.dot(v_body);
        if (incidence <= 0.0) {
            continue;
        }

        summed_lever += (face.CD * face.area * incidence) * face.centroid;
    }

    return -0.5 * summed_lever.cross(v_body);
}

DragDisturbance::Mat34 DragDisturbance::dtorque_dq(const BaseState& /*x*/, const DisturbanceConfig& dist_cfg,
                                                   const Vec3& v_body, const Mat34& dV_dq) const {
    Mat34 J = Mat34::Zero();
    if (!active_ || !dist_cfg.plan_for_aero) {
        return J;
    }

    const double v_norm = v_body.norm();
    if (!std::isfinite(v_norm) || v_norm <= 0.0) {
        return J;
    }

    Vec3 C = Vec3::Zero();
    Mat34 dC_dq = Mat34::Zero();
    const size_t n_faces = config_.numFaces();

    for (size_t i = 0; i < n_faces; ++i) {
        const GeometryFace& face = config_.getFace(i);
        if (face.area <= 0.0 || face.CD <= 0.0) {
            continue;
        }

        const double incidence = face.normal.dot(v_body);
        if (incidence <= 0.0) {
            continue;
        }

        const double coeff = face.CD * face.area;
        const double F = coeff * incidence;
        const Eigen::RowVector4d dF_dq = coeff * (face.normal.transpose() * dV_dq);

        C += F * face.centroid;
        dC_dq += face.centroid * dF_dq;
    }

    for (int j = 0; j < 4; ++j) {
        J.col(j) = -(0.5 * (dC_dq.col(j).cross(v_body) + C.cross(dV_dq.col(j))));
    }

    return J;
}

DragDisturbance::T443 DragDisturbance::ddtorque_dqdq(const BaseState& /*x*/, const DisturbanceConfig& dist_cfg,
                                                     const Vec3& v_body, const Mat34& dV_dq,
                                                     const std::array<Mat44, 3>& d2V_dq2) const {
    T443 H = T443::Zero();
    if (!active_ || !dist_cfg.plan_for_aero) {
        return H;
    }

    const double v_norm = v_body.norm();
    if (!std::isfinite(v_norm) || v_norm <= 0.0) {
        return H;
    }

    Vec3 C = Vec3::Zero();
    Mat34 dC_dq = Mat34::Zero();
    std::array<Mat44, 3> ddC_dqdq = {Mat44::Zero(), Mat44::Zero(), Mat44::Zero()};
    const size_t n_faces = config_.numFaces();

    for (size_t i = 0; i < n_faces; ++i) {
        const GeometryFace& face = config_.getFace(i);
        if (face.area <= 0.0 || face.CD <= 0.0) {
            continue;
        }

        const double incidence = face.normal.dot(v_body);
        if (incidence <= 0.0) {
            continue;
        }

        const double coeff = face.CD * face.area;
        const double F = coeff * incidence;
        const Eigen::RowVector4d dF_dq = coeff * (face.normal.transpose() * dV_dq);

        Mat44 d2F_dq2 = Mat44::Zero();
        for (int k = 0; k < 3; ++k) {
            d2F_dq2 += coeff * face.normal(k) * d2V_dq2[static_cast<size_t>(k)];
        }

        C += F * face.centroid;
        dC_dq += face.centroid * dF_dq;
        for (int k = 0; k < 3; ++k) {
            ddC_dqdq[static_cast<size_t>(k)] += face.centroid(k) * d2F_dq2;
        }
    }

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            const Vec3 ddC_ij(ddC_dqdq[0](i, j), ddC_dqdq[1](i, j), ddC_dqdq[2](i, j));
            const Vec3 d2V_ij(d2V_dq2[0](i, j), d2V_dq2[1](i, j), d2V_dq2[2](i, j));

            Vec3 term = Vec3::Zero();
            term += ddC_ij.cross(v_body);
            term += dC_dq.col(i).cross(dV_dq.col(j));
            term += dC_dq.col(j).cross(dV_dq.col(i));
            term += C.cross(d2V_ij);

            H.slice(0)(i, j) = -0.5 * term(0);
            H.slice(1)(i, j) = -0.5 * term(1);
            H.slice(2)(i, j) = -0.5 * term(2);
        }
    }

    return H;
}

}  // namespace saltro::disturbances
