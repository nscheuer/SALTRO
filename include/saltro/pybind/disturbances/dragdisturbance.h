#pragma once

#include <array>
#include <Eigen/Dense>
#include <saltro/pybind/disturbances/disturbance.h>
#include <saltro/pybind/disturbances/geometryconfig.h>

namespace saltro::disturbances {

class DragDisturbance : public ::Disturbance {
public:
    using Vec3 = Eigen::Vector3d;
    using BaseState = ::Disturbance::BaseState;
    using Mat34 = ::Disturbance::Mat34;
    using Mat44 = ::Disturbance::Mat44;
    using T443 = ::Disturbance::T443;

    DragDisturbance() = default;
    explicit DragDisturbance(const GeometryConfig& config);

    void setGeometryConfig(const GeometryConfig& config) { config_ = config; }
    const GeometryConfig& geometryConfig() const { return config_; }

    Vec3 torque(const BaseState& x, const DisturbanceConfig& dist_cfg) const override;

    Vec3 torque(const BaseState& x, const DisturbanceConfig& dist_cfg, const Vec3& v_body) const;
    Mat34 dtorque_dq(const BaseState& x, const DisturbanceConfig& dist_cfg, const Vec3& v_body,
                     const Mat34& dV_dq) const;
    T443 ddtorque_dqdq(const BaseState& x, const DisturbanceConfig& dist_cfg, const Vec3& v_body,
                       const Mat34& dV_dq, const std::array<Mat44, 3>& d2V_dq2) const;

private:
    GeometryConfig config_;
};

}  // namespace saltro::disturbances
