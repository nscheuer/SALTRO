#pragma once

#include <array>
#include <Eigen/Dense>
#include <saltro/pybind/disturbances/disturbance.h>

namespace saltro::disturbances {

class GGDisturbance : public ::Disturbance {
public:
    using Vec3 = Eigen::Vector3d;
    using Mat33 = Eigen::Matrix3d;
    using BaseState = ::Disturbance::BaseState;
    using Mat34 = ::Disturbance::Mat34;
    using Mat44 = ::Disturbance::Mat44;
    using T443 = ::Disturbance::T443;

    GGDisturbance() = default;
    explicit GGDisturbance(const Mat33& inertia);

    void setInertia(const Mat33& inertia) { inertia_ = inertia; }
    const Mat33& inertia() const { return inertia_; }

    Vec3 torque(const BaseState& x, const DisturbanceConfig& dist_cfg) const override;

    Vec3 torque(const BaseState& x, const DisturbanceConfig& dist_cfg, const Vec3& r_body,
                const Mat33& J) const;
    Mat34 dtorque_dq(const BaseState& x, const DisturbanceConfig& dist_cfg, const Vec3& r_body,
                     const Mat33& J, const Mat34& dr_dq) const;
    T443 ddtorque_dqdq(const BaseState& x, const DisturbanceConfig& dist_cfg, const Vec3& r_body,
                       const Mat33& J, const Mat34& dr_dq,
                       const std::array<Mat44, 3>& d2r_dq2) const;

private:
    Mat33 inertia_ = Mat33::Identity();
};

}  // namespace saltro::disturbances
