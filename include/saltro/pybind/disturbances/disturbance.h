#pragma once

#include <Eigen/Dense>
#include <saltro/math/tensor.h>
#include <saltro/pybind/plannersettings.h>

class Disturbance {
public:
    using Vec3 = Eigen::Vector3d;
    using Vec4 = Eigen::Vector4d;
    using BaseState = Eigen::Matrix<double, 7, 1>;
    using Mat34 = Eigen::Matrix<double, 3, 4>;
    using Mat44 = Eigen::Matrix<double, 4, 4>;
    using T443 = saltro::math::Tensor3<4, 4, 3>;  // 3 slices of 4×4 matrices for Hessians

    virtual ~Disturbance() = default;

    virtual Vec3 torque(const BaseState& x, const DisturbanceConfig& dist_cfg) const = 0;

    virtual Mat34 dtorque_dq(const BaseState& x, const DisturbanceConfig& dist_cfg) const {
        return Mat34::Zero();
    }

    virtual T443 ddtorque_dqdq(const BaseState& x, const DisturbanceConfig& dist_cfg) const {
        return T443::Zero();
    }

    void setActive(bool active) { active_ = active; }
    bool isActive() const { return active_; }

protected:
    bool active_ = true;
};