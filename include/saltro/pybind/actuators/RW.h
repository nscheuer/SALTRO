#pragma once

#include <saltro/pybind/actuators/actuator.h>
#include <Eigen/Dense>

class RW final : public Actuator {
public:
    using Vec3 = Actuator::Vec3;
    using BaseState = Actuator::BaseState;

    using Mat13 = Actuator::Mat13;
    using Mat73 = Actuator::Mat73;
    using Mat11 = Eigen::Matrix<double, 1, 1>;
    using Mat17 = Eigen::Matrix<double, 1, 7>;
    using Mat71 = Eigen::Matrix<double, 7, 1>;

    using T113 = Actuator::T113;
    using T173 = Actuator::T173;
    using T773 = Actuator::T773;

    RW(const Vec3& axis, double max_torque, double J, double h0, double h_max);
    ~RW() override = default;

    double wheelInertia() const noexcept;
    double momentum() const noexcept;
    double momentumMax() const noexcept;
    void setMomentum(double h) noexcept;

    // Torque methods (follow base class signature)
    Vec3 torque(double u, const BaseState& x) const override;

    // Storage (momentum rate) methods - RW-specific
    Mat11 storageTorque(double u, const BaseState& x) const;

    // Jacobian methods - torque derivatives
    Mat13 dtorq_du(double u, const BaseState& x) const override;
    Mat73 dtorq_dbasestate(double u, const BaseState& x) const override;

    // Jacobian methods - storage derivatives (RW-specific)
    Mat11 dstor_torq_du(double u, const BaseState& x) const;
    Mat71 dstor_torq_dbasestate(double u, const BaseState& x) const;

    // Hessian methods inherited from base class
    using Actuator::ddtorq_dudu;
    using Actuator::ddtorq_dudbasestate;
    using Actuator::ddtorq_dbasestatedbasestate;

private:
    static constexpr int kBaseStateDim = 7;
    static constexpr int kStorageDim = 1;

    double J_{0.0};
    double h_{0.0};
    double h_max_{0.0};
};