#pragma once

#include <saltro/pybind/actuators/actuator.h>
#include <Eigen/Dense>

class MTQ final : public Actuator {
public:
    using Vec3 = Actuator::Vec3;
    using BaseState = Actuator::BaseState;

    using Mat13 = Actuator::Mat13;
    using Mat73 = Actuator::Mat73;

    using T113 = Actuator::T113;
    using T173 = Actuator::T173;
    using T773 = Actuator::T773;

    MTQ(const Vec3 &axis, double max_dipole);

    Vec3 torque(double u, const BaseState& x, const Vec3& B_body) const;

    Mat13 dtorq_du(double u, const BaseState& x, const Vec3& B_body) const;
    Mat73 dtorq_dbasestate(double u, const BaseState& x, const Vec3& B_body, const Eigen::Matrix<double, 4, 3>& dB_dq) const;

    T113 ddtorq_dudu(double u, const BaseState& x, const Vec3& B_body) const;

    T173 ddtorq_dudbasestate(double u, const BaseState& x, const Vec3& B_body, const Eigen::Matrix<double,4,3>& dB_dq) const;

    T773 ddtorq_dbasestatedbasestate(double u, const BaseState& x, const Eigen::Matrix<double,4,3>& dB_dq, const std::array<Eigen::Matrix<double,4,4>,3>& d2B_dq2) const;
};