#pragma once

#include <Eigen/Dense>
#include <saltro/math/tensor.h>

class Actuator {
public:
    using Vec3 = Eigen::Matrix<double, 3, 1>;
    using BaseState = Eigen::Matrix<double, 7, 1>;

    using Mat13 = Eigen::Matrix<double, 1, 3>;
    using Mat73 = Eigen::Matrix<double, 7, 3>;

    using T113 = saltro::math::Tensor3<1, 1, 3>;
    using T173 = saltro::math::Tensor3<1, 7, 3>;
    using T773 = saltro::math::Tensor3<7, 7, 3>;

    static constexpr int input_len = 1;

    Actuator() = delete;
    Actuator(const Vec3 &axis, double u_max);
    virtual ~Actuator() = default;

    const Vec3& axis() const noexcept;
    double u_max() const noexcept;

    double clamp(double u) const noexcept;

    virtual Vec3 torque(double u, const BaseState& x) const;

    virtual Mat13 dtorq_du(double u, const BaseState& x) const;
    virtual Mat73 dtorq_dbasestate(double u, const BaseState& x) const;

    virtual T113 ddtorq_dudu(double u, const BaseState& x) const;
    virtual T173 ddtorq_dudbasestate(double u, const BaseState& x) const;
    virtual T773 ddtorq_dbasestatedbasestate(double u, const BaseState& x) const;

protected:
    static Vec3 normalize(const Vec3& v);

    Vec3 axis_;
    double u_max_;
};