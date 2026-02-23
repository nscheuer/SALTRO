#include <saltro/pybind/actuators/actuator.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

Actuator::Actuator(const Vec3 &axis, double u_max) : axis_(normalize(axis)), u_max_(std::abs(u_max)) {
    if (!std::isfinite(u_max_)) {
        throw std::invalid_argument("u_max must be finite");
    }
}

const Actuator::Vec3& Actuator::axis() const noexcept {
    return axis_;
}

double Actuator::u_max() const noexcept {
    return u_max_;
}

double Actuator::clamp(double u) const noexcept {
    return std::clamp(u, -u_max_, u_max_);
}

Actuator::Vec3 Actuator::normalize(const Vec3& v) {
    if (!v.allFinite()) {
        throw std::invalid_argument("Axis vector must be finite");
    }

    const double n = v.norm();
    if (n < std::numeric_limits<double>::epsilon()) {
        throw std::invalid_argument("Axis vector must be non-zero");
    }
    return v / n;
}

Actuator::Vec3 Actuator::torque(double, const BaseState&) const {
    return Vec3::Zero();
}

Actuator::Mat13 Actuator::dtorq_du(double, const BaseState&) const {
    return Mat13::Zero();
}

Actuator::Mat73 Actuator::dtorq_dbasestate(double, const BaseState&) const {
    return Mat73::Zero();
}

Actuator::T113 Actuator::ddtorq_dudu(double, const BaseState&) const {
    return T113::Zero();
}

Actuator::T173 Actuator::ddtorq_dudbasestate(double, const BaseState&) const {
    return T173::Zero();
}

Actuator::T773 Actuator::ddtorq_dbasestatedbasestate(double, const BaseState&) const {
    return T773::Zero();
}