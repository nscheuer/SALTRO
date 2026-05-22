#include <saltro/pybind/actuators/Magic.h>

Magic::Magic(const Vec3 &axis, double max_torque) : Actuator(axis, max_torque) {}

Magic::Vec3 Magic::torque(double u, const BaseState& x) const {
    (void)x;
    return axis_ * u;
}

Magic::Mat13 Magic::dtorq_du(double u, const BaseState& x) const {
    (void)u;
    (void)x;
    return axis_.transpose();
}

Magic::Mat73 Magic::dtorq_dbasestate(double u, const BaseState& x) const {
    (void)u;
    (void)x;
    return Mat73::Zero();
}

Magic::T113 Magic::ddtorq_dudu(double u, const BaseState& x) const {
    (void)u;
    (void)x;
    return T113::Zero();
}

Magic::T173 Magic::ddtorq_dudbasestate(double u, const BaseState& x) const {
    (void)u;
    (void)x;
    return T173::Zero();
}

Magic::T773 Magic::ddtorq_dbasestatedbasestate(double u, const BaseState& x) const {
    (void)u;
    (void)x;
    return T773::Zero();
}
