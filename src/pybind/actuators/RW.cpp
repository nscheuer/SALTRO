#include <saltro/pybind/actuators/RW.h>

#include <stdexcept>
#include <cmath>

RW::RW(const Vec3& axis, double max_torque, double J, double h0, double h_max) 
    : Actuator(axis, max_torque), J_(J), h_(h0), h_max_(h_max) {
  // Axis and u_max validation/normalization is handled by base class Actuator constructor
  if (!(J_ > 0.0)) {
    throw std::invalid_argument("RW inertia J must be > 0");
  }
  if (!(h_max_ >= 0.0)) {
    throw std::invalid_argument("RW h_max must be >= 0");
  }
}

double RW::wheelInertia() const noexcept { 
  return J_; 
}

double RW::momentum() const noexcept { 
  return h_; 
}

double RW::momentumMax() const noexcept { 
  return h_max_; 
}

void RW::setMomentum(double h) noexcept {
  h_ = h;
}

RW::Vec3 RW::torque(double u, const BaseState& x) const {
  (void)x;  // unused
  return axis_ * u;
}

RW::Mat11 RW::storageTorque(double u, const BaseState& x) const {
  (void)x;  // unused
  Mat11 out;
  out(0, 0) = -u;
  return out;
}

RW::Mat13 RW::dtorq_du(double u, const BaseState& x) const {
  (void)u;  // unused
  (void)x;  // unused
  Mat13 J;
  J.row(0) = axis_.transpose();
  return J;
}

RW::Mat73 RW::dtorq_dbasestate(double u, const BaseState& x) const {
  (void)u;  // unused
  (void)x;  // unused
  return Mat73::Zero();
}

RW::Mat11 RW::dstor_torq_du(double u, const BaseState& x) const {
  (void)u;  // unused
  (void)x;  // unused
  Mat11 J;
  J(0, 0) = -1.0;
  return J;
}

RW::Mat71 RW::dstor_torq_dbasestate(double u, const BaseState& x) const {
  (void)u;  // unused
  (void)x;  // unused
  return Mat71::Zero();
}