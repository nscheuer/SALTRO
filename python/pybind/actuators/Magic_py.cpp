// PYBIND_DEPENDS: actuator

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include "../tensor_py.h"  // Tensor3 type caster for ddtorq_* Hessian return types
#include <saltro/pybind/actuators/Magic.h>

namespace py = pybind11;

void bind_Magic(py::module_& m) {
    py::class_<Magic, Actuator>(m, "Magic")
        .def(py::init<const Magic::Vec3&, double>(),
             py::arg("axis"),
             py::arg("max_torque"),
             R"doc(
Construct a "magic" (direct body-torque) actuator.

Magic actuators apply a body-frame torque ``τ = u * axis`` directly along
a fixed body axis, with no environmental dependence and no
momentum-storage state. They are typically used to model thrusters or
as test fixtures (idealised body-torque commanders without the
``m × B`` rank deficiency of an MTQ or the back-reaction inertia of a
reaction wheel).

Parameters
----------
axis : ndarray (3,)
    Torque axis direction in body frame (will be normalized).
max_torque : float
    Maximum torque magnitude (N·m).
)doc")
        .def("torque",
             py::overload_cast<double, const Magic::BaseState&>(&Magic::torque, py::const_),
             py::arg("u"),
             py::arg("x"),
             R"doc(
Compute body-frame torque produced by this magic actuator.

Returns ``u * axis`` -- exactly linear in the control input, with no
state or environment dependence.

Parameters
----------
u : float
    Control input (torque magnitude, N·m).
x : ndarray (7,)
    Base state (unused; included for interface consistency).

Returns
-------
ndarray (3,)
    Torque vector in body frame (N·m).
)doc")
        .def("dtorq_du",
             py::overload_cast<double, const Magic::BaseState&>(&Magic::dtorq_du, py::const_),
             py::arg("u"),
             py::arg("x"),
             "Jacobian of torque with respect to control input "
             "(constant ``axis^T`` row).")
        .def("dtorq_dbasestate",
             py::overload_cast<double, const Magic::BaseState&>(&Magic::dtorq_dbasestate, py::const_),
             py::arg("u"),
             py::arg("x"),
             "Jacobian of torque with respect to base state (zero — "
             "magic torque is independent of ω and q).")
        .def("ddtorq_dudu",
             py::overload_cast<double, const Magic::BaseState&>(&Magic::ddtorq_dudu, py::const_),
             py::arg("u"),
             py::arg("x"),
             "Hessian of torque w.r.t. control input (zero — affine in u).")
        .def("ddtorq_dudbasestate",
             py::overload_cast<double, const Magic::BaseState&>(&Magic::ddtorq_dudbasestate, py::const_),
             py::arg("u"),
             py::arg("x"),
             "Mixed Hessian ∂²τ/(∂u ∂x) (zero — no state dependence).")
        .def("ddtorq_dbasestatedbasestate",
             py::overload_cast<double, const Magic::BaseState&>(&Magic::ddtorq_dbasestatedbasestate, py::const_),
             py::arg("u"),
             py::arg("x"),
             "Hessian of torque w.r.t. base state (zero — no state dependence).");
}
