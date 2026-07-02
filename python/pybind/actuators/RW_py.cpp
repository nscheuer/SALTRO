// PYBIND_DEPENDS: actuator

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include "../tensor_py.h"  // Tensor3 type caster for ddtorq_* Hessian return types
#include <saltro/pybind/actuators/RW.h>

namespace py = pybind11;

void bind_RW(py::module_& m) {
    py::class_<RW, Actuator>(m, "RW")
        .def(py::init<const RW::Vec3&, double, double, double, double>(),
             py::arg("axis"),
             py::arg("max_torque"),
             py::arg("J"),
             py::arg("h0"),
             py::arg("h_max"),
             R"doc(
Construct a Reaction Wheel (RW).

Parameters
----------
axis : ndarray (3,)
    Wheel spin axis direction (will be normalized)
max_torque : float
    Maximum torque output
J : float
    Wheel moment of inertia (must be > 0)
h0 : float
    Initial angular momentum
h_max : float
    Maximum angular momentum (must be >= 0)
)doc")
        .def_property_readonly("wheelInertia", &RW::wheelInertia,
                               "Get the wheel moment of inertia")
        .def_property("momentum", &RW::momentum, &RW::setMomentum,
                      "Get or set the wheel angular momentum")
        .def_property_readonly("momentumMax", &RW::momentumMax,
                               "Get the maximum angular momentum")
        .def("torque",
             py::overload_cast<double, const RW::BaseState&>(&RW::torque, py::const_),
             py::arg("u"),
             py::arg("x"),
             R"doc(
Compute RW torque on spacecraft.

The torque is: τ = axis * u

Parameters
----------
u : float
    Control input (commanded torque)
x : ndarray (7,)
    Base state [position (3), quaternion (4)]

Returns
-------
ndarray (3,)
    Torque vector in body frame
)doc")
        .def("storageTorque", &RW::storageTorque,
             py::arg("u"),
             py::arg("x"),
             R"doc(
Compute the rate of change of stored angular momentum.

Returns -u (momentum increases opposite to spacecraft torque).

Parameters
----------
u : float
    Control input
x : ndarray (7,)
    Base state

Returns
-------
ndarray (1, 1)
    Momentum storage rate
)doc")
        .def("dtorq_du",
             py::overload_cast<double, const RW::BaseState&>(&RW::dtorq_du, py::const_),
             py::arg("u"),
             py::arg("x"),
             "Jacobian of torque with respect to control input")
        .def("dtorq_dbasestate",
             py::overload_cast<double, const RW::BaseState&>(&RW::dtorq_dbasestate, py::const_),
             py::arg("u"),
             py::arg("x"),
             "Jacobian of torque with respect to base state")
        .def("dstor_torq_du", &RW::dstor_torq_du,
             py::arg("u"),
             py::arg("x"),
             "Jacobian of storage torque with respect to control input")
        .def("dstor_torq_dbasestate", &RW::dstor_torq_dbasestate,
             py::arg("u"),
             py::arg("x"),
             "Jacobian of storage torque with respect to base state")
        .def("ddtorq_dudu",
             py::overload_cast<double, const RW::BaseState&>(&RW::ddtorq_dudu, py::const_),
             py::arg("u"),
             py::arg("x"),
             "Hessian of torque w.r.t. control input (zero — RW torque is affine in u).")
        .def("ddtorq_dudbasestate",
             py::overload_cast<double, const RW::BaseState&>(&RW::ddtorq_dudbasestate, py::const_),
             py::arg("u"),
             py::arg("x"),
             "Mixed Hessian ∂²τ/(∂u ∂x) (zero — RW torque has no base-state dependence).")
        .def("ddtorq_dbasestatedbasestate",
             py::overload_cast<double, const RW::BaseState&>(&RW::ddtorq_dbasestatedbasestate, py::const_),
             py::arg("u"),
             py::arg("x"),
             "Hessian of torque w.r.t. base state (zero — RW torque has no base-state dependence).");
}
