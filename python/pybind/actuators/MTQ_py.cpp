// PYBIND_DEPENDS: actuator

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "../tensor_py.h"  // Tensor3 type caster for ddtorq_* Hessian return types
#include <saltro/pybind/actuators/MTQ.h>

namespace py = pybind11;

void bind_MTQ(py::module_& m) {
    py::class_<MTQ, Actuator>(m, "MTQ")
        .def(py::init<const MTQ::Vec3&, double>(),
             py::arg("axis"),
             py::arg("max_dipole"),
             R"doc(
Construct a Magnetic Torquer (MTQ).

Parameters
----------
axis : ndarray (3,)
    MTQ axis direction (will be normalized)
max_dipole : float
    Maximum magnetic dipole moment
)doc")
        .def("torque",
             py::overload_cast<double, const MTQ::BaseState&, const MTQ::Vec3&>(&MTQ::torque, py::const_),
             py::arg("u"),
             py::arg("x"),
             py::arg("B_body"),
             R"doc(
Compute MTQ torque from magnetic interaction.

The torque is: τ = -B × (axis * u)

Parameters
----------
u : float
    Control input (magnetic dipole moment)
x : ndarray (7,)
    Base state [position (3), quaternion (4)]
B_body : ndarray (3,)
    Magnetic field vector in body frame

Returns
-------
ndarray (3,)
    Torque vector in body frame
)doc")
        .def("dtorq_du",
             py::overload_cast<double, const MTQ::BaseState&, const MTQ::Vec3&>(&MTQ::dtorq_du, py::const_),
             py::arg("u"),
             py::arg("x"),
             py::arg("B_body"),
             "Jacobian of torque with respect to control input")
        .def("dtorq_dbasestate",
             py::overload_cast<double, const MTQ::BaseState&, const MTQ::Vec3&, const Eigen::Matrix<double,4,3>&>(&MTQ::dtorq_dbasestate, py::const_),
             py::arg("u"),
             py::arg("x"),
             py::arg("B_body"),
             py::arg("dB_dq"),
             R"doc(
Jacobian of torque with respect to base state.

Parameters
----------
u : float
    Control input
x : ndarray (7,)
    Base state
B_body : ndarray (3,)
    Magnetic field in body frame
dB_dq : ndarray (4, 3)
    Jacobian of B with respect to quaternion

Returns
-------
ndarray (7, 3)
    Jacobian matrix
)doc")
        .def("ddtorq_dudu",
             py::overload_cast<double, const MTQ::BaseState&, const MTQ::Vec3&>(&MTQ::ddtorq_dudu, py::const_),
             py::arg("u"),
             py::arg("x"),
             py::arg("B_body"),
             "Hessian of torque with respect to control input (twice)")
        .def("ddtorq_dudbasestate",
             py::overload_cast<double, const MTQ::BaseState&, const MTQ::Vec3&, const Eigen::Matrix<double,4,3>&>(&MTQ::ddtorq_dudbasestate, py::const_),
             py::arg("u"),
             py::arg("x"),
             py::arg("B_body"),
             py::arg("dB_dq"),
             "Hessian of torque with respect to control input and base state")
        .def("ddtorq_dbasestatedbasestate",
             py::overload_cast<double, const MTQ::BaseState&, const Eigen::Matrix<double,4,3>&, const std::array<Eigen::Matrix<double,4,4>,3>&>(&MTQ::ddtorq_dbasestatedbasestate, py::const_),
             py::arg("u"),
             py::arg("x"),
             py::arg("dB_dq"),
             py::arg("d2B_dq2"),
             "Hessian of torque with respect to base state (twice)");
}
