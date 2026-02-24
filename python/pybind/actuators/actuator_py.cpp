#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <saltro/pybind/actuators/actuator.h>

namespace py = pybind11;

void bind_actuator(py::module_& m) {
    py::class_<Actuator>(m, "Actuator")
        .def(py::init<const Actuator::Vec3&, double>(),
             py::arg("axis"),
             py::arg("u_max"),
             R"doc(
Construct an Actuator base class.

Parameters
----------
axis : ndarray (3,)
    Actuator axis direction (will be normalized)
u_max : float
    Maximum control input magnitude
)doc")
        .def_property_readonly("axis", &Actuator::axis,
                               "Get the normalized actuator axis")
        .def_property_readonly("u_max", &Actuator::u_max,
                               "Get the maximum control input")
        .def("clamp", &Actuator::clamp,
             py::arg("u"),
             "Clamp control input to valid range [-u_max, u_max]")
        .def("torque", &Actuator::torque,
             py::arg("u"),
             py::arg("x"),
             "Compute actuator torque (base implementation returns zero)")
        .def("dtorq_du", &Actuator::dtorq_du,
             py::arg("u"),
             py::arg("x"),
             "Jacobian of torque with respect to control input")
        .def("dtorq_dbasestate", &Actuator::dtorq_dbasestate,
             py::arg("u"),
             py::arg("x"),
             "Jacobian of torque with respect to base state")
        .def("ddtorq_dudu", &Actuator::ddtorq_dudu,
             py::arg("u"),
             py::arg("x"),
             "Hessian of torque with respect to control input (twice)")
        .def("ddtorq_dudbasestate", &Actuator::ddtorq_dudbasestate,
             py::arg("u"),
             py::arg("x"),
             "Hessian of torque with respect to control input and base state")
        .def("ddtorq_dbasestatedbasestate", &Actuator::ddtorq_dbasestatedbasestate,
             py::arg("u"),
             py::arg("x"),
             "Hessian of torque with respect to base state (twice)")
        .def_readonly_static("input_len", &Actuator::input_len,
                             "Number of control inputs (always 1)");
}
