#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <saltro/pybind/disturbances/disturbance.h>

namespace py = pybind11;

void bind_disturbance(py::module_& m) {
    py::class_<Disturbance>(m, "Disturbance")
        .def("torque", &Disturbance::torque,
             py::arg("x"),
             py::arg("dist_cfg"),
             "Compute disturbance torque (base is abstract)")
        .def("dtorque_dq", &Disturbance::dtorque_dq,
             py::arg("x"),
             py::arg("dist_cfg"),
             "Jacobian of torque with respect to quaternion")
        .def("ddtorque_dqdq", &Disturbance::ddtorque_dqdq,
             py::arg("x"),
             py::arg("dist_cfg"),
             "Hessian of torque with respect to quaternion")
        .def("setActive", &Disturbance::setActive,
             py::arg("active"),
             "Enable or disable the disturbance")
        .def("isActive", &Disturbance::isActive,
             "Check whether the disturbance is active");
}
