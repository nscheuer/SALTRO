// PYBIND_DEPENDS: disturbance

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <saltro/pybind/disturbances/ggdisturbance.h>

namespace py = pybind11;
using namespace saltro::disturbances;

void bind_ggdisturbance(py::module_& m) {
    py::class_<GGDisturbance, Disturbance>(m, "GGDisturbance")
        .def(py::init<>(), "Construct a GGDisturbance with default configuration")
        .def(py::init<const GGDisturbance::Mat33&>(),
             py::arg("inertia"),
             "Construct a GGDisturbance with an inertia tensor")
        .def("setInertia", &GGDisturbance::setInertia,
             py::arg("inertia"),
             "Set the inertia tensor")
        .def_property_readonly("inertia", &GGDisturbance::inertia,
                               "Get the inertia tensor")
        .def("torque",
             py::overload_cast<const GGDisturbance::BaseState&, const DisturbanceConfig&>(
                 &GGDisturbance::torque, py::const_),
             py::arg("x"),
             py::arg("dist_cfg"),
             "Compute gravity-gradient torque using the config position")
        .def("torque",
             py::overload_cast<const GGDisturbance::BaseState&, const DisturbanceConfig&, const GGDisturbance::Vec3&, const GGDisturbance::Mat33&>(
                 &GGDisturbance::torque, py::const_),
             py::arg("x"),
             py::arg("dist_cfg"),
             py::arg("r_body"),
             py::arg("J"),
             "Compute gravity-gradient torque using provided body-frame position and inertia")
        .def("dtorque_dq",
             py::overload_cast<const GGDisturbance::BaseState&, const DisturbanceConfig&, const GGDisturbance::Vec3&, const GGDisturbance::Mat33&, const GGDisturbance::Mat34&>(
                 &GGDisturbance::dtorque_dq, py::const_),
             py::arg("x"),
             py::arg("dist_cfg"),
             py::arg("r_body"),
             py::arg("J"),
             py::arg("dr_dq"),
             "Jacobian of gravity-gradient torque with respect to quaternion")
        .def("ddtorque_dqdq",
                [](const GGDisturbance& self,
                    const GGDisturbance::BaseState& x,
                    const DisturbanceConfig& dist_cfg,
                    const GGDisturbance::Vec3& r_body,
                    const GGDisturbance::Mat33& J,
                    const GGDisturbance::Mat34& dr_dq,
                    const std::array<GGDisturbance::Mat44, 3>& d2r_dq2) {
                     const auto H = self.ddtorque_dqdq(x, dist_cfg, r_body, J, dr_dq, d2r_dq2);
                     return std::array<GGDisturbance::Mat44, 3>{H.slice(0), H.slice(1), H.slice(2)};
                },
                py::arg("x"),
                py::arg("dist_cfg"),
                py::arg("r_body"),
                py::arg("J"),
                py::arg("dr_dq"),
                py::arg("d2r_dq2"),
                "Hessian of gravity-gradient torque with respect to quaternion");
}
