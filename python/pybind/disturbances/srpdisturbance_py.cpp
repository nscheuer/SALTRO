// PYBIND_DEPENDS: disturbance geometryconfig

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <saltro/pybind/disturbances/srpdisturbance.h>

namespace py = pybind11;
using namespace saltro::disturbances;

void bind_srpdisturbance(py::module_& m) {
    py::class_<SRPDisturbance, Disturbance>(m, "SRPDisturbance")
        .def(py::init<>(), "Construct a SRPDisturbance with default configuration")
        .def(py::init<const GeometryConfig&>(),
             py::arg("config"),
             "Construct a SRPDisturbance with a geometry configuration")
        .def("setGeometryConfig", &SRPDisturbance::setGeometryConfig,
             py::arg("config"),
             "Set the geometry configuration")
        .def_property_readonly("geometryConfig", &SRPDisturbance::geometryConfig,
                               "Get the geometry configuration")
        .def("torque",
             py::overload_cast<const SRPDisturbance::BaseState&, const DisturbanceConfig&>(
                 &SRPDisturbance::torque, py::const_),
             py::arg("x"),
             py::arg("dist_cfg"),
                "Compute SRP torque (zero if inactive or in eclipse)")
        .def("torque",
             py::overload_cast<const SRPDisturbance::BaseState&, const DisturbanceConfig&, const SRPDisturbance::Vec3&>(
                 &SRPDisturbance::torque, py::const_),
             py::arg("x"),
             py::arg("dist_cfg"),
             py::arg("v_body"),
             "Compute SRP torque using provided body-frame sun vector; zero if near-zero")
        .def("dtorque_dq",
             py::overload_cast<const SRPDisturbance::BaseState&, const DisturbanceConfig&, const SRPDisturbance::Vec3&, const SRPDisturbance::Mat34&>(
                 &SRPDisturbance::dtorque_dq, py::const_),
             py::arg("x"),
             py::arg("dist_cfg"),
             py::arg("v_body"),
             py::arg("dV_dq"),
             "Jacobian of SRP torque with respect to quaternion")
        .def("ddtorque_dqdq",
             [](const SRPDisturbance& self,
                const SRPDisturbance::BaseState& x,
                const DisturbanceConfig& dist_cfg,
                const SRPDisturbance::Vec3& v_body,
                const SRPDisturbance::Mat34& dV_dq,
                const std::array<SRPDisturbance::Mat44, 3>& d2V_dq2) {
                 const auto H = self.ddtorque_dqdq(x, dist_cfg, v_body, dV_dq, d2V_dq2);
                 return std::array<SRPDisturbance::Mat44, 3>{H.slice(0), H.slice(1), H.slice(2)};
             },
             py::arg("x"),
             py::arg("dist_cfg"),
             py::arg("v_body"),
             py::arg("dV_dq"),
             py::arg("d2V_dq2"),
             "Hessian of SRP torque with respect to quaternion");
}
