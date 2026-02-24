// PYBIND_DEPENDS: disturbance geometryconfig

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <saltro/pybind/disturbances/dragdisturbance.h>

namespace py = pybind11;
using namespace saltro::disturbances;

void bind_dragdisturbance(py::module_& m) {
    py::class_<DragDisturbance, Disturbance>(m, "DragDisturbance")
        .def(py::init<>(), "Construct a DragDisturbance with default configuration")
        .def(py::init<const GeometryConfig&>(),
             py::arg("config"),
             "Construct a DragDisturbance with a geometry configuration")
        .def("setGeometryConfig", &DragDisturbance::setGeometryConfig,
             py::arg("config"),
             "Set the geometry configuration")
        .def_property_readonly("geometryConfig", &DragDisturbance::geometryConfig,
                               "Get the geometry configuration")
        .def("torque",
             py::overload_cast<const DragDisturbance::BaseState&, const DisturbanceConfig&>(
                 &DragDisturbance::torque, py::const_),
             py::arg("x"),
             py::arg("dist_cfg"),
             "Compute drag torque using the config velocity")
        .def("torque",
             py::overload_cast<const DragDisturbance::BaseState&, const DisturbanceConfig&, const DragDisturbance::Vec3&>(
                 &DragDisturbance::torque, py::const_),
             py::arg("x"),
             py::arg("dist_cfg"),
             py::arg("v_body"),
             "Compute drag torque using provided body-frame velocity")
        .def("dtorque_dq",
             py::overload_cast<const DragDisturbance::BaseState&, const DisturbanceConfig&, const DragDisturbance::Vec3&, const DragDisturbance::Mat34&>(
                 &DragDisturbance::dtorque_dq, py::const_),
             py::arg("x"),
             py::arg("dist_cfg"),
             py::arg("v_body"),
             py::arg("dV_dq"),
             "Jacobian of drag torque with respect to quaternion")
        .def("ddtorque_dqdq",
                [](const DragDisturbance& self,
                    const DragDisturbance::BaseState& x,
                    const DisturbanceConfig& dist_cfg,
                    const DragDisturbance::Vec3& v_body,
                    const DragDisturbance::Mat34& dV_dq,
                    const std::array<DragDisturbance::Mat44, 3>& d2V_dq2) {
                     const auto H = self.ddtorque_dqdq(x, dist_cfg, v_body, dV_dq, d2V_dq2);
                     return std::array<DragDisturbance::Mat44, 3>{H.slice(0), H.slice(1), H.slice(2)};
                },
                py::arg("x"),
                py::arg("dist_cfg"),
                py::arg("v_body"),
                py::arg("dV_dq"),
                py::arg("d2V_dq2"),
                "Hessian of drag torque with respect to quaternion");
}
