// PYBIND_DEPENDS: satellite

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <saltro/pybind/controller/pdcontroller.h>

namespace py = pybind11;

void bind_pdcontroller(py::module_& m) {
    using saltro::controller::PDController;

    py::class_<PDController>(m, "PDController")
        .def(py::init<const Satellite&>(),
             py::arg("satellite"),
             R"doc(
Construct a quaternion PD controller for the given satellite.

Parameters
----------
satellite : Satellite
    Satellite model that provides inertia and actuator configuration.
)doc")
        .def("find_u",
             &PDController::find_u,
             py::arg("x"),
             py::arg("B_eci"),
             py::arg("q_goal"),
             py::arg("boresight_body"),
             R"doc(
Compute the control command for the current state and environment.

Parameters
----------
x : ndarray
    Attitude state vector ``[ω, q, h_rw]``.
B_eci : ndarray (3,)
    Magnetic field in ECI frame.
q_goal : ndarray (4,)
    Goal quaternion or vector-goal sentinel ``[NaN, x, y, z]``.
boresight_body : ndarray (3,)
    Desired boresight direction in body frame.

Returns
-------
ndarray
    Control vector in actuator coordinates.
)doc")
        .def("setGains",
             &PDController::setGains,
             py::arg("kp_q"),
             py::arg("kd_w"),
             "Override the proportional and rate-damping gains.")
        .def("setRWScale",
             &PDController::setRWScale,
             py::arg("rw_scale"),
             "Set the reaction-wheel weighting preference in the allocator.")
        .def("setGoalRate",
             &PDController::setGoalRate,
             py::arg("omega_des"),
             "Set an optional body-rate feedforward target for damping.")
        .def_property_readonly("kp_q", &PDController::kp_q)
        .def_property_readonly("kd_w", &PDController::kd_w)
        .def_property_readonly("rwScale", &PDController::rwScale);
}
