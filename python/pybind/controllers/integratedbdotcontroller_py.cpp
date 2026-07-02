// PYBIND_DEPENDS: satellite

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <saltro/pybind/controller/integratedbdotcontroller.h>

namespace py = pybind11;

void bind_integratedbdotcontroller(py::module_& m) {
    using saltro::controller::IntegratedBdotController;

    py::class_<IntegratedBdotController>(m, "IntegratedBdotController")
        .def(py::init<const Satellite&>(),
             py::arg("satellite"),
             R"doc(
Construct an integrated B-dot controller for the given satellite.

Parameters
----------
satellite : Satellite
    Satellite model that provides inertia and actuator configuration.
)doc")
        .def("find_u",
             &IntegratedBdotController::find_u,
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
    Goal quaternion or vector-goal sentinel.
boresight_body : ndarray (3,)
    Desired boresight direction in body frame.

Returns
-------
ndarray
    Control vector in actuator coordinates.
)doc");
}
