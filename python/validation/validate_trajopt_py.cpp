// PYBIND_DEPENDS: plannersettings satellite

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <string>

#include <saltro/validation/validate_trajOpt.h>

namespace py = pybind11;

py::tuple validatetrajOpt_py(
    const PlannerSettings& settings,
    const Satellite& satellite,
    const Eigen::Ref<const Eigen::VectorXd>& x0,
    const Eigen::Vector3d& r0,
    const Eigen::Vector3d& v0,
    const Eigen::Ref<const Eigen::VectorXd>& jtime,
    const Eigen::Ref<const Eigen::MatrixXd>& q_goal,
    const Eigen::Ref<const Eigen::MatrixXd>& boresight,
    int state_dim,
    int input_dim,
    int N
) {
    std::string error_msg;
    const bool ok = saltro::validation::validatetrajOpt(
        settings, satellite, x0, r0, v0, jtime, q_goal, boresight,
        state_dim, input_dim, N, error_msg);
    return py::make_tuple(ok, error_msg);
}

void bind_validate_trajopt(py::module_& m) {
    m.def(
        "validatetrajOpt",
        &validatetrajOpt_py,
        py::arg("settings"),
        py::arg("satellite"),
        py::arg("x0"),
        py::arg("r0"),
        py::arg("v0"),
        py::arg("jtime"),
        py::arg("q_goal"),
        py::arg("boresight"),
        py::arg("state_dim"),
        py::arg("input_dim"),
        py::arg("N"),
        R"doc(
Validate all inputs to the trajectory optimization problem.

Top-level gate-keeper for a planning problem: chains the per-field
validators (settings/satellite/x0/orbit/jtime/q_goal/boresight) and then
the cross-context dimension checks (N, state_dim, input_dim, x0 size,
jtime length, q_goal/boresight shapes).

Parameters
----------
settings : PlannerSettings
satellite : Satellite
x0 : ndarray (state_dim,)
    Initial state vector
r0 : ndarray (3,)
    Initial ECI position [m]
v0 : ndarray (3,)
    Initial ECI velocity [m/s]
jtime : ndarray (N,)
    Julian times (centuries since J2000)
q_goal : ndarray (4,N)
    Goal quaternion sequence
boresight : ndarray (3,N)
    Boresight direction sequence
state_dim : int
    Expected state dimension (must equal satellite.stateDim)
input_dim : int
    Expected control dimension (must equal satellite.controlDim)
N : int
    Expected horizon length

Returns
-------
tuple[bool, str]
    (is_valid, error_message)
)doc");
}
