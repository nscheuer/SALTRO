#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <saltro/validation/validate_initialstate.h>

namespace py = pybind11;

py::tuple validateInitialState_py(const Eigen::Ref<const Eigen::VectorXd>& x0) {
    std::string error_msg;
    bool ok = saltro::validation::validateInitialState(x0, error_msg);
    return py::make_tuple(ok, error_msg);
}

void bind_validate_initialstate(py::module_& m) {
    m.def("validateInitialState", &validateInitialState_py, py::arg("x0"),
          R"doc(
Validate the initial state vector for trajectory optimization.

The state vector x0 contains:
- [0:3] Angular velocity (rad/s) in body frame
- [3:7] Quaternion (q0, q1, q2, q3) representing attitude
- [7:] (optional) Reaction wheel angular momenta

Validation checks:
1. Angular velocity:
   - Is finite (not NaN or infinity)
   - Magnitude is within reasonable limits (< 10 rad/s)

2. Quaternion:
   - Components are finite (not NaN)
   - Is normalized: |norm - 1.0| <= 1e-6

3. Reaction wheel momenta (if present):
   - Are finite (not NaN or infinity)

Parameters
----------
x0 : ndarray
    Initial state vector (minimum size 7)
    [angular_velocity (3), quaternion (4), rw_momenta (..., optional)]

Returns
-------
tuple[bool, str]
    (is_valid, error_message) where is_valid is True if state is valid,
    and error_message contains details if validation failed

Raises
------
ValueError
    If x0 has fewer than 7 elements
)doc");
}
