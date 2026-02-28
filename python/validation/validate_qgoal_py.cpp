#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <saltro/validation/validate_qgoal.h>

namespace py = pybind11;

py::tuple validateQGoal_py(const Eigen::Ref<const Eigen::MatrixXd>& q_goal) {
    std::string error_msg;
    const bool ok = saltro::validation::validateQGoal(q_goal, error_msg);
    return py::make_tuple(ok, error_msg);
}

void bind_validate_qgoal(py::module_& m) {
    m.def("validateQGoal", &validateQGoal_py, py::arg("q_goal"),
          R"doc(
Validate attitude goal matrix q_goal.

Input format per column:
- Quaternion goal: [q0, qx, qy, qz]
- ECI vector goal: [nan, x, y, z]

Both formats can be mixed across columns.

Returns
-------
tuple[bool, str]
    (is_valid, error_message)
)doc");
}
