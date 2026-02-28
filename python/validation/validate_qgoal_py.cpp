#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <saltro/optimizer/validation/validate_qgoal.h>

namespace py = pybind11;

py::tuple validateQGoal_py(
    const Eigen::Ref<const Eigen::MatrixXd>& q_goal_in,
    int jtime_length
) {
    if (q_goal_in.rows() != 4) {
        return py::make_tuple(false, std::string("q_goal must have 4 rows"));
    }

    if (jtime_length > q_goal_in.cols()) {
        return py::make_tuple(false, std::string("jtime_length exceeds q_goal columns"));
    }

    Eigen::Matrix<double, 4, saltro::limits::MAX_LENGTH_TRAJ> q_goal;
    q_goal.setZero();
    q_goal.leftCols(jtime_length) = q_goal_in.leftCols(jtime_length);

    std::string error_msg;
    bool ok = saltro::optimizer::validation::validateQGoal(q_goal, jtime_length, error_msg);
    return py::make_tuple(ok, error_msg);
}

void bind_validate_qgoal(py::module_& m) {
    m.def("validateQGoal", &validateQGoal_py, 
          py::arg("q_goal"), py::arg("jtime_length"));
}
