#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <saltro/optimizer/validation/validate_orbit.h>

namespace py = pybind11;

py::tuple validateOrbitInitialConditions_py(const Eigen::Vector3d& r0, const Eigen::Vector3d& v0) {
    std::string error_msg;
    bool ok = saltro::optimizer::validation::validateOrbitInitialConditions(r0, v0, error_msg);
    return py::make_tuple(ok, error_msg);
}

void bind_validate_orbit(py::module_& m) {
    m.def("validateOrbitInitialConditions", &validateOrbitInitialConditions_py, 
          py::arg("r0"), py::arg("v0"));
}
