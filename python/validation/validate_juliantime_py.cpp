#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <saltro/validation/validate_juliantime.h>

namespace py = pybind11;

py::tuple validateJulianTime_py(const Eigen::Ref<const Eigen::VectorXd>& jtime) {
    std::string error_msg;
    const bool ok = saltro::validation::validateJulianTime(jtime, error_msg);
    return py::make_tuple(ok, error_msg);
}

void bind_validate_juliantime(py::module_& m) {
    m.def("validateJulianTime", &validateJulianTime_py, py::arg("jtime"),
          R"doc(
Validate a time vector in Julian centuries.

Expected mission bounds are [0.20, 0.40] Julian centuries.
The vector must be finite, non-zero, and strictly increasing.

Returns
-------
tuple[bool, str]
    (is_valid, error_message)
)doc");
}
