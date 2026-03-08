#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <saltro/validation/validate_orbitstate.h>

namespace py = pybind11;

py::tuple validateOrbitState_py(
    const Eigen::Vector3d& r0,
    const Eigen::Vector3d& v0
) {
    std::string error_msg;
    bool ok = saltro::validation::validateOrbitState(r0, v0, error_msg);
    return py::make_tuple(ok, error_msg);
}

void bind_validate_orbitstate(py::module_& m) {
    m.def("validateOrbitState", &validateOrbitState_py,
          py::arg("r0"), py::arg("v0"),
          R"doc(
Validate initial orbit state for LEO use.

Expected units:
- r0 in meters
- v0 in meters per second

Checks include:
- finite vectors
- likely unit mistakes (km/km/s vs m/m/s)
- bound Earth orbit (negative specific energy)
- low eccentricity (not highly elliptical)
- LEO envelope (perigee/apogee altitude bounds)

Returns
-------
tuple[bool, str]
    (is_valid, error_message)
)doc");
}