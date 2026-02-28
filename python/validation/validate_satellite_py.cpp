// PYBIND_DEPENDS: satellite

#include <pybind11/pybind11.h>
#include <saltro/optimizer/validation/validate_satellite.h>

namespace py = pybind11;

py::tuple validateSatellite_py(const Satellite& satellite) {
    std::string error_msg;
    bool ok = saltro::optimizer::validation::validateSatellite(satellite, error_msg);
    return py::make_tuple(ok, error_msg);
}

void bind_validate_satellite(py::module_& m) {
    m.def("validateSatellite", &validateSatellite_py, py::arg("satellite"));
}
