// PYBIND_DEPENDS: plannersettings

#include <pybind11/pybind11.h>
#include <saltro/validation/validate_plannersettings.h>

namespace py = pybind11;

py::tuple validatePlannerSettings_py(const PlannerSettings& settings) {
    std::string error_msg;
    bool ok = saltro::validation::validatePlannerSettings(settings, error_msg);
    return py::make_tuple(ok, error_msg);
}

void bind_validate_plannersettings(py::module_& m) {
    m.def("validatePlannerSettings", &validatePlannerSettings_py, py::arg("settings"));
}
