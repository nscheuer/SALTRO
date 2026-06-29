#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <saltro/math/frames.h>

namespace py = pybind11;

void bind_frames(py::module_& m) {
    using namespace saltro::math;

    m.def("gmst_rad", &gmst_rad, py::arg("jcentury"),
          "Greenwich Mean Sidereal Time in radians, wrapped to [0, 2*pi).");

    m.def("eci_to_ecef_dcm", &eci_to_ecef_dcm, py::arg("jcentury"),
          "Direction cosine matrix rotating ECI vectors into the ECEF frame.");

    m.def("ecef_to_eci_dcm", &ecef_to_eci_dcm, py::arg("jcentury"),
          "Direction cosine matrix rotating ECEF vectors into the ECI frame.");
}
