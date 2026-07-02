#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <saltro/math/frames.h>

namespace py = pybind11;

void bind_frames(py::module_& m) {
    m.def("gmst_rad",
          &saltro::math::gmst_rad,
          py::arg("jcentury"),
          "Compute Greenwich Mean Sidereal Time in radians.");

    m.def("eci_to_ecef_dcm",
          &saltro::math::eci_to_ecef_dcm,
          py::arg("jcentury"),
          "Return the ECI-to-ECEF direction cosine matrix.");

    m.def("ecef_to_eci_dcm",
          &saltro::math::ecef_to_eci_dcm,
          py::arg("jcentury"),
          "Return the ECEF-to-ECI direction cosine matrix.");
}
