#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <saltro/math/mrp.h>

namespace py = pybind11;

void bind_mrp(py::module_& m) {
    m.def("quatError",
          &saltro::math::quatError,
          py::arg("q_goal"),
          py::arg("q"),
          "Compute the quaternion error q_goal^{-1} x q.");

    m.def("quatToMRP",
          &saltro::math::quatToMRP,
          py::arg("q_err"),
          "Convert a quaternion error into Modified Rodrigues Parameters.");

    m.def("findGMat",
          &saltro::math::findGMat,
          py::arg("q"),
          py::arg("nRW"),
          "Build the full-state to reduced-state projection matrix.");
}
