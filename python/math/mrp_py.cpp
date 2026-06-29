#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <saltro/math/mrp.h>

namespace py = pybind11;

void bind_mrp(py::module_& m) {
    using namespace saltro::math;

    m.def("quatError", &quatError, py::arg("q_goal"), py::arg("q"),
          "Error quaternion q_goal^{-1} (x) q, sign-fixed so the scalar part >= 0.");

    m.def("quatToMRP", &quatToMRP, py::arg("q_err"),
          "Modified Rodrigues parameters 2 q_v / (1 + q0) for the error quaternion.");

    m.def("findGMat", &findGMat, py::arg("q"), py::arg("nRW"),
          "Reduced-state attitude map G(q): (6+nRW) x (7+nRW), with W^T(q) in the "
          "attitude block and identity blocks for angular velocity and RW momenta.");
}
