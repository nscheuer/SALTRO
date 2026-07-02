#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <saltro/math/angles.h>

namespace py = pybind11;

void bind_angles(py::module_& m) {
    m.def("wrap_to_2pi",
          &saltro::math::wrap_to_2pi,
          py::arg("rad"),
          "Wrap an angle in radians to the interval [0, 2pi).");

    m.def("wrap_to_360",
          &saltro::math::wrap_to_360,
          py::arg("deg"),
          "Wrap an angle in degrees to the interval [0, 360).");

    m.def("deg2rad",
          &saltro::math::deg2rad,
          py::arg("deg"),
          "Convert degrees to radians.");

    m.def("rad2deg",
          &saltro::math::rad2deg,
          py::arg("rad"),
          "Convert radians to degrees.");

    m.def("rot_z",
          &saltro::math::rot_z,
          py::arg("rad"),
          "Return the right-handed rotation matrix about the z axis.");

    m.def("rotate_about_z",
          &saltro::math::rotate_about_z,
          py::arg("v"),
          py::arg("angle_rad"),
          "Rotate a 3D vector about the z axis.");

    m.def(
        "unit_vector",
        [](const Eigen::Vector3d& v) {
            Eigen::Vector3d out = Eigen::Vector3d::Zero();
            const bool ok = saltro::math::unit_vector(v, out);
            return py::make_tuple(ok, out);
        },
        py::arg("v"),
        "Return (ok, unit_v) for a 3D vector normalization request."
    );

    m.def("clamp",
          &saltro::math::clamp,
          py::arg("x"),
          py::arg("lo"),
          py::arg("hi"),
          "Clamp a scalar to the closed interval [lo, hi].");
}
