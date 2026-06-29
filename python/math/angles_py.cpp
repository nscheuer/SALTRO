#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <saltro/math/angles.h>

namespace py = pybind11;

void bind_angles(py::module_& m) {
    using namespace saltro::math;

    m.def("wrap_to_2pi", &wrap_to_2pi, py::arg("rad"),
          "Wrap an angle in radians into [0, 2*pi).");

    m.def("wrap_to_360", &wrap_to_360, py::arg("deg"),
          "Wrap an angle in degrees into [0, 360).");

    m.def("deg2rad", &deg2rad, py::arg("deg"), "Convert degrees to radians.");

    m.def("rad2deg", &rad2deg, py::arg("rad"), "Convert radians to degrees.");

    m.def("rot_z", &rot_z, py::arg("rad"),
          "Rotation matrix for a rotation of `rad` about the z axis.");

    m.def("rotate_about_z", &rotate_about_z, py::arg("v"), py::arg("angle_rad"),
          "Rotate vector v about the z axis by angle_rad.");

    m.def(
        "unit_vector",
        [](const Eigen::Vector3d& v) {
            Eigen::Vector3d out;
            bool ok = unit_vector(v, out);
            return py::make_tuple(ok, out);
        },
        py::arg("v"),
        "Return (ok, unit_vector). ok is False if v is too small to normalize.");

    m.def("clamp", &clamp, py::arg("x"), py::arg("lo"), py::arg("hi"),
          "Clamp x to the interval [lo, hi].");
}
