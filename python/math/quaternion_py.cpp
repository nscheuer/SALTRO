#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <saltro/math/quaternion.h>

namespace py = pybind11;

void bind_quaternion(py::module_& m) {
    m.def("normalizeQuat",
          &saltro::math::normalizeQuat,
          py::arg("q"),
          "Normalize a quaternion to unit length.");

    m.def("rotationMatrix",
          &saltro::math::rotationMatrix,
          py::arg("q"),
          "Convert a quaternion into a 3x3 rotation matrix.");

    m.def("findWMat",
          &saltro::math::findWMat,
          py::arg("q"),
          "Return the 4x3 quaternion kinematics matrix W(q).");

    m.def("quatNormJacobian",
          &saltro::math::quatNormJacobian,
          py::arg("q"),
          "Return the quaternion-normalization Jacobian used by the C++ math layer.");

    m.def("skewSymmetric",
          &saltro::math::skewSymmetric,
          py::arg("v"),
          "Return the skew-symmetric cross-product matrix for a 3D vector.");

    m.def("drotmatTvecdq",
          &saltro::math::drotmatTvecdq,
          py::arg("q"),
          py::arg("v"),
          "Return the derivative of R(q)^T v with respect to q.");

    m.def("ddrotmatTvecdqdq",
          &saltro::math::ddrotmatTvecdqdq,
          py::arg("q"),
          py::arg("v"),
          "Return the 3-component quaternion Hessian of R(q)^T v.");
}
