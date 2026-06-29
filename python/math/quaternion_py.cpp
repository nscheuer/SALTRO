#include <array>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <saltro/math/quaternion.h>

namespace py = pybind11;

void bind_quaternion(py::module_& m) {
    using namespace saltro::math;

    m.def("normalizeQuat", &normalizeQuat, py::arg("q"),
          "Return q scaled to unit norm (throws if q is near zero).");

    m.def("rotationMatrix", &rotationMatrix, py::arg("q"),
          "Rotation matrix R(q) for the (normalized) scalar-first quaternion q.");

    m.def("findWMat", &findWMat, py::arg("q"),
          "4x3 attitude kinematics matrix W(q) with qdot = 0.5 W(q) omega.");

    m.def("quatNormJacobian", &quatNormJacobian, py::arg("q"),
          "Jacobian d(q/|q|)/dq (4x3 block form).");

    m.def("skewSymmetric", &skewSymmetric, py::arg("v"),
          "Skew-symmetric matrix [v]_x such that [v]_x w == v cross w.");

    m.def("drotmatTvecdq", &drotmatTvecdq, py::arg("q"), py::arg("v"),
          "Jacobian d(R(q)^T v)/dq (4x3, unit-norm-constrained).");

    m.def(
        "ddrotmatTvecdqdq",
        [](const Vec4& q, const Vec3& v) {
            std::array<Mat44, 3> H = ddrotmatTvecdqdq(q, v);
            // Return as a list of three 4x4 matrices (one per output component).
            return std::vector<Mat44>(H.begin(), H.end());
        },
        py::arg("q"), py::arg("v"),
        "Second derivative d^2(R(q)^T v)/dq^2 as a list of three 4x4 matrices.");
}
