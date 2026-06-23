#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <saltro/math/matrix.h>

namespace py = pybind11;

void bind_matrix(py::module_& m) {
    m.def(
        "psd_clip",
        [](Eigen::MatrixXd matrix) {
            saltro::math::psd_clip(matrix);
            return matrix;
        },
        py::arg("matrix"),
        "Return a symmetric matrix with negative eigenvalues clipped to zero."
    );
}
