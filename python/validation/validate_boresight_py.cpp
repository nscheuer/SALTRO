#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <saltro/validation/validate_boresight.h>

namespace py = pybind11;

py::tuple validateBoresight_py(const Eigen::Ref<const Eigen::MatrixXd>& boresight) {
    std::string error_msg;
    const bool ok = saltro::validation::validateBoresight(boresight, error_msg);
    return py::make_tuple(ok, error_msg);
}

void bind_validate_boresight(py::module_& m) {
    m.def("validateBoresight", &validateBoresight_py, py::arg("boresight"),
          R"doc(
Validate boresight history matrix.

Expected shape is (3, N), one boresight direction per column.
Each column must be a finite unit vector (norm ~= 1).

Validation checks include:
- matrix has 3 rows and at least 1 column
- all elements are finite (no NaN/Inf)
- each column has norm ~= 1 (within tolerance)

Parameters
----------
boresight : ndarray
    Boresight history matrix of shape (3, N).
    Each column is a unit vector representing the boresight direction.

Returns
-------
tuple[bool, str]
    (is_valid, error_message) where is_valid is True if boresight is valid,
    and error_message contains details if validation failed
)doc");
}
