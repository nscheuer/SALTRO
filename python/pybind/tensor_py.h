#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <saltro/math/tensor.h>

namespace py = pybind11;

namespace pybind11 { namespace detail {

/**
 * @brief Type caster for saltro::math::Tensor3<R,C,D> to/from numpy arrays
 * 
 * Automatically converts between C++ Tensor3 objects and numpy 3D arrays.
 * The numpy array has shape (D, R, C) where D is the depth (number of slices),
 * R is rows, and C is columns.
 */
template <int R, int C, int D>
struct type_caster<saltro::math::Tensor3<R, C, D>> {
    using TensorType = saltro::math::Tensor3<R, C, D>;
    
    PYBIND11_TYPE_CASTER(TensorType, _("numpy.ndarray[float64]"));

    /**
     * @brief Convert Python numpy array to C++ Tensor3
     * 
     * @param src Python object (should be a numpy array)
     * @param convert Whether to allow type conversions
     * @return true if conversion succeeded
     */
    bool load(handle src, bool convert) {
        auto arr = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(src);
        if (!arr)
            return false;

        // Check dimensions
        if (arr.ndim() != 3)
            return false;
        
        if (arr.shape(0) != D || arr.shape(1) != R || arr.shape(2) != C)
            return false;

        // Copy data from numpy array to Tensor3
        auto buf = arr.unchecked<3>();
        for (int k = 0; k < D; ++k) {
            for (int i = 0; i < R; ++i) {
                for (int j = 0; j < C; ++j) {
                    value(i, j, k) = buf(k, i, j);
                }
            }
        }

        return true;
    }

    /**
     * @brief Convert C++ Tensor3 to Python numpy array
     * 
     * @param src C++ Tensor3 object
     * @param policy Return value policy
     * @param parent Parent object handle
     * @return Python numpy array with shape (D, R, C)
     */
    static handle cast(const TensorType& src, return_value_policy policy, handle parent) {
        // Create numpy array with shape (D, R, C)
        py::array_t<double> arr({D, R, C});
        auto buf = arr.mutable_unchecked<3>();

        // Copy data from Tensor3 to numpy array
        for (int k = 0; k < D; ++k) {
            const auto& slice = src.slice(k);
            for (int i = 0; i < R; ++i) {
                for (int j = 0; j < C; ++j) {
                    buf(k, i, j) = slice(i, j);
                }
            }
        }

        return arr.release();
    }
};

}} // namespace pybind11::detail
