#pragma once

#include <array>
#include <Eigen/Dense>

namespace saltro::math {

/**
 * @brief A 3D tensor with compile-time dimensions stored as an array of slices.
 * 
 * This template provides a fixed-size 3D tensor where each "slice" along the
 * third dimension is an R×C Eigen matrix. Indexing follows (i, j, k) where
 * k selects the slice.
 * 
 * @tparam R Number of rows in each slice
 * @tparam C Number of columns in each slice  
 * @tparam D Number of slices (depth)
 */
template<int R, int C, int D>
class Tensor3 {
public:
    using Slice = Eigen::Matrix<double, R, C>;

    /**
     * @brief Default constructor; initializes all elements to zero.
     */
    Tensor3() { setZero(); }

    /**
     * @brief Static factory method to create a zero-initialized tensor.
     * 
     * @return A new Tensor3 with all elements set to zero.
     */
    static Tensor3 Zero() { return Tensor3(); }

    /**
     * @brief Set all elements of the tensor to zero.
     */
    void setZero() {
        for (auto& s : data_) s.setZero();
    }

    /**
     * @brief Check if all elements of the tensor are zero.
     * 
     * @return True if every element is zero, false otherwise.
     */
    bool isZero() const {
        for (const auto& s : data_) {
            if (!s.isZero()) return false;
        }
        return true;
    }

    /**
     * @brief Access a slice (2D matrix) along the third dimension.
     * 
     * Returns a reference to the k-th R×C matrix slice.
     * 
     * @param k Slice index (0 ≤ k < D).
     * @return Mutable reference to the k-th slice.
     */
    Slice& slice(int k) { return data_[static_cast<std::size_t>(k)]; }
    
    /**
     * @brief Access a slice (2D matrix) along the third dimension (const version).
     * 
     * @param k Slice index (0 ≤ k < D).
     * @return Const reference to the k-th slice.
     */
    const Slice& slice(int k) const { return data_[static_cast<std::size_t>(k)]; }

    /**
     * @brief Access a single element by 3D index (i, j, k).
     * 
     * Returns a mutable reference to element at row i, column j, slice k.
     * 
     * @param i Row index (0 ≤ i < R).
     * @param j Column index (0 ≤ j < C).
     * @param k Slice index (0 ≤ k < D).
     * @return Mutable reference to the element.
     */
    double& operator()(int i, int j, int k) {
        return data_[static_cast<std::size_t>(k)](i, j);
    }

    /**
     * @brief Access a single element by 3D index (const version).
     * 
     * @param i Row index (0 ≤ i < R).
     * @param j Column index (0 ≤ j < C).
     * @param k Slice index (0 ≤ k < D).
     * @return Const reference to the element.
     */
    const double& operator()(int i, int j, int k) const {
        return data_[static_cast<std::size_t>(k)](i, j);
    }

private:
    std::array<Slice, D> data_{};
};

}  // namespace saltro::math
