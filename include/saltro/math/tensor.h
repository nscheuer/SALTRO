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

    Tensor3() { setZero(); }

    static Tensor3 Zero() { return Tensor3(); }

    void setZero() {
        for (auto& s : data_) s.setZero();
    }

    Slice& slice(int k) { return data_[static_cast<std::size_t>(k)]; }
    const Slice& slice(int k) const { return data_[static_cast<std::size_t>(k)]; }

    double& operator()(int i, int j, int k) {
        return data_[static_cast<std::size_t>(k)](i, j);
    }

    const double& operator()(int i, int j, int k) const {
        return data_[static_cast<std::size_t>(k)](i, j);
    }

private:
    std::array<Slice, D> data_{};
};

}  // namespace saltro::math
