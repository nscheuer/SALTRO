#pragma once

#include <cassert>
#include <vector>
#include <Eigen/Dense>

/**
 * @file tensor_ops.h
 * @brief Third-order tensor algebra used by DDP backward-pass and RK4 Hessian
 *        composition.
 *
 * We represent a rank-3 tensor H with output-last indexing as
 *   std::vector<Eigen::MatrixXd>  // size D slices, each R × C
 * where `H[k]` is the k-th "slice" interpreted as the Hessian matrix whose
 * rows are the first input and whose columns are the second input, for output
 * equation `k`. This matches the convention used in `Satellite::dynamicsHessians`
 * (Tensor3<R, C, D>::slice(k)) and in the PhD reference PlannerUtil.cpp
 * (armadillo `cube.slice(k)`).
 *
 * Four helpers port the PhD cube operations:
 *   - matTimesCube(M, H)     : slice-wise left-multiply,  M @ H[k]
 *   - cubeTimesMat(H, M)     : slice-wise right-multiply, H[k] @ M
 *   - matTimesCubeT(M, H)    : slice-wise M @ H[k]^T      (for mixed-partial
 *                              symmetrization terms)
 *   - matOverCube(M, H)      : output-index contraction, (M @ H)[l] =
 *                              Σ_i M(l, i) * H[i]
 *
 * All operations produce a new tensor (value semantics). They are simple
 * loops over slices; there is no clever blocking. This keeps the port
 * auditable against the PhD source at the cost of some temporaries.
 */

namespace saltro::math {

/// H[k] = M @ H[k]  for each k in [0, D).
/// Shapes: M is (p × r); each H[k] is (r × c); result[k] is (p × c).
inline std::vector<Eigen::MatrixXd> matTimesCube(
    const Eigen::Ref<const Eigen::MatrixXd>& M,
    const std::vector<Eigen::MatrixXd>& H
) {
    std::vector<Eigen::MatrixXd> out;
    out.reserve(H.size());
    for (const auto& Hk : H) {
        out.emplace_back(M * Hk);
    }
    return out;
}

/// H[k] = H[k] @ M  for each k in [0, D).
/// Shapes: each H[k] is (r × c); M is (c × p); result[k] is (r × p).
inline std::vector<Eigen::MatrixXd> cubeTimesMat(
    const std::vector<Eigen::MatrixXd>& H,
    const Eigen::Ref<const Eigen::MatrixXd>& M
) {
    std::vector<Eigen::MatrixXd> out;
    out.reserve(H.size());
    for (const auto& Hk : H) {
        out.emplace_back(Hk * M);
    }
    return out;
}

/// H[k] = M @ H[k]^T  for each k in [0, D).
/// Shapes: each H[k] is (r × c); M is (p × c); result[k] is (p × r).
/// Used for symmetrization terms in the mixed-partials chain rule.
inline std::vector<Eigen::MatrixXd> matTimesCubeT(
    const Eigen::Ref<const Eigen::MatrixXd>& M,
    const std::vector<Eigen::MatrixXd>& H
) {
    std::vector<Eigen::MatrixXd> out;
    out.reserve(H.size());
    for (const auto& Hk : H) {
        out.emplace_back(M * Hk.transpose());
    }
    return out;
}

/// Output-index contraction: result[l] = Σ_i M(l, i) * H[i].
/// Implements the "first-derivative × second-derivative" term of the Hessian
/// chain rule for a composition y = f(g(x)): ∂²f_l/∂x² gets a term
/// (∂f_l/∂g_i) · ∂²g_i/∂x².
/// Shapes: M is (p × m) where m = H.size(); each H[i] is (r × c); result has
/// p slices each (r × c).
inline std::vector<Eigen::MatrixXd> matOverCube(
    const Eigen::Ref<const Eigen::MatrixXd>& M,
    const std::vector<Eigen::MatrixXd>& H
) {
    assert(M.cols() == static_cast<Eigen::Index>(H.size()) && "matOverCube: M.cols() must equal H.size()");
    const int p = static_cast<int>(M.rows());
    const int r = H.empty() ? 0 : static_cast<int>(H[0].rows());
    const int c = H.empty() ? 0 : static_cast<int>(H[0].cols());

    std::vector<Eigen::MatrixXd> out(static_cast<std::size_t>(p),
                                     Eigen::MatrixXd::Zero(r, c));
    for (int l = 0; l < p; ++l) {
        for (std::size_t i = 0; i < H.size(); ++i) {
            out[static_cast<std::size_t>(l)].noalias() += M(l, static_cast<Eigen::Index>(i)) * H[i];
        }
    }
    return out;
}

/// Element-wise tensor addition: result[k] = A[k] + B[k].
inline std::vector<Eigen::MatrixXd> cubeAdd(
    const std::vector<Eigen::MatrixXd>& A,
    const std::vector<Eigen::MatrixXd>& B
) {
    assert(A.size() == B.size() && "cubeAdd: size mismatch");
    std::vector<Eigen::MatrixXd> out;
    out.reserve(A.size());
    for (std::size_t k = 0; k < A.size(); ++k) {
        out.emplace_back(A[k] + B[k]);
    }
    return out;
}

/// Scalar multiply: result[k] = s * H[k].
inline std::vector<Eigen::MatrixXd> cubeScale(
    double s,
    const std::vector<Eigen::MatrixXd>& H
) {
    std::vector<Eigen::MatrixXd> out;
    out.reserve(H.size());
    for (const auto& Hk : H) {
        out.emplace_back(s * Hk);
    }
    return out;
}

}  // namespace saltro::math
