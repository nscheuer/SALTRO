#include <saltro/math/mrp.h>
#include <saltro/math/quaternion.h>
#include <cmath>

namespace saltro::math {

Eigen::Vector4d quatError(const Eigen::Vector4d& q_goal, const Eigen::Vector4d& q) {
    // q_err = q_goal^{-1} ⊗ q  (for unit quaternions, inverse = conjugate)
    Eigen::Vector4d q_err = quatMult(quatConj(q_goal), q);
    // Ensure shortest-path: q_err(0) >= 0
    if (q_err(0) < 0.0) {
        q_err = -q_err;
    }
    return q_err;
}

Eigen::Vector3d quatToMRP(const Eigen::Vector4d& q_err) {
    // MRP = 2 * q_vec / (1 + q0)
    const double denom = 1.0 + q_err(0);
    return 2.0 * q_err.tail<3>() / denom;
}

Eigen::MatrixXd findGMat(const Eigen::Vector4d& q, int nRW) {
    const int nx = 7 + nRW;   // full state dim
    const int nxr = 6 + nRW;  // reduced state dim

    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(nxr, nx);

    // Top-left: I_3 (angular velocity maps to itself)
    G.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();

    // Middle: W^T(q)  (3×4), maps quaternion perturbations to attitude angle perturbations
    // W is 4×3 from findWMat, so W^T is 3×4
    Mat43 W = findWMat(q);
    G.block<3, 4>(3, 3) = W.transpose();

    // Bottom-right: I_{nRW} (RW momenta map to themselves)
    for (int i = 0; i < nRW; ++i) {
        G(6 + i, 7 + i) = 1.0;
    }

    return G;
}

} // namespace saltro::math
