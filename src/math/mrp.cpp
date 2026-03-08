#include <saltro/math/mrp.h>
#include <saltro/math/quaternion.h>
#include <cmath>

namespace saltro::math {

Eigen::Vector4d quatError(const Eigen::Vector4d& q_goal, const Eigen::Vector4d& q) {
    // q_err = q_goal^{-1} ⊗ q
    // For unit quaternions, q^{-1} = [q0, -q1, -q2, -q3]
    // Quaternion multiplication: p ⊗ r
    const double p0 = q_goal(0), p1 = -q_goal(1), p2 = -q_goal(2), p3 = -q_goal(3);
    const double r0 = q(0), r1 = q(1), r2 = q(2), r3 = q(3);

    Eigen::Vector4d q_err;
    q_err(0) = p0*r0 - p1*r1 - p2*r2 - p3*r3;
    q_err(1) = p0*r1 + p1*r0 + p2*r3 - p3*r2;
    q_err(2) = p0*r2 - p1*r3 + p2*r0 + p3*r1;
    q_err(3) = p0*r3 + p1*r2 - p2*r1 + p3*r0;

    // Ensure shortest-path: q0 >= 0
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
