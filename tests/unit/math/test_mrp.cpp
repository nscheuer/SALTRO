#include <catch2/catch_test_macros.hpp>
#include <Eigen/Dense>
#include <cmath>

#include <saltro/math/mrp.h>
#include <saltro/math/quaternion.h>

using namespace saltro::math;

namespace {
Eigen::Vector4d normQ(const Eigen::Vector4d& q) { return q / q.norm(); }
}  // namespace

// ============================================================================
// quatError
// ============================================================================

TEST_CASE("quatError of identical quaternions is identity", "[math][mrp][quatError]") {
    Eigen::Vector4d q = normQ(Eigen::Vector4d(0.7, 0.2, -0.5, 0.4));
    Eigen::Vector4d e = quatError(q, q);
    REQUIRE(std::abs(e(0) - 1.0) < 1e-12);
    REQUIRE(e.tail<3>().norm() < 1e-12);
}

TEST_CASE("quatError composes back to current quaternion", "[math][mrp][quatError]") {
    // q = q_goal (x) q_err  =>  with q_err = q_goal^{-1} (x) q.
    Eigen::Vector4d q_goal = normQ(Eigen::Vector4d(0.9, 0.1, 0.2, -0.3));
    Eigen::Vector4d q = normQ(Eigen::Vector4d(0.6, -0.4, 0.5, 0.3));
    Eigen::Vector4d e = quatError(q_goal, q);

    // Reconstruct q from q_goal (x) e (scalar-first Hamilton product).
    double a0 = q_goal(0), a1 = q_goal(1), a2 = q_goal(2), a3 = q_goal(3);
    double b0 = e(0), b1 = e(1), b2 = e(2), b3 = e(3);
    Eigen::Vector4d q_rec;
    q_rec(0) = a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3;
    q_rec(1) = a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2;
    q_rec(2) = a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1;
    q_rec(3) = a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0;

    // q and q_rec equal up to global sign.
    double err = std::min((q_rec - q).norm(), (q_rec + q).norm());
    REQUIRE(err < 1e-12);
}

TEST_CASE("quatError enforces q0 >= 0", "[math][mrp][quatError]") {
    Eigen::Vector4d q_goal = normQ(Eigen::Vector4d(1.0, 0.0, 0.0, 0.0));
    // A large rotation whose raw error has negative scalar part.
    Eigen::Vector4d q = normQ(Eigen::Vector4d(-0.2, 0.8, 0.3, -0.4));
    Eigen::Vector4d e = quatError(q_goal, q);
    REQUIRE(e(0) >= 0.0);
}

// ============================================================================
// quatToMRP
// ============================================================================

TEST_CASE("quatToMRP of identity is zero", "[math][mrp][quatToMRP]") {
    Eigen::Vector3d s = quatToMRP(Eigen::Vector4d(1.0, 0.0, 0.0, 0.0));
    REQUIRE(s.norm() < 1e-14);
}

TEST_CASE("quatToMRP small-angle limit is (theta/2) * axis", "[math][mrp][quatToMRP]") {
    // Small rotation theta about axis n: q = [cos(t/2), sin(t/2) n].
    // MRP = 2 q_v/(1+q0) = 2 sin(t/2)/(1+cos(t/2)) n = 2 tan(t/4) n -> (t/2) n.
    // (This convention yields half the rotation-vector magnitude in the limit.)
    double theta = 1e-3;
    Eigen::Vector3d axis = Eigen::Vector3d(0.3, -0.6, 0.7).normalized();
    Eigen::Vector4d q;
    q(0) = std::cos(theta / 2.0);
    q.tail<3>() = std::sin(theta / 2.0) * axis;
    Eigen::Vector3d s = quatToMRP(q);
    Eigen::Vector3d expected = (theta / 2.0) * axis;
    REQUIRE((s - expected).norm() < 1e-9);
}

TEST_CASE("quatToMRP round-trips through quatError", "[math][mrp][quatToMRP]") {
    // For a moderate rotation, MRP should equal tan(theta/4)*axis*... check
    // the closed form: MRP = 2 q_v/(1+q0) and reconstruct q_v.
    double theta = 0.5;
    Eigen::Vector3d axis = Eigen::Vector3d(1.0, 2.0, -1.0).normalized();
    Eigen::Vector4d q;
    q(0) = std::cos(theta / 2.0);
    q.tail<3>() = std::sin(theta / 2.0) * axis;

    Eigen::Vector3d s = quatToMRP(q);
    // Invert: q0 = (4 - |s|^2)/(4 + |s|^2), q_v = s (1+q0)/2.
    double s2 = s.squaredNorm();
    double q0_rec = (4.0 - s2) / (4.0 + s2);
    Eigen::Vector3d qv_rec = s * (1.0 + q0_rec) / 2.0;
    REQUIRE(std::abs(q0_rec - q(0)) < 1e-12);
    REQUIRE((qv_rec - q.tail<3>()).norm() < 1e-12);
}

// ============================================================================
// findGMat
// ============================================================================

TEST_CASE("findGMat has correct dimensions", "[math][mrp][gmat]") {
    int nRW = 3;
    Eigen::Vector4d q = normQ(Eigen::Vector4d(0.8, 0.2, -0.4, 0.3));
    Eigen::MatrixXd G = findGMat(q, nRW);
    REQUIRE(G.rows() == 6 + nRW);
    REQUIRE(G.cols() == 7 + nRW);
}

TEST_CASE("findGMat identity and RW blocks are correct", "[math][mrp][gmat]") {
    int nRW = 2;
    Eigen::Vector4d q = normQ(Eigen::Vector4d(0.5, 0.5, 0.5, 0.5));
    Eigen::MatrixXd G = findGMat(q, nRW);

    // Top-left 3x3 identity.
    REQUIRE((G.block(0, 0, 3, 3) - Eigen::Matrix3d::Identity()).norm() < 1e-14);
    // Bottom-right nRW identity (rows 6.., cols 7..).
    for (int i = 0; i < nRW; ++i) {
        REQUIRE(std::abs(G(6 + i, 7 + i) - 1.0) < 1e-14);
    }
}

TEST_CASE("findGMat attitude block is W^T(q) with G q-block * q == 0",
          "[math][mrp][gmat]") {
    int nRW = 1;
    Eigen::Vector4d q = normQ(Eigen::Vector4d(0.8, 0.2, -0.4, 0.3));
    Eigen::MatrixXd G = findGMat(q, nRW);

    // The attitude (delta-theta) block multiplied by the quaternion itself is
    // the tangent projection W^T q, which is zero for a unit quaternion.
    Eigen::Matrix<double, 3, 4> Wt = G.block(3, 3, 3, 4);
    Eigen::Vector3d proj = Wt * q;
    REQUIRE(proj.norm() < 1e-12);

    // And it equals findWMat(q)^T explicitly.
    Mat43 W = findWMat(q);
    REQUIRE((Wt - W.transpose()).norm() < 1e-14);
}

TEST_CASE("findGMat W^T block linearizes quaternion kinematics",
          "[math][mrp][gmat][finite-diff]") {
    // The attitude row of G maps quaternion rate to body-rate-like coordinates:
    // since qdot = 0.5 W omega and W^T W = I_3 (for unit q), we have
    // omega = 2 W^T qdot. Check W^T (0.5 W omega) == 0.5 omega.
    Eigen::Vector4d q = normQ(Eigen::Vector4d(0.9, -0.1, 0.2, 0.3));
    Eigen::Vector3d omega(0.11, -0.07, 0.05);
    Eigen::MatrixXd G = findGMat(q, 0);
    Eigen::Matrix<double, 3, 4> Wt = G.block(3, 3, 3, 4);

    Mat43 W = findWMat(q);
    Eigen::Vector4d qdot = 0.5 * W * omega;
    Eigen::Vector3d recovered = Wt * qdot;
    REQUIRE((recovered - 0.5 * omega).norm() < 1e-12);
}
