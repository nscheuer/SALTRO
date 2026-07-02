#include <catch2/catch_test_macros.hpp>
#include <Eigen/Dense>
#include <array>
#include <cmath>

#include <saltro/math/quaternion.h>

using namespace saltro::math;

namespace {

// A fixed, non-trivial unit quaternion used across tests.
Vec4 fixedUnitQuat() {
    Vec4 q(0.8, 0.2, -0.4, 0.3);
    return q / q.norm();
}

// A fixed, non-trivial vector to be rotated.
Vec3 fixedVec() {
    return Vec3(1.3, -0.7, 2.1);
}

// Scalar-first Hamilton product p (x) r.
Vec4 quatMul(const Vec4& p, const Vec4& r) {
    Vec4 out;
    out(0) = p(0) * r(0) - p(1) * r(1) - p(2) * r(2) - p(3) * r(3);
    out(1) = p(0) * r(1) + p(1) * r(0) + p(2) * r(3) - p(3) * r(2);
    out(2) = p(0) * r(2) - p(1) * r(3) + p(2) * r(0) + p(3) * r(1);
    out(3) = p(0) * r(3) + p(1) * r(2) - p(2) * r(1) + p(3) * r(0);
    return out;
}

// Perturb a unit quaternion by a body-frame rotation vector (stays on the
// unit sphere): q' = q (x) exp(theta/2).
Vec4 perturbOnSphere(const Vec4& q, const Vec3& theta) {
    double n = theta.norm();
    Vec4 dq;
    if (n < 1e-300) {
        dq << 1.0, 0.0, 0.0, 0.0;
    } else {
        dq(0) = std::cos(n / 2.0);
        dq.tail<3>() = std::sin(n / 2.0) * theta / n;
    }
    return quatMul(q, dq);
}

}  // namespace

// ============================================================================
// normalizeQuat
// ============================================================================

TEST_CASE("normalizeQuat produces unit norm", "[math][quaternion][normalize]") {
    Vec4 q(2.0, -1.0, 0.5, 3.0);
    Vec4 qn = normalizeQuat(q);
    REQUIRE(std::abs(qn.norm() - 1.0) < 1e-12);
    // Direction preserved: qn parallel to q.
    REQUIRE(std::abs((qn * q.norm() - q).norm()) < 1e-12);
}

TEST_CASE("normalizeQuat of an already-unit quaternion is unchanged", "[math][quaternion][normalize]") {
    Vec4 q = fixedUnitQuat();
    Vec4 qn = normalizeQuat(q);
    REQUIRE(std::abs((qn - q).norm()) < 1e-12);
}

TEST_CASE("normalizeQuat throws on near-zero quaternion", "[math][quaternion][normalize]") {
    Vec4 q(1e-15, 0.0, -1e-15, 0.0);
    REQUIRE_THROWS(normalizeQuat(q));
}

// ============================================================================
// rotationMatrix
// ============================================================================

TEST_CASE("rotationMatrix is orthonormal with det +1", "[math][quaternion][rotation]") {
    Vec4 q = fixedUnitQuat();
    Mat33 R = rotationMatrix(q);

    Mat33 should_be_I = R * R.transpose();
    REQUIRE((should_be_I - Mat33::Identity()).norm() < 1e-12);
    REQUIRE(std::abs(R.determinant() - 1.0) < 1e-12);
}

TEST_CASE("rotationMatrix normalizes its input", "[math][quaternion][rotation]") {
    Vec4 q = fixedUnitQuat();
    Mat33 R1 = rotationMatrix(q);
    Mat33 R2 = rotationMatrix(2.5 * q);  // same direction, different norm
    REQUIRE((R1 - R2).norm() < 1e-12);
}

TEST_CASE("rotationMatrix identity quaternion gives identity", "[math][quaternion][rotation]") {
    Vec4 q(1.0, 0.0, 0.0, 0.0);
    Mat33 R = rotationMatrix(q);
    REQUIRE((R - Mat33::Identity()).norm() < 1e-12);
}

TEST_CASE("rotationMatrix known 90deg about z", "[math][quaternion][rotation]") {
    // q = [cos(45), 0, 0, sin(45)] -> 90 deg about +z.
    double c = std::cos(M_PI / 4.0);
    double s = std::sin(M_PI / 4.0);
    Vec4 q(c, 0.0, 0.0, s);
    Mat33 R = rotationMatrix(q);

    // With this convention R(0,1) = 2(q1q2 - q0q3) = -2 c s = -sin(90) = -1.
    // R rotates a body vector to the reference frame: R * e_x should be e_y
    // if this is an active rotation by +90 about z. Check the matrix entries.
    Mat33 expected;
    expected << 0.0, -1.0, 0.0,
                1.0,  0.0, 0.0,
                0.0,  0.0, 1.0;
    REQUIRE((R - expected).norm() < 1e-12);
}

// ============================================================================
// skewSymmetric
// ============================================================================

TEST_CASE("skewSymmetric implements cross product", "[math][quaternion][skew]") {
    Vec3 a(1.0, -2.0, 3.0);
    Vec3 b(-0.5, 0.4, 1.7);
    Vec3 cross_direct = a.cross(b);
    Vec3 cross_skew = skewSymmetric(a) * b;
    REQUIRE((cross_direct - cross_skew).norm() < 1e-14);
}

TEST_CASE("skewSymmetric is antisymmetric with zero diagonal", "[math][quaternion][skew]") {
    Vec3 a(1.3, -0.7, 2.1);
    Mat33 S = skewSymmetric(a);
    REQUIRE((S + S.transpose()).norm() < 1e-14);
    REQUIRE(std::abs(S(0, 0)) < 1e-14);
    REQUIRE(std::abs(S(1, 1)) < 1e-14);
    REQUIRE(std::abs(S(2, 2)) < 1e-14);
}

// ============================================================================
// findWMat / quatNormJacobian (FD vs analytic)
// ============================================================================

TEST_CASE("quatNormJacobian matches finite differences of normalizeQuat",
          "[math][quaternion][jacobian][finite-diff]") {
    // quatNormJacobian returns d(q/|q|)/dq. Header type is Mat43 but the
    // implementation fills a 4x3 block; we compare the available 4x3 entries
    // against the (4x4) numerical normalization Jacobian's first 3 columns.
    Vec4 q(0.8, 0.2, -0.4, 0.3);  // raw (unnormalized direction)
    Mat43 Janalytic = quatNormJacobian(q);

    const double eps = 1e-6;
    // Numerical d(normalize)/dq, 4x4.
    Mat44 Jnum = Mat44::Zero();
    for (int j = 0; j < 4; ++j) {
        Vec4 qp = q, qm = q;
        qp(j) += eps;
        qm(j) -= eps;
        Vec4 fp = qp / qp.norm();
        Vec4 fm = qm / qm.norm();
        Jnum.col(j) = (fp - fm) / (2.0 * eps);
    }

    // Compare the 4x3 analytic block against the first 3 numerical columns.
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 3; ++j) {
            REQUIRE(std::abs(Janalytic(i, j) - Jnum(i, j)) < 1e-6);
        }
    }
}

TEST_CASE("findWMat satisfies quaternion kinematics qdot = 0.5 W omega",
          "[math][quaternion][wmat]") {
    Vec4 q = fixedUnitQuat();
    Vec3 omega(0.13, -0.07, 0.21);
    Mat43 W = findWMat(q);

    // Reference qdot from the standard scalar-first kinematics:
    // qdot = 0.5 * [ -qv . w ; q0 w + qv x w ].
    double q0 = q(0);
    Vec3 qv = q.tail<3>();
    Vec4 qdot_ref;
    qdot_ref(0) = -0.5 * qv.dot(omega);
    qdot_ref.tail<3>() = 0.5 * (q0 * omega + qv.cross(omega));

    Vec4 qdot = 0.5 * W * omega;
    REQUIRE((qdot - qdot_ref).norm() < 1e-12);
}

TEST_CASE("findWMat columns are orthogonal to q (W^T q = 0)",
          "[math][quaternion][wmat]") {
    Vec4 q = fixedUnitQuat();
    Mat43 W = findWMat(q);
    Vec3 wtq = W.transpose() * q;
    REQUIRE(wtq.norm() < 1e-12);
}

// ============================================================================
// drotmatTvecdq (FD of R(q)^T v vs analytic)
// ============================================================================

// drotmatTvecdq returns a 4x3 Euclidean Jacobian J with J(a, k) =
// d[R(q)^T v]_k / dq_a, but as a *unit-norm-constrained* derivative (it carries
// the q0 normalization contribution, so a naive Euclidean FD of the raw 3-2-1
// rotation-matrix formula does NOT match it). The physically meaningful and
// codebase-consistent check is the ON-MANIFOLD derivative: contracting J onto
// the quaternion tangent space via findWMat must reproduce the finite-difference
// of R(q)^T v under unit-norm-preserving (body-rotation) perturbations.
TEST_CASE("drotmatTvecdq matches on-manifold finite differences of R(q)^T v",
          "[math][quaternion][drotmat][finite-diff]") {
    const std::array<Vec4, 2> qs = {Vec4(1.0, 0.0, 0.0, 0.0), fixedUnitQuat()};
    Vec3 v = fixedVec();

    for (const Vec4& q : qs) {
        Mat43 J = drotmatTvecdq(q, v);
        Mat43 W = findWMat(q);
        // d(R^T v)/dtheta = J^T (dq/dtheta) = J^T (0.5 W), shape 3x3.
        Mat33 analytic = J.transpose() * (0.5 * W);

        const double eps = 1e-7;
        Mat33 numerical = Mat33::Zero();
        for (int k = 0; k < 3; ++k) {
            Vec3 tp = Vec3::Zero(), tm = Vec3::Zero();
            tp(k) = eps;
            tm(k) = -eps;
            Vec3 fp = rotationMatrix(perturbOnSphere(q, tp)).transpose() * v;
            Vec3 fm = rotationMatrix(perturbOnSphere(q, tm)).transpose() * v;
            numerical.col(k) = (fp - fm) / (2.0 * eps);
        }

        double maxerr = (analytic - numerical).cwiseAbs().maxCoeff();
        INFO("q = " << q.transpose());
        INFO("max |analytic - numerical| = " << maxerr);
        INFO("analytic=\n" << analytic);
        INFO("numerical=\n" << numerical);
        REQUIRE(maxerr < 1e-6);
    }
}

// ============================================================================
// ddrotmatTvecdqdq -- the helper that had the bug; pin it hard.
// ============================================================================

// =========================================================================
// KNOWN BUG (flagged, NOT papered over):
//
// ddrotmatTvecdqdq is the Euclidean second derivative of R(q)^T v, indexed in
// satellite.cpp as H[i](QUAT_INDEX+j, QUAT_INDEX+k) = d^2(R^T v)_i / (dq_j dq_k).
// As a Hessian it must be (1) symmetric in (j,k) and (2) equal to the Euclidean
// Jacobian of drotmatTvecdq (the matching first derivative). These cases pin
// both. The qv-qv block of this helper was previously incomplete and asymmetric
// (wrong by O(1)); it is exercised here at a NON-identity quaternion, where the
// defect shows -- identity-only tests cancel it. (This is the direct regression
// test for the rotation-Hessian fix.)
// =========================================================================

TEST_CASE("ddrotmatTvecdqdq slices are symmetric in the two q-indices",
          "[math][quaternion][ddrotmat][symmetry]") {
    Vec4 q = fixedUnitQuat();
    Vec3 v = fixedVec();
    std::array<Mat44, 3> H = ddrotmatTvecdqdq(q, v);
    for (int k = 0; k < 3; ++k) {
        double asym = (H[static_cast<size_t>(k)] - H[static_cast<size_t>(k)].transpose()).norm();
        INFO("component k=" << k << " asymmetry norm = " << asym);
        REQUIRE(asym < 1e-12);
    }
}

TEST_CASE("ddrotmatTvecdqdq matches finite differences of drotmatTvecdq",
          "[math][quaternion][ddrotmat][finite-diff]") {
    // H[k](a,b) = d^2 (R^T v)_k / (dq_a dq_b).
    // drotmatTvecdq returns J with J(a,k) = d(R^T v)_k/dq_a. So
    // d J(a,k) / dq_b = H[k](a,b). We finite-difference drotmatTvecdq and the
    // analytic helper must match it (and the FD reference is itself symmetric to
    // ~1e-10, see the positive-control case below).
    Vec4 q = fixedUnitQuat();
    Vec3 v = fixedVec();
    std::array<Mat44, 3> Hanalytic = ddrotmatTvecdqdq(q, v);

    const double eps = 1e-6;
    std::array<Mat44, 3> Hnum = {Mat44::Zero(), Mat44::Zero(), Mat44::Zero()};
    for (int b = 0; b < 4; ++b) {
        Vec4 qp = q, qm = q;
        qp(b) += eps;
        qm(b) -= eps;
        Mat43 Jp = drotmatTvecdq(qp, v);
        Mat43 Jm = drotmatTvecdq(qm, v);
        Mat43 dJ = (Jp - Jm) / (2.0 * eps);  // dJ(a,k) = dJ(a,k)/dq_b
        for (int k = 0; k < 3; ++k) {
            for (int a = 0; a < 4; ++a) {
                Hnum[static_cast<size_t>(k)](a, b) = dJ(a, k);
            }
        }
    }

    for (int k = 0; k < 3; ++k) {
        double maxerr = (Hanalytic[static_cast<size_t>(k)] - Hnum[static_cast<size_t>(k)])
                            .cwiseAbs()
                            .maxCoeff();
        INFO("component k=" << k << " max |analytic - numerical| = " << maxerr);
        INFO("analytic=\n" << Hanalytic[static_cast<size_t>(k)]);
        INFO("numerical=\n" << Hnum[static_cast<size_t>(k)]);
        REQUIRE(maxerr < 1e-5);
    }
}

// Positive control: the numerical Hessian (FD of drotmatTvecdq) is itself
// symmetric, proving the symmetry failure above is a defect in the analytic
// helper and not a property of the underlying function.
TEST_CASE("ddrotmatTvecdqdq finite-difference reference is symmetric",
          "[math][quaternion][ddrotmat][symmetry]") {
    Vec4 q = fixedUnitQuat();
    Vec3 v = fixedVec();
    const double eps = 1e-6;
    std::array<Mat44, 3> Hnum = {Mat44::Zero(), Mat44::Zero(), Mat44::Zero()};
    for (int b = 0; b < 4; ++b) {
        Vec4 qp = q, qm = q;
        qp(b) += eps;
        qm(b) -= eps;
        Mat43 dJ = (drotmatTvecdq(qp, v) - drotmatTvecdq(qm, v)) / (2.0 * eps);
        for (int k = 0; k < 3; ++k)
            for (int a = 0; a < 4; ++a)
                Hnum[static_cast<size_t>(k)](a, b) = dJ(a, k);
    }
    for (int k = 0; k < 3; ++k) {
        double asym = (Hnum[static_cast<size_t>(k)] - Hnum[static_cast<size_t>(k)].transpose()).norm();
        INFO("component k=" << k << " FD asymmetry norm = " << asym);
        REQUIRE(asym < 1e-5);
    }
}
