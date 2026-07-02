#include <catch2/catch_test_macros.hpp>
#include <Eigen/Dense>
#include <cmath>

#include <saltro/math/matrix.h>

using namespace saltro::math;

namespace {

// Smallest eigenvalue of a symmetric matrix.
double minEig(const Eigen::MatrixXd& M) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(0.5 * (M + M.transpose()));
    return es.eigenvalues().minCoeff();
}

}  // namespace

TEST_CASE("psd_clip result is symmetric", "[math][matrix][psd_clip]") {
    Eigen::MatrixXd M(3, 3);
    M << 2.0, -1.0, 0.5,
        -1.0, 1.0, 0.3,
         0.5, 0.3, -2.0;  // indefinite
    psd_clip(M);
    REQUIRE((M - M.transpose()).norm() < 1e-12);
}

TEST_CASE("psd_clip clamps negative eigenvalues to zero", "[math][matrix][psd_clip]") {
    // Construct an indefinite matrix with a known negative eigenvalue.
    Eigen::MatrixXd M(3, 3);
    M << 1.0, 0.0, 0.0,
         0.0, -3.0, 0.0,
         0.0, 0.0, 2.0;
    psd_clip(M);
    // All eigenvalues should now be >= 0 (within tolerance).
    REQUIRE(minEig(M) >= -1e-12);
    // The clipped matrix should be diag(1, 0, 2).
    Eigen::MatrixXd expected = Eigen::MatrixXd::Zero(3, 3);
    expected(0, 0) = 1.0;
    expected(1, 1) = 0.0;
    expected(2, 2) = 2.0;
    REQUIRE((M - expected).norm() < 1e-12);
}

TEST_CASE("psd_clip leaves an already-PSD matrix essentially unchanged",
          "[math][matrix][psd_clip]") {
    // A^T A is PSD by construction.
    Eigen::MatrixXd A(4, 4);
    A << 1.0, 2.0, 0.0, -1.0,
         0.5, 1.0, 3.0, 0.2,
        -0.3, 0.4, 1.0, 0.7,
         0.1, -0.2, 0.5, 2.0;
    Eigen::MatrixXd P = A.transpose() * A;
    Eigen::MatrixXd P0 = P;
    psd_clip(P);
    REQUIRE((P - P0).norm() < 1e-10);
}

TEST_CASE("psd_clip is idempotent", "[math][matrix][psd_clip]") {
    Eigen::MatrixXd M(3, 3);
    M << 2.0, -1.0, 0.5,
        -1.0, -1.0, 0.3,
         0.5, 0.3, -2.0;
    psd_clip(M);
    Eigen::MatrixXd M1 = M;
    psd_clip(M);  // applying again should change nothing
    REQUIRE((M - M1).norm() < 1e-10);
}

TEST_CASE("psd_clip symmetrizes an asymmetric input", "[math][matrix][psd_clip]") {
    // Input is not symmetric; psd_clip should operate on the symmetric part.
    Eigen::MatrixXd M(2, 2);
    M << 3.0, 2.0,
         0.0, 3.0;  // asymmetric, symmetric part has eigenvalues 3 +/- 1 -> [2,4] PSD
    Eigen::MatrixXd Msym = 0.5 * (M + M.transpose());
    psd_clip(M);
    REQUIRE((M - M.transpose()).norm() < 1e-12);
    // Symmetric part was already PSD, so result equals the symmetric part.
    REQUIRE((M - Msym).norm() < 1e-10);
}
