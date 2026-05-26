#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <memory>

#include <Eigen/Dense>

#include <saltro/limits.h>
#include <saltro/math/quaternion.h>
#include <saltro/optimizer/warm_start.h>
#include <saltro/pybind/controller/pdcontroller.h>
#include <saltro/pybind/satellite.h>

using namespace saltro;

namespace {

constexpr double PI = 3.14159265358979323846;

// Build a minimal 1-MTQ-per-axis + 1-RW satellite.  Inertia is a small
// isotropic body; actuators are deliberately weak so the PD output is well
// inside the saturation regime.
std::unique_ptr<Satellite> makeSimpleSatellite() {
    auto s = std::make_unique<Satellite>();
    Eigen::Matrix3d J;
    J << 0.05, 0.0, 0.0,
         0.0, 0.06, 0.0,
         0.0, 0.0, 0.04;
    s->setInertia(J);
    s->addMTQ(Eigen::Vector3d::UnitX(), 0.2);
    s->addMTQ(Eigen::Vector3d::UnitY(), 0.2);
    s->addMTQ(Eigen::Vector3d::UnitZ(), 0.2);
    s->addRW(Eigen::Vector3d::UnitZ(), 1.0e-3, 1.0e-5, 0.0, 1.6e-2);
    return s;
}

Satellite::VecX makeState(const Eigen::Vector3d& omega, const Eigen::Vector4d& q) {
    Satellite::VecX x = Satellite::VecX::Zero(8);
    x.segment<3>(Satellite::AV_INDEX) = omega;
    x.segment<4>(Satellite::QUAT_INDEX) = q;
    return x;
}

}  // namespace

// ----------------------------------------------------------------------------
// Quaternion-goal regression: aligned state should produce ~zero torque demand.
// ----------------------------------------------------------------------------
TEST_CASE("PDController returns small command when aligned (quaternion goal)",
          "[controller][pd][quat]") {
    auto sat = makeSimpleSatellite();
    controller::PDController pd(*sat);

    // Aligned: ω = 0, q = identity, goal = identity.
    const auto x = makeState(Eigen::Vector3d::Zero(),
                             Eigen::Vector4d(1.0, 0.0, 0.0, 0.0));
    const Eigen::Vector3d B_eci(2.2e-5, -1.6e-5, 3.1e-5);
    const Eigen::Vector4d q_goal(1.0, 0.0, 0.0, 0.0);
    const Eigen::Vector3d boresight = Eigen::Vector3d::UnitZ();

    const Satellite::VecX u = pd.find_u(x, B_eci, q_goal, boresight);
    REQUIRE(u.size() == 4);
    REQUIRE(u.norm() < 1e-6);
}

// ----------------------------------------------------------------------------
// Vector-goal sentinel handling (the bug this PR fixes).
//
// Pre-fix: q_goal[0] = NaN propagated through quatError → tau_des = NaN →
// the !u_raw.allFinite() guard returned u = 0 at every knot, silently
// degenerating to the ZeroController behavior.  Post-fix: PDController must
// detect the [NaN, r̂_eci] sentinel and produce a finite, direction-correct
// torque demand.
// ----------------------------------------------------------------------------
TEST_CASE("PDController accepts NaN-flagged vector goal and produces finite output",
          "[controller][pd][vec]") {
    auto sat = makeSimpleSatellite();
    controller::PDController pd(*sat);

    const auto x = makeState(Eigen::Vector3d::Zero(),
                             Eigen::Vector4d(1.0, 0.0, 0.0, 0.0));
    // Use a relatively strong B field so the MTQ torque is well above the
    // actuator-authority Tikhonov noise floor — at LEO magnitudes (~25 µT)
    // the auth-weighted regularization squashes a 1 mN·m demand down to
    // ~10^-11 A·m^2 even with the fix in place, which makes a "non-zero"
    // assertion fragile.  100× LEO is well within validity of the bilinear
    // model and gives a representative MTQ output magnitude (~10^-5 A·m^2).
    const Eigen::Vector3d B_eci(2.2e-3, -1.6e-3, 3.1e-3);

    // Vector goal: ECI +x.  Boresight is body +z.  At q = identity:
    //   r_body = R(q)^T · r̂_eci = +x_body.
    //   bs × r_body = ẑ × x̂ = ŷ.
    // Pre-fix, this would NaN out and the safe-guard would return u = 0.
    const Eigen::Vector4d q_goal_vec(std::nan(""), 1.0, 0.0, 0.0);
    const Eigen::Vector3d boresight = Eigen::Vector3d::UnitZ();

    const Satellite::VecX u = pd.find_u(x, B_eci, q_goal_vec, boresight);
    REQUIRE(u.size() == 4);
    REQUIRE(u.allFinite());
    // Output is non-trivial (would be zero if the vec-mode branch had failed
    // and the !u_raw.allFinite() guard kicked in).
    REQUIRE(u.norm() > 1e-9);

    // Direction sanity: with bs = +ẑ_body, target = +x̂_body (= +x̂_eci at q=I),
    // tau_des points in +ŷ_body.  The only y-axis-producing actuator at
    // q=identity is MTQ_x (since m × B for m=x̂ has a y-component proportional
    // to -Bz, which is positive when Bz < 0; for our B with Bz > 0, MTQ_x
    // contributes a negative-y component, so the solver picks u_mtq_x < 0).
    // The other MTQs and the z-RW contribute negligibly to ŷ given the
    // authority weighting, so MTQ_x should dominate the response.
    REQUIRE(std::abs(u(0)) > std::abs(u(1)));
    REQUIRE(std::abs(u(0)) > std::abs(u(3)));   // bigger than the RW command
}

// ----------------------------------------------------------------------------
// Vector-goal alignment: zero error → ~zero command.
// ----------------------------------------------------------------------------
TEST_CASE("PDController vec goal returns small command when boresight aligned",
          "[controller][pd][vec]") {
    auto sat = makeSimpleSatellite();
    controller::PDController pd(*sat);

    // Boresight body +z, goal ECI +z, q = identity → r_body = +z = bs.
    // bs × r_body = 0  ⇒  tau_des = -kd_w · ω = 0 at ω=0.
    const auto x = makeState(Eigen::Vector3d::Zero(),
                             Eigen::Vector4d(1.0, 0.0, 0.0, 0.0));
    const Eigen::Vector3d B_eci(2.2e-5, -1.6e-5, 3.1e-5);
    const Eigen::Vector4d q_goal_vec(std::nan(""), 0.0, 0.0, 1.0);
    const Eigen::Vector3d boresight = Eigen::Vector3d::UnitZ();

    const Satellite::VecX u = pd.find_u(x, B_eci, q_goal_vec, boresight);
    REQUIRE(u.size() == 4);
    REQUIRE(u.allFinite());
    REQUIRE(u.norm() < 1e-6);
}

// ----------------------------------------------------------------------------
// Vector-goal antipodal: bs × r_body = 0 at 180° (cross product of parallel
// vectors).  We tolerate the zero output here -- the cross-product PD law
// cannot distinguish 0° from 180° (a well-known singularity that needs
// rate-only damping or a perturbation to escape).  Verify it at least
// produces finite output and lets `ω` carry the recovery via the damping
// term.
// ----------------------------------------------------------------------------
TEST_CASE("PDController vec goal handles antipodal singularity finitely",
          "[controller][pd][vec]") {
    auto sat = makeSimpleSatellite();
    controller::PDController pd(*sat);

    // ω = +z, q = identity, boresight = +z, goal = ECI -z  →  bs × r_body = 0,
    // tau_des = -kd_w · ω = -kd_w · (0,0,ω_z)  →  pure rate damping.
    const auto x = makeState(Eigen::Vector3d(0.0, 0.0, 0.05),
                             Eigen::Vector4d(1.0, 0.0, 0.0, 0.0));
    const Eigen::Vector3d B_eci(2.2e-5, -1.6e-5, 3.1e-5);
    const Eigen::Vector4d q_goal_vec(std::nan(""), 0.0, 0.0, -1.0);
    const Eigen::Vector3d boresight = Eigen::Vector3d::UnitZ();

    const Satellite::VecX u = pd.find_u(x, B_eci, q_goal_vec, boresight);
    REQUIRE(u.size() == 4);
    REQUIRE(u.allFinite());
}

// ----------------------------------------------------------------------------
// warm_start dispatch wiring: settings.init_traj.initcontroller = 3 must
// select PDController (was previously case 3 → return false on main).
// ----------------------------------------------------------------------------
TEST_CASE("warm_start dispatches initcontroller=3 to PDController",
          "[controller][pd][warm_start]") {
    auto sat = makeSimpleSatellite();

    PlannerSettings settings;
    settings.init_traj.initcontroller = 3;
    // Need one pass for the dt source.
    settings.num_passes = 1;
    settings.passes[0].dt = 1.0;

    const int N = 5;
    const double dt_sec = 1.0;
    const double sec_per_century = 36525.0 * 86400.0;

    Eigen::VectorXd jtime(N);
    Eigen::MatrixXd q_goal = Eigen::MatrixXd::Zero(4, N);
    Eigen::MatrixXd boresight = Eigen::MatrixXd::Zero(3, N);
    Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> R; R.setZero();
    Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> V; V.setZero();
    Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> B; B.setZero();
    Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> S; S.setZero();
    Eigen::Matrix<double, 1, limits::MAX_LENGTH_TRAJ> rho; rho.setZero();

    for (int k = 0; k < N; ++k) {
        jtime(k) = 0.25 + (k * dt_sec) / sec_per_century;
        q_goal(0, k) = 1.0;                         // quaternion goal = identity
        boresight.col(k) = Eigen::Vector3d::UnitZ();
        B.col(k) = Eigen::Vector3d(2.2e-5, -1.6e-5, 3.1e-5);
    }

    // Start tumbling so PD actually has something to do.
    Satellite::VecX x0 = makeState(Eigen::Vector3d(0.05, -0.03, 0.04),
                                   Eigen::Vector4d(1.0, 0.0, 0.0, 0.0));

    Eigen::MatrixXd X = Eigen::MatrixXd::Zero(sat->stateDim(), N);
    Eigen::MatrixXd U = Eigen::MatrixXd::Zero(sat->controlDim(), N);

    const bool ok = optimizer::warm_start(
        settings, *sat, x0, jtime, q_goal, boresight, N, R, V, B, S, rho, X, U);
    REQUIRE(ok);
    REQUIRE(X.allFinite());
    REQUIRE(U.allFinite());
}
