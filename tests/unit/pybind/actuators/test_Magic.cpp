/**
 * @file test_Magic.cpp
 * @brief Catch2 tests for the Magic (direct body-torque) actuator class.
 *
 * Magic actuators model an idealised body-torque commander:
 *   τ = u * axis
 * with no environmental dependence and no internal momentum-storage state.
 * They exist to express thrusters or as test fixtures (no MTQ rank
 * deficiency, no RW back-reaction inertia).
 *
 * These tests pin down the linear, state-independent torque law and
 * its derivative structure (axis^T for ∂τ/∂u, zeros for everything
 * else). If any of these break, the Magic actuator has stopped being
 * "magic" -- it's gained coupling that the planner doesn't know about.
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <Eigen/Dense>
#include <saltro/pybind/actuators/Magic.h>
#include <cmath>

using Vec3 = Eigen::Vector3d;
using BaseState = Eigen::Matrix<double, 7, 1>;

static Vec3 valid_axis() {
    Vec3 axis;
    axis << 1.0, 0.0, 0.0;
    return axis;
}

static BaseState valid_base_state() {
    BaseState x;
    x << 0.0, 0.0, 0.0,
         1.0, 0.0, 0.0, 0.0;
    return x;
}

TEST_CASE("Magic constructor with valid inputs", "[magic]") {
    Vec3 axis = valid_axis();
    double max_torque = 0.01;

    REQUIRE_NOTHROW(Magic(axis, max_torque));

    Magic magic(axis, max_torque);
    REQUIRE(magic.axis().isApprox(axis));
    REQUIRE(magic.u_max() == max_torque);
}

TEST_CASE("Magic normalizes non-unit axis", "[magic]") {
    Vec3 axis;
    axis << 2.0, 0.0, 0.0;
    Magic magic(axis, 0.05);
    REQUIRE(magic.axis().norm() == Catch::Approx(1.0));
    REQUIRE(magic.axis()(0) == Catch::Approx(1.0));
}

TEST_CASE("Magic torque is linear in u along axis", "[magic][torque]") {
    Vec3 axis;
    axis << 0.0, 1.0, 0.0;
    Magic magic(axis, 0.1);

    BaseState x = valid_base_state();
    const double u = 0.04;

    Vec3 tau = magic.torque(u, x);
    Vec3 expected = axis * u;

    REQUIRE(tau.isApprox(expected));
}

TEST_CASE("Magic torque is independent of base state", "[magic][torque]") {
    // The "magic" property: same u → same τ regardless of ω or q.
    Vec3 axis;
    axis << 1.0, 0.0, 0.0;
    Magic magic(axis, 0.1);

    BaseState x1 = valid_base_state();
    BaseState x2;
    // Different ω and a nontrivial quaternion (90° about z).
    x2 << 0.3, -0.1, 0.7,
          std::sqrt(0.5), 0.0, 0.0, std::sqrt(0.5);

    const double u = 0.02;
    Vec3 tau1 = magic.torque(u, x1);
    Vec3 tau2 = magic.torque(u, x2);

    REQUIRE(tau1.isApprox(tau2));
    REQUIRE(tau1.isApprox(axis * u));
}

TEST_CASE("Magic torque vanishes at zero command", "[magic][torque]") {
    Magic magic(valid_axis(), 0.1);
    Vec3 tau = magic.torque(0.0, valid_base_state());
    REQUIRE(tau.norm() == Catch::Approx(0.0).margin(1e-15));
}

TEST_CASE("Magic dtorq_du equals axis^T", "[magic][jacobian]") {
    Vec3 axis;
    axis << 0.0, 0.0, 1.0;
    Magic magic(axis, 0.05);

    auto jac = magic.dtorq_du(0.03, valid_base_state());
    REQUIRE(jac.rows() == 1);
    REQUIRE(jac.cols() == 3);
    REQUIRE(jac(0, 0) == Catch::Approx(axis(0)));
    REQUIRE(jac(0, 1) == Catch::Approx(axis(1)));
    REQUIRE(jac(0, 2) == Catch::Approx(axis(2)));
}

TEST_CASE("Magic dtorq_du is constant in u and x", "[magic][jacobian]") {
    Vec3 axis;
    axis << 0.5, 0.5, std::sqrt(0.5);  // unit-norm body axis
    Magic magic(axis, 0.1);

    auto jac0 = magic.dtorq_du(0.0, valid_base_state());

    BaseState x_other;
    x_other << 0.1, 0.2, 0.3,
               std::sqrt(0.5), std::sqrt(0.5), 0.0, 0.0;
    auto jac1 = magic.dtorq_du(0.07, x_other);

    REQUIRE(jac0.isApprox(jac1));
}

TEST_CASE("Magic dtorq_dbasestate is zero", "[magic][jacobian]") {
    Magic magic(valid_axis(), 0.1);
    auto J = magic.dtorq_dbasestate(0.05, valid_base_state());
    REQUIRE(J.rows() == 7);
    REQUIRE(J.cols() == 3);
    REQUIRE(J.norm() == Catch::Approx(0.0).margin(1e-15));
}

TEST_CASE("Magic ddtorq_dudu is zero (torque is affine in u)", "[magic][hessian]") {
    Magic magic(valid_axis(), 0.1);
    auto H = magic.ddtorq_dudu(0.02, valid_base_state());
    for (int i = 0; i < 3; ++i) {
        REQUIRE(H.slice(i).norm() == Catch::Approx(0.0).margin(1e-15));
    }
}

TEST_CASE("Magic ddtorq_dudbasestate is zero", "[magic][hessian]") {
    Magic magic(valid_axis(), 0.1);
    auto H = magic.ddtorq_dudbasestate(0.02, valid_base_state());
    for (int i = 0; i < 3; ++i) {
        REQUIRE(H.slice(i).norm() == Catch::Approx(0.0).margin(1e-15));
    }
}

TEST_CASE("Magic ddtorq_dbasestatedbasestate is zero", "[magic][hessian]") {
    Magic magic(valid_axis(), 0.1);
    auto H = magic.ddtorq_dbasestatedbasestate(0.02, valid_base_state());
    for (int i = 0; i < 3; ++i) {
        REQUIRE(H.slice(i).norm() == Catch::Approx(0.0).margin(1e-15));
    }
}

TEST_CASE("Magic torque is sign-preserving and odd in u", "[magic][torque]") {
    Magic magic(valid_axis(), 0.1);
    BaseState x = valid_base_state();
    Vec3 tau_pos = magic.torque(+0.03, x);
    Vec3 tau_neg = magic.torque(-0.03, x);
    REQUIRE(tau_pos.isApprox(-tau_neg));
}
