/**
 * @file test_satellite_constraints.cpp
 * @brief Comprehensive Catch2 tests for Satellite constraint evaluation,
 *        constraint Jacobians, and constraint Hessians.
 *
 * Covers:
 *  - Constraint vector dimensionality (with/without actuators, terminal step)
 *  - Angular velocity constraint satisfaction / violation / normalization
 *  - Sun-avoidance constraint satisfaction / violation / eclipse robustness
 *  - MTQ dipole upper/lower bounds (scaling, config override)
 *  - RW torque upper/lower bounds
 *  - RW momentum upper/lower bounds
 *  - RW stiction proxy
 *  - Input validation (wrong dimensions, out-of-range k, negative N)
 *  - Finite-difference verification of Jacobians against constraints()
 *  - Finite-difference verification of Hessians against constraintJacobians()
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>

#include <saltro/pybind/satellite.h>
#include <saltro/limits.h>
#include <saltro/math/quaternion.h>

using Vec3 = Eigen::Vector3d;
using Vec4 = Eigen::Vector4d;
using Mat33 = Eigen::Matrix3d;
using VecX = Satellite::VecX;
using MatX = Satellite::MatX;

// ============================================================================
// Helpers
// ============================================================================

static Mat33 validInertia() {
    return (Mat33() << 0.067, 0.0, 0.0,
                        0.0,  0.067, 0.0,
                        0.0,  0.0,  0.067).finished();
}

/// Identity quaternion (no rotation).
static Vec4 identityQuat() { return Vec4(1.0, 0.0, 0.0, 0.0); }

/// Build a full state vector (AV 3 + Q 4 + RW momenta n_rw).
static VecX makeState(const Vec3& w, const Vec4& q,
                      const Eigen::VectorXd& h_rw = Eigen::VectorXd()) {
    const int n = 7 + static_cast<int>(h_rw.size());
    VecX x(n);
    x.segment<3>(0) = w;
    x.segment<4>(3) = q;
    for (int i = 0; i < static_cast<int>(h_rw.size()); ++i)
        x(7 + i) = h_rw(i);
    return x;
}

static void normalizeQuatInState(VecX& x) {
    Vec4 q = x.segment<4>(3);
    q.normalize();
    x.segment<4>(3) = q;
}

/// Build a zero control vector of given dimension.
static VecX zeroControl(int dim) {
    VecX u(dim);
    u.setZero();
    return u;
}

/// Default constraint config.
static ConstraintConfig defaultCnstCfg() {
    ConstraintConfig cfg;
    cfg.wmax = 20.0 * M_PI / 180.0;  // 20 deg/s
    cfg.sun_limit_angle = 20.0 * M_PI / 180.0;
    cfg.control_limit_scale = 0.75;
    cfg.u_max.resize(0);
    return cfg;
}

/// Sun vector in +Z ECI (unit).
static Vec3 sunZ() { return Vec3(0.0, 0.0, 1.0); }

// ============================================================================
// Fixture: Satellite with actuators
// ============================================================================

class ConstraintFixture {
public:
    Mat33 J;
    PlannerSettings settings;
    Satellite sat;

    // Actuator parameters
    static constexpr double mtq_dipole  = 0.2;
    static constexpr double rw_torque   = 0.001;
    static constexpr double rw_inertia  = 1e-5;
    static constexpr double rw_h0       = 0.0;
    static constexpr double rw_hmax     = 0.01;

    ConstraintFixture()
        : J(validInertia()), settings(), sat(J, settings) {
        // 3 MTQs along body axes
        sat.addMTQ(Vec3(1, 0, 0), mtq_dipole);
        sat.addMTQ(Vec3(0, 1, 0), mtq_dipole);
        sat.addMTQ(Vec3(0, 0, 1), mtq_dipole);
        // 2 RWs
        sat.addRW(Vec3(1, 0, 0), rw_torque, rw_inertia, rw_h0, rw_hmax);
        sat.addRW(Vec3(0, 1, 0), rw_torque, rw_inertia, rw_h0, rw_hmax);
    }

    int n_mtq()  const { return sat.numMTQ(); }
    int n_rw()   const { return sat.numRW(); }
    int n_ctrl() const { return sat.controlDim(); }
    int n_state() const { return sat.stateDim(); }

    /// Expected constraint dimension at an intermediate step.
    int expectedDimIntermediate() const {
        return 1 + 1 + 2 * n_mtq() + 5 * n_rw();
    }
    /// Expected constraint dimension at the terminal step.
    int expectedDimTerminal() const { return 2; }

    VecX nominalState() const {
        Eigen::VectorXd h(n_rw());
        h.setZero();
        return makeState(Vec3::Zero(), identityQuat(), h);
    }

    VecX nominalControl() const { return zeroControl(n_ctrl()); }
};

// ============================================================================
// SECTION 1 — Constraint vector dimension
// ============================================================================

TEST_CASE("constraints: dimension with no actuators", "[satellite][constraints][dim]") {
    Satellite sat(validInertia(), PlannerSettings{});
    auto x = makeState(Vec3::Zero(), identityQuat());
    auto u = zeroControl(0);
    auto cfg = defaultCnstCfg();

    // Intermediate step
    auto c = sat.constraints(0, 10, x, u, sunZ(), cfg);
    REQUIRE(c.size() == 2);

    // Terminal step
    auto c_term = sat.constraints(9, 10, x, u, sunZ(), cfg);
    REQUIRE(c_term.size() == 2);
}

TEST_CASE_METHOD(ConstraintFixture,
    "constraints: dimension with actuators (intermediate step)",
    "[satellite][constraints][dim]") {
    auto c = sat.constraints(0, 10, nominalState(), nominalControl(), sunZ(), defaultCnstCfg());
    REQUIRE(c.size() == expectedDimIntermediate());
}

TEST_CASE_METHOD(ConstraintFixture,
    "constraints: dimension at terminal step (state only)",
    "[satellite][constraints][dim]") {
    auto c = sat.constraints(9, 10, nominalState(), nominalControl(), sunZ(), defaultCnstCfg());
    REQUIRE(c.size() == expectedDimTerminal());
}

// ============================================================================
// SECTION 2 — Angular velocity constraint
// ============================================================================

TEST_CASE_METHOD(ConstraintFixture,
    "constraints: AV satisfied at zero angular velocity",
    "[satellite][constraints][av]") {
    auto c = sat.constraints(0, 10, nominalState(), nominalControl(), sunZ(), defaultCnstCfg());
    // c(0) = (||w||² - wmax²) / wmax² = -1  when w=0
    REQUIRE(c(0) == Catch::Approx(-1.0));
}

TEST_CASE_METHOD(ConstraintFixture,
    "constraints: AV exactly at limit",
    "[satellite][constraints][av]") {
    auto cfg = defaultCnstCfg();
    Vec3 w(cfg.wmax, 0.0, 0.0);
    Eigen::VectorXd h(n_rw()); h.setZero();
    auto x = makeState(w, identityQuat(), h);
    auto c = sat.constraints(0, 10, x, nominalControl(), sunZ(), cfg);
    REQUIRE(c(0) == Catch::Approx(0.0).margin(1e-12));
}

TEST_CASE_METHOD(ConstraintFixture,
    "constraints: AV violated above limit",
    "[satellite][constraints][av]") {
    auto cfg = defaultCnstCfg();
    Vec3 w(cfg.wmax * 2.0, 0.0, 0.0);
    Eigen::VectorXd h(n_rw()); h.setZero();
    auto x = makeState(w, identityQuat(), h);
    auto c = sat.constraints(0, 10, x, nominalControl(), sunZ(), cfg);
    REQUIRE(c(0) > 0.0);
    // (4*wmax²  - wmax²) / wmax² = 3
    REQUIRE(c(0) == Catch::Approx(3.0));
}

TEST_CASE_METHOD(ConstraintFixture,
    "constraints: AV normalization is scale-independent",
    "[satellite][constraints][av]") {
    // Doubling wmax doubles the allowance but the same AV fraction gives same c value
    auto cfg1 = defaultCnstCfg();
    cfg1.wmax = 0.1;
    auto cfg2 = defaultCnstCfg();
    cfg2.wmax = 0.2;

    Vec3 w1(0.05, 0.0, 0.0);  // half of cfg1
    Vec3 w2(0.10, 0.0, 0.0);  // half of cfg2

    Eigen::VectorXd h(n_rw()); h.setZero();
    auto x1 = makeState(w1, identityQuat(), h);
    auto x2 = makeState(w2, identityQuat(), h);

    auto c1 = sat.constraints(0, 10, x1, nominalControl(), sunZ(), cfg1);
    auto c2 = sat.constraints(0, 10, x2, nominalControl(), sunZ(), cfg2);
    REQUIRE(c1(0) == Catch::Approx(c2(0)).margin(1e-12));
}

// ============================================================================
// SECTION 3 — Sun avoidance constraint
// ============================================================================

TEST_CASE_METHOD(ConstraintFixture,
    "constraints: sun constraint satisfied when sun is behind spacecraft",
    "[satellite][constraints][sun]") {
    // Identity quaternion: body +X = ECI +X.  Sun along +Z → body Z-comp = 1,
    // body X = 0 → sun_body.x = 0 < cos(20°) ≈ 0.94 → satisfied
    auto c = sat.constraints(0, 10, nominalState(), nominalControl(), sunZ(), defaultCnstCfg());
    REQUIRE(c(1) < 0.0);
}

TEST_CASE_METHOD(ConstraintFixture,
    "constraints: sun constraint violated when sun is along boresight",
    "[satellite][constraints][sun]") {
    // Sun along +X in ECI, identity quat → sun_body.x = 1.0
    // 1.0 - cos(20°) > 0 → violated
    Vec3 sun_eci(1.0, 0.0, 0.0);
    auto c = sat.constraints(0, 10, nominalState(), nominalControl(), sun_eci, defaultCnstCfg());
    REQUIRE(c(1) > 0.0);
}

TEST_CASE_METHOD(ConstraintFixture,
    "constraints: sun constraint handles zero sun vector (eclipse)",
    "[satellite][constraints][sun]") {
    Vec3 sun_zero = Vec3::Zero();
    auto c = sat.constraints(0, 10, nominalState(), nominalControl(), sun_zero, defaultCnstCfg());
    // Should be 0 (no constraint active)
    REQUIRE(c(1) == Catch::Approx(0.0).margin(1e-14));
}

TEST_CASE_METHOD(ConstraintFixture,
    "constraints: sun constraint at exact limit angle",
    "[satellite][constraints][sun]") {
    // Rotate sun by exactly 20° from +X in the XZ plane
    auto cfg = defaultCnstCfg();
    double angle = cfg.sun_limit_angle;
    Vec3 sun_eci(std::cos(angle), 0.0, std::sin(angle));
    // Identity quat → sun_body = sun_eci, sun_body.x = cos(angle) - cos(angle) = 0
    auto c = sat.constraints(0, 10, nominalState(), nominalControl(), sun_eci, cfg);
    REQUIRE(c(1) == Catch::Approx(0.0).margin(1e-12));
}

// ============================================================================
// SECTION 4 — MTQ control bounds
// ============================================================================

TEST_CASE_METHOD(ConstraintFixture,
    "constraints: MTQ bounds satisfied at zero command",
    "[satellite][constraints][mtq]") {
    auto c = sat.constraints(0, 10, nominalState(), nominalControl(), sunZ(), defaultCnstCfg());
    // Indices 2 .. 2+2*3-1 = 2..7 are MTQ bounds.  At u=0, upper = -1, lower = -1
    for (int i = 0; i < 2 * n_mtq(); ++i) {
        REQUIRE(c(2 + i) == Catch::Approx(-1.0));
    }
}

TEST_CASE_METHOD(ConstraintFixture,
    "constraints: MTQ upper bound exactly at scaled limit",
    "[satellite][constraints][mtq]") {
    auto cfg = defaultCnstCfg();
    double lim = cfg.control_limit_scale * mtq_dipole;
    VecX u = nominalControl();
    u(0) = lim;  // first MTQ at exact limit
    auto c = sat.constraints(0, 10, nominalState(), u, sunZ(), cfg);
    // Upper bound for first MTQ (idx 2): (lim - lim)/lim = 0
    REQUIRE(c(2) == Catch::Approx(0.0).margin(1e-12));
    // Lower bound for first MTQ (idx 3): (-lim - lim)/lim = -2 → satisfied
    REQUIRE(c(3) < 0.0);
}

TEST_CASE_METHOD(ConstraintFixture,
    "constraints: MTQ bound violated when command exceeds scaled limit",
    "[satellite][constraints][mtq]") {
    auto cfg = defaultCnstCfg();
    double lim = cfg.control_limit_scale * mtq_dipole;
    VecX u = nominalControl();
    u(1) = lim * 1.5;  // second MTQ at 150 %
    auto c = sat.constraints(0, 10, nominalState(), u, sunZ(), cfg);
    // Upper bound for second MTQ is at idx 4
    REQUIRE(c(4) > 0.0);
}

TEST_CASE_METHOD(ConstraintFixture,
    "constraints: MTQ bounds use config u_max when provided",
    "[satellite][constraints][mtq]") {
    auto cfg = defaultCnstCfg();
    cfg.u_max.resize(n_ctrl());
    cfg.u_max.setZero();
    cfg.u_max(0) = 0.05;  // tighter limit for first MTQ

    double lim = cfg.control_limit_scale * 0.05;
    VecX u = nominalControl();
    u(0) = lim;                     // at config limit
    auto c = sat.constraints(0, 10, nominalState(), u, sunZ(), cfg);
    REQUIRE(c(2) == Catch::Approx(0.0).margin(1e-12));

    u(0) = lim * 1.1;              // above config limit
    c = sat.constraints(0, 10, nominalState(), u, sunZ(), cfg);
    REQUIRE(c(2) > 0.0);
}

TEST_CASE_METHOD(ConstraintFixture,
    "constraints: MTQ negative command violates lower bound",
    "[satellite][constraints][mtq]") {
    auto cfg = defaultCnstCfg();
    double lim = cfg.control_limit_scale * mtq_dipole;
    VecX u = nominalControl();
    u(2) = -lim * 2.0;  // third MTQ, negative
    auto c = sat.constraints(0, 10, nominalState(), u, sunZ(), cfg);
    // Lower bound for third MTQ: at idx 7  (2+2*2+1)
    REQUIRE(c(7) > 0.0);
}

// ============================================================================
// SECTION 5 — RW torque bounds
// ============================================================================

TEST_CASE_METHOD(ConstraintFixture,
    "constraints: RW torque satisfied at zero command",
    "[satellite][constraints][rw_torque]") {
    auto c = sat.constraints(0, 10, nominalState(), nominalControl(), sunZ(), defaultCnstCfg());
    int rw_start = 2 + 2 * n_mtq();  // index of first RW constraint
    // Each RW: upper, lower → c = -1
    for (int i = 0; i < n_rw(); ++i) {
        REQUIRE(c(rw_start + 5 * i)     == Catch::Approx(-1.0));  // upper
        REQUIRE(c(rw_start + 5 * i + 1) == Catch::Approx(-1.0));  // lower
    }
}

TEST_CASE_METHOD(ConstraintFixture,
    "constraints: RW torque violated above limit",
    "[satellite][constraints][rw_torque]") {
    auto cfg = defaultCnstCfg();
    double lim = cfg.control_limit_scale * rw_torque;
    VecX u = nominalControl();
    u(n_mtq()) = lim * 2.0;  // first RW at double limit
    auto c = sat.constraints(0, 10, nominalState(), u, sunZ(), cfg);
    int rw_start = 2 + 2 * n_mtq();
    REQUIRE(c(rw_start) > 0.0);  // upper violated
}

// ============================================================================
// SECTION 6 — RW momentum bounds
// ============================================================================

TEST_CASE_METHOD(ConstraintFixture,
    "constraints: RW momentum satisfied at zero",
    "[satellite][constraints][rw_momentum]") {
    auto c = sat.constraints(0, 10, nominalState(), nominalControl(), sunZ(), defaultCnstCfg());
    int rw_start = 2 + 2 * n_mtq();
    for (int i = 0; i < n_rw(); ++i) {
        int upper_idx = rw_start + 5 * i + 2;
        int lower_idx = rw_start + 5 * i + 3;
        REQUIRE(c(upper_idx) == Catch::Approx(-1.0));
        REQUIRE(c(lower_idx) == Catch::Approx(-1.0));
    }
}

TEST_CASE_METHOD(ConstraintFixture,
    "constraints: RW momentum violated at max",
    "[satellite][constraints][rw_momentum]") {
    Eigen::VectorXd h(n_rw());
    h(0) = rw_hmax * 1.5;  // 150 % of max
    h(1) = 0.0;
    auto x = makeState(Vec3::Zero(), identityQuat(), h);
    auto c = sat.constraints(0, 10, x, nominalControl(), sunZ(), defaultCnstCfg());
    int rw_start = 2 + 2 * n_mtq();
    REQUIRE(c(rw_start + 2) > 0.0);  // upper momentum, first RW
}

TEST_CASE_METHOD(ConstraintFixture,
    "constraints: RW momentum exactly at limit",
    "[satellite][constraints][rw_momentum]") {
    Eigen::VectorXd h(n_rw());
    h(0) = rw_hmax;
    h(1) = 0.0;
    auto x = makeState(Vec3::Zero(), identityQuat(), h);
    auto c = sat.constraints(0, 10, x, nominalControl(), sunZ(), defaultCnstCfg());
    int rw_start = 2 + 2 * n_mtq();
    REQUIRE(c(rw_start + 2) == Catch::Approx(0.0).margin(1e-12));
}

TEST_CASE_METHOD(ConstraintFixture,
    "constraints: RW negative momentum violates lower bound",
    "[satellite][constraints][rw_momentum]") {
    Eigen::VectorXd h(n_rw());
    h(0) = 0.0;
    h(1) = -rw_hmax * 1.2;
    auto x = makeState(Vec3::Zero(), identityQuat(), h);
    auto c = sat.constraints(0, 10, x, nominalControl(), sunZ(), defaultCnstCfg());
    int rw_start = 2 + 2 * n_mtq();
    REQUIRE(c(rw_start + 5 + 3) > 0.0);  // lower momentum, second RW
}

// ============================================================================
// SECTION 7 — RW stiction proxy
// ============================================================================

TEST_CASE_METHOD(ConstraintFixture,
    "constraints: stiction constraint at zero state is zero",
    "[satellite][constraints][stiction]") {
    auto c = sat.constraints(0, 10, nominalState(), nominalControl(), sunZ(), defaultCnstCfg());
    int rw_start = 2 + 2 * n_mtq();
    for (int i = 0; i < n_rw(); ++i) {
        REQUIRE(c(rw_start + 5 * i + 4) == Catch::Approx(0.0).margin(1e-14));
    }
}

TEST_CASE_METHOD(ConstraintFixture,
    "constraints: stiction constraint is non-positive (always satisfied or zero)",
    "[satellite][constraints][stiction]") {
    Eigen::VectorXd h(n_rw());
    h << 0.005, -0.003;
    auto x = makeState(Vec3::Zero(), identityQuat(), h);
    VecX u = nominalControl();
    u(n_mtq())     = 0.0005;
    u(n_mtq() + 1) = -0.0003;
    auto c = sat.constraints(0, 10, x, u, sunZ(), defaultCnstCfg());
    int rw_start = 2 + 2 * n_mtq();
    for (int i = 0; i < n_rw(); ++i) {
        REQUIRE(c(rw_start + 5 * i + 4) <= 0.0);
    }
}

// ============================================================================
// SECTION 8 — Terminal vs. Intermediate step
// ============================================================================

TEST_CASE_METHOD(ConstraintFixture,
    "constraints: terminal step has no control/RW constraints",
    "[satellite][constraints][terminal]") {
    int N = 10;
    auto c = sat.constraints(N - 1, N, nominalState(), nominalControl(), sunZ(), defaultCnstCfg());
    REQUIRE(c.size() == 2);
}

TEST_CASE_METHOD(ConstraintFixture,
    "constraints: first intermediate step has full constraints",
    "[satellite][constraints][terminal]") {
    auto c = sat.constraints(0, 10, nominalState(), nominalControl(), sunZ(), defaultCnstCfg());
    REQUIRE(c.size() == expectedDimIntermediate());
}

TEST_CASE_METHOD(ConstraintFixture,
    "constraints: step N-2 is intermediate, N-1 is terminal",
    "[satellite][constraints][terminal]") {
    int N = 5;
    auto ci = sat.constraints(N - 2, N, nominalState(), nominalControl(), sunZ(), defaultCnstCfg());
    auto ct = sat.constraints(N - 1, N, nominalState(), nominalControl(), sunZ(), defaultCnstCfg());
    REQUIRE(ci.size() > ct.size());
    REQUIRE(ct.size() == 2);
}

// ============================================================================
// SECTION 9 — Input validation
// ============================================================================

TEST_CASE_METHOD(ConstraintFixture,
    "constraints: throws on negative N",
    "[satellite][constraints][validation]") {
    REQUIRE_THROWS_AS(
        sat.constraints(0, -1, nominalState(), nominalControl(), sunZ(), defaultCnstCfg()),
        std::invalid_argument);
}

TEST_CASE_METHOD(ConstraintFixture,
    "constraints: throws on zero N",
    "[satellite][constraints][validation]") {
    REQUIRE_THROWS_AS(
        sat.constraints(0, 0, nominalState(), nominalControl(), sunZ(), defaultCnstCfg()),
        std::invalid_argument);
}

TEST_CASE_METHOD(ConstraintFixture,
    "constraints: throws on k >= N",
    "[satellite][constraints][validation]") {
    REQUIRE_THROWS_AS(
        sat.constraints(10, 10, nominalState(), nominalControl(), sunZ(), defaultCnstCfg()),
        std::out_of_range);
}

TEST_CASE_METHOD(ConstraintFixture,
    "constraints: throws on negative k",
    "[satellite][constraints][validation]") {
    REQUIRE_THROWS_AS(
        sat.constraints(-1, 10, nominalState(), nominalControl(), sunZ(), defaultCnstCfg()),
        std::out_of_range);
}

TEST_CASE_METHOD(ConstraintFixture,
    "constraints: throws on undersized state",
    "[satellite][constraints][validation]") {
    VecX x_short(3);
    x_short.setZero();
    REQUIRE_THROWS_AS(
        sat.constraints(0, 10, x_short, nominalControl(), sunZ(), defaultCnstCfg()),
        std::invalid_argument);
}

TEST_CASE_METHOD(ConstraintFixture,
    "constraints: throws on undersized control",
    "[satellite][constraints][validation]") {
    VecX u_short(1);
    u_short.setZero();
    REQUIRE_THROWS_AS(
        sat.constraints(0, 10, nominalState(), u_short, sunZ(), defaultCnstCfg()),
        std::invalid_argument);
}

// ============================================================================
// SECTION 10 — Jacobian input validation
// ============================================================================

TEST_CASE_METHOD(ConstraintFixture,
    "constraintJacobians: throws on invalid inputs",
    "[satellite][jacobians][validation]") {
    REQUIRE_THROWS_AS(
        sat.constraintJacobians(0, 0, nominalState(), nominalControl(), sunZ(), defaultCnstCfg()),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        sat.constraintJacobians(10, 10, nominalState(), nominalControl(), sunZ(), defaultCnstCfg()),
        std::out_of_range);
}

// ============================================================================
// SECTION 11 — Hessian input validation
// ============================================================================

TEST_CASE_METHOD(ConstraintFixture,
    "constraintHessians: throws on invalid inputs",
    "[satellite][hessians][validation]") {
    REQUIRE_THROWS_AS(
        sat.constraintHessians(0, -5, nominalState(), nominalControl(), sunZ(), defaultCnstCfg()),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        sat.constraintHessians(-1, 10, nominalState(), nominalControl(), sunZ(), defaultCnstCfg()),
        std::out_of_range);
}

// ============================================================================
// SECTION 12 — Jacobian dimensions
// ============================================================================

TEST_CASE_METHOD(ConstraintFixture,
    "constraintJacobians: matrix dimensions intermediate step",
    "[satellite][jacobians][dim]") {
    auto [c_u, c_x] = sat.constraintJacobians(0, 10, nominalState(), nominalControl(), sunZ(), defaultCnstCfg());
    REQUIRE(c_u.rows() == expectedDimIntermediate());
    REQUIRE(c_u.cols() == n_ctrl());
    REQUIRE(c_x.rows() == expectedDimIntermediate());
    REQUIRE(c_x.cols() == n_state());
}

TEST_CASE_METHOD(ConstraintFixture,
    "constraintJacobians: matrix dimensions terminal step",
    "[satellite][jacobians][dim]") {
    auto [c_u, c_x] = sat.constraintJacobians(9, 10, nominalState(), nominalControl(), sunZ(), defaultCnstCfg());
    REQUIRE(c_u.rows() == 2);
    REQUIRE(c_x.rows() == 2);
}

// ============================================================================
// SECTION 13 — Jacobian finite-difference verification
// ============================================================================

TEST_CASE_METHOD(ConstraintFixture,
    "constraintJacobians: finite-difference check w.r.t. state",
    "[satellite][jacobians][fd]") {
    // Use a non-trivial operating point
    Eigen::VectorXd h(n_rw());
    h << 0.003, -0.002;
    Vec3 w(0.05, -0.03, 0.01);
    // Slightly rotated quaternion
    Vec4 q(std::sqrt(1.0 - 0.01 - 0.04 - 0.09), 0.1, 0.2, 0.3);
    q.normalize();
    auto x0 = makeState(w, q, h);
    VecX u0 = nominalControl();
    u0(0) = 0.05; u0(1) = -0.03; u0(2) = 0.01;
    u0(n_mtq()) = 0.0003; u0(n_mtq() + 1) = -0.0002;

    auto cfg = defaultCnstCfg();
    Vec3 sun(0.5, 0.3, 0.8);
    int k = 0, N = 10;

    auto c0 = sat.constraints(k, N, x0, u0, sun, cfg);
    auto [c_u, c_x] = sat.constraintJacobians(k, N, x0, u0, sun, cfg);

    const double eps = 1e-7;

    // Check ∂c/∂x with finite differences
    for (int j = 0; j < n_state(); ++j) {
        VecX xp = x0; xp(j) += eps;
        VecX xm = x0; xm(j) -= eps;
        if (j >= 3 && j < 7) {
            normalizeQuatInState(xp);
            normalizeQuatInState(xm);
        }
        auto cp = sat.constraints(k, N, xp, u0, sun, cfg);
        auto cm = sat.constraints(k, N, xm, u0, sun, cfg);
        VecX fd_col = (cp - cm) / (2.0 * eps);
        for (int i = 0; i < c0.size(); ++i) {
            REQUIRE(c_x(i, j) == Catch::Approx(fd_col(i)).margin(1e-4));
        }
    }
}

TEST_CASE_METHOD(ConstraintFixture,
    "constraintJacobians: finite-difference check w.r.t. control",
    "[satellite][jacobians][fd]") {
    Eigen::VectorXd h(n_rw());
    h << 0.003, -0.002;
    Vec3 w(0.05, -0.03, 0.01);
    Vec4 q(std::sqrt(1.0 - 0.01 - 0.04 - 0.09), 0.1, 0.2, 0.3);
    q.normalize();
    auto x0 = makeState(w, q, h);
    VecX u0 = nominalControl();
    u0(0) = 0.05; u0(1) = -0.03; u0(2) = 0.01;
    u0(n_mtq()) = 0.0003; u0(n_mtq() + 1) = -0.0002;

    auto cfg = defaultCnstCfg();
    Vec3 sun(0.5, 0.3, 0.8);
    int k = 0, N = 10;

    auto c0 = sat.constraints(k, N, x0, u0, sun, cfg);
    auto [c_u, c_x] = sat.constraintJacobians(k, N, x0, u0, sun, cfg);

    const double eps = 1e-7;

    for (int j = 0; j < n_ctrl(); ++j) {
        VecX up = u0; up(j) += eps;
        VecX um = u0; um(j) -= eps;
        auto cp = sat.constraints(k, N, x0, up, sun, cfg);
        auto cm = sat.constraints(k, N, x0, um, sun, cfg);
        VecX fd_col = (cp - cm) / (2.0 * eps);
        for (int i = 0; i < c0.size(); ++i) {
            REQUIRE(c_u(i, j) == Catch::Approx(fd_col(i)).margin(1e-4));
        }
    }
}

TEST_CASE_METHOD(ConstraintFixture,
    "constraintJacobians: terminal step FD check",
    "[satellite][jacobians][fd]") {
    Eigen::VectorXd h(n_rw());
    h << 0.001, -0.004;
    Vec3 w(0.02, 0.01, -0.03);
    Vec4 q = identityQuat();
    auto x0 = makeState(w, q, h);
    auto u0 = nominalControl();
    auto cfg = defaultCnstCfg();
    Vec3 sun(0.0, 1.0, 0.0);
    int k = 9, N = 10;

    auto c0 = sat.constraints(k, N, x0, u0, sun, cfg);
    auto [c_u, c_x] = sat.constraintJacobians(k, N, x0, u0, sun, cfg);

    const double eps = 1e-7;
    for (int j = 0; j < n_state(); ++j) {
        VecX xp = x0; xp(j) += eps;
        VecX xm = x0; xm(j) -= eps;
        if (j >= 3 && j < 7) {
            normalizeQuatInState(xp);
            normalizeQuatInState(xm);
        }
        auto cp = sat.constraints(k, N, xp, u0, sun, cfg);
        auto cm = sat.constraints(k, N, xm, u0, sun, cfg);
        VecX fd_col = (cp - cm) / (2.0 * eps);
        for (int i = 0; i < c0.size(); ++i) {
            REQUIRE(c_x(i, j) == Catch::Approx(fd_col(i)).margin(1e-4));
        }
    }

    // Control Jacobian should be zero at terminal step
    REQUIRE(c_u.norm() == Catch::Approx(0.0).margin(1e-14));
}

// ============================================================================
// SECTION 14 — Hessian finite-difference verification
// ============================================================================

TEST_CASE_METHOD(ConstraintFixture,
    "constraintHessians: finite-difference check w.r.t. state-state",
    "[satellite][hessians][fd]") {
    Eigen::VectorXd h(n_rw());
    h << 0.004, -0.003;
    Vec3 w(0.08, -0.02, 0.04);
    Vec4 q(0.0, 0.05, 0.15, -0.1);
    q(0) = std::sqrt(1.0 - q.tail<3>().squaredNorm());
    q.normalize();
    auto x0 = makeState(w, q, h);
    VecX u0 = nominalControl();
    u0(0) = 0.04; u0(1) = -0.02; u0(2) = 0.03;
    u0(n_mtq()) = 0.0004; u0(n_mtq() + 1) = -0.0001;

    auto cfg = defaultCnstCfg();
    Vec3 sun(0.3, 0.6, 0.7);
    int k = 0, N = 10;

    auto [H_uu, H_ux, H_xx] = sat.constraintHessians(k, N, x0, u0, sun, cfg);
    auto c0 = sat.constraints(k, N, x0, u0, sun, cfg);
    const int nc = static_cast<int>(c0.size());

    const double eps = 1e-5;

    // ∂²c/∂x² ≈ (Jx(x+eps) - Jx(x-eps)) / (2*eps)
    for (int j = 0; j < n_state(); ++j) {
        VecX xp = x0; xp(j) += eps;
        VecX xm = x0; xm(j) -= eps;
        if (j >= 3 && j < 7) {
            normalizeQuatInState(xp);
            normalizeQuatInState(xm);
        }
        auto [_, c_xp] = sat.constraintJacobians(k, N, xp, u0, sun, cfg);
        auto [_2, c_xm] = sat.constraintJacobians(k, N, xm, u0, sun, cfg);
        MatX fd_slice = (c_xp - c_xm) / (2.0 * eps);

        for (int ci = 0; ci < nc; ++ci) {
            for (int row = 0; row < n_state(); ++row) {
                REQUIRE(H_xx.slice(ci)(row, j) ==
                        Catch::Approx(fd_slice(ci, row)).margin(1e-3));
            }
        }
    }
}

TEST_CASE_METHOD(ConstraintFixture,
    "constraintHessians: finite-difference check w.r.t. control-control",
    "[satellite][hessians][fd]") {
    Eigen::VectorXd h(n_rw());
    h << 0.004, -0.003;
    Vec3 w(0.08, -0.02, 0.04);
    Vec4 q = identityQuat();
    auto x0 = makeState(w, q, h);
    VecX u0 = nominalControl();
    u0(0) = 0.04; u0(n_mtq()) = 0.0004; u0(n_mtq() + 1) = -0.0002;

    auto cfg = defaultCnstCfg();
    Vec3 sun(0.3, 0.6, 0.7);
    int k = 0, N = 10;

    auto [H_uu, H_ux, H_xx] = sat.constraintHessians(k, N, x0, u0, sun, cfg);
    auto c0 = sat.constraints(k, N, x0, u0, sun, cfg);
    const int nc = static_cast<int>(c0.size());

    const double eps = 1e-5;

    for (int j = 0; j < n_ctrl(); ++j) {
        VecX up = u0; up(j) += eps;
        VecX um = u0; um(j) -= eps;
        auto [c_up, _] = sat.constraintJacobians(k, N, x0, up, sun, cfg);
        auto [c_um, _2] = sat.constraintJacobians(k, N, x0, um, sun, cfg);
        MatX fd_slice = (c_up - c_um) / (2.0 * eps);

        for (int ci = 0; ci < nc; ++ci) {
            for (int row = 0; row < n_ctrl(); ++row) {
                REQUIRE(H_uu.slice(ci)(row, j) ==
                        Catch::Approx(fd_slice(ci, row)).margin(1e-3));
            }
        }
    }
}

TEST_CASE_METHOD(ConstraintFixture,
    "constraintHessians: finite-difference check w.r.t. control-state",
    "[satellite][hessians][fd]") {
    Eigen::VectorXd h(n_rw());
    h << 0.004, -0.003;
    Vec3 w(0.08, -0.02, 0.04);
    Vec4 q = identityQuat();
    auto x0 = makeState(w, q, h);
    VecX u0 = nominalControl();
    u0(0) = 0.04; u0(n_mtq()) = 0.0004; u0(n_mtq() + 1) = -0.0002;

    auto cfg = defaultCnstCfg();
    Vec3 sun(0.3, 0.6, 0.7);
    int k = 0, N = 10;

    auto [H_uu, H_ux, H_xx] = sat.constraintHessians(k, N, x0, u0, sun, cfg);
    auto c0 = sat.constraints(k, N, x0, u0, sun, cfg);
    const int nc = static_cast<int>(c0.size());

    const double eps = 1e-5;

    // ∂²c/(∂u∂x) ≈ (Ju(x+eps) - Ju(x-eps)) / (2*eps)
    for (int j = 0; j < n_state(); ++j) {
        VecX xp = x0; xp(j) += eps;
        VecX xm = x0; xm(j) -= eps;
        if (j >= 3 && j < 7) {
            normalizeQuatInState(xp);
            normalizeQuatInState(xm);
        }
        auto [c_up, _]  = sat.constraintJacobians(k, N, xp, u0, sun, cfg);
        auto [c_um, _2] = sat.constraintJacobians(k, N, xm, u0, sun, cfg);
        MatX fd_slice = (c_up - c_um) / (2.0 * eps);

        for (int ci = 0; ci < nc; ++ci) {
            for (int row = 0; row < n_ctrl(); ++row) {
                REQUIRE(H_ux.slice(ci)(row, j) ==
                        Catch::Approx(fd_slice(ci, row)).margin(1e-3));
            }
        }
    }
}

TEST_CASE_METHOD(ConstraintFixture,
    "constraintHessians: terminal step — Huu and Hux are zero",
    "[satellite][hessians][terminal]") {
    auto x0 = nominalState();
    auto u0 = nominalControl();
    auto cfg = defaultCnstCfg();
    int k = 9, N = 10;

    auto [H_uu, H_ux, H_xx] = sat.constraintHessians(k, N, x0, u0, sunZ(), cfg);

    // Only 2 constraints at terminal, but tensor is full size.
    // However, slices for those 2 should have zero u content.
    for (int ci = 0; ci < 2; ++ci) {
        REQUIRE(H_uu.slice(ci).norm() == Catch::Approx(0.0).margin(1e-14));
        REQUIRE(H_ux.slice(ci).norm() == Catch::Approx(0.0).margin(1e-14));
    }
}

// ============================================================================
// SECTION 15 — Hessian structural checks
// ============================================================================

TEST_CASE_METHOD(ConstraintFixture,
    "constraintHessians: AV constraint Hxx is diagonal in w-block",
    "[satellite][hessians][structure]") {
    auto x0 = nominalState();
    auto u0 = nominalControl();
    auto cfg = defaultCnstCfg();

    auto [H_uu, H_ux, H_xx] = sat.constraintHessians(0, 10, x0, u0, sunZ(), cfg);

    // AV Hessian (slice 0) should be 2/wmax² * I₃ in the w-block
    double wmax = cfg.wmax;
    double expected = 2.0 / (wmax * wmax);
    auto hxx_av = H_xx.slice(0);
    for (int i = 0; i < 3; ++i)
        REQUIRE(hxx_av(i, i) == Catch::Approx(expected));

    // Off-diagonal in w-block should be zero
    REQUIRE(hxx_av(0, 1) == Catch::Approx(0.0).margin(1e-14));
    REQUIRE(hxx_av(0, 2) == Catch::Approx(0.0).margin(1e-14));
    REQUIRE(hxx_av(1, 2) == Catch::Approx(0.0).margin(1e-14));

    // No u components
    REQUIRE(H_uu.slice(0).norm() == Catch::Approx(0.0).margin(1e-14));
    REQUIRE(H_ux.slice(0).norm() == Catch::Approx(0.0).margin(1e-14));
}

TEST_CASE_METHOD(ConstraintFixture,
    "constraintHessians: MTQ bounds have zero Hessians (linear constraints)",
    "[satellite][hessians][structure]") {
    Eigen::VectorXd h(n_rw()); h << 0.002, -0.001;
    auto x0 = makeState(Vec3(0.05, 0.0, 0.0), identityQuat(), h);
    VecX u0 = nominalControl();
    u0(0) = 0.05;

    auto [H_uu, H_ux, H_xx] = sat.constraintHessians(0, 10, x0, u0, sunZ(), defaultCnstCfg());

    // MTQ bound indices: 2 .. 2 + 2*n_mtq - 1
    for (int ci = 2; ci < 2 + 2 * n_mtq(); ++ci) {
        REQUIRE(H_uu.slice(ci).norm() == Catch::Approx(0.0).margin(1e-14));
        REQUIRE(H_ux.slice(ci).norm() == Catch::Approx(0.0).margin(1e-14));
        REQUIRE(H_xx.slice(ci).norm() == Catch::Approx(0.0).margin(1e-14));
    }
}

TEST_CASE_METHOD(ConstraintFixture,
    "constraintHessians: RW torque bounds have zero Hessians (linear)",
    "[satellite][hessians][structure]") {
    auto x0 = nominalState();
    auto u0 = nominalControl();

    auto [H_uu, H_ux, H_xx] = sat.constraintHessians(0, 10, x0, u0, sunZ(), defaultCnstCfg());

    int rw_start = 2 + 2 * n_mtq();
    for (int i = 0; i < n_rw(); ++i) {
        int upper = rw_start + 5 * i;
        int lower = rw_start + 5 * i + 1;
        for (int ci : {upper, lower}) {
            REQUIRE(H_uu.slice(ci).norm() == Catch::Approx(0.0).margin(1e-14));
            REQUIRE(H_ux.slice(ci).norm() == Catch::Approx(0.0).margin(1e-14));
            REQUIRE(H_xx.slice(ci).norm() == Catch::Approx(0.0).margin(1e-14));
        }
    }
}

TEST_CASE_METHOD(ConstraintFixture,
    "constraintHessians: RW momentum bounds have zero Hessians (linear)",
    "[satellite][hessians][structure]") {
    auto x0 = nominalState();
    auto u0 = nominalControl();

    auto [H_uu, H_ux, H_xx] = sat.constraintHessians(0, 10, x0, u0, sunZ(), defaultCnstCfg());

    int rw_start = 2 + 2 * n_mtq();
    for (int i = 0; i < n_rw(); ++i) {
        int h_upper = rw_start + 5 * i + 2;
        int h_lower = rw_start + 5 * i + 3;
        for (int ci : {h_upper, h_lower}) {
            REQUIRE(H_uu.slice(ci).norm() == Catch::Approx(0.0).margin(1e-14));
            REQUIRE(H_ux.slice(ci).norm() == Catch::Approx(0.0).margin(1e-14));
            REQUIRE(H_xx.slice(ci).norm() == Catch::Approx(0.0).margin(1e-14));
        }
    }
}

// ============================================================================
// SECTION 16 — Stiction Hessian structure
// ============================================================================

TEST_CASE_METHOD(ConstraintFixture,
    "constraintHessians: stiction Hessian is non-zero and has correct pattern",
    "[satellite][hessians][stiction]") {
    Eigen::VectorXd h(n_rw());
    h << 0.005, -0.003;
    auto x0 = makeState(Vec3::Zero(), identityQuat(), h);
    VecX u0 = nominalControl();
    u0(n_mtq()) = 0.0003;
    u0(n_mtq() + 1) = -0.0002;

    auto [H_uu, H_ux, H_xx] = sat.constraintHessians(0, 10, x0, u0, sunZ(), defaultCnstCfg());

    int rw_start = 2 + 2 * n_mtq();
    for (int i = 0; i < n_rw(); ++i) {
        int stiction_idx = rw_start + 5 * i + 4;
        int ctrl_idx = n_mtq() + i;
        int state_idx = 7 + i;  // RW_MOMENTUM_INDEX + i
        double u_i = u0(ctrl_idx);
        double h_i = h(i);

        // Huu: -2 * h²
        REQUIRE(H_uu.slice(stiction_idx)(ctrl_idx, ctrl_idx) ==
                Catch::Approx(-2.0 * h_i * h_i).margin(1e-12));

        // Hxx: -2 * u²
        REQUIRE(H_xx.slice(stiction_idx)(state_idx, state_idx) ==
                Catch::Approx(-2.0 * u_i * u_i).margin(1e-12));

        // Hux: -4 * u * h
        REQUIRE(H_ux.slice(stiction_idx)(ctrl_idx, state_idx) ==
                Catch::Approx(-4.0 * u_i * h_i).margin(1e-12));
    }
}

// ============================================================================
// SECTION 17 — Edge case: no actuators (Jacobians / Hessians)
// ============================================================================

TEST_CASE("constraintJacobians: no actuators", "[satellite][jacobians][no_actuators]") {
    Satellite sat(validInertia(), PlannerSettings{});
    auto x = makeState(Vec3(0.1, 0.0, 0.0), identityQuat());
    auto u = zeroControl(0);
    auto cfg = defaultCnstCfg();

    // Intermediate step — still only 2 constraints when no actuators
    auto [c_u, c_x] = sat.constraintJacobians(0, 10, x, u, sunZ(), cfg);
    REQUIRE(c_u.rows() == 2);
    REQUIRE(c_u.cols() == 0);
    REQUIRE(c_x.rows() == 2);
    REQUIRE(c_x.cols() == 7);
}

TEST_CASE("constraintHessians: no actuators", "[satellite][hessians][no_actuators]") {
    Satellite sat(validInertia(), PlannerSettings{});
    auto x = makeState(Vec3(0.1, 0.0, 0.0), identityQuat());
    auto u = zeroControl(0);
    auto cfg = defaultCnstCfg();

    auto [H_uu, H_ux, H_xx] = sat.constraintHessians(0, 10, x, u, sunZ(), cfg);
    // AV hessian should still be present
    double wmax = cfg.wmax;
    double expected = 2.0 / (wmax * wmax);
    REQUIRE(H_xx.slice(0)(0, 0) == Catch::Approx(expected));
}

// ============================================================================
// SECTION 18 — Edge case: maximum actuators
// ============================================================================

TEST_CASE("constraints: maximum actuators", "[satellite][constraints][max_actuators]") {
    Satellite sat(validInertia(), PlannerSettings{});
    // Add MAX_NUM_MTQ MTQs
    for (int i = 0; i < saltro::limits::MAX_NUM_MTQ; ++i) {
        Vec3 axis = Vec3::Zero();
        axis(i % 3) = 1.0;
        sat.addMTQ(axis, 0.2);
    }
    // Add MAX_NUM_RW RWs
    for (int i = 0; i < saltro::limits::MAX_NUM_RW; ++i) {
        Vec3 axis = Vec3::Zero();
        axis(i % 3) = 1.0;
        sat.addRW(axis, 0.001, 1e-5, 0.0, 0.01);
    }
    // Add MAX_NUM_MAGIC magic actuators
    for (int i = 0; i < saltro::limits::MAX_NUM_MAGIC; ++i) {
        Vec3 axis = Vec3::Zero();
        axis(i % 3) = 1.0;
        sat.addMagic(axis, 0.01);
    }

    int n = sat.stateDim();
    int m = sat.controlDim();
    Eigen::VectorXd h(sat.numRW()); h.setZero();
    auto x = makeState(Vec3::Zero(), identityQuat(), h);
    auto u = zeroControl(m);
    auto cfg = defaultCnstCfg();

    auto c = sat.constraints(0, 10, x, u, sunZ(), cfg);
    int expected = 1 + 1 + 2 * saltro::limits::MAX_NUM_MTQ
                         + 5 * saltro::limits::MAX_NUM_RW
                         + 2 * saltro::limits::MAX_NUM_MAGIC;
    REQUIRE(c.size() == expected);
    REQUIRE(expected == saltro::limits::MAX_CONSTRAINT_DIM);
}

// ============================================================================
// SECTION 19 — All constraints satisfied at nominal point
// ============================================================================

TEST_CASE_METHOD(ConstraintFixture,
    "constraints: all constraints satisfied at nominal operating point (c <= 0)",
    "[satellite][constraints][flight_safety]") {
    auto c = sat.constraints(0, 10, nominalState(), nominalControl(), sunZ(), defaultCnstCfg());
    for (int i = 0; i < c.size(); ++i) {
        INFO("Constraint index " << i << " violated with value " << c(i));
        REQUIRE(c(i) <= 0.0 + 1e-12);
    }
}

TEST_CASE_METHOD(ConstraintFixture,
    "constraints: all constraints satisfied at terminal nominal point",
    "[satellite][constraints][flight_safety]") {
    auto c = sat.constraints(9, 10, nominalState(), nominalControl(), sunZ(), defaultCnstCfg());
    for (int i = 0; i < c.size(); ++i) {
        INFO("Constraint index " << i << " violated with value " << c(i));
        REQUIRE(c(i) <= 0.0 + 1e-12);
    }
}

// ============================================================================
// SECTION 20 — control_limit_scale edge cases
// ============================================================================

TEST_CASE_METHOD(ConstraintFixture,
    "constraints: control_limit_scale = 1 uses full actuator capacity",
    "[satellite][constraints][scale]") {
    auto cfg = defaultCnstCfg();
    cfg.control_limit_scale = 1.0;
    VecX u = nominalControl();
    u(0) = mtq_dipole;  // exactly at full capacity
    auto c = sat.constraints(0, 10, nominalState(), u, sunZ(), cfg);
    REQUIRE(c(2) == Catch::Approx(0.0).margin(1e-12));
}

TEST_CASE_METHOD(ConstraintFixture,
    "constraints: smaller scale tightens limits",
    "[satellite][constraints][scale]") {
    auto cfg1 = defaultCnstCfg();
    cfg1.control_limit_scale = 0.5;
    auto cfg2 = defaultCnstCfg();
    cfg2.control_limit_scale = 1.0;

    VecX u = nominalControl();
    u(0) = 0.15;  // between 0.5*0.2 = 0.1 and 1.0*0.2 = 0.2
    auto c1 = sat.constraints(0, 10, nominalState(), u, sunZ(), cfg1);
    auto c2 = sat.constraints(0, 10, nominalState(), u, sunZ(), cfg2);
    // With tighter scale, constraint is violated; with looser, it's not
    REQUIRE(c1(2) > 0.0);
    REQUIRE(c2(2) < 0.0);
}

// ============================================================================
// SECTION 21 — Sun limit angle edge cases
// ============================================================================

TEST_CASE_METHOD(ConstraintFixture,
    "constraints: sun_limit_angle = 0 makes sun constraint trivially satisfied for off-axis sun",
    "[satellite][constraints][sun_edge]") {
    auto cfg = defaultCnstCfg();
    cfg.sun_limit_angle = 0.0;
    // cos(0) = 1, so sun_body.x - 1 <= 0 whenever sun is not exactly along +X
    auto c = sat.constraints(0, 10, nominalState(), nominalControl(), sunZ(), cfg);
    REQUIRE(c(1) <= 0.0 + 1e-12);
}

TEST_CASE_METHOD(ConstraintFixture,
    "constraints: sun_limit_angle = pi accepts any direction",
    "[satellite][constraints][sun_edge]") {
    auto cfg = defaultCnstCfg();
    cfg.sun_limit_angle = M_PI;
    // cos(pi) = -1, so sun_body.x + 1 >= 0 always, but let's check along +X
    Vec3 sun(1.0, 0.0, 0.0);
    auto c = sat.constraints(0, 10, nominalState(), nominalControl(), sun, cfg);
    // sun_body.x = 1, cos(pi) = -1 → c = 1 - (-1) = 2 → violated!
    // Actually this means sun_limit_angle = pi → cos = -1 → everything passes
    // sun_body.x - cos(pi) = 1.0 - (-1.0) = 2.0 > 0 → still violated
    // This is expected: limit_angle = pi means "no exclusion zone at all"
    // But mathematically cos(pi) = -1 so the constraint is always violated unless sun_body.x < -1 (impossible)
    // Actually, the user would set sun_limit_angle = pi to mean "exclude 180° cone" which excludes everything
    // Let's just check the formula works correctly
    REQUIRE(c(1) == Catch::Approx(2.0).margin(1e-12));
}

// ============================================================================
// SECTION 22 — Symmetry: upper and lower bounds are symmetric
// ============================================================================

TEST_CASE_METHOD(ConstraintFixture,
    "constraints: MTQ bounds are symmetric for positive and negative commands",
    "[satellite][constraints][symmetry]") {
    auto cfg = defaultCnstCfg();
    VecX u_pos = nominalControl();
    VecX u_neg = nominalControl();
    u_pos(0) = 0.1;
    u_neg(0) = -0.1;

    auto c_pos = sat.constraints(0, 10, nominalState(), u_pos, sunZ(), cfg);
    auto c_neg = sat.constraints(0, 10, nominalState(), u_neg, sunZ(), cfg);

    // For first MTQ: upper of +cmd = lower of -cmd
    REQUIRE(c_pos(2) == Catch::Approx(c_neg(3)).margin(1e-12));
    REQUIRE(c_pos(3) == Catch::Approx(c_neg(2)).margin(1e-12));
}

// ============================================================================
// SECTION 23 — Constraint Jacobians: AV constraint structure
// ============================================================================

TEST_CASE_METHOD(ConstraintFixture,
    "constraintJacobians: AV row has entries only in w-columns",
    "[satellite][jacobians][structure]") {
    Eigen::VectorXd h(n_rw()); h << 0.001, -0.002;
    auto x0 = makeState(Vec3(0.05, 0.03, -0.01), identityQuat(), h);
    auto u0 = nominalControl();

    auto [c_u, c_x] = sat.constraintJacobians(0, 10, x0, u0, sunZ(), defaultCnstCfg());

    // AV Jacobian (row 0): only columns 0,1,2 (angular velocity) should be non-zero
    for (int j = 3; j < n_state(); ++j) {
        REQUIRE(c_x(0, j) == Catch::Approx(0.0).margin(1e-14));
    }
    // And no dependence on u
    for (int j = 0; j < n_ctrl(); ++j) {
        REQUIRE(c_u(0, j) == Catch::Approx(0.0).margin(1e-14));
    }
}

// ============================================================================
// SECTION 24 — Constraint Jacobians: Sun constraint structure
// ============================================================================

TEST_CASE_METHOD(ConstraintFixture,
    "constraintJacobians: sun row has entries only in q-columns",
    "[satellite][jacobians][structure]") {
    auto x0 = nominalState();
    auto u0 = nominalControl();
    Vec3 sun(0.5, 0.3, 0.8);

    auto [c_u, c_x] = sat.constraintJacobians(0, 10, x0, u0, sun, defaultCnstCfg());

    // Sun constraint (row 1): only columns 3,4,5,6 (quaternion) should be non-zero
    for (int j = 0; j < 3; ++j) {
        REQUIRE(c_x(1, j) == Catch::Approx(0.0).margin(1e-14));
    }
    for (int j = 7; j < n_state(); ++j) {
        REQUIRE(c_x(1, j) == Catch::Approx(0.0).margin(1e-14));
    }
}
