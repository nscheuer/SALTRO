#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Dense>

#include <saltro/pybind/disturbances/ggdisturbance.h>
#include <saltro/pybind/plannersettings.h>
#include <saltro/constants/constants.h>

using namespace saltro::disturbances;
using Vec3 = Eigen::Vector3d;
using Mat33 = Eigen::Matrix3d;

namespace {
DisturbanceConfig makeDistCfg(bool plan_for_gg) {
    DisturbanceConfig cfg;
    cfg.plan_for_gg = plan_for_gg;
    return cfg;
}
}

TEST_CASE("GGDisturbance returns zero when disabled", "[ggdisturbance]") {
    Mat33 J = Mat33::Identity();
    GGDisturbance gg(J);
    DisturbanceConfig dist = makeDistCfg(false);

    GGDisturbance::BaseState x = GGDisturbance::BaseState::Zero();
    Vec3 r_body(1.0e6, 0.0, 0.0);

    Vec3 torque = gg.torque(x, dist, r_body, J);
    REQUIRE(torque.isZero());
}

TEST_CASE("GGDisturbance returns zero when inactive", "[ggdisturbance]") {
    Mat33 J = Mat33::Identity();
    GGDisturbance gg(J);
    gg.setActive(false);
    DisturbanceConfig dist = makeDistCfg(true);

    GGDisturbance::BaseState x = GGDisturbance::BaseState::Zero();
    Vec3 r_body(1.0e6, 0.0, 0.0);

    Vec3 torque = gg.torque(x, dist, r_body, J);
    REQUIRE(torque.isZero());
}

TEST_CASE("GGDisturbance returns zero for zero position", "[ggdisturbance]") {
    Mat33 J = Mat33::Identity();
    GGDisturbance gg(J);
    DisturbanceConfig dist = makeDistCfg(true);

    GGDisturbance::BaseState x = GGDisturbance::BaseState::Zero();
    Vec3 r_body = Vec3::Zero();

    Vec3 torque = gg.torque(x, dist, r_body, J);
    REQUIRE(torque.isZero());
}

TEST_CASE("GGDisturbance returns zero for spherical inertia", "[ggdisturbance]") {
    // Spherical satellite (J ∝ I) produces zero GG torque
    Mat33 J = 10.0 * Mat33::Identity();
    GGDisturbance gg(J);
    DisturbanceConfig dist = makeDistCfg(true);

    GGDisturbance::BaseState x = GGDisturbance::BaseState::Zero();
    Vec3 r_body(6.5e6, 1.0e6, 2.0e6);

    Vec3 torque = gg.torque(x, dist, r_body, J);
    
    // For spherical inertia, J*n = c*n, so n × (J*n) = n × (c*n) = 0
    REQUIRE_THAT(torque.norm(), Catch::Matchers::WithinAbs(0.0, 1e-6));
}

TEST_CASE("GGDisturbance torque for elongated satellite", "[ggdisturbance]") {
    // Elongated satellite along x-axis
    Mat33 J;
    J << 1.0, 0.0, 0.0,
         0.0, 10.0, 0.0,
         0.0, 0.0, 10.0;

    GGDisturbance gg(J);
    DisturbanceConfig dist = makeDistCfg(true);

    GGDisturbance::BaseState x = GGDisturbance::BaseState::Zero();
    
    // Position along z-axis (nadir = -z direction)
    const double r = 7.0e6;  // meters
    Vec3 r_body(0.0, 0.0, r);

    Vec3 torque = gg.torque(x, dist, r_body, J);

    // nadir = (0, 0, -1)
    // J * nadir = (0, 0, -10)
    // nadir × (J * nadir) = (0, 0, -1) × (0, 0, -10) = (0, 0, 0)
    // Torque should be zero when aligned with principal axis
    REQUIRE_THAT(torque.norm(), Catch::Matchers::WithinAbs(0.0, 1e-6));
}

TEST_CASE("GGDisturbance torque for misaligned satellite", "[ggdisturbance]") {
    // Elongated satellite along x-axis
    Mat33 J;
    J << 1.0, 0.0, 0.0,
         0.0, 10.0, 0.0,
         0.0, 0.0, 10.0;

    GGDisturbance gg(J);
    DisturbanceConfig dist = makeDistCfg(true);

    GGDisturbance::BaseState x = GGDisturbance::BaseState::Zero();
    
    // Position at 45 degrees in x-z plane
    const double r = 7.0e6;
    Vec3 r_body(r / std::sqrt(2.0), 0.0, r / std::sqrt(2.0));

    Vec3 torque = gg.torque(x, dist, r_body, J);

    // Compute expected torque manually
    Vec3 r_hat = r_body.normalized();
    Vec3 nadir = -r_hat;
    const double mu_e = saltro::constants::MU_EARTH;
    const double const_term = 3.0 * mu_e / (r * r * r);
    Vec3 expected = const_term * nadir.cross(J * nadir);

    REQUIRE_THAT(torque(0), Catch::Matchers::WithinRel(expected(0), 1e-10));
    REQUIRE_THAT(torque(1), Catch::Matchers::WithinRel(expected(1), 1e-10));
    REQUIRE_THAT(torque(2), Catch::Matchers::WithinRel(expected(2), 1e-10));
}

TEST_CASE("GGDisturbance torque magnitude scales with 1/r^3", "[ggdisturbance]") {
    Mat33 J;
    J << 1.0, 0.0, 0.0,
         0.0, 5.0, 0.0,
         0.0, 0.0, 8.0;

    GGDisturbance gg(J);
    DisturbanceConfig dist = makeDistCfg(true);

    GGDisturbance::BaseState x = GGDisturbance::BaseState::Zero();
    
    const double r1 = 7.0e6;
    const double r2 = 14.0e6;  // 2x distance
    
    Vec3 r_body1(r1 / std::sqrt(3.0), r1 / std::sqrt(3.0), r1 / std::sqrt(3.0));
    Vec3 r_body2(r2 / std::sqrt(3.0), r2 / std::sqrt(3.0), r2 / std::sqrt(3.0));

    Vec3 torque1 = gg.torque(x, dist, r_body1, J);
    Vec3 torque2 = gg.torque(x, dist, r_body2, J);

    // Torque should scale as 1/r^3, so doubling distance should reduce by factor of 8
    double ratio = torque1.norm() / torque2.norm();
    REQUIRE_THAT(ratio, Catch::Matchers::WithinRel(8.0, 1e-10));
}

TEST_CASE("GGDisturbance jacobian is zero when dr_dq is zero", "[ggdisturbance]") {
    Mat33 J;
    J << 2.0, 0.0, 0.0,
         0.0, 3.0, 0.0,
         0.0, 0.0, 4.0;

    GGDisturbance gg(J);
    DisturbanceConfig dist = makeDistCfg(true);

    GGDisturbance::BaseState x = GGDisturbance::BaseState::Zero();
    Vec3 r_body(7.0e6, 0.0, 0.0);
    GGDisturbance::Mat34 dr_dq = GGDisturbance::Mat34::Zero();

    GGDisturbance::Mat34 jac = gg.dtorque_dq(x, dist, r_body, J, dr_dq);
    
    REQUIRE(jac.isZero());
}

TEST_CASE("GGDisturbance jacobian returns zero when disabled", "[ggdisturbance]") {
    Mat33 J = Mat33::Identity();
    GGDisturbance gg(J);
    DisturbanceConfig dist = makeDistCfg(false);

    GGDisturbance::BaseState x = GGDisturbance::BaseState::Zero();
    Vec3 r_body(7.0e6, 1.0e6, 0.0);
    GGDisturbance::Mat34 dr_dq = GGDisturbance::Mat34::Random();

    GGDisturbance::Mat34 jac = gg.dtorque_dq(x, dist, r_body, J, dr_dq);
    
    REQUIRE(jac.isZero());
}

TEST_CASE("GGDisturbance jacobian has correct dimensions", "[ggdisturbance]") {
    Mat33 J;
    J << 2.0, 0.1, 0.0,
         0.1, 3.0, 0.2,
         0.0, 0.2, 4.0;

    GGDisturbance gg(J);
    DisturbanceConfig dist = makeDistCfg(true);

    GGDisturbance::BaseState x = GGDisturbance::BaseState::Zero();
    Vec3 r_body(6.8e6, 1.0e6, 0.5e6);
    
    // Create a non-trivial dr_dq
    GGDisturbance::Mat34 dr_dq;
    dr_dq << 1.0, 0.5, 0.2, 0.1,
             0.3, 1.0, 0.4, 0.2,
             0.1, 0.2, 1.0, 0.5;
    dr_dq *= 1.0e5;  // Scale to reasonable values

    GGDisturbance::Mat34 jac = gg.dtorque_dq(x, dist, r_body, J, dr_dq);
    
    // Jacobian should be 3x4
    REQUIRE(jac.rows() == 3);
    REQUIRE(jac.cols() == 4);
    
    // Should not be all zeros for non-trivial inputs
    REQUIRE(jac.norm() > 0.0);
}

TEST_CASE("GGDisturbance jacobian numerical check", "[ggdisturbance]") {
    Mat33 J;
    J << 5.0, 0.0, 0.0,
         0.0, 8.0, 0.0,
         0.0, 0.0, 10.0;

    GGDisturbance gg(J);
    DisturbanceConfig dist = makeDistCfg(true);

    GGDisturbance::BaseState x = GGDisturbance::BaseState::Zero();
    Vec3 r_body(7.0e6, 1.0e6, 0.5e6);
    
    GGDisturbance::Mat34 dr_dq;
    dr_dq << 2.0e5, 1.0e5, 0.5e5, 0.2e5,
             1.0e5, 2.0e5, 1.0e5, 0.5e5,
             0.5e5, 1.0e5, 2.0e5, 1.0e5;

    GGDisturbance::Mat34 jac = gg.dtorque_dq(x, dist, r_body, J, dr_dq);
    
    // Numerical derivative check
    const double eps = 1.0e-6;
    for (int j = 0; j < 4; ++j) {
        Vec3 r_plus = r_body + eps * dr_dq.col(j);
        Vec3 r_minus = r_body - eps * dr_dq.col(j);
        
        Vec3 torque_plus = gg.torque(x, dist, r_plus, J);
        Vec3 torque_minus = gg.torque(x, dist, r_minus, J);
        
        Vec3 numerical_deriv = (torque_plus - torque_minus) / (2.0 * eps);
        
        REQUIRE_THAT(jac(0, j), Catch::Matchers::WithinRel(numerical_deriv(0), 1e-5));
        REQUIRE_THAT(jac(1, j), Catch::Matchers::WithinRel(numerical_deriv(1), 1e-5));
        REQUIRE_THAT(jac(2, j), Catch::Matchers::WithinRel(numerical_deriv(2), 1e-5));
    }
}

TEST_CASE("GGDisturbance hessian is zero when dr_dq and d2r_dq2 are zero", "[ggdisturbance]") {
    Mat33 J;
    J << 2.0, 0.0, 0.0,
         0.0, 3.0, 0.0,
         0.0, 0.0, 4.0;

    GGDisturbance gg(J);
    DisturbanceConfig dist = makeDistCfg(true);

    GGDisturbance::BaseState x = GGDisturbance::BaseState::Zero();
    Vec3 r_body(7.0e6, 1.0e6, 0.0);
    GGDisturbance::Mat34 dr_dq = GGDisturbance::Mat34::Zero();
    std::array<GGDisturbance::Mat44, 3> d2r_dq2 = {
        GGDisturbance::Mat44::Zero(),
        GGDisturbance::Mat44::Zero(),
        GGDisturbance::Mat44::Zero()
    };

    GGDisturbance::T443 H = gg.ddtorque_dqdq(x, dist, r_body, J, dr_dq, d2r_dq2);

    for (int k = 0; k < 3; ++k) {
        REQUIRE(H.slice(k).isZero());
    }
}

TEST_CASE("GGDisturbance hessian returns zero when disabled", "[ggdisturbance]") {
    Mat33 J = Mat33::Identity();
    GGDisturbance gg(J);
    DisturbanceConfig dist = makeDistCfg(false);

    GGDisturbance::BaseState x = GGDisturbance::BaseState::Zero();
    Vec3 r_body(7.0e6, 1.0e6, 0.0);
    GGDisturbance::Mat34 dr_dq = GGDisturbance::Mat34::Random();
    std::array<GGDisturbance::Mat44, 3> d2r_dq2 = {
        GGDisturbance::Mat44::Random(),
        GGDisturbance::Mat44::Random(),
        GGDisturbance::Mat44::Random()
    };

    GGDisturbance::T443 H = gg.ddtorque_dqdq(x, dist, r_body, J, dr_dq, d2r_dq2);

    for (int k = 0; k < 3; ++k) {
        REQUIRE(H.slice(k).isZero());
    }
}

TEST_CASE("GGDisturbance hessian has correct dimensions", "[ggdisturbance]") {
    Mat33 J;
    J << 3.0, 0.1, 0.0,
         0.1, 5.0, 0.2,
         0.0, 0.2, 7.0;

    GGDisturbance gg(J);
    DisturbanceConfig dist = makeDistCfg(true);

    GGDisturbance::BaseState x = GGDisturbance::BaseState::Zero();
    Vec3 r_body(6.8e6, 1.0e6, 0.5e6);
    
    GGDisturbance::Mat34 dr_dq;
    dr_dq << 1.0e5, 0.5e5, 0.2e5, 0.1e5,
             0.3e5, 1.0e5, 0.4e5, 0.2e5,
             0.1e5, 0.2e5, 1.0e5, 0.5e5;

    std::array<GGDisturbance::Mat44, 3> d2r_dq2;
    for (int k = 0; k < 3; ++k) {
        d2r_dq2[static_cast<size_t>(k)] = 1.0e4 * GGDisturbance::Mat44::Random();
    }

    GGDisturbance::T443 H = gg.ddtorque_dqdq(x, dist, r_body, J, dr_dq, d2r_dq2);
    
    // Each slice should be 4x4
    for (int k = 0; k < 3; ++k) {
        REQUIRE(H.slice(k).rows() == 4);
        REQUIRE(H.slice(k).cols() == 4);
    }
}

TEST_CASE("GGDisturbance hessian is symmetric", "[ggdisturbance]") {
    Mat33 J;
    J << 4.0, 0.0, 0.0,
         0.0, 6.0, 0.0,
         0.0, 0.0, 9.0;

    GGDisturbance gg(J);
    DisturbanceConfig dist = makeDistCfg(true);

    GGDisturbance::BaseState x = GGDisturbance::BaseState::Zero();
    Vec3 r_body(6.8e6, 1.2e6, 0.8e6);
    
    GGDisturbance::Mat34 dr_dq;
    dr_dq << 2.0e5, 1.0e5, 0.5e5, 0.3e5,
             1.5e5, 2.0e5, 1.0e5, 0.6e5,
             0.8e5, 1.2e5, 2.0e5, 1.5e5;

    // Create symmetric second derivatives
    std::array<GGDisturbance::Mat44, 3> d2r_dq2;
    for (int k = 0; k < 3; ++k) {
        GGDisturbance::Mat44 temp = GGDisturbance::Mat44::Random();
        d2r_dq2[static_cast<size_t>(k)] = 1.0e4 * (temp + temp.transpose());
    }

    GGDisturbance::T443 H = gg.ddtorque_dqdq(x, dist, r_body, J, dr_dq, d2r_dq2);
    
    // Each slice should be symmetric
    for (int k = 0; k < 3; ++k) {
        auto slice = H.slice(k);
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                REQUIRE_THAT(slice(i, j), Catch::Matchers::WithinAbs(slice(j, i), 1e-10));
            }
        }
    }
}

TEST_CASE("GGDisturbance hessian numerical check", "[ggdisturbance]") {
    Mat33 J;
    J << 3.0, 0.0, 0.0,
         0.0, 5.0, 0.0,
         0.0, 0.0, 7.0;

    GGDisturbance gg(J);
    DisturbanceConfig dist = makeDistCfg(true);

    GGDisturbance::BaseState x = GGDisturbance::BaseState::Zero();
    Vec3 r_body(7.0e6, 1.0e6, 0.5e6);
    
    GGDisturbance::Mat34 dr_dq;
    dr_dq << 1.5e5, 0.8e5, 0.4e5, 0.2e5,
             0.9e5, 1.5e5, 0.7e5, 0.4e5,
             0.4e5, 0.6e5, 1.5e5, 0.8e5;

    std::array<GGDisturbance::Mat44, 3> d2r_dq2;
    for (int k = 0; k < 3; ++k) {
        d2r_dq2[static_cast<size_t>(k)] = GGDisturbance::Mat44::Zero();
    }

    GGDisturbance::T443 H = gg.ddtorque_dqdq(x, dist, r_body, J, dr_dq, d2r_dq2);
    
    // Numerical derivative check (2nd derivative w.r.t. first quaternion element)
    const double eps = 1.0e-6;
    Vec3 r_plus = r_body + eps * dr_dq.col(0);
    Vec3 r_minus = r_body - eps * dr_dq.col(0);
    
    GGDisturbance::Mat34 jac_plus = gg.dtorque_dq(x, dist, r_plus, J, dr_dq);
    GGDisturbance::Mat34 jac_minus = gg.dtorque_dq(x, dist, r_minus, J, dr_dq);
    
    GGDisturbance::Mat34 numerical_hess_slice = (jac_plus - jac_minus) / (2.0 * eps);
    
    // Check first row of Hessian (corresponds to first component of torque)
    for (int j = 0; j < 4; ++j) {
        // Use absolute tolerance for values near zero, relative for larger values
        double analytical = H.slice(0)(0, j);
        double numerical = numerical_hess_slice(0, j);
        if (std::abs(analytical) < 1e-6 && std::abs(numerical) < 1e-6) {
            REQUIRE_THAT(analytical, Catch::Matchers::WithinAbs(numerical, 1e-6));
        } else {
            REQUIRE_THAT(analytical, Catch::Matchers::WithinRel(numerical, 1e-3));
        }
    }
}
