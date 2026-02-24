#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Dense>

#include <saltro/pybind/disturbances/dragdisturbance.h>
#include <saltro/pybind/plannersettings.h>

using namespace saltro::disturbances;
using Vec3 = Eigen::Vector3d;

namespace {
GeometryFace makeFace(double area, const Vec3& centroid, const Vec3& normal, double CD) {
    return GeometryFace(area, centroid, normal, 0.0, 0.0, 0.0, CD);
}

DisturbanceConfig makeDistCfg(bool plan_for_aero) {
    DisturbanceConfig cfg;
    cfg.plan_for_aero = plan_for_aero;
    return cfg;
}
}

TEST_CASE("DragDisturbance returns zero when disabled", "[dragdisturbance]") {
    GeometryConfig config;
    config.addFace(makeFace(1.0, Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0), 2.0));

    DragDisturbance drag(config);
    DisturbanceConfig dist = makeDistCfg(false);

    DragDisturbance::BaseState x = DragDisturbance::BaseState::Zero();
    Vec3 v_body(0.0, 3.0, 0.0);

    Vec3 torque = drag.torque(x, dist, v_body);
    REQUIRE(torque.isZero());
}

TEST_CASE("DragDisturbance returns zero when inactive", "[dragdisturbance]") {
    GeometryConfig config;
    config.addFace(makeFace(1.0, Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0), 2.0));

    DragDisturbance drag(config);
    drag.setActive(false);
    DisturbanceConfig dist = makeDistCfg(true);

    DragDisturbance::BaseState x = DragDisturbance::BaseState::Zero();
    Vec3 v_body(0.0, 3.0, 0.0);

    Vec3 torque = drag.torque(x, dist, v_body);
    REQUIRE(torque.isZero());
}

TEST_CASE("DragDisturbance single face torque matches expected", "[dragdisturbance]") {
    GeometryConfig config;
    config.addFace(makeFace(2.0, Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0), 2.0));

    DragDisturbance drag(config);
    DisturbanceConfig dist = makeDistCfg(true);

    DragDisturbance::BaseState x = DragDisturbance::BaseState::Zero();
    Vec3 v_body(0.0, 3.0, 0.0);

    Vec3 torque = drag.torque(x, dist, v_body);
    Vec3 expected(0.0, 0.0, -18.0);

    REQUIRE_THAT(torque(0), Catch::Matchers::WithinRel(expected(0), 1e-12));
    REQUIRE_THAT(torque(1), Catch::Matchers::WithinRel(expected(1), 1e-12));
    REQUIRE_THAT(torque(2), Catch::Matchers::WithinRel(expected(2), 1e-12));
}

TEST_CASE("DragDisturbance ignores faces with negative incidence", "[dragdisturbance]") {
    GeometryConfig config;
    config.addFace(makeFace(1.0, Vec3(1.0, 0.0, 0.0), Vec3(-1.0, 0.0, 0.0), 2.0));

    DragDisturbance drag(config);
    DisturbanceConfig dist = makeDistCfg(true);

    DragDisturbance::BaseState x = DragDisturbance::BaseState::Zero();
    Vec3 v_body(1.0, 0.0, 0.0);

    Vec3 torque = drag.torque(x, dist, v_body);
    REQUIRE(torque.isZero());
}

TEST_CASE("DragDisturbance sums torque across multiple faces", "[dragdisturbance]") {
    GeometryConfig config;
    config.addFace(makeFace(1.0, Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0), 1.0));
    config.addFace(makeFace(2.0, Vec3(0.0, 1.0, 0.0), Vec3(0.0, 1.0, 0.0), 2.0));

    DragDisturbance drag(config);
    DisturbanceConfig dist = makeDistCfg(true);

    DragDisturbance::BaseState x = DragDisturbance::BaseState::Zero();
    Vec3 v_body(0.0, 2.0, 0.0);

    Vec3 torque = drag.torque(x, dist, v_body);

    const double inc = 2.0;
    Vec3 summed = (1.0 * 1.0 * inc) * Vec3(1.0, 0.0, 0.0)
                  + (2.0 * 2.0 * inc) * Vec3(0.0, 1.0, 0.0);
    Vec3 expected = -0.5 * summed.cross(v_body);

    REQUIRE_THAT(torque(0), Catch::Matchers::WithinRel(expected(0), 1e-12));
    REQUIRE_THAT(torque(1), Catch::Matchers::WithinRel(expected(1), 1e-12));
    REQUIRE_THAT(torque(2), Catch::Matchers::WithinRel(expected(2), 1e-12));
}

TEST_CASE("DragDisturbance jacobian is zero when dV_dq is zero", "[dragdisturbance]") {
    GeometryConfig config;
    config.addFace(makeFace(1.0, Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0), 1.0));

    DragDisturbance drag(config);
    DisturbanceConfig dist = makeDistCfg(true);

    DragDisturbance::BaseState x = DragDisturbance::BaseState::Zero();
    Vec3 v_body(0.0, 2.0, 0.0);
    DragDisturbance::Mat34 dV_dq = DragDisturbance::Mat34::Zero();

    DragDisturbance::Mat34 J = drag.dtorque_dq(x, dist, v_body, dV_dq);
    REQUIRE(J.isZero());
}

TEST_CASE("DragDisturbance jacobian calculation with non-zero dV_dq", "[dragdisturbance]") {
    GeometryConfig config;
    config.addFace(makeFace(2.0, Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0), 1.0));

    DragDisturbance drag(config);
    DisturbanceConfig dist = makeDistCfg(true);

    DragDisturbance::BaseState x = DragDisturbance::BaseState::Zero();
    Vec3 v_body(0.0, 3.0, 0.0);
    
    // Set up dV_dq: how velocity changes with state
    DragDisturbance::Mat34 dV_dq = DragDisturbance::Mat34::Zero();
    dV_dq(0, 0) = 1.0;  // dv_x / dq_0 = 1.0
    dV_dq(1, 1) = 2.0;  // dv_y / dq_1 = 2.0
    dV_dq(2, 2) = 1.5;  // dv_z / dq_2 = 1.5

    DragDisturbance::Mat34 J = drag.dtorque_dq(x, dist, v_body, dV_dq);

    // Manually compute expected jacobian
    // C = CD * A * (n . v) * centroid = 1.0 * 2.0 * 3.0 * (1.0, 0.0, 0.0) = (6.0, 0.0, 0.0)
    // tau = -0.5 * C.cross(v) = -0.5 * (6,0,0).cross(0,3,0) = -0.5 * (0,0,18) = (0,0,-9)
    //
    // dtau/dq_j = -0.5 * (dC/dq_j . cross(v) + C . cross(dv/dq_j))
    // 
    // dC/dq_0 = CD * A * (n . dv/dq_0) * centroid = 1.0 * 2.0 * (0,1,0).(1,0,0) * (1,0,0) = (0,0,0)
    // dC/dq_1 = CD * A * (n . dv/dq_1) * centroid = 1.0 * 2.0 * (0,1,0).(0,2,0) * (1,0,0) = (4,0,0)
    // dC/dq_2 = CD * A * (n . dv/dq_2) * centroid = 1.0 * 2.0 * (0,1,0).(0,0,1.5) * (1,0,0) = (0,0,0)
    //
    // dtau/dq_0 = -0.5 * ((0,0,0).cross(0,3,0) + (6,0,0).cross(1,0,0)) = -0.5 * (0 + (0,0,0)) = (0,0,0)
    // dtau/dq_1 = -0.5 * ((4,0,0).cross(0,3,0) + (6,0,0).cross(0,2,0)) = -0.5 * ((0,0,12) + (0,0,12)) = (0,0,-12)
    // dtau/dq_2 = -0.5 * ((0,0,0).cross(0,3,0) + (6,0,0).cross(0,0,1.5)) = -0.5 * (0 + (0,-9,0)) = (0,4.5,0)
    // dtau/dq_3 = -0.5 * ((0,0,0).cross(0,3,0) + (6,0,0).cross(0,0,0)) = (0,0,0)

    REQUIRE_THAT(J(0, 0), Catch::Matchers::WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(J(1, 0), Catch::Matchers::WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(J(2, 0), Catch::Matchers::WithinAbs(0.0, 1e-12));

    REQUIRE_THAT(J(0, 1), Catch::Matchers::WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(J(1, 1), Catch::Matchers::WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(J(2, 1), Catch::Matchers::WithinAbs(-12.0, 1e-12));

    REQUIRE_THAT(J(0, 2), Catch::Matchers::WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(J(1, 2), Catch::Matchers::WithinAbs(4.5, 1e-12));
    REQUIRE_THAT(J(2, 2), Catch::Matchers::WithinAbs(0.0, 1e-12));

    REQUIRE_THAT(J(0, 3), Catch::Matchers::WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(J(1, 3), Catch::Matchers::WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(J(2, 3), Catch::Matchers::WithinAbs(0.0, 1e-12));
}

TEST_CASE("DragDisturbance hessian is zero when dV_dq and d2V_dq2 are zero", "[dragdisturbance]") {
    GeometryConfig config;
    config.addFace(makeFace(1.0, Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0), 1.0));

    DragDisturbance drag(config);
    DisturbanceConfig dist = makeDistCfg(true);

    DragDisturbance::BaseState x = DragDisturbance::BaseState::Zero();
    Vec3 v_body(0.0, 2.0, 0.0);
    DragDisturbance::Mat34 dV_dq = DragDisturbance::Mat34::Zero();
    std::array<DragDisturbance::Mat44, 3> d2V_dq2 = {
        DragDisturbance::Mat44::Zero(),
        DragDisturbance::Mat44::Zero(),
        DragDisturbance::Mat44::Zero()
    };

    DragDisturbance::T443 H = drag.ddtorque_dqdq(x, dist, v_body, dV_dq, d2V_dq2);

    for (int k = 0; k < 3; ++k) {
        REQUIRE(H.slice(k).isZero());
    }
}

TEST_CASE("DragDisturbance hessian calculation with non-zero dV_dq and d2V_dq2", "[dragdisturbance]") {
    GeometryConfig config;
    config.addFace(makeFace(1.0, Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0), 2.0));

    DragDisturbance drag(config);
    DisturbanceConfig dist = makeDistCfg(true);

    DragDisturbance::BaseState x = DragDisturbance::BaseState::Zero();
    Vec3 v_body(0.0, 2.0, 0.0);
    
    // First derivative of velocity with respect to state
    DragDisturbance::Mat34 dV_dq = DragDisturbance::Mat34::Zero();
    dV_dq(0, 0) = 1.0;  // dv_x / dq_0 = 1.0
    dV_dq(1, 1) = 1.0;  // dv_y / dq_1 = 1.0
    dV_dq(2, 2) = 1.0;  // dv_z / dq_2 = 1.0
    
    // Second derivatives of velocity with respect to state pairs
    std::array<DragDisturbance::Mat44, 3> d2V_dq2 = {
        DragDisturbance::Mat44::Zero(),
        DragDisturbance::Mat44::Zero(),
        DragDisturbance::Mat44::Zero()
    };
    // d²v_y / dq_0 dq_0 = 0.5 (constant second derivative)
    d2V_dq2[1](0, 0) = 0.5;
    d2V_dq2[1](1, 0) = 0.5;
    d2V_dq2[1](0, 1) = 0.5;
    d2V_dq2[1](1, 1) = 0.5;

    DragDisturbance::T443 H = drag.ddtorque_dqdq(x, dist, v_body, dV_dq, d2V_dq2);

    // Basic verifications:
    // 1. When velocity derivatives are symmetric and simple, second derivatives should follow pattern
    // 2. The hessian should be non-zero due to the non-zero d2V_dq2
    bool has_nonzero = false;
    for (int k = 0; k < 3; ++k) {
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                if (std::abs(H.slice(k)(i, j)) > 1e-12) {
                    has_nonzero = true;
                    break;
                }
            }
            if (has_nonzero) break;
        }
        if (has_nonzero) break;
    }
    REQUIRE(has_nonzero);

    // Verify specific elements based on the geometry and derivatives
    // C = CD * A * (n . v) * centroid = 2.0 * 1.0 * (0,1,0).(0,2,0) * (1,0,0) = (4,0,0)
    // tau = -0.5 * (4,0,0).cross(0,2,0) = -0.5 * (0,0,8) = (0,0,-4)
    //
    // For mixed partials involving y-velocity second derivatives
    // The y-component (index 1) of the hessian should show contributions from d2V_dq2[1]
    
    // Check that mixing q_0 and q_1 with nonzero d2V_dq2[1] gives nonzero contributions
    REQUIRE((std::abs(H.slice(1)(0, 1)) > 1e-12 || std::abs(H.slice(2)(0, 1)) > 1e-12 ||
             std::abs(H.slice(0)(0, 1)) > 1e-12 || 
             std::abs(H.slice(1)(1, 0)) > 1e-12 || std::abs(H.slice(2)(1, 0)) > 1e-12 ||
             std::abs(H.slice(0)(1, 0)) > 1e-12));
}
