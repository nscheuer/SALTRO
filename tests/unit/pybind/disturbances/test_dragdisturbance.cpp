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
