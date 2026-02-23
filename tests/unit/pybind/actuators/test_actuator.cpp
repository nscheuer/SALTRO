#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Dense>
#include <saltro/pybind/actuators/actuator.h>
#include <cmath>
#include <limits>

using Vec3 = Eigen::Vector3d;
using BaseState = Eigen::Matrix<double, 7, 1>;

static Vec3 valid_axis()
{
    Vec3 axis;
    axis << 1.0, 0.0, 0.0;
    return axis;
}

static BaseState valid_base_state()
{
    BaseState x;
    x << 0.0, 0.0, 0.0,
         1.0, 0.0, 0.0, 0.0;
    return x;
}

TEST_CASE("Actuator constructor with valid inputs", "[actuator]")
{
    Vec3 axis = valid_axis();
    double u_max = 1.0;

    REQUIRE_NOTHROW(Actuator(axis, u_max));

    Actuator act(axis, u_max);
    REQUIRE(act.axis().isApprox(axis));
    REQUIRE(act.u_max() == u_max);
}

TEST_CASE("Actuator normalizes axis", "[actuator]")
{
    Vec3 axis;
    axis << 3.0, 4.0, 0.0;
    double u_max = 1.0;

    Actuator act(axis, u_max);
    
    REQUIRE_THAT(act.axis().norm(), Catch::Matchers::WithinRel(1.0, 1e-10));
    REQUIRE(act.axis().isApprox(axis / axis.norm()));
}

TEST_CASE("Actuator takes absolute value of u_max", "[actuator]")
{
    Vec3 axis = valid_axis();
    
    Actuator act1(axis, -5.0);
    REQUIRE(act1.u_max() == 5.0);
    
    Actuator act2(axis, 5.0);
    REQUIRE(act2.u_max() == 5.0);
}

TEST_CASE("Actuator rejects zero axis", "[actuator]")
{
    Vec3 axis = Vec3::Zero();
    double u_max = 1.0;

    REQUIRE_THROWS_AS(Actuator(axis, u_max), std::invalid_argument);
}

TEST_CASE("Actuator rejects non-finite axis", "[actuator]")
{
    Vec3 axis;
    axis << std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0;
    double u_max = 1.0;

    REQUIRE_THROWS_AS(Actuator(axis, u_max), std::invalid_argument);
}

TEST_CASE("Actuator rejects non-finite u_max", "[actuator]")
{
    Vec3 axis = valid_axis();
    
    REQUIRE_THROWS_AS(Actuator(axis, std::numeric_limits<double>::quiet_NaN()), 
                      std::invalid_argument);
    REQUIRE_THROWS_AS(Actuator(axis, std::numeric_limits<double>::infinity()), 
                      std::invalid_argument);
}

TEST_CASE("Actuator clamp function", "[actuator]")
{
    Vec3 axis = valid_axis();
    double u_max = 2.5;
    Actuator act(axis, u_max);

    REQUIRE(act.clamp(1.0) == 1.0);
    REQUIRE(act.clamp(-1.0) == -1.0);
    REQUIRE(act.clamp(2.5) == 2.5);
    REQUIRE(act.clamp(-2.5) == -2.5);
    REQUIRE(act.clamp(5.0) == 2.5);
    REQUIRE(act.clamp(-5.0) == -2.5);
}

TEST_CASE("Actuator base class torque returns zero", "[actuator]")
{
    Vec3 axis = valid_axis();
    Actuator act(axis, 1.0);
    BaseState x = valid_base_state();

    Vec3 tau = act.torque(0.5, x);
    REQUIRE(tau.isZero());
}

TEST_CASE("Actuator base class derivatives return zero", "[actuator]")
{
    Vec3 axis = valid_axis();
    Actuator act(axis, 1.0);
    BaseState x = valid_base_state();

    auto J_u = act.dtorq_du(0.5, x);
    auto J_x = act.dtorq_dbasestate(0.5, x);
    auto H_uu = act.ddtorq_dudu(0.5, x);
    auto H_ux = act.ddtorq_dudbasestate(0.5, x);
    auto H_xx = act.ddtorq_dbasestatedbasestate(0.5, x);

    REQUIRE(J_u.isZero());
    REQUIRE(J_x.isZero());
    REQUIRE(H_uu.isZero());
    REQUIRE(H_ux.isZero());
    REQUIRE(H_xx.isZero());
}

TEST_CASE("Actuator input_len constant", "[actuator]")
{
    REQUIRE(Actuator::input_len == 1);
}
