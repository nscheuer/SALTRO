#include <catch2/catch_test_macros.hpp>
#include <Eigen/Dense>
#include <saltro/pybind/actuators/RW.h>
#include <cmath>

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

TEST_CASE("RW constructor with valid inputs", "[rw]")
{
    Vec3 axis = valid_axis();
    double max_torque = 0.01;
    double J = 2.0e-5;
    double h0 = 0.0;
    double h_max = 0.1;

    REQUIRE_NOTHROW(RW(axis, max_torque, J, h0, h_max));

    RW rw(axis, max_torque, J, h0, h_max);
    REQUIRE(rw.axis().isApprox(axis));
    REQUIRE(rw.u_max() == max_torque);
    REQUIRE(rw.wheelInertia() == J);
    REQUIRE(rw.momentum() == h0);
    REQUIRE(rw.momentumMax() == h_max);
}

TEST_CASE("RW rejects non-positive inertia", "[rw]")
{
    Vec3 axis = valid_axis();
    
    REQUIRE_THROWS_AS(RW(axis, 0.01, 0.0, 0.0, 0.1), std::invalid_argument);
    REQUIRE_THROWS_AS(RW(axis, 0.01, -1.0e-5, 0.0, 0.1), std::invalid_argument);
}

TEST_CASE("RW rejects negative h_max", "[rw]")
{
    Vec3 axis = valid_axis();
    
    REQUIRE_THROWS_AS(RW(axis, 0.01, 2.0e-5, 0.0, -0.1), std::invalid_argument);
}

TEST_CASE("RW accepts zero h_max", "[rw]")
{
    Vec3 axis = valid_axis();
    
    REQUIRE_NOTHROW(RW(axis, 0.01, 2.0e-5, 0.0, 0.0));
}

TEST_CASE("RW setMomentum and getMomentum", "[rw]")
{
    Vec3 axis = valid_axis();
    RW rw(axis, 0.01, 2.0e-5, 0.0, 0.1);

    REQUIRE(rw.momentum() == 0.0);
    
    rw.setMomentum(0.05);
    REQUIRE(rw.momentum() == 0.05);
    
    rw.setMomentum(-0.03);
    REQUIRE(rw.momentum() == -0.03);
}

TEST_CASE("RW torque computation", "[rw]")
{
    Vec3 axis;
    axis << 1.0, 0.0, 0.0;
    RW rw(axis, 0.01, 2.0e-5, 0.0, 0.1);

    BaseState x = valid_base_state();
    double u = 0.005;

    Vec3 tau = rw.torque(u, x);

    Vec3 expected = axis * u;
    
    REQUIRE(tau.isApprox(expected));
}

TEST_CASE("RW torque with different axis", "[rw]")
{
    Vec3 axis;
    axis << 0.0, 0.0, 1.0;
    RW rw(axis, 0.01, 2.0e-5, 0.0, 0.1);

    BaseState x = valid_base_state();
    double u = -0.003;

    Vec3 tau = rw.torque(u, x);

    Vec3 expected;
    expected << 0.0, 0.0, -0.003;
    
    REQUIRE(tau.isApprox(expected));
}

TEST_CASE("RW storageTorque computation", "[rw]")
{
    Vec3 axis = valid_axis();
    RW rw(axis, 0.01, 2.0e-5, 0.0, 0.1);

    BaseState x = valid_base_state();
    double u = 0.005;

    auto h_dot = rw.storageTorque(u, x);

    REQUIRE(h_dot(0, 0) == -u);
}

TEST_CASE("RW dtorq_du", "[rw]")
{
    Vec3 axis;
    axis << 0.0, 1.0, 0.0;
    RW rw(axis, 0.01, 2.0e-5, 0.0, 0.1);

    BaseState x = valid_base_state();

    auto J = rw.dtorq_du(0.5, x);

    Eigen::RowVector3d expected = axis.transpose();
    
    REQUIRE(J.isApprox(expected));
}

TEST_CASE("RW dtorq_dbasestate is zero", "[rw]")
{
    Vec3 axis = valid_axis();
    RW rw(axis, 0.01, 2.0e-5, 0.0, 0.1);

    BaseState x = valid_base_state();

    auto J = rw.dtorq_dbasestate(0.5, x);

    REQUIRE(J.isZero());
}

TEST_CASE("RW dstor_torq_du", "[rw]")
{
    Vec3 axis = valid_axis();
    RW rw(axis, 0.01, 2.0e-5, 0.0, 0.1);

    BaseState x = valid_base_state();

    auto J = rw.dstor_torq_du(0.5, x);

    REQUIRE(J(0, 0) == -1.0);
}

TEST_CASE("RW dstor_torq_dbasestate is zero", "[rw]")
{
    Vec3 axis = valid_axis();
    RW rw(axis, 0.01, 2.0e-5, 0.0, 0.1);

    BaseState x = valid_base_state();

    auto J = rw.dstor_torq_dbasestate(0.5, x);

    REQUIRE(J.isZero());
}

TEST_CASE("RW base class hessians are zero", "[rw]")
{
    Vec3 axis = valid_axis();
    RW rw(axis, 0.01, 2.0e-5, 0.0, 0.1);

    BaseState x = valid_base_state();

    auto H_uu = rw.ddtorq_dudu(0.5, x);
    auto H_ux = rw.ddtorq_dudbasestate(0.5, x);
    auto H_xx = rw.ddtorq_dbasestatedbasestate(0.5, x);

    REQUIRE(H_uu.isZero());
    REQUIRE(H_ux.isZero());
    REQUIRE(H_xx.isZero());
}
