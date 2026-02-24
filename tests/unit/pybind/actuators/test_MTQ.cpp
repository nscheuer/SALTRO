#include <catch2/catch_test_macros.hpp>
#include <Eigen/Dense>
#include <saltro/pybind/actuators/MTQ.h>
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

static Vec3 valid_magnetic_field()
{
    Vec3 B;
    B << 0.0, 1.0e-5, 0.0;
    return B;
}

TEST_CASE("MTQ constructor", "[mtq]")
{
    Vec3 axis = valid_axis();
    double max_dipole = 0.2;

    REQUIRE_NOTHROW(MTQ(axis, max_dipole));

    MTQ mtq(axis, max_dipole);
    REQUIRE(mtq.axis().isApprox(axis));
    REQUIRE(mtq.u_max() == max_dipole);
}

TEST_CASE("MTQ torque computation", "[mtq]")
{
    Vec3 axis;
    axis << 1.0, 0.0, 0.0;
    MTQ mtq(axis, 0.2);

    BaseState x = valid_base_state();
    Vec3 B_body;
    B_body << 0.0, 1.0e-5, 0.0;
    double u = 0.1;

    Vec3 tau = mtq.torque(u, x, B_body);

    Vec3 expected;
    expected << 0.0, 0.0, 1.0e-6;
    
    REQUIRE(tau.isApprox(expected, 1e-12));
}

TEST_CASE("MTQ torque with different configurations", "[mtq]")
{
    Vec3 axis;
    axis << 0.0, 1.0, 0.0;
    MTQ mtq(axis, 1.0);

    BaseState x = valid_base_state();
    
    Vec3 B;
    B << 2.0e-5, 0.0, 0.0;
    
    Vec3 tau = mtq.torque(0.5, x, B);
    
    Vec3 expected;
    expected << 0.0, 0.0, -1.0e-5;
    
    REQUIRE(tau.isApprox(expected, 1e-12));
}

TEST_CASE("MTQ torque is zero when B parallel to axis", "[mtq]")
{
    Vec3 axis;
    axis << 1.0, 0.0, 0.0;
    MTQ mtq(axis, 1.0);

    BaseState x = valid_base_state();
    Vec3 B;
    B << 1.0e-5, 0.0, 0.0;

    Vec3 tau = mtq.torque(1.0, x, B);
    
    REQUIRE(tau.norm() < 1e-15);
}

TEST_CASE("MTQ torque is zero when u is zero", "[mtq]")
{
    Vec3 axis = valid_axis();
    MTQ mtq(axis, 1.0);

    BaseState x = valid_base_state();
    Vec3 B = valid_magnetic_field();

    Vec3 tau = mtq.torque(0.0, x, B);
    
    REQUIRE(tau.isZero());
}

TEST_CASE("MTQ dtorq_du", "[mtq]")
{
    Vec3 axis;
    axis << 1.0, 0.0, 0.0;
    MTQ mtq(axis, 1.0);

    BaseState x = valid_base_state();
    Vec3 B;
    B << 0.0, 1.0e-5, 0.0;

    auto J = mtq.dtorq_du(0.5, x, B);

    Vec3 B_cross_axis = B.cross(axis);
    Eigen::RowVector3d expected = -B_cross_axis.transpose();
    
    REQUIRE(J.isApprox(expected, 1e-12));
}

TEST_CASE("MTQ dtorq_dbasestate with zero dB_dq", "[mtq]")
{
    Vec3 axis = valid_axis();
    MTQ mtq(axis, 1.0);

    BaseState x = valid_base_state();
    Vec3 B = valid_magnetic_field();
    Eigen::Matrix<double, 4, 3> dB_dq = Eigen::Matrix<double, 4, 3>::Zero();

    auto J = mtq.dtorq_dbasestate(0.5, x, B, dB_dq);

    REQUIRE(J.isZero());
}

TEST_CASE("MTQ ddtorq_dudu is zero", "[mtq]")
{
    Vec3 axis = valid_axis();
    MTQ mtq(axis, 1.0);

    BaseState x = valid_base_state();
    Vec3 B = valid_magnetic_field();

    auto H = mtq.ddtorq_dudu(0.5, x, B);

    REQUIRE(H.isZero());
}

TEST_CASE("MTQ ddtorq_dudbasestate with non-zero dB_dq", "[mtq]")
{
    Vec3 axis = valid_axis();
    MTQ mtq(axis, 1.0);

    BaseState x = valid_base_state();
    Vec3 B = valid_magnetic_field();
    
    Eigen::Matrix<double, 4, 3> dB_dq;
    dB_dq.setOnes();

    auto H = mtq.ddtorq_dudbasestate(0.5, x, B, dB_dq);

    REQUIRE(H.slice(0).rows() == 1);
    REQUIRE(H.slice(0).cols() == 7);
}
