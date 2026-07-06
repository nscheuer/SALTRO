#include <catch2/catch_test_macros.hpp>
#include <Eigen/Dense>
#include <limits>
#include <string>

#include <saltro/validation/validate_trajOpt.h>
#include <saltro/pybind/satellite.h>
#include <saltro/pybind/plannersettings.h>

// validatetrajOpt() is the top-level input gate-keeper for a planning problem:
// it chains the per-field validators (settings/satellite/x0/orbit/jtime/q_goal/
// boresight) and then validateTrajOptCrossContext() for dimension consistency.
// The per-field validators are tested individually; this pins the orchestration
// and, in particular, the cross-context dimension checks (N, state_dim,
// input_dim, x0 size, jtime length), which had no coverage.

using namespace saltro;
using saltro::validation::validatetrajOpt;

namespace {

PlannerSettings validSettings() {
    PlannerSettings s;
    s.num_passes = 1;
    s.constraints.control_limit_scale = 0.75;
    s.constraints.u_max = Eigen::VectorXd::Constant(3, 1.0);
    s.constraints.wmax = 0.3;
    s.constraints.sun_limit_angle = 0.35;
    s.disturbances.coeff_N = 3;
    s.init_traj.initcontroller = 0;
    s.passes[0].dt = 1.0;
    return s;
}

Eigen::Matrix3d validInertia() {
    Eigen::Matrix3d J;
    J << 0.05, 0.0, 0.0,
         0.0, 0.06, 0.0,
         0.0, 0.0, 0.07;
    return J;
}

// 3 MTQs -> controlDim 3 (matches u_max size 3), no RW -> stateDim 7.
struct Fixture {
    PlannerSettings settings = validSettings();
    Satellite sat{validInertia(), settings};
    Eigen::VectorXd x0;
    Eigen::Vector3d r0{7000e3, 0.0, 0.0};
    Eigen::Vector3d v0{0.0, 7546.0, 0.0};
    Eigen::VectorXd jtime;
    Eigen::MatrixXd q_goal;
    Eigen::MatrixXd boresight;
    int N = 2;

    Fixture() {
        sat.addMTQ(Eigen::Vector3d(1, 0, 0), 0.2);
        sat.addMTQ(Eigen::Vector3d(0, 1, 0), 0.2);
        sat.addMTQ(Eigen::Vector3d(0, 0, 1), 0.2);
        x0 = Eigen::VectorXd(7);
        x0 << 0.1, 0.05, -0.02, 1.0, 0.0, 0.0, 0.0;
        jtime = Eigen::VectorXd(2);
        jtime << 0.22, 0.22 + 1e-6;  // Julian centuries since J2000, in [0.20, 0.40]
        q_goal = Eigen::MatrixXd(4, 2);
        q_goal.col(0) << 1.0, 0.0, 0.0, 0.0;
        q_goal.col(1) << 0.0, 1.0, 0.0, 0.0;
        boresight = Eigen::MatrixXd(3, 2);
        boresight.col(0) << 1.0, 0.0, 0.0;
        boresight.col(1) << 0.0, 1.0, 0.0;
    }

    bool run(std::string& err, int sd, int id, int n) const {
        return validatetrajOpt(settings, sat, x0, r0, v0, jtime, q_goal, boresight, sd, id, n, err);
    }
};

}  // namespace

TEST_CASE("validatetrajOpt: a consistent problem passes", "[validation][trajopt]") {
    Fixture f;
    std::string err;
    const bool ok = f.run(err, f.sat.stateDim(), f.sat.controlDim(), f.N);
    INFO("stateDim=" << f.sat.stateDim() << " controlDim=" << f.sat.controlDim()
         << " err=" << err);
    REQUIRE(ok);
    REQUIRE(err.empty());
}

TEST_CASE("validatetrajOpt: N must be > 1", "[validation][trajopt]") {
    Fixture f;
    std::string err;
    REQUIRE_FALSE(f.run(err, f.sat.stateDim(), f.sat.controlDim(), 1));
    REQUIRE(err.find("N must be > 1") != std::string::npos);
}

TEST_CASE("validatetrajOpt: state_dim must match the satellite", "[validation][trajopt]") {
    Fixture f;
    std::string err;
    REQUIRE_FALSE(f.run(err, f.sat.stateDim() + 1, f.sat.controlDim(), f.N));
    REQUIRE(err.find("state_dim") != std::string::npos);
}

TEST_CASE("validatetrajOpt: input_dim must match the satellite", "[validation][trajopt]") {
    Fixture f;
    std::string err;
    REQUIRE_FALSE(f.run(err, f.sat.stateDim(), f.sat.controlDim() + 1, f.N));
    REQUIRE(err.find("input_dim") != std::string::npos);
}

TEST_CASE("validatetrajOpt: jtime length must match N", "[validation][trajopt]") {
    Fixture f;
    std::string err;
    // jtime has 2 entries; claim N=3 -> cross-context length mismatch.
    REQUIRE_FALSE(f.run(err, f.sat.stateDim(), f.sat.controlDim(), 3));
}
