#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Dense>

#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <utility>

#include <saltro/pybind/satellite.h>

using namespace saltro;

namespace {

constexpr double kSqrtHalf = 0.70710678118654752440;
constexpr double kPi = 3.14159265358979323846;

Eigen::Vector4d quatConj(const Eigen::Vector4d& q) {
    return Eigen::Vector4d(q(0), -q(1), -q(2), -q(3));
}

Eigen::Vector4d quatMult(const Eigen::Vector4d& a, const Eigen::Vector4d& b) {
    return Eigen::Vector4d(
        a(0) * b(0) - a(1) * b(1) - a(2) * b(2) - a(3) * b(3),
        a(0) * b(1) + a(1) * b(0) + a(2) * b(3) - a(3) * b(2),
        a(0) * b(2) - a(1) * b(3) + a(2) * b(0) + a(3) * b(1),
        a(0) * b(3) + a(1) * b(2) - a(2) * b(1) + a(3) * b(0));
}

Eigen::Matrix3d quatRotMat(const Eigen::Vector4d& q) {
    const double q0 = q(0);
    const double qx = q(1);
    const double qy = q(2);
    const double qz = q(3);
    const Eigen::Vector3d qv = q.tail<3>();

    Eigen::Matrix3d skew;
    skew << 0.0, -qz,  qy,
            qz,  0.0, -qx,
           -qy,  qx,  0.0;

    return (q0 * q0 - qv.dot(qv)) * Eigen::Matrix3d::Identity()
         + 2.0 * qv * qv.transpose()
         + 2.0 * q0 * skew;
}

}  // namespace

class SatelliteCostOmegaFixture {
public:
    Satellite sat;
    const Eigen::Vector3d boresight = Eigen::Vector3d::UnitX();
    const Eigen::Vector4d target_q = Eigen::Vector4d(kSqrtHalf, 0.0, 0.0, kSqrtHalf);
    const Eigen::Vector4d target_vec =
        Eigen::Vector4d(std::numeric_limits<double>::quiet_NaN(),
                        std::cos(30.0 * kPi / 180.0),
                        std::sin(30.0 * kPi / 180.0),
                        0.0);
    const Eigen::Vector3d B_eci = Eigen::Vector3d(2.5e-5, -1.5e-5, 3.0e-5);

    SatelliteCostOmegaFixture()
        : sat(Eigen::Vector3d(0.067, 0.071, 0.069).asDiagonal(), PlannerSettings()) {
        for (const Eigen::Vector3d& axis : std::array<Eigen::Vector3d, 3>{
                 Eigen::Vector3d::UnitX(),
                 Eigen::Vector3d::UnitY(),
                 Eigen::Vector3d::UnitZ()}) {
            sat.addMTQ(axis, 0.2);
            sat.addRW(axis, 0.001, 1e-5, 0.0, 0.02);
        }
    }

    CostConfig costCfg(double angle = 1e3,
                       double ang_vel = 1e4,
                       double ang_vel_err_dir = 0.0,
                       double ang_vel_err_dir_ratio = 0.0,
                       double ang_vel_roll_ratio = 1.0,
                       bool use_hess = true) const {
        CostConfig cfg;
        cfg.angle = angle;
        cfg.ang_vel = ang_vel;
        cfg.ang_vel_err_dir = ang_vel_err_dir;
        cfg.ang_vel_err_dir_ratio = ang_vel_err_dir_ratio;
        cfg.ang_vel_roll_ratio = ang_vel_roll_ratio;
        cfg.ang_vel_mag = 0.0;
        cfg.mtq_control_weight = 0.0;
        cfg.rw_control_weight = 0.0;
        cfg.rw_AM_weight = 0.0;
        cfg.rw_stic_weight = 0.0;
        cfg.RWh_stiction_mult = 0.0;
        cfg.use_cost_hess = use_hess;
        cfg.setTerminalEmphasis(1.0);
        return cfg;
    }

    std::pair<Satellite::VecX, Satellite::VecX> nominalStateAndControl() const {
        Satellite::VecX x = Satellite::VecX::Zero(sat.stateDim());
        x.segment<3>(Satellite::AV_INDEX) << 0.01, -0.005, 0.008;
        Eigen::Vector4d q(0.92, 0.10, -0.20, 0.32);
        x.segment<4>(Satellite::QUAT_INDEX) = q.normalized();
        x.segment<3>(Satellite::RW_MOMENTUM_INDEX) << 1e-4, -5e-5, 2e-5;

        Satellite::VecX u = Satellite::VecX::Zero(sat.controlDim());
        u << 0.05, -0.02, 0.04, 1e-5, -5e-6, 2e-6;
        return {x, u};
    }

    Satellite::VecX projectedGradFD(const Satellite::VecX& x,
                                    const Satellite::VecX& u,
                                    const Eigen::Vector4d& target,
                                    const CostConfig& cfg,
                                    double eps = 1e-6) const {
        Satellite::VecX g = Satellite::VecX::Zero(x.size());
        for (int i = 0; i < x.size(); ++i) {
            Satellite::VecX xp = x;
            Satellite::VecX xm = x;
            xp(i) += eps;
            xm(i) -= eps;
            const double cp = sat.stageCost(0, 100, xp, u, boresight, target, B_eci, cfg);
            const double cm = sat.stageCost(0, 100, xm, u, boresight, target, B_eci, cfg);
            g(i) = (cp - cm) / (2.0 * eps);
        }

        const Eigen::Vector4d q = x.segment<4>(Satellite::QUAT_INDEX);
        const Eigen::Matrix4d P = Eigen::Matrix4d::Identity() - q * q.transpose();
        g.segment<4>(Satellite::QUAT_INDEX) = P * g.segment<4>(Satellite::QUAT_INDEX);
        return g;
    }

    Satellite::MatX hessFD(const Satellite::VecX& x,
                           const Satellite::VecX& u,
                           const Eigen::Vector4d& target,
                           const CostConfig& cfg,
                           double eps = 1e-4) const {
        Satellite::MatX H = Satellite::MatX::Zero(x.size(), x.size());
        for (int i = 0; i < x.size(); ++i) {
            for (int j = 0; j < x.size(); ++j) {
                Satellite::VecX xpp = x;
                Satellite::VecX xmm = x;
                Satellite::VecX xpm = x;
                Satellite::VecX xmp = x;
                xpp(i) += eps; xpp(j) += eps;
                xmm(i) -= eps; xmm(j) -= eps;
                xpm(i) += eps; xpm(j) -= eps;
                xmp(i) -= eps; xmp(j) += eps;
                const double cpp = sat.stageCost(0, 100, xpp, u, boresight, target, B_eci, cfg);
                const double cmm = sat.stageCost(0, 100, xmm, u, boresight, target, B_eci, cfg);
                const double cpm = sat.stageCost(0, 100, xpm, u, boresight, target, B_eci, cfg);
                const double cmp = sat.stageCost(0, 100, xmp, u, boresight, target, B_eci, cfg);
                H(i, j) = (cpp + cmm - cpm - cmp) / (4.0 * eps * eps);
            }
        }
        return H;
    }
};

TEST_CASE_METHOD(SatelliteCostOmegaFixture,
                 "Omega crossterm paths are callable in quaternion mode",
                 "[cost][omega_ff][quat]") {
    const auto [x, u] = nominalStateAndControl();
    for (const CostConfig& cfg : {
             costCfg(1e3, 1e4, 0.7, 0.0, 1.0),
             costCfg(1e3, 1e4, 0.0, 0.5, 1.0) }) {
        const double cost = sat.stageCost(0, 100, x, u, boresight, target_q, B_eci, cfg);
        REQUIRE(cost >= 0.0);
    }
}

TEST_CASE_METHOD(SatelliteCostOmegaFixture,
                 "Quaternion crossterm rewards error-reducing angular velocity",
                 "[cost][omega_ff][quat]") {
    auto [x, u] = nominalStateAndControl();
    u.setZero();
    const CostConfig cfg = costCfg(1e3, 1e4, 0.0, 0.5, 1.0);

    Eigen::Vector4d q_goal = target_q;
    const Eigen::Vector4d q = x.segment<4>(Satellite::QUAT_INDEX);
    if (q.dot(q_goal) < 0.0) q_goal = -q_goal;
    Eigen::Vector3d err_axis = quatMult(quatConj(q), q_goal).tail<3>();
    err_axis.normalize();

    Satellite::VecX x_plus = x;
    Satellite::VecX x_minus = x;
    x_plus.segment<3>(Satellite::AV_INDEX) = 1e-4 * err_axis;
    x_minus.segment<3>(Satellite::AV_INDEX) = -1e-4 * err_axis;

    const double c_plus = sat.stageCost(0, 100, x_plus, u, boresight, target_q, B_eci, cfg);
    const double c_minus = sat.stageCost(0, 100, x_minus, u, boresight, target_q, B_eci, cfg);
    REQUIRE(c_plus < c_minus);
}

TEST_CASE_METHOD(SatelliteCostOmegaFixture,
                 "Quaternion crossterm gradients match finite differences",
                 "[jacobians][omega_ff][quat]") {
    const auto [x, u] = nominalStateAndControl();
    for (const auto& [cfg, label] : std::array<std::pair<CostConfig, std::string>, 3>{
             std::make_pair(costCfg(1e3, 1e4, 0.0, 0.0, 1.0), "new-default"),
             std::make_pair(costCfg(1e3, 1e4, 0.0, 0.5, 1.0), "new-ratio"),
             std::make_pair(costCfg(1e3, 1e4, 0.7, 0.0, 1.0), "legacy")}) {
        DYNAMIC_SECTION(label) {
            const auto [grad_ana, grad_u, lux] =
                sat.stageCostJacobians(0, 100, x, u, boresight, target_q, B_eci, cfg);
            (void)grad_u;
            (void)lux;
            const Satellite::VecX grad_fd = projectedGradFD(x, u, target_q, cfg);
            for (int i = 0; i < 3; ++i) {
                REQUIRE_THAT(grad_ana(i), Catch::Matchers::WithinAbs(grad_fd(i), 5e-4));
            }
            for (int i = 0; i < 4; ++i) {
                REQUIRE_THAT(grad_ana(Satellite::QUAT_INDEX + i),
                             Catch::Matchers::WithinAbs(grad_fd(Satellite::QUAT_INDEX + i), 5e-4));
            }
        }
    }
}

TEST_CASE_METHOD(SatelliteCostOmegaFixture,
                 "Quaternion crossterm Hessian blocks match finite differences",
                 "[hessians][omega_ff][quat]") {
    const auto [x, u] = nominalStateAndControl();
    const CostConfig cfg = costCfg(1e3, 1e4, 0.0, 0.5, 1.0);

    const auto [Hxx, Huu, Hux] = sat.stageCostHessians(0, 100, x, u, boresight, target_q, B_eci, cfg);
    (void)Huu;
    (void)Hux;
    const Satellite::MatX Hfd = hessFD(x, u, target_q, cfg);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            const double expected = (i == j) ? cfg.ang_vel : 0.0;
            REQUIRE_THAT(Hxx(i, j), Catch::Matchers::WithinAbs(expected, 1e-6));
        }
    }
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            REQUIRE_THAT(Hxx(i, Satellite::QUAT_INDEX + j),
                         Catch::Matchers::WithinAbs(Hfd(i, Satellite::QUAT_INDEX + j), 2e-2));
            REQUIRE_THAT(Hxx(Satellite::QUAT_INDEX + j, i),
                         Catch::Matchers::WithinAbs(Hxx(i, Satellite::QUAT_INDEX + j), 1e-10));
        }
    }
}

TEST_CASE_METHOD(SatelliteCostOmegaFixture,
                 "Vector roll-ratio semantics are branch-specific",
                 "[cost][omega_ff][vec]") {
    auto [x, u] = nominalStateAndControl();
    const CostConfig cfg_uniform = costCfg(1e3, 1e4, 0.0, 0.0, 1.0);
    const CostConfig cfg_low_roll = costCfg(1e3, 1e4, 0.0, 0.0, 0.05);

    Satellite::VecX x_roll = x;
    x_roll.segment<3>(Satellite::AV_INDEX) << 0.05, 0.0, 0.0;
    Satellite::VecX x_perp = x;
    x_perp.segment<3>(Satellite::AV_INDEX) << 0.0, 0.05, 0.0;

    const double c_same = sat.stageCost(0, 100, x, u, boresight, target_vec, B_eci, cfg_uniform);
    const double c_same2 = sat.stageCost(0, 100, x, u, boresight, target_vec, B_eci, costCfg(1e3, 1e4, 0.0, 0.0, 1.0));
    const double c_roll_uniform = sat.stageCost(0, 100, x_roll, u, boresight, target_vec, B_eci, cfg_uniform);
    const double c_roll_low = sat.stageCost(0, 100, x_roll, u, boresight, target_vec, B_eci, cfg_low_roll);
    const double c_perp_uniform = sat.stageCost(0, 100, x_perp, u, boresight, target_vec, B_eci, cfg_uniform);
    const double c_perp_low = sat.stageCost(0, 100, x_perp, u, boresight, target_vec, B_eci, cfg_low_roll);
    const double c_quat_uniform = sat.stageCost(0, 100, x, u, boresight, target_q, B_eci, cfg_uniform);
    const double c_quat_low = sat.stageCost(0, 100, x, u, boresight, target_q, B_eci, cfg_low_roll);

    REQUIRE_THAT(c_same, Catch::Matchers::WithinAbs(c_same2, 1e-12));
    REQUIRE(c_roll_low < c_roll_uniform);
    REQUIRE_THAT(c_perp_uniform, Catch::Matchers::WithinAbs(c_perp_low, 1e-12));
    REQUIRE_THAT(c_quat_uniform, Catch::Matchers::WithinAbs(c_quat_low, 1e-12));
}

TEST_CASE_METHOD(SatelliteCostOmegaFixture,
                 "Vector crossterm rewards error-reducing angular velocity",
                 "[cost][omega_ff][vec]") {
    auto [x, u] = nominalStateAndControl();
    u.setZero();
    const CostConfig cfg = costCfg(1e3, 1e4, 0.0, 0.5, 0.5);

    const Eigen::Vector3d r_eci = target_vec.tail<3>().normalized();
    Eigen::Vector3d err_axis = boresight.cross(quatRotMat(x.segment<4>(Satellite::QUAT_INDEX)).transpose() * r_eci);
    err_axis.normalize();

    Satellite::VecX x_plus = x;
    Satellite::VecX x_minus = x;
    x_plus.segment<3>(Satellite::AV_INDEX) = 1e-4 * err_axis;
    x_minus.segment<3>(Satellite::AV_INDEX) = -1e-4 * err_axis;

    const double c_plus = sat.stageCost(0, 100, x_plus, u, boresight, target_vec, B_eci, cfg);
    const double c_minus = sat.stageCost(0, 100, x_minus, u, boresight, target_vec, B_eci, cfg);
    REQUIRE(c_plus < c_minus);
}

TEST_CASE_METHOD(SatelliteCostOmegaFixture,
                 "Vector roll-ratio and crossterm derivatives match finite differences",
                 "[jacobians][hessians][omega_ff][vec]") {
    const auto [x, u] = nominalStateAndControl();

    for (double ratio : {0.0, 0.5}) {
        DYNAMIC_SECTION("grad ratio=" << ratio) {
            const CostConfig cfg = costCfg(1e3, 1e4, 0.0, ratio, 0.5);
            const auto [grad_ana, grad_u, lux] =
                sat.stageCostJacobians(0, 100, x, u, boresight, target_vec, B_eci, cfg);
            (void)grad_u;
            (void)lux;
            const Satellite::VecX grad_fd = projectedGradFD(x, u, target_vec, cfg);
            for (int i = 0; i < 3; ++i) {
                REQUIRE_THAT(grad_ana(i), Catch::Matchers::WithinAbs(grad_fd(i), 5e-4));
            }
            for (int i = 0; i < 4; ++i) {
                REQUIRE_THAT(grad_ana(Satellite::QUAT_INDEX + i),
                             Catch::Matchers::WithinAbs(grad_fd(Satellite::QUAT_INDEX + i), 5e-4));
            }
        }
    }

    const CostConfig cfg = costCfg(1e3, 1e4, 0.0, 0.5, 0.5);
    const auto [Hxx, Huu, Hux] = sat.stageCostHessians(0, 100, x, u, boresight, target_vec, B_eci, cfg);
    (void)Huu;
    (void)Hux;
    const Satellite::MatX Hfd = hessFD(x, u, target_vec, cfg);
    const Eigen::Matrix3d expected =
        cfg.ang_vel * Eigen::Matrix3d::Identity() - cfg.ang_vel * 0.5 * (boresight * boresight.transpose());

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            REQUIRE_THAT(Hxx(i, j), Catch::Matchers::WithinAbs(expected(i, j), 1e-8));
        }
    }
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            REQUIRE_THAT(Hxx(i, Satellite::QUAT_INDEX + j),
                         Catch::Matchers::WithinAbs(Hfd(i, Satellite::QUAT_INDEX + j), 2e-2));
            REQUIRE_THAT(Hxx(Satellite::QUAT_INDEX + j, i),
                         Catch::Matchers::WithinAbs(Hxx(i, Satellite::QUAT_INDEX + j), 1e-10));
        }
    }
}

TEST_CASE_METHOD(SatelliteCostOmegaFixture,
                 "Vector angle cost retains 2-DOF pointing semantics",
                 "[cost][omega_ff][vec]") {
    auto [x, u] = nominalStateAndControl();
    x.segment<3>(Satellite::AV_INDEX).setZero();
    x.segment<3>(Satellite::RW_MOMENTUM_INDEX).setZero();
    const double th = 30.0 * kPi / 180.0 / 2.0;
    x.segment<4>(Satellite::QUAT_INDEX) = Eigen::Vector4d(std::cos(th), 0.0, 0.0, std::sin(th));

    for (const int act : {0, 1, 3}) {  // implemented set (type 2 removed)
        CostConfig cfg = costCfg(1e2, 0.0, 0.0, 0.0, 1.0);
        cfg.ang_cost_func_type = act;
        const double cost = sat.stageCost(0, 100, x, u, boresight, target_vec, B_eci, cfg);
        REQUIRE_THAT(cost, Catch::Matchers::WithinAbs(0.0, 1e-8));
    }

    CostConfig cfg = costCfg(1e2, 0.0, 0.0, 0.0, 1.0);
    cfg.ang_cost_func_type = 3;
    const double roll_half = 45.0 * kPi / 180.0 / 2.0;
    const Eigen::Vector4d q_roll(std::cos(roll_half), std::sin(roll_half), 0.0, 0.0);
    Satellite::VecX x_rolled = x;
    x_rolled.segment<4>(Satellite::QUAT_INDEX) =
        quatMult(x.segment<4>(Satellite::QUAT_INDEX), q_roll).normalized();
    const double cost_rolled = sat.stageCost(0, 100, x_rolled, u, boresight, target_vec, B_eci, cfg);
    REQUIRE_THAT(cost_rolled, Catch::Matchers::WithinAbs(0.0, 1e-8));
}
