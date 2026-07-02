#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <vector>

#include <Eigen/Dense>

#include <saltro/math/mrp.h>
#include <saltro/math/quaternion.h>
#include <saltro/optimizer/spike_removal.h>
#include <saltro/pybind/controller/pdcontroller.h>
#include <saltro/pybind/satellite.h>
#include <saltro/pybind/plannersettings.h>

// C++ twin for the spike-removal detector. The Python suite
// (tests/unit/optimizer/test_spike_removal.py) exercises a *reference*
// implementation under tests/debug/; the shipped C++ detectSpikes() in
// src/optimizer/spike_removal.cpp had no direct test. These cases drive it with
// hand-built pointing-error profiles so the detection logic is pinned.

using saltro::optimizer::detectSpikes;
using saltro::optimizer::SpikeCandidate;

namespace {

// Satellite is non-copyable, so configure one in place.
void setupSpikeSat(Satellite& sat) {
    sat.addMTQ(Eigen::Vector3d(1.0, 0.0, 0.0), 0.2);
    sat.addMTQ(Eigen::Vector3d(0.0, 1.0, 0.0), 0.2);
    sat.addMTQ(Eigen::Vector3d(0.0, 0.0, 1.0), 0.2);
    sat.addRW(Eigen::Vector3d(1.0, 0.0, 0.0), 0.001, 1e-5, 0.0, 0.02);
    sat.addRW(Eigen::Vector3d(0.0, 1.0, 0.0), 0.001, 1e-5, 0.0, 0.02);
    sat.addRW(Eigen::Vector3d(0.0, 0.0, 1.0), 0.001, 1e-5, 0.0, 0.02);
}

// Build trajectory matrices realizing a given pointing-error profile. The
// attitude target is the identity quaternion (held constant so no spurious goal
// transitions are flagged), and X carries a rotation of theta about +Z, whose
// rotation angle relative to identity is exactly theta — so the detector's
// pointingError(q, q_target) == theta. Controls are left at zero so the
// actuation/physics filters never veto (and the second, control-effort-gated
// pass never fires).
struct Traj {
    Eigen::MatrixXd X, U, attitude_target, boresight, B;
};

// phi[k] is the target's own rotation about +Z at knot k (lets a test move the
// goal — smoothly for tracking, or with a discrete jump for a re-tasking). The
// boresight is placed at angle phi[k]+theta[k], so the pointing error stays
// exactly theta[k] regardless of how the target moves. phi empty => held target.
Traj buildTrajectory(const Satellite& sat, const std::vector<double>& theta,
                     bool vector_mode = false, const std::vector<double>& phi = {}) {
    const int N = static_cast<int>(theta.size());
    Traj t;
    t.X = Eigen::MatrixXd::Zero(sat.stateDim(), N);
    t.U = Eigen::MatrixXd::Zero(sat.controlDim(), N);
    t.attitude_target = Eigen::MatrixXd::Zero(4, N);
    t.boresight = Eigen::MatrixXd::Zero(3, N);
    t.B = Eigen::MatrixXd::Zero(3, N);
    for (int k = 0; k < N; ++k) {
        const double ph = phi.empty() ? 0.0 : phi[static_cast<std::size_t>(k)];
        const double ang = ph + theta[static_cast<std::size_t>(k)];
        t.X(3, k) = std::cos(ang / 2.0);  // q0  (boresight at phi+theta about +Z)
        t.X(6, k) = std::sin(ang / 2.0);  // q3
        if (vector_mode) {
            // Vector-pointing target [NaN, x, y, z]: ECI direction at angle phi.
            t.attitude_target(0, k) = std::nan("");
            t.attitude_target(1, k) = std::cos(ph);
            t.attitude_target(2, k) = std::sin(ph);
        } else {
            // Quaternion target: rotation about +Z by phi.
            t.attitude_target(0, k) = std::cos(ph / 2.0);
            t.attitude_target(3, k) = std::sin(ph / 2.0);
        }
        t.boresight(0, k) = 1.0;
        t.B(0, k) = 2.5e-5;
        t.B(1, k) = -1.5e-5;
        t.B(2, k) = 3.0e-5;
    }
    return t;
}

// The converge -> spike -> return pointing-error profile used by the detection
// tests (entry 0.05, peak 0.30 = 6x, return below entry*exit_fudge).
std::vector<double> spikeProfile() {
    std::vector<double> theta;
    for (int k = 0; k <= 14; ++k) theta.push_back(0.50 - k * (0.45 / 14.0));
    for (int k = 1; k <= 8; ++k) theta.push_back(0.05 + k * (0.25 / 8.0));
    for (int k = 1; k <= 9; ++k) theta.push_back(0.30 - k * (0.26 / 9.0));
    return theta;
}

}  // namespace

TEST_CASE("detectSpikes flags a converge -> spike -> return window", "[spike_removal][detect]") {
    Eigen::Matrix3d J = Eigen::Vector3d(0.067, 0.071, 0.069).asDiagonal();
    PlannerSettings settings;
    Satellite sat(J, settings);
    setupSpikeSat(sat);
    SpikeRemovalConfig cfg;   // defaults: min_consecutive=7, min_prior_decrease_knots=10,
                              //           min_spike_ratio=3, exit_fudge=2
    ConstraintConfig cnst;

    const std::vector<double> theta = spikeProfile();
    Traj t = buildTrajectory(sat, theta);
    std::vector<SpikeCandidate> spikes =
        detectSpikes(sat, t.X, t.U, t.attitude_target, t.boresight, t.B, cnst, cfg);

    REQUIRE(spikes.size() == 1);
    const auto& s = spikes.front();
    // Enter at the convergence minimum (~knot 14), exit after the peak (knot 22).
    REQUIRE(s.first >= 13);
    REQUIRE(s.first <= 15);
    REQUIRE(s.second > 22);
    REQUIRE(s.second < static_cast<int>(theta.size()));
}

TEST_CASE("detectSpikes works in vector-pointing mode (NaN-sentinel target)",
          "[spike_removal][detect][vector_pointing]") {
    // Regression guard for findGoalTransitions: a held vector-pointing target is
    // [NaN, x, y, z]. A naive (col(k)-col(k-1)).isZero() check is ALWAYS false
    // because NaN != NaN, so every knot was flagged as a goal transition and the
    // whole horizon was buffered out of detection — making the detector blind to
    // every spike in vector-pointing mode. The same spike must now be found.
    Eigen::Matrix3d J = Eigen::Vector3d(0.067, 0.071, 0.069).asDiagonal();
    PlannerSettings settings;
    Satellite sat(J, settings);
    setupSpikeSat(sat);
    SpikeRemovalConfig cfg;
    ConstraintConfig cnst;

    const std::vector<double> theta = spikeProfile();
    Traj t = buildTrajectory(sat, theta, /*vector_mode=*/true);
    std::vector<SpikeCandidate> spikes =
        detectSpikes(sat, t.X, t.U, t.attitude_target, t.boresight, t.B, cnst, cfg);

    REQUIRE(spikes.size() == 1);
    REQUIRE(spikes.front().first >= 13);
    REQUIRE(spikes.front().first <= 15);
}

TEST_CASE("detectSpikes still fires when the goal is slowly slewing (not a transition)",
          "[spike_removal][detect][goal_transition]") {
    // The key case: a continuously slewing target (e.g. nadir tracking) must NOT
    // be mistaken for a goal transition. Here the vector target rotates a steady
    // 0.5 deg/knot — well below the 5 deg floor — so findGoalTransitions flags
    // nothing and the spike is still detected. (Under the old "any change is a
    // transition" rule this slow drift buffered the whole horizon and the spike
    // was missed.)
    Eigen::Matrix3d J = Eigen::Vector3d(0.067, 0.071, 0.069).asDiagonal();
    PlannerSettings settings;
    Satellite sat(J, settings);
    setupSpikeSat(sat);
    SpikeRemovalConfig cfg;
    ConstraintConfig cnst;

    const std::vector<double> theta = spikeProfile();
    std::vector<double> phi(theta.size(), 0.0);
    const double slew_per_knot = 0.5 * M_PI / 180.0;  // 0.5 deg/knot
    for (std::size_t k = 0; k < phi.size(); ++k) phi[k] = static_cast<double>(k) * slew_per_knot;

    Traj t = buildTrajectory(sat, theta, /*vector_mode=*/true, phi);
    std::vector<SpikeCandidate> spikes =
        detectSpikes(sat, t.X, t.U, t.attitude_target, t.boresight, t.B, cnst, cfg);

    REQUIRE(spikes.size() == 1);
    REQUIRE(spikes.front().first >= 13);
    REQUIRE(spikes.front().first <= 15);
}

TEST_CASE("detectSpikes buffers a discrete goal re-tasking (nadir -> sun style jump)",
          "[spike_removal][detect][goal_transition]") {
    // The complementary case: a genuine discrete goal switch (here a 60 deg jump
    // in the target direction at knot 20, mid-spike) MUST be treated as a
    // transition so the slew-induced excursion isn't removed as a spike. The
    // goal-switch buffer (+/-15 knots) then covers the whole window -> no
    // candidate. The identical excursion with a held goal is detected (the case
    // above), so this isolates the transition handling.
    Eigen::Matrix3d J = Eigen::Vector3d(0.067, 0.071, 0.069).asDiagonal();
    PlannerSettings settings;
    Satellite sat(J, settings);
    setupSpikeSat(sat);
    SpikeRemovalConfig cfg;
    ConstraintConfig cnst;

    const std::vector<double> theta = spikeProfile();
    std::vector<double> phi(theta.size(), 0.0);
    const double jump = 60.0 * M_PI / 180.0;
    for (std::size_t k = 20; k < phi.size(); ++k) phi[k] = jump;  // discrete step at knot 20

    Traj t = buildTrajectory(sat, theta, /*vector_mode=*/true, phi);
    std::vector<SpikeCandidate> spikes =
        detectSpikes(sat, t.X, t.U, t.attitude_target, t.boresight, t.B, cnst, cfg);

    REQUIRE(spikes.empty());
}

TEST_CASE("detectSpikes returns nothing for a monotonically converging trajectory",
          "[spike_removal][detect]") {
    Eigen::Matrix3d J = Eigen::Vector3d(0.067, 0.071, 0.069).asDiagonal();
    PlannerSettings settings;
    Satellite sat(J, settings);
    setupSpikeSat(sat);
    SpikeRemovalConfig cfg;
    ConstraintConfig cnst;

    // Pure convergence: strictly decreasing, never re-rises -> no spike under either pass.
    std::vector<double> theta;
    const int N = 32;
    for (int k = 0; k < N; ++k) theta.push_back(0.50 - k * (0.49 / (N - 1)));

    Traj t = buildTrajectory(sat, theta);
    std::vector<SpikeCandidate> spikes =
        detectSpikes(sat, t.X, t.U, t.attitude_target, t.boresight, t.B, cnst, cfg);

    REQUIRE(spikes.empty());
}

// ===========================================================================
// Substitution-machinery twins of the Python reference suite
// (tests/unit/optimizer/test_spike_removal.py), driving the shipped C++
// implementation through saltro::optimizer::spike_removal_detail.
// ===========================================================================

namespace {

namespace detail = saltro::optimizer::spike_removal_detail;

// Python make_satellite(): 3 MTQ + 1 RW hybrid (sat_3_1_hybrid.py).
void setupHybrid31Sat(Satellite& sat) {
    sat.addMTQ(Eigen::Vector3d(1.0, 0.0, 0.0), 0.2);
    sat.addMTQ(Eigen::Vector3d(0.0, 1.0, 0.0), 0.2);
    sat.addMTQ(Eigen::Vector3d(0.0, 0.0, 1.0), 0.2);
    sat.addRW(Eigen::Vector3d(1.0, 0.0, 0.0), 5.7e-6, 0.0023, 0.0, 0.0036);
}

Eigen::Matrix3d hybridInertia() {
    Eigen::Matrix3d J;
    J <<  0.03136490806,  5.88304e-05,  -0.00671361357,
          5.88304e-05,    0.03409127827, -0.00012334756,
         -0.00671361357, -0.00012334756,  0.01004091997;
    return J;
}

// Python make_env(): constant representative-LEO environment (3 x N).
struct EnvMats {
    Eigen::MatrixXd B, S, R, V, rho;
};

EnvMats makeEnv(int N) {
    EnvMats e;
    e.B = Eigen::MatrixXd::Zero(3, N);
    e.B.row(2).setConstant(3.1e-5);  // 31 uT in Z
    e.S = Eigen::MatrixXd::Zero(3, N);
    e.S.row(2).setConstant(1.0);
    e.R = Eigen::MatrixXd::Zero(3, N);
    e.R.row(0).setConstant(7000e3);
    e.V = Eigen::MatrixXd::Zero(3, N);
    e.V.row(1).setConstant(7.5e3);
    e.rho = Eigen::MatrixXd::Zero(1, N);
    return e;
}

}  // namespace

// Twin of test_actuation_filter_physics_limited: a spike whose excursion is
// explained by saturated actuators fighting the error correctly must be
// discarded by the physics-limited filter.  A Magic (direct body-torque)
// actuator makes the geometry exact: tau = u * axis, independent of attitude
// and B-field.  Before the Magic-aware isSaturated fix, the filter could
// never vote when the opposing torque came from a Magic channel, so the
// candidate was (wrongly) kept.
TEST_CASE("actuation filter discards a physics-limited spike (Magic torque opposing error)",
          "[spike_removal][actuation_filter]") {
    Eigen::Matrix3d J = Eigen::Vector3d(0.067, 0.071, 0.069).asDiagonal();
    PlannerSettings settings;
    Satellite sat(J, settings);
    sat.addMTQ(Eigen::Vector3d(1.0, 0.0, 0.0), 0.2);
    sat.addMTQ(Eigen::Vector3d(0.0, 1.0, 0.0), 0.2);
    sat.addMTQ(Eigen::Vector3d(0.0, 0.0, 1.0), 0.2);
    sat.addMagic(Eigen::Vector3d(0.0, 0.0, 1.0), 0.01);
    SpikeRemovalConfig cfg;
    ConstraintConfig cnst;

    const std::vector<double> theta = spikeProfile();
    Traj t = buildTrajectory(sat, theta);

    // Control: with zero controls the spike is detected.
    REQUIRE(detectSpikes(sat, t.X, t.U, t.attitude_target, t.boresight, t.B, cnst, cfg).size() == 1);

    // Saturate the Magic channel with a torque that CORRECTS the error at
    // every knot (alpha opposes the error axis).  The trajectory rotates
    // about +Z relative to the identity target, so the correcting torque is
    // -sign(q_err_z) about Z.
    const int magic_idx = sat.numMTQ() + sat.numRW();  // control index 3
    const Eigen::Vector4d q_identity(1.0, 0.0, 0.0, 0.0);
    for (int k = 0; k < t.U.cols(); ++k) {
        const Eigen::Vector4d q_k = t.X.col(k).segment<4>(3);
        const Eigen::Vector4d q_err = saltro::math::quatError(q_identity, q_k);
        if (q_err.tail<3>().norm() < 1e-10) continue;
        const double s = (q_err(3) > 0.0) ? 1.0 : -1.0;
        t.U(magic_idx, k) = -s * 0.01;  // hardware-saturated, correcting
    }

    // The exposed helpers agree the mid-spike knot is physics-limited.
    const int k_mid = 18;
    const Eigen::VectorXd u_mid = t.U.col(k_mid);
    const Eigen::Vector3d tau_mid = sat.actuatorTorque(t.X.col(k_mid), u_mid, t.B.col(k_mid));
    REQUIRE(detail::isSaturated(u_mid, sat, cnst.control_limit_scale));
    REQUIRE(detail::torqueOpposesError(t.X.col(k_mid), tau_mid,
                                       t.attitude_target.col(k_mid),
                                       Eigen::Vector3d(t.boresight.col(k_mid)), sat));

    // ... so the candidate is discarded.
    REQUIRE(detectSpikes(sat, t.X, t.U, t.attitude_target, t.boresight, t.B, cnst, cfg).empty());
}

// Twin of test_pd_sim_normalization: the PD rollout must produce unit
// quaternions and finite states at every step.
TEST_CASE("PD sim keeps quaternions normalized and states finite",
          "[spike_removal][pd_sim]") {
    PlannerSettings settings;
    Satellite sat(hybridInertia(), settings);
    setupHybrid31Sat(sat);

    saltro::controller::PDController pd(sat);
    pd.setGains(2.0, 5.0);

    const int n_steps = 10;
    EnvMats env = makeEnv(n_steps);

    Eigen::VectorXd x_start = Eigen::VectorXd::Zero(sat.stateDim());
    x_start(3) = 1.0;   // identity quaternion
    x_start(0) = 0.05;  // small angular velocity
    x_start(1) = 0.02;

    // Target: 90-deg rotation about Z
    const double h = M_PI / 4.0;
    const Eigen::Vector4d q_target(std::cos(h), 0.0, 0.0, std::sin(h));

    SpikeRemovalConfig cfg;  // omega_max = 0 -> no clamping
    auto [X_pd, U_pd] = detail::simulatePDSegment(
        sat, pd, x_start, q_target, n_steps,
        env.B, env.S, env.R, env.V, env.rho,
        settings.disturbances, 10.0, cfg);

    REQUIRE(X_pd.rows() == sat.stateDim());
    REQUIRE(X_pd.cols() == n_steps + 1);
    REQUIRE(U_pd.rows() == sat.controlDim());
    REQUIRE(U_pd.cols() == n_steps);
    REQUIRE(X_pd.allFinite());
    REQUIRE(U_pd.allFinite());
    for (int k = 0; k < X_pd.cols(); ++k) {
        const double qn = X_pd.col(k).segment<4>(3).norm();
        REQUIRE(std::abs(qn - 1.0) < 1e-9);
    }
}

// Regression guard for the inherited Magic OOB crash: with a Magic-carrying
// satellite, PDController::find_u must return a full controlDim()-sized
// vector (main's PR #28 behavior) so the PD rollout's U_pd column assignment
// does not go out of bounds — and the substitution machinery must produce a
// valid trajectory driven by the Magic channels.
TEST_CASE("PD sim is well-formed on a Magic-carrying satellite",
          "[spike_removal][pd_sim][magic]") {
    Eigen::Matrix3d J = Eigen::Vector3d(0.067, 0.071, 0.069).asDiagonal();
    PlannerSettings settings;
    Satellite sat(J, settings);
    sat.addMagic(Eigen::Vector3d(1.0, 0.0, 0.0), 0.02);
    sat.addMagic(Eigen::Vector3d(0.0, 1.0, 0.0), 0.02);
    sat.addMagic(Eigen::Vector3d(0.0, 0.0, 1.0), 0.02);

    saltro::controller::PDController pd(sat);
    pd.setGains(2.0, 5.0);
    pd.setRWScale(1.0);

    const int n_steps = 10;
    EnvMats env = makeEnv(n_steps);

    Eigen::VectorXd x_start = Eigen::VectorXd::Zero(sat.stateDim());
    x_start(3) = 1.0;
    x_start(0) = 0.05;

    const double h = M_PI / 4.0;
    const Eigen::Vector4d q_target(std::cos(h), 0.0, 0.0, std::sin(h));

    SpikeRemovalConfig cfg;
    auto [X_pd, U_pd] = detail::simulatePDSegment(
        sat, pd, x_start, q_target, n_steps,
        env.B, env.S, env.R, env.V, env.rho,
        settings.disturbances, 10.0, cfg);

    REQUIRE(U_pd.rows() == sat.controlDim());  // includes the Magic channels
    REQUIRE(X_pd.allFinite());
    REQUIRE(U_pd.allFinite());
    // PD actually commands the Magic actuators (nonzero effort somewhere).
    REQUIRE(U_pd.cwiseAbs().maxCoeff() > 0.0);
    for (int k = 0; k < X_pd.cols(); ++k) {
        const double qn = X_pd.col(k).segment<4>(3).norm();
        REQUIRE(std::abs(qn - 1.0) < 1e-9);
    }
}

// Twin of test_tail_rerollout_gain_correction: applying the iLQR feedback
// gain to the reduced state error at a stitch point must track the nominal
// trajectory at least as well as pure open-loop propagation.
TEST_CASE("tail re-rollout gain correction tracks nominal at least as well as open-loop",
          "[spike_removal][tail_rerollout]") {
    PlannerSettings settings;
    Satellite sat(hybridInertia(), settings);
    setupHybrid31Sat(sat);

    const int N = 20;
    EnvMats env = makeEnv(N);
    const double dt = 10.0;
    const int nu = sat.controlDim();
    const int nxr = sat.reducedStateDim();
    const int n_rw = sat.numRW();

    Eigen::MatrixXd X = Eigen::MatrixXd::Zero(sat.stateDim(), N);
    X.row(3).setConstant(1.0);
    Eigen::MatrixXd X_nominal = X;
    X(0, 5) = 0.01;  // small omega_x deviation at the stitch point

    // Simple stabilizing diagonal gain (nu x nxr).
    Eigen::MatrixXd K_k = Eigen::MatrixXd::Zero(nu, nxr);
    for (int i = 0; i < std::min(nu, nxr); ++i) K_k(i, i) = -0.1;

    // Reduced state error [delta_omega, mrp_err, delta_h] (iLQR convention).
    Eigen::VectorXd dx = Eigen::VectorXd::Zero(nxr);
    dx.head<3>() = X.col(5).head<3>() - X_nominal.col(5).head<3>();
    const Eigen::Vector4d q_err = saltro::math::quatError(
        X_nominal.col(5).segment<4>(3), X.col(5).segment<4>(3));
    dx.segment<3>(3) = saltro::math::quatToMRP(q_err);
    for (int i = 0; i < n_rw; ++i) dx(6 + i) = X(7 + i, 5) - X_nominal(7 + i, 5);

    const Eigen::VectorXd u_bar = Eigen::VectorXd::Zero(nu);
    const Eigen::VectorXd u_with_gain = u_bar + K_k * dx;

    const Eigen::Vector3d R5(env.R.col(5)), B5(env.B.col(5)), S5(env.S.col(5)), V5(env.V.col(5));
    const int rho5 = 0;
    const Eigen::VectorXd x_gain =
        sat.dynamicsStepRK4(X.col(5), u_with_gain, dt, settings.disturbances, R5, B5, S5, V5, rho5);
    const Eigen::VectorXd x_open =
        sat.dynamicsStepRK4(X.col(5), u_bar, dt, settings.disturbances, R5, B5, S5, V5, rho5);
    const Eigen::VectorXd x_nominal_6 =
        sat.dynamicsStepRK4(X_nominal.col(5), u_bar, dt, settings.disturbances, R5, B5, S5, V5, rho5);

    const double err_gain = (x_gain.head<3>() - x_nominal_6.head<3>()).norm();
    const double err_open = (x_open.head<3>() - x_nominal_6.head<3>()).norm();
    REQUIRE(err_gain <= err_open + 1e-10);
}

// Twin of test_compare_costs_pd_cheaper, plus the rejection direction that is
// the point of the new backstop in applySpikeRemoval.
TEST_CASE("compareCosts accepts a cheaper PD window and rejects a costlier one",
          "[spike_removal][compare_costs]") {
    PlannerSettings settings;
    Satellite sat(hybridInertia(), settings);
    setupHybrid31Sat(sat);

    const int N = 30;
    EnvMats env = makeEnv(N);
    const int nu = sat.controlDim();
    const int nx = sat.stateDim();

    CostConfig cost_cfg;
    cost_cfg.angle = 1e2;
    cost_cfg.ang_vel = 1.0;
    cost_cfg.control_mult = 1.0;
    cost_cfg.mtq_control_weight = 0.1;
    cost_cfg.rw_control_weight = 1.0;
    cost_cfg.ang_cost_func_type = 3;

    Eigen::MatrixXd attitude_target = Eigen::MatrixXd::Zero(4, N);
    attitude_target.row(0).setConstant(1.0);  // identity quaternion goal
    Eigen::MatrixXd boresight = Eigen::MatrixXd::Zero(3, N);
    boresight.row(0).setConstant(1.0);

    const int t_enter = 5;
    const int t_exit = 20;
    const int n_pd = t_exit - t_enter;

    // Original: 180-deg detour inside the spike window.
    Eigen::MatrixXd X_detour = Eigen::MatrixXd::Zero(nx, N);
    X_detour.row(3).setConstant(1.0);
    for (int k = t_enter; k < t_exit; ++k) {
        X_detour(3, k) = std::cos(M_PI / 2.0);
        X_detour(6, k) = std::sin(M_PI / 2.0);
    }
    Eigen::MatrixXd U_zero = Eigen::MatrixXd::Zero(nu, N);

    // PD: stays at the goal.
    Eigen::MatrixXd X_at_goal_pd = Eigen::MatrixXd::Zero(nx, n_pd + 1);
    X_at_goal_pd.row(3).setConstant(1.0);
    Eigen::MatrixXd U_pd = Eigen::MatrixXd::Zero(nu, n_pd);

    SECTION("PD at goal is cheaper than a 180-deg detour -> accepted") {
        double cost_orig = 0.0, cost_pd = 0.0;
        const bool accept = detail::compareCosts(
            sat, X_detour, U_zero, X_at_goal_pd, U_pd, t_enter, t_exit,
            env.B, boresight, attitude_target, cost_cfg, N, &cost_orig, &cost_pd);
        REQUIRE(accept);
        REQUIRE(cost_pd < cost_orig);
        REQUIRE(std::isfinite(cost_orig));
        REQUIRE(std::isfinite(cost_pd));
    }

    SECTION("PD on a 180-deg detour is costlier than staying at goal -> rejected") {
        Eigen::MatrixXd X_at_goal = Eigen::MatrixXd::Zero(nx, N);
        X_at_goal.row(3).setConstant(1.0);
        Eigen::MatrixXd X_detour_pd = Eigen::MatrixXd::Zero(nx, n_pd + 1);
        X_detour_pd.row(3).setConstant(1.0);
        for (int i = 0; i < n_pd; ++i) {
            X_detour_pd(3, i) = std::cos(M_PI / 2.0);
            X_detour_pd(6, i) = std::sin(M_PI / 2.0);
        }
        double cost_orig = 0.0, cost_pd = 0.0;
        const bool accept = detail::compareCosts(
            sat, X_at_goal, U_zero, X_detour_pd, U_pd, t_enter, t_exit,
            env.B, boresight, attitude_target, cost_cfg, N, &cost_orig, &cost_pd);
        REQUIRE_FALSE(accept);
        REQUIRE(cost_pd > cost_orig);
    }

    SECTION("equal-cost substitution is rejected (strict comparison)") {
        const bool accept = detail::compareCosts(
            sat, X_detour, U_zero,
            Eigen::MatrixXd(X_detour.block(0, t_enter, nx, n_pd + 1)),
            Eigen::MatrixXd(U_zero.block(0, t_enter, nu, n_pd)),
            t_enter, t_exit,
            env.B, boresight, attitude_target, cost_cfg, N);
        REQUIRE_FALSE(accept);
    }
}

// Twin of test_keepout_sun_violation: a PD segment whose attitude puts the
// sun inside the keep-out cone must be flagged.
TEST_CASE("keep-out check flags a sun-in-boresight PD segment",
          "[spike_removal][keepout]") {
    PlannerSettings settings;
    Satellite sat(hybridInertia(), settings);
    setupHybrid31Sat(sat);
    ConstraintConfig cnst;

    const int N = 10;
    const int n_pd_steps = 5;
    const int nu = sat.controlDim();

    // Identity attitude + sun along ECI +X -> sun_body.x() = 1 > cos(20 deg).
    Eigen::VectorXd x_test = Eigen::VectorXd::Zero(sat.stateDim());
    x_test(3) = 1.0;
    const Eigen::Vector3d sun_eci(1.0, 0.0, 0.0);

    // Sanity: the sun-avoidance constraint (index 1) fires at this attitude.
    const Eigen::VectorXd c_test =
        sat.constraints(0, N, x_test, Eigen::VectorXd::Zero(nu), sun_eci, cnst);
    REQUIRE(c_test.size() > 1);
    REQUIRE(c_test(1) > 0.0);

    Eigen::MatrixXd X_pd = Eigen::MatrixXd::Zero(sat.stateDim(), n_pd_steps + 1);
    X_pd.row(3).setConstant(1.0);
    Eigen::MatrixXd U_pd = Eigen::MatrixXd::Zero(nu, n_pd_steps);

    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(3, N);
    S.row(0).setConstant(1.0);
    REQUIRE_FALSE(detail::keepoutClear(sat, X_pd, U_pd, S, cnst, N, 0));

    // Sun on the anti-boresight side is clear.
    Eigen::MatrixXd S_clear = Eigen::MatrixXd::Zero(3, N);
    S_clear.row(0).setConstant(-1.0);
    REQUIRE(detail::keepoutClear(sat, X_pd, U_pd, S_clear, cnst, N, 0));
}

// New coverage for the Magic-aware saturation fix: Magic control channels
// (indices n_mtq + n_rw + i) must be visible to isSaturated, both on hybrid
// actuator sets and on Magic-only sets (where the pass-2 control-effort gate
// was previously permanently false).
TEST_CASE("isSaturated sees Magic actuator channels", "[spike_removal][magic]") {
    Eigen::Matrix3d J = Eigen::Vector3d(0.067, 0.071, 0.069).asDiagonal();
    PlannerSettings settings;

    SECTION("hybrid MTQ + Magic set") {
        Satellite sat(J, settings);
        sat.addMTQ(Eigen::Vector3d(1.0, 0.0, 0.0), 0.2);
        sat.addMTQ(Eigen::Vector3d(0.0, 1.0, 0.0), 0.2);
        sat.addMTQ(Eigen::Vector3d(0.0, 0.0, 1.0), 0.2);
        sat.addMagic(Eigen::Vector3d(0.0, 0.0, 1.0), 0.01);

        Eigen::VectorXd u = Eigen::VectorXd::Zero(sat.controlDim());
        REQUIRE_FALSE(detail::isSaturated(u, sat, 0.75));

        // Magic channel at 75% of hardware max >= 0.9 * 0.75 ceiling -> saturated.
        u(3) = 0.75 * 0.01;
        REQUIRE(detail::isSaturated(u, sat, 0.75));

        // Below the ceiling -> not saturated.
        u(3) = 0.5 * 0.01;
        REQUIRE_FALSE(detail::isSaturated(u, sat, 0.75));

        // Sign-symmetric.
        u(3) = -0.75 * 0.01;
        REQUIRE(detail::isSaturated(u, sat, 0.75));
    }

    SECTION("Magic-only set (pass-2 control-effort gate)") {
        Satellite sat(J, settings);
        sat.addMagic(Eigen::Vector3d(1.0, 0.0, 0.0), 0.01);
        sat.addMagic(Eigen::Vector3d(0.0, 1.0, 0.0), 0.01);
        sat.addMagic(Eigen::Vector3d(0.0, 0.0, 1.0), 0.01);

        Eigen::VectorXd u = Eigen::VectorXd::Zero(3);
        // Pass-2 heuristic parameters: hardware scale 1.0, 50% threshold.
        REQUIRE_FALSE(detail::isSaturated(u, sat, 1.0, 0.5));
        u(2) = 0.006;  // 60% of u_max
        REQUIRE(detail::isSaturated(u, sat, 1.0, 0.5));
        u(2) = 0.004;  // 40% of u_max
        REQUIRE_FALSE(detail::isSaturated(u, sat, 1.0, 0.5));
    }
}

// New coverage for the vec-mode error axis: torqueOpposesError must work for
// vector-pointing ([NaN, x, y, z]) targets, using boresight x r_body — the
// axis that rotates the boresight toward the target — as the error axis.
TEST_CASE("torqueOpposesError handles vector-pointing (NaN-sentinel) targets",
          "[spike_removal][vector_pointing]") {
    // Diagonal inertia so alpha = J^-1 tau stays axis-aligned.
    Eigen::Matrix3d J = Eigen::Vector3d(0.03, 0.04, 0.05).asDiagonal();
    PlannerSettings settings;
    Satellite sat(J, settings);
    sat.addMTQ(Eigen::Vector3d(1.0, 0.0, 0.0), 0.2);
    sat.addMTQ(Eigen::Vector3d(0.0, 1.0, 0.0), 0.2);
    sat.addMTQ(Eigen::Vector3d(0.0, 0.0, 1.0), 0.2);

    Eigen::VectorXd x = Eigen::VectorXd::Zero(sat.stateDim());
    x(3) = 1.0;  // identity attitude
    const Eigen::Vector3d bs(1.0, 0.0, 0.0);  // boresight +X

    SECTION("target at ECI +Y: torque about +Z corrects, about -Z drives away") {
        // r_body = +Y at identity attitude; bs x r_body = +Z rotates the
        // boresight toward the target.
        Eigen::Vector4d target(std::nan(""), 0.0, 1.0, 0.0);
        REQUIRE(detail::torqueOpposesError(x, Eigen::Vector3d(0.0, 0.0, 1e-3), target, bs, sat));
        REQUIRE_FALSE(detail::torqueOpposesError(x, Eigen::Vector3d(0.0, 0.0, -1e-3), target, bs, sat));
        // Torque perpendicular to the error axis is not "opposing".
        REQUIRE_FALSE(detail::torqueOpposesError(x, Eigen::Vector3d(1e-3, 0.0, 0.0), target, bs, sat));
    }

    SECTION("boresight already on target: no error axis -> false") {
        Eigen::Vector4d target(std::nan(""), 1.0, 0.0, 0.0);
        REQUIRE_FALSE(detail::torqueOpposesError(x, Eigen::Vector3d(0.0, 0.0, 1e-3), target, bs, sat));
        REQUIRE_FALSE(detail::torqueOpposesError(x, Eigen::Vector3d(0.0, 0.0, -1e-3), target, bs, sat));
    }

    SECTION("rotated attitude: error axis follows the body frame") {
        // Attitude rotated 90 deg about +Z: body +X boresight points at ECI +Y.
        // Target at ECI -X lies +90 deg further about +Z: r_body = R(q)^T*(-x)
        // = +Y_body, so bs x r_body = +Z_body still corrects.
        Eigen::VectorXd x_rot = x;
        x_rot(3) = std::cos(M_PI / 4.0);
        x_rot(6) = std::sin(M_PI / 4.0);
        Eigen::Vector4d target(std::nan(""), -1.0, 0.0, 0.0);
        REQUIRE(detail::torqueOpposesError(x_rot, Eigen::Vector3d(0.0, 0.0, 1e-3), target, bs, sat));
        REQUIRE_FALSE(detail::torqueOpposesError(x_rot, Eigen::Vector3d(0.0, 0.0, -1e-3), target, bs, sat));
    }

    SECTION("quaternion-mode behavior is unchanged") {
        // Attitude rotated +0.3 rad about +Z relative to an identity target.
        Eigen::VectorXd x_q = x;
        x_q(3) = std::cos(0.15);
        x_q(6) = std::sin(0.15);
        const Eigen::Vector4d target_q(1.0, 0.0, 0.0, 0.0);
        const Eigen::Vector4d q_err =
            saltro::math::quatError(target_q, Eigen::Vector4d(x_q.segment<4>(3)));
        const double s = (q_err(3) > 0.0) ? 1.0 : -1.0;
        // Correcting torque is -s about Z; the anti-correcting torque is +s.
        REQUIRE(detail::torqueOpposesError(x_q, Eigen::Vector3d(0.0, 0.0, -s * 1e-3), target_q, bs, sat));
        REQUIRE_FALSE(detail::torqueOpposesError(x_q, Eigen::Vector3d(0.0, 0.0, s * 1e-3), target_q, bs, sat));
    }
}
