#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <vector>

#include <Eigen/Dense>

#include <saltro/optimizer/spike_removal.h>
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
