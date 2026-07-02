#pragma once

#include <Eigen/Dense>
#include <vector>
#include <utility>

#include <saltro/pybind/satellite.h>
#include <saltro/pybind/plannersettings.h>
#include <saltro/pybind/controller/pdcontroller.h>

namespace saltro::optimizer {

/// A spike candidate: (t_enter, t_exit) knot indices.
using SpikeCandidate = std::pair<int, int>;

/**
 * Internal building blocks of the spike-removal pipeline, exposed so the
 * C++ unit tests (tests/unit/optimizer/test_spike_removal.cpp) can pin their
 * behavior directly — mirroring the Python reference suite in
 * tests/unit/optimizer/test_spike_removal.py.  Not part of the stable API.
 */
namespace spike_removal_detail {

/// True if any dominant control channel (MTQ, RW, or Magic) is saturated
/// against the effective AL-imposed ceiling (u_max × control_limit_scale).
bool isSaturated(
	const Eigen::VectorXd& u,
	const Satellite& satellite,
	double control_limit_scale,
	double ratio_of_al_ceiling = 0.9
);

/// True if the actuator torque is driving the attitude toward the pointing
/// goal ("opposing the error").  Works for both quaternion targets and
/// vector-pointing targets ([NaN, x, y, z] NaN-sentinel), where the error
/// axis is boresight_body × r_body — the axis that rotates the boresight
/// toward the target direction.
bool torqueOpposesError(
	const Eigen::VectorXd& x,
	const Eigen::Vector3d& tau_act,
	const Eigen::Vector4d& target,
	const Eigen::Vector3d& boresight_body,
	const Satellite& satellite
);

/// Simulate a PD-controlled trajectory over n_steps knots.
/// Returns (X_pd: nx × n_steps+1, U_pd: nu × n_steps).
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> simulatePDSegment(
	const Satellite& satellite,
	const saltro::controller::PDController& pd,
	const Eigen::VectorXd& x_start,
	const Eigen::Vector4d& q_target,
	int n_steps,
	const Eigen::Ref<const Eigen::MatrixXd>& B_slice,
	const Eigen::Ref<const Eigen::MatrixXd>& S_slice,
	const Eigen::Ref<const Eigen::MatrixXd>& R_slice,
	const Eigen::Ref<const Eigen::MatrixXd>& V_slice,
	const Eigen::Ref<const Eigen::MatrixXd>& rho_slice,
	const DisturbanceConfig& dist_cfg,
	double dt,
	const SpikeRemovalConfig& cfg
);

/// True if the PD trajectory is free of sun-avoidance (keep-out) violations.
bool keepoutClear(
	const Satellite& satellite,
	const Eigen::MatrixXd& X_pd,
	const Eigen::MatrixXd& U_pd,
	const Eigen::Ref<const Eigen::MatrixXd>& S,
	const ConstraintConfig& cnst_cfg,
	int N,
	int t_enter
);

/// Cost backstop (port of the Python prototype's compare_costs): returns true
/// iff the PD-substituted window is strictly cheaper (stage cost summed over
/// [t_enter, t_exit)) than the original window.  For all-no-goal windows the
/// comparison window is extended 15% into the next goal segment using the
/// original tail trajectory on both sides.  Optionally reports both costs.
bool compareCosts(
	const Satellite& satellite,
	const Eigen::Ref<const Eigen::MatrixXd>& X_orig,
	const Eigen::Ref<const Eigen::MatrixXd>& U_orig,
	const Eigen::MatrixXd& X_pd,
	const Eigen::MatrixXd& U_pd,
	int t_enter,
	int t_exit,
	const Eigen::Ref<const Eigen::MatrixXd>& B,
	const Eigen::Ref<const Eigen::MatrixXd>& boresight,
	const Eigen::Ref<const Eigen::MatrixXd>& attitude_target,
	const CostConfig& cost_cfg,
	int N,
	double* cost_orig_out = nullptr,
	double* cost_pd_out = nullptr
);

} // namespace spike_removal_detail

/**
 * @brief Detect spike candidate windows in a trajectory.
 *
 * Scans pointing-error time history for runs of consecutive increasing error
 * that (a) follow a period of convergence, (b) have a large peak-to-entry
 * ratio, (c) eventually return to near the entry error level, and (d) are
 * not physics-limited (actuator saturated fighting correctly).
 *
 * @return List of (t_enter, t_exit) pairs.
 */
std::vector<SpikeCandidate> detectSpikes(
	const Satellite& satellite,
	const Eigen::Ref<const Eigen::MatrixXd>& X,
	const Eigen::Ref<const Eigen::MatrixXd>& U,
	const Eigen::Ref<const Eigen::MatrixXd>& attitude_target,
	const Eigen::Ref<const Eigen::MatrixXd>& boresight,
	const Eigen::Ref<const Eigen::MatrixXd>& B,
	const ConstraintConfig& cnst_cfg,
	const SpikeRemovalConfig& cfg
);

/**
 * @brief Detect and remove one trajectory spike after an accepted forward pass.
 *
 * Should be called immediately after a successful forward pass, before the
 * next backward pass.  Detects at most one spike, substitutes a PD segment,
 * blends into the tail, and re-rolls out with iLQR gain correction.
 *
 * @param X            State trajectory (nx × N), modified in place.
 * @param U            Control trajectory (nu × N), modified in place.
 * @param U_bar        Nominal controls saved BEFORE the forward pass.
 * @param K            Feedback gains from the last backward pass.
 * @param satellite    Satellite model.
 * @param settings     Planner settings.
 * @param pass_idx     Current pass index.
 * @param R,V,B,S,rho  Environment matrices.
 * @param jtime        Julian time vector.
 * @param boresight    Body-frame boresight (3 × N).
 * @param attitude_target  Goal quaternions (4 × N).
 * @param iteration    Current iLQR iteration index.
 * @param cfg          Spike removal configuration.
 *
 * @return true if a substitution occurred.
 */
bool applySpikeRemoval(
	const Satellite& satellite,
	Eigen::Ref<Eigen::MatrixXd> X,
	Eigen::Ref<Eigen::MatrixXd> U,
	const Eigen::Ref<const Eigen::MatrixXd>& U_bar,
	const std::vector<Eigen::MatrixXd>& K,
	const PlannerSettings& settings,
	int pass_idx,
	const Eigen::Ref<const Eigen::MatrixXd>& R,
	const Eigen::Ref<const Eigen::MatrixXd>& V,
	const Eigen::Ref<const Eigen::MatrixXd>& B,
	const Eigen::Ref<const Eigen::MatrixXd>& S,
	const Eigen::Ref<const Eigen::MatrixXd>& rho,
	const Eigen::Ref<const Eigen::VectorXd>& jtime,
	const Eigen::Ref<const Eigen::MatrixXd>& boresight,
	const Eigen::Ref<const Eigen::MatrixXd>& attitude_target,
	int iteration,
	const SpikeRemovalConfig& cfg
);

} // namespace saltro::optimizer
