#pragma once

#include <Eigen/Dense>
#include <vector>
#include <utility>

#include <saltro/pybind/satellite.h>
#include <saltro/pybind/plannersettings.h>

namespace saltro::optimizer {

/// A spike candidate: (t_enter, t_exit) knot indices.
using SpikeCandidate = std::pair<int, int>;

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
