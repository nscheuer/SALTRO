#include <saltro/optimizer/spike_removal.h>

#include <saltro/limits.h>
#include <saltro/math/attitude.h>
#include <saltro/math/mrp.h>
#include <saltro/math/quaternion.h>
#include <saltro/pybind/controller/pdcontroller.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <set>
#include <vector>

namespace saltro::optimizer {

namespace {

/// Pointing error at knot k (radians) — thin wrapper over saltro::math::pointingError
/// that indexes into the trajectory matrices.
double pointingError(
	const Eigen::Ref<const Eigen::MatrixXd>& X,
	const Eigen::Ref<const Eigen::MatrixXd>& attitude_target,
	const Eigen::Ref<const Eigen::MatrixXd>& boresight,
	int k
) {
	const Eigen::Vector4d target = attitude_target.col(k);
	const Eigen::Vector4d q = X.col(k).segment<4>(3);
	const Eigen::Vector3d bs = boresight.col(k).head<3>();
	return saltro::math::pointingError(q, target, bs);
}

/// Find set of knot indices where goal changes.
/// True if two attitude-target columns represent the SAME goal. Components are
/// compared element-wise, treating the NaN sentinel as equal to NaN: in
/// vector-pointing mode the target is [NaN, x, y, z], and NaN != NaN would make
/// a plain (a - b).isZero() check ALWAYS false, flagging every knot as a goal
/// transition and buffering the whole horizon out of spike detection. A goal
/// change is a change in the NaN pattern or in any finite component.
bool sameGoalTarget(const Eigen::Ref<const Eigen::VectorXd>& a,
                    const Eigen::Ref<const Eigen::VectorXd>& b) {
	for (int i = 0; i < a.size(); ++i) {
		const bool a_nan = std::isnan(a(i));
		const bool b_nan = std::isnan(b(i));
		if (a_nan != b_nan) return false;
		if (!a_nan && std::abs(a(i) - b(i)) > 1e-9) return false;
	}
	return true;
}

std::set<int> findGoalTransitions(const Eigen::Ref<const Eigen::MatrixXd>& attitude_target) {
	const int N = static_cast<int>(attitude_target.cols());
	std::set<int> transitions;
	for (int k = 1; k < N; ++k) {
		if (!sameGoalTarget(attitude_target.col(k), attitude_target.col(k - 1))) {
			transitions.insert(k);
		}
	}
	return transitions;
}

/// Check if any dominant control channel is saturated against the effective
/// AL-imposed ceiling (u_max × control_limit_scale).
///
/// The AL penalty drives |u| to at most `control_limit_scale × u_max`
/// (default 0.75).  Measuring saturation as a fraction of the hardware
/// u_max was dead code — AL never let |u| reach 0.95·u_max.  We now scale
/// by control_limit_scale so the threshold tracks whatever the user sets
/// (e.g., 0.75 or 0.9).  `ratio_of_al_ceiling` defaults to 0.9, meaning
/// "within 10% of the effective AL ceiling" ≈ saturated — gives a bit
/// of margin to catch cases that are clearly AL-pegged but not
/// numerically at the exact ceiling.
bool isSaturated(
	const Eigen::VectorXd& u,
	const Satellite& satellite,
	double control_limit_scale,
	double ratio_of_al_ceiling = 0.9
) {
	const int n_mtq = satellite.numMTQ();
	const int n_rw = satellite.numRW();
	const double thresh = ratio_of_al_ceiling * control_limit_scale;
	for (int i = 0; i < n_mtq; ++i) {
		const double u_max = satellite.getMTQ(i).u_max();
		if (u_max > 0 && std::abs(u(i)) >= thresh * u_max) return true;
	}
	for (int i = 0; i < n_rw; ++i) {
		const double u_max = satellite.getRW(i).u_max();
		if (u_max > 0 && std::abs(u(n_mtq + i)) >= thresh * u_max) return true;
	}
	return false;
}

/// Check if actuator torque opposes the pointing error.
bool torqueOpposesError(
	const Eigen::VectorXd& x,
	const Eigen::Vector3d& tau_act,
	const Eigen::Vector4d& target,
	const Satellite& satellite
) {
	if (std::isnan(target(0))) return false;

	const Eigen::Vector4d q = x.segment<4>(3);
	const Eigen::Vector4d q_err = saltro::math::quatError(target, q);
	const Eigen::Vector3d err_axis = q_err.tail<3>();
	const double err_norm = err_axis.norm();
	if (err_norm < 1e-8) return false;

	const Eigen::Vector3d n_err = err_axis / err_norm;
	const Eigen::Vector3d alpha = satellite.invInertiaNoRW() * tau_act;
	return alpha.dot(n_err) < 0.0;
}

/// Extract environment vectors at knot k.
struct Env {
	Eigen::Vector3d R, B, S, V;
	int rho;
};

Env getEnv(
	const Eigen::Ref<const Eigen::MatrixXd>& R,
	const Eigen::Ref<const Eigen::MatrixXd>& B,
	const Eigen::Ref<const Eigen::MatrixXd>& S,
	const Eigen::Ref<const Eigen::MatrixXd>& V,
	const Eigen::Ref<const Eigen::MatrixXd>& rho,
	int k
) {
	const int N = static_cast<int>(R.cols());
	const int kc = std::min(k, N - 1);
	return {
		Eigen::Vector3d(R.col(kc)),
		Eigen::Vector3d(B.col(kc)),
		Eigen::Vector3d(S.col(kc)),
		Eigen::Vector3d(V.col(kc)),
		static_cast<int>(std::max(0.0, std::round(rho(0, kc)))),
	};
}

// ---------------------------------------------------------------------------
// PD segment simulation
// ---------------------------------------------------------------------------

/// Simulate PD-controlled trajectory over n_steps knots.
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
) {
	const int nx = satellite.stateDim();
	const int nu = satellite.controlDim();

	Eigen::MatrixXd X_pd = Eigen::MatrixXd::Zero(nx, n_steps + 1);
	Eigen::MatrixXd U_pd = Eigen::MatrixXd::Zero(nu, n_steps);
	X_pd.col(0) = x_start;

	for (int k = 0; k < n_steps; ++k) {
		const Eigen::VectorXd& x_k = X_pd.col(k);
		const Eigen::Vector3d B_k(B_slice.col(k));
		const Eigen::Vector3d S_k(S_slice.col(k));
		const Eigen::Vector3d R_k(R_slice.col(k));
		const Eigen::Vector3d V_k(V_slice.col(k));
		const int rho_k = static_cast<int>(std::max(0.0, std::round(rho_slice(0, k))));

		// PDController handles allocation, authority weighting, and scale-to-max
		// saturation internally.  boresight is unused by PD but required by
		// the Controller interface.
		Eigen::VectorXd u_k = pd.find_u(x_k, B_k, q_target, Eigen::Vector3d::Zero());
		U_pd.col(k) = u_k;

		Eigen::VectorXd x_next = satellite.dynamicsStepRK4(x_k, u_k, dt, dist_cfg, R_k, B_k, S_k, V_k, rho_k);

		// Clamp angular velocity
		if (cfg.omega_max > 0.0) {
			const double omega_norm = x_next.head<3>().norm();
			if (omega_norm > cfg.omega_max) {
				x_next.head<3>() *= cfg.omega_max / omega_norm;
			}
		}

		X_pd.col(k + 1) = x_next;
	}

	return {X_pd, U_pd};
}

// ---------------------------------------------------------------------------
// Keep-out check
// ---------------------------------------------------------------------------

bool keepoutClear(
	const Satellite& satellite,
	const Eigen::MatrixXd& X_pd,
	const Eigen::MatrixXd& U_pd,
	const Eigen::Ref<const Eigen::MatrixXd>& S,
	const ConstraintConfig& cnst_cfg,
	int N,
	int t_enter
) {
	const int n_pd = static_cast<int>(X_pd.cols());
	const int nu = satellite.controlDim();

	for (int i = 0; i < n_pd; ++i) {
		const int k = t_enter + i;
		const Eigen::VectorXd u_k = (i < U_pd.cols()) ? Eigen::VectorXd(U_pd.col(i)) : Eigen::VectorXd::Zero(nu);
		const Eigen::VectorXd S_k = (k < S.cols()) ? Eigen::VectorXd(S.col(k)) : Eigen::VectorXd(S.col(S.cols() - 1));

		const Eigen::VectorXd c_k = satellite.constraints(k, N, X_pd.col(i), u_k, S_k, cnst_cfg);
		// Index 1 = sun avoidance
		if (c_k.size() > 1 && c_k(1) > 0.0) {
			return false;
		}
	}
	return true;
}

// ---------------------------------------------------------------------------
// Substitute + blend + tail re-rollout
// ---------------------------------------------------------------------------

/// Scale-to-max-saturation clamp.  Uniform scaling preserves torque direction.
void scaleToMax(Eigen::VectorXd& u, const Satellite& satellite) {
	const int n_mtq = satellite.numMTQ();
	const int n_rw = satellite.numRW();
	double max_ratio = 1.0;
	for (int i = 0; i < n_mtq; ++i) {
		const double u_max = std::abs(satellite.getMTQ(i).u_max());
		if (u_max > 0.0) {
			const double r = std::abs(u(i)) / u_max;
			if (r > max_ratio) max_ratio = r;
		}
	}
	for (int i = 0; i < n_rw; ++i) {
		const double u_max = std::abs(satellite.getRW(i).u_max());
		if (u_max > 0.0) {
			const double r = std::abs(u(n_mtq + i)) / u_max;
			if (r > max_ratio) max_ratio = r;
		}
	}
	if (max_ratio > 1.0) {
		u /= max_ratio;
	}
}

void substituteAndBlend(
	const Satellite& satellite,
	const saltro::controller::PDController& pd,
	Eigen::Ref<Eigen::MatrixXd> X,
	Eigen::Ref<Eigen::MatrixXd> U,
	const Eigen::MatrixXd& X_pd,
	const Eigen::MatrixXd& U_pd,
	int t_enter, int t_exit, int blend_len,
	const Eigen::MatrixXd& X_nominal_pre,
	const Eigen::Ref<const Eigen::MatrixXd>& U_bar,
	const std::vector<Eigen::MatrixXd>& K,
	const DisturbanceConfig& dist_cfg,
	const Eigen::Ref<const Eigen::MatrixXd>& R,
	const Eigen::Ref<const Eigen::MatrixXd>& B,
	const Eigen::Ref<const Eigen::MatrixXd>& S,
	const Eigen::Ref<const Eigen::MatrixXd>& V,
	const Eigen::Ref<const Eigen::MatrixXd>& rho,
	double dt,
	const SpikeRemovalConfig& cfg
) {
	(void)cfg;
	const int N = static_cast<int>(X.cols());
	const int nu = static_cast<int>(U.rows());
	const int n_pd = t_exit - t_enter;

	// --- Substitution region [t_enter, t_exit) ---
	X.block(0, t_enter, X.rows(), n_pd) = X_pd.leftCols(n_pd);
	U.block(0, t_enter, nu, n_pd) = U_pd.leftCols(n_pd);

	// Anchor t_exit from PD trajectory
	if (X_pd.cols() > n_pd) {
		X.col(t_exit) = X_pd.col(n_pd);
	}

	// Target quaternion for blend PD
	const Eigen::Vector4d q_exit_target = X_nominal_pre.col(t_exit).segment<4>(3);

	const int blend_end = std::min(t_exit + blend_len, N - 1);

	// --- Blend zone [t_exit, blend_end) ---
	for (int k = t_exit; k < blend_end; ++k) {
		const double lam = static_cast<double>(k - t_exit) / static_cast<double>(blend_len);
		const auto env = getEnv(R, B, S, V, rho, k);

		// PD contribution (already scale-to-max clamped inside PDController)
		Eigen::VectorXd u_pd_k = pd.find_u(X.col(k), env.B, q_exit_target, Eigen::Vector3d::Zero());
		// iLQR open-loop nominal
		const Eigen::VectorXd u_ilqr_k = (k < U_bar.cols())
			? Eigen::VectorXd(U_bar.col(k))
			: Eigen::VectorXd::Zero(nu);

		Eigen::VectorXd u_blend = (1.0 - lam) * u_pd_k + lam * u_ilqr_k;
		scaleToMax(u_blend, satellite);

		U.col(k) = u_blend;
		if (k + 1 < N) {
			X.col(k + 1) = satellite.dynamicsStepRK4(X.col(k), u_blend, dt, dist_cfg,
			                                         env.R, env.B, env.S, env.V, env.rho);
		}
	}

	// Tail re-rollout [blend_end, N-1) — OPEN-LOOP propagation with U_bar.
	// Must integrate to produce a feasible trajectory (X[k+1] = f(X[k], U[k])),
	// otherwise the next backward_pass linearizes around invalid states.
	// Applying feedback K·dx from the pre-substitution gains to the large dx
	// at the blend boundary blows up (state past RK4 stability), so we use
	// pure open-loop U_bar.  The safety valve catches divergence.
	(void)K;
	for (int k = blend_end; k < N - 1; ++k) {
		const auto env = getEnv(R, B, S, V, rho, k);
		const Eigen::VectorXd u_bar_k = (k < U_bar.cols())
			? Eigen::VectorXd(U_bar.col(k))
			: Eigen::VectorXd::Zero(nu);
		U.col(k) = u_bar_k;
		if (k + 1 < N) {
			X.col(k + 1) = satellite.dynamicsStepRK4(
				X.col(k), u_bar_k, dt, dist_cfg,
				env.R, env.B, env.S, env.V, env.rho);
		}
	}
}

} // anonymous namespace

// ===========================================================================
// Public API
// ===========================================================================

std::vector<SpikeCandidate> detectSpikes(
	const Satellite& satellite,
	const Eigen::Ref<const Eigen::MatrixXd>& X,
	const Eigen::Ref<const Eigen::MatrixXd>& U,
	const Eigen::Ref<const Eigen::MatrixXd>& attitude_target,
	const Eigen::Ref<const Eigen::MatrixXd>& boresight,
	const Eigen::Ref<const Eigen::MatrixXd>& B,
	const ConstraintConfig& cnst_cfg,
	const SpikeRemovalConfig& cfg
) {
	(void)cnst_cfg; // reserved for future constraint-based filtering

	const int N = static_cast<int>(X.cols());
	const auto transitions = findGoalTransitions(attitude_target);

	// Buffer zone
	std::set<int> buffered;
	for (int t : transitions) {
		for (int dk = -cfg.goal_switch_buffer; dk <= cfg.goal_switch_buffer; ++dk) {
			buffered.insert(t + dk);
		}
	}

	// Compute pointing error
	std::vector<double> theta(N, std::numeric_limits<double>::quiet_NaN());
	for (int k = 0; k < N; ++k) {
		if (buffered.count(k) == 0) {
			theta[k] = pointingError(X, attitude_target, boresight, k);
		}
	}

	std::vector<SpikeCandidate> candidates;
	int k = 0;

	while (k < N - 1) {
		if (std::isnan(theta[k])) { ++k; continue; }

		// Count consecutive increasing-error knots
		int run_start = k;
		int run_len = 0;
		int j = k;
		while (j + 1 < N && !std::isnan(theta[j + 1]) && theta[j + 1] > theta[j]) {
			++run_len;
			++j;
		}

		if (run_len < cfg.min_consecutive) { ++k; continue; }

		const int t_enter = run_start;
		const double entry_error = theta[t_enter];

		// Prior-decrease filter
		int prior_decrease = 0;
		int pk = t_enter - 1;
		while (pk > 0 && std::isnan(theta[pk])) --pk;
		while (pk > 0 && !std::isnan(theta[pk]) && !std::isnan(theta[pk - 1])) {
			if (theta[pk] < theta[pk - 1]) {
				++prior_decrease;
				--pk;
			} else {
				break;
			}
		}
		if (prior_decrease < cfg.min_prior_decrease_knots) {
			k = j + 1;
			continue;
		}

		// Spike-magnitude filter
		const double peak = *std::max_element(theta.begin() + run_start, theta.begin() + j + 1);
		if (peak < entry_error * cfg.min_spike_ratio) { ++k; continue; }

		// Find exit
		int t_exit = -1;
		for (int m = j + 1; m < N; ++m) {
			if (std::isnan(theta[m])) break;
			if (theta[m] <= entry_error * cfg.exit_fudge) {
				t_exit = m;
				break;
			}
		}
		if (t_exit < 0) { k = j + 1; continue; }

		// Max spike window filter
		if (cfg.max_spike_knots > 0 && (t_exit - t_enter) > cfg.max_spike_knots) {
			k = j + 1;
			continue;
		}

		// Actuation-driven filter
		const int mid = (t_enter + t_exit) / 2;
		const int check_knots[] = {
			t_enter,
			mid,
			std::min(mid + (t_exit - t_enter) / 4, t_exit - 1),
		};
		int physics_limited_votes = 0;
		int n_checked = 0;
		for (int ck : check_knots) {
			if (ck < 0 || ck >= U.cols()) continue;
			++n_checked;
			const Eigen::VectorXd u_k = U.col(ck);
			const Eigen::Vector3d tau = satellite.actuatorTorque(X.col(ck), u_k, B.col(ck));
			if (isSaturated(u_k, satellite, cnst_cfg.control_limit_scale) &&
			    torqueOpposesError(X.col(ck), tau, attitude_target.col(ck), satellite)) {
				++physics_limited_votes;
			}
		}
		if (physics_limited_votes >= std::max(1, n_checked / 2 + 1)) {
			k = j + 1;
			continue;
		}

		candidates.emplace_back(SpikeCandidate{t_enter, t_exit});
		k = t_exit + 1;
	}

	// =====================================================================
	// Second detection pass: local PE outliers.
	// Detect regions where the PE is significantly above the surrounding
	// baseline.  This catches spikes that don't have the classical
	// "monotonically decreasing → sudden increase" pattern — e.g. when
	// the PE oscillates at ~10° between spikes.
	//
	// For each knot k, compute the local minimum PE in a window around it.
	// If theta[k] > local_min * min_spike_ratio, it's a spike candidate.
	// =====================================================================
	const int window_half = std::max(cfg.min_consecutive, 5);
	for (int kk = window_half; kk < N - window_half; ++kk) {
		if (std::isnan(theta[kk])) continue;
		if (buffered.count(kk) != 0) continue;

		// Find minimum PE in the windows before and after this knot
		double min_before = std::numeric_limits<double>::max();
		for (int j = kk - window_half; j < kk; ++j) {
			if (!std::isnan(theta[j]) && theta[j] < min_before) min_before = theta[j];
		}
		double min_after = std::numeric_limits<double>::max();
		for (int j = kk + 1; j <= kk + window_half && j < N; ++j) {
			if (!std::isnan(theta[j]) && theta[j] < min_after) min_after = theta[j];
		}

		const double local_baseline = std::min(min_before, min_after);
		if (local_baseline >= std::numeric_limits<double>::max()) continue;

		// Is this knot significantly above the local baseline?
		if (theta[kk] < local_baseline * cfg.min_spike_ratio) continue;
		// Require absolute minimum spike height — only flag genuine SO(3) wraps,
		// not mild oscillations in an already-converging trajectory.
		// pi/3 ≈ 60° ensures we only catch large deviations.
		if (theta[kk] < M_PI / 3.0) continue;
		// Also require the local baseline to be reasonably low — if the whole
		// trajectory is at 80°+ PE, there's no "spike" to remove.
		if (local_baseline > M_PI / 4.0) continue;  // ~45° baseline max

		// Expand the spike window: walk outward until PE drops below
		// baseline * exit_fudge
		const double exit_thresh = local_baseline * cfg.exit_fudge;
		int t_enter_s = kk;
		while (t_enter_s > 0 && !std::isnan(theta[t_enter_s - 1]) && theta[t_enter_s - 1] > exit_thresh) {
			--t_enter_s;
		}
		int t_exit_s = kk;
		while (t_exit_s < N - 1 && !std::isnan(theta[t_exit_s + 1]) && theta[t_exit_s + 1] > exit_thresh) {
			++t_exit_s;
		}
		t_exit_s = std::min(t_exit_s + 1, N - 1);  // exit is exclusive

		// Max spike window filter
		if (cfg.max_spike_knots > 0 && (t_exit_s - t_enter_s) > cfg.max_spike_knots) {
			continue;
		}

		// Control-effort filter: verify the spike has significant control effort.
		// Genuine homotopy spikes have the optimizer actively commanding torque
		// to drive the wrap. If controls are small, it's not a spike — it's just
		// the trajectory passing through a high-error region naturally.
		bool has_control_effort = false;
		for (int ck = t_enter_s; ck < t_exit_s && ck < U.cols(); ++ck) {
			// Pass 2 control-effort heuristic: "is there meaningful command at this knot?"
			// Use hardware u_max (scale=1.0) + 50% threshold — this is about detecting
			// any significant effort, not proximity to the AL ceiling.
			if (isSaturated(U.col(ck), satellite, 1.0, 0.5)) {
				has_control_effort = true;
				break;
			}
		}
		if (!has_control_effort) continue;

		// Check this doesn't overlap with an existing candidate
		bool overlaps = false;
		for (const auto& c : candidates) {
			if (t_enter_s < c.second && t_exit_s > c.first) {
				overlaps = true;
				break;
			}
		}
		if (overlaps) continue;

		candidates.emplace_back(SpikeCandidate{t_enter_s, t_exit_s});
		kk = t_exit_s;  // skip past this spike
	}

	// =====================================================================
	// Third detection pass: hemisphere flip detection.
	// The fundamental signature of a winding spike: the trajectory crosses
	// to the opposite hemisphere of S³ relative to itself.
	// s_k = sign(q_k · q_{k-1}).  A sign change in s indicates the
	// quaternion crossed through q·q_prev = 0 (a 180° rotation in one step).
	//
	// Filters:
	// 1. Local context is well-converged: PE at k±m is below a threshold
	// 2. Symmetric: PE at k-m and k+m are both lower than at k
	// 3. Deadband: neighbors must be solidly on one hemisphere (|q·q_prev| > min)
	// =====================================================================
	{
		const double converge_thresh = M_PI / 4.0;  // 45° — context must be below this
		const double deadband = 0.3;                 // min |q·q_next| for solid hemisphere
		const int max_search = N / 2;                // max distance to search for context

		for (int kk = 1; kk < N - 1; ++kk) {
			if (std::isnan(theta[kk])) continue;
			if (buffered.count(kk) != 0) continue;
			if (theta[kk] < M_PI / 6.0) continue;  // 30° minimum PE to consider

			// Search LEFT for the nearest well-converged knot
			int left_idx = -1;
			for (int j = kk - 1; j >= 0 && (kk - j) < max_search; --j) {
				if (std::isnan(theta[j])) continue;
				if (theta[j] < converge_thresh) {
					left_idx = j;
					break;
				}
			}
			if (left_idx < 0) continue;

			// Search RIGHT for the nearest well-converged knot
			int right_idx = -1;
			for (int j = kk + 1; j < N && (j - kk) < max_search; ++j) {
				if (std::isnan(theta[j])) continue;
				if (theta[j] < converge_thresh) {
					right_idx = j;
					break;
				}
			}
			if (right_idx < 0) continue;

			// Check hemisphere flip: left and right context on opposite hemispheres
			const Eigen::Vector4d q_left = X.col(left_idx).segment<4>(3);
			const Eigen::Vector4d q_right = X.col(right_idx).segment<4>(3);
			if (q_left.dot(q_right) > -deadband) continue;

			// Check neighbors are solidly on their hemisphere (no noise)
			if (left_idx > 0) {
				if (std::abs(X.col(left_idx).segment<4>(3).dot(
					X.col(left_idx - 1).segment<4>(3))) < deadband) continue;
			}
			if (right_idx < N - 1) {
				if (std::abs(X.col(right_idx).segment<4>(3).dot(
					X.col(right_idx + 1).segment<4>(3))) < deadband) continue;
			}

			// Filter 2: spike is symmetric — PE at center higher than context
			if (theta[kk] <= theta[left_idx] || theta[kk] <= theta[right_idx]) continue;

			// Find spike window using context PE as threshold
			const double exit_pe = std::max(theta[left_idx], theta[right_idx]);
			int t_enter_h = kk;
			while (t_enter_h > left_idx && !std::isnan(theta[t_enter_h - 1]) &&
			       theta[t_enter_h - 1] > exit_pe) {
				--t_enter_h;
			}
			int t_exit_h = kk;
			while (t_exit_h < right_idx && !std::isnan(theta[t_exit_h + 1]) &&
			       theta[t_exit_h + 1] > exit_pe) {
				++t_exit_h;
			}
			t_exit_h = std::min(t_exit_h + 1, N - 1);

			// Max spike window filter
			if (cfg.max_spike_knots > 0 && (t_exit_h - t_enter_h) > cfg.max_spike_knots) {
				continue;
			}

			// Check no overlap
			bool overlaps = false;
			for (const auto& c : candidates) {
				if (t_enter_h < c.second && t_exit_h > c.first) {
					overlaps = true;
					break;
				}
			}
			if (overlaps) continue;

			candidates.emplace_back(SpikeCandidate{t_enter_h, t_exit_h});
			kk = t_exit_h;
		}
	}

	// Sort by t_enter
	std::sort(candidates.begin(), candidates.end(),
	          [](const SpikeCandidate& a, const SpikeCandidate& b) {
	              return a.first < b.first;
	          });

	return candidates;
}

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
) {
	if (!cfg.enabled) return false;

	if (iteration < cfg.start_at_iter) {
		if (cfg.verbose) {
			std::cout << "[SpikeRemoval] iter=" << iteration
			          << ": skipping (before start_at_iter=" << cfg.start_at_iter << ")\n";
		}
		return false;
	}
	if (iteration >= cfg.start_at_iter + cfg.max_intervention_iters) {
		if (cfg.verbose) {
			std::cout << "[SpikeRemoval] iter=" << iteration
			          << ": skipping (past max_intervention_iters)\n";
		}
		return false;
	}

	const auto& pass = settings.passes[pass_idx];
	const auto& cnst_cfg = settings.constraints;
	const auto& dist_cfg = settings.disturbances;
	const int N = static_cast<int>(X.cols());

	// Compute dt
	double dt = pass.dt;
	if (jtime.size() >= 2) {
		dt = (jtime(1) - jtime(0)) * 36525.0 * 86400.0;
	}

	// Compute current mean PE
	double mean_pe = 0.0;
	int pe_count = 0;
	for (int k = 0; k < N; ++k) {
		const double pe = pointingError(X, attitude_target, boresight, k);
		if (std::isfinite(pe)) {
			mean_pe += pe;
			++pe_count;
		}
	}
	mean_pe = (pe_count > 0) ? mean_pe / pe_count : 0.0;

	// Guard: don't intervene if mean PE is low (< 5°) — the trajectory
	// is well-converged and spike removal would likely degrade it.
	if (mean_pe < 5.0 * M_PI / 180.0) {
		if (cfg.verbose) {
			std::cout << "[SpikeRemoval] iter=" << iteration
			          << ": skipping (mean PE " << (mean_pe * 180.0 / M_PI)
			          << "° is low)\n";
		}
		return false;
	}

	// Build the PD controller once per call.  Config overrides the auto-tuned
	// default gains; spike removal has been tuned with more aggressive values
	// than a nominal-response PD would use.
	saltro::controller::PDController pd(satellite);
	pd.setGains(cfg.kp_q, cfg.kd_w);

	// Auto-set rw_scale from satellite topology when user passes sentinel (-1).
	// Fraction-of-actuators: rw_scale = numRW / (numMTQ + numRW).
	//   3+0 MTQ-only  → 0.0   (no RW to use)
	//   0+3 RW-only   → 1.0   (must use RW; otherwise PD produces zero torque)
	//   3+1 hybrid    → 0.25  (MTQ-dominant, RW assists)
	//   3+3 hybrid    → 0.5   (balanced)
	double effective_rw_scale = cfg.rw_scale;
	if (effective_rw_scale < 0.0) {
		const int n_mtq_sat = satellite.numMTQ();
		const int n_rw_sat = satellite.numRW();
		const int tot = n_mtq_sat + n_rw_sat;
		effective_rw_scale = (tot > 0) ? (static_cast<double>(n_rw_sat) / tot) : 0.0;
		if (cfg.verbose) {
			std::cout << "[SpikeRemoval] iter=" << iteration
			          << ": auto rw_scale=" << effective_rw_scale
			          << " (nMTQ=" << n_mtq_sat << ", nRW=" << n_rw_sat << ")\n";
		}
	}
	pd.setRWScale(effective_rw_scale);

	// Detect candidates
	const auto candidates = detectSpikes(satellite, X, U, attitude_target, boresight, B, cnst_cfg, cfg);

	if (cfg.verbose && candidates.empty()) {
		std::cout << "[SpikeRemoval] iter=" << iteration << ": no candidates\n";
	}
	if (cfg.verbose && !candidates.empty()) {
		std::cout << "[SpikeRemoval] iter=" << iteration << ": "
		          << candidates.size() << " candidate(s)\n";
	}

	if (candidates.empty()) return false;

	for (const auto& [t_enter, t_exit] : candidates) {
		const int n_steps = t_exit - t_enter;
		if (n_steps < 2) continue;

		// Slice environment for PD sim
		const int t_end_pd = std::min(t_exit + 1, N);
		const Eigen::MatrixXd B_slice = B.block(0, t_enter, 3, t_end_pd - t_enter);
		const Eigen::MatrixXd S_slice = S.block(0, t_enter, S.rows(), t_end_pd - t_enter);
		const Eigen::MatrixXd R_slice = R.block(0, t_enter, 3, t_end_pd - t_enter);
		const Eigen::MatrixXd V_slice = V.block(0, t_enter, 3, t_end_pd - t_enter);
		const Eigen::MatrixXd rho_slice = rho.block(0, t_enter, rho.rows(), t_end_pd - t_enter);

		// PD target: spike exit state
		const Eigen::Vector4d q_target = X.col(t_exit).segment<4>(3);

		auto [X_pd, U_pd] = simulatePDSegment(
			satellite, pd, X.col(t_enter), q_target, n_steps,
			B_slice, S_slice, R_slice, V_slice, rho_slice,
			dist_cfg, dt, cfg
		);

		// Validate PD sim output.  At large dt the PD rollout can diverge
		// (RK4 stiffness), producing NaN/Inf quaternions that crash
		// keepoutClear / satellite.constraints downstream.
		bool pd_bad = !X_pd.allFinite() || !U_pd.allFinite();
		if (!pd_bad) {
			for (int k = 0; k < X_pd.cols(); ++k) {
				const double qn = X_pd.col(k).segment<4>(3).norm();
				if (!std::isfinite(qn) || std::abs(qn - 1.0) > 1e-3) {
					pd_bad = true; break;
				}
			}
		}
		if (pd_bad) {
			if (cfg.verbose) {
				std::cout << "[SpikeRemoval]   (" << t_enter << "," << t_exit
				          << "): PD sim diverged -- skipping\n";
			}
			continue;
		}

		// Keep-out check
		if (!keepoutClear(satellite, X_pd, U_pd, S, cnst_cfg, N, t_enter)) {
			if (cfg.verbose) {
				std::cout << "[SpikeRemoval]   (" << t_enter << "," << t_exit
				          << "): keep-out violation -- skipping\n";
			}
			continue;
		}

		if (cfg.verbose) {
			std::cout << "[SpikeRemoval]   (" << t_enter << "," << t_exit
			          << "): substituting PD segment\n";
		}

		// Save pre-substitution state for safety-valve rollback.
		const Eigen::MatrixXd X_nominal_pre = X;
		const Eigen::MatrixXd X_saved = X;
		const Eigen::MatrixXd U_saved = U;

		substituteAndBlend(
			satellite, pd, X, U, X_pd, U_pd,
			t_enter, t_exit, cfg.blend_len,
			X_nominal_pre, U_bar, K,
			dist_cfg, R, B, S, V, rho, dt, cfg
		);

		// Safety valve: reject if substitution produced a degenerate
		// trajectory.  Degenerate states crash the next backward_pass on
		// "Quaternion norm is too small to normalize".
		constexpr double kWmaxSafe = 2.0 * 20.0 * M_PI / 180.0;  // 40°/s
		bool degenerate = !X.allFinite() || !U.allFinite();
		if (!degenerate) {
			for (int k = 0; k < N; ++k) {
				const double qn = X.col(k).segment<4>(3).norm();
				if (!std::isfinite(qn) || std::abs(qn - 1.0) > 1e-3) {
					degenerate = true; break;
				}
				if (X.col(k).head<3>().cwiseAbs().maxCoeff() > kWmaxSafe) {
					degenerate = true; break;
				}
			}
		}
		if (degenerate) {
			if (cfg.verbose) {
				std::cout << "[SpikeRemoval]   (" << t_enter << "," << t_exit
				          << "): safety valve -- reverting substitution\n";
			}
			X = X_saved;
			U = U_saved;
			continue;
		}

		// Only substitute one spike per call
		return true;
	}

	return false;
}

} // namespace saltro::optimizer
