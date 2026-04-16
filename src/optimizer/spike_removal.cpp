#include <saltro/optimizer/spike_removal.h>

#include <saltro/math/integrators/rk4.h>
#include <saltro/math/mrp.h>
#include <saltro/math/quaternion.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <set>
#include <vector>

namespace saltro::optimizer {

// ---------------------------------------------------------------------------
// Quaternion helpers
// ---------------------------------------------------------------------------

namespace {

Eigen::Vector4d quatConj(const Eigen::Vector4d& q) {
	return {q(0), -q(1), -q(2), -q(3)};
}

Eigen::Vector4d quatMult(const Eigen::Vector4d& q1, const Eigen::Vector4d& q2) {
	const double w1 = q1(0), x1 = q1(1), y1 = q1(2), z1 = q1(3);
	const double w2 = q2(0), x2 = q2(1), y2 = q2(2), z2 = q2(3);
	return {
		w1*w2 - x1*x2 - y1*y2 - z1*z2,
		w1*x2 + x1*w2 + y1*z2 - z1*y2,
		w1*y2 - x1*z2 + y1*w2 + z1*x2,
		w1*z2 + x1*y2 - y1*x2 + z1*w2,
	};
}

Eigen::Vector4d quatErrorShortest(const Eigen::Vector4d& q_target,
                                  const Eigen::Vector4d& q_current) {
	Eigen::Vector4d q_err = quatMult(quatConj(q_target), q_current);
	if (q_err(0) < 0.0) q_err = -q_err;
	return q_err;
}

double quatAngle(const Eigen::Vector4d& q_err) {
	const double w = std::clamp(std::abs(q_err(0)), 0.0, 1.0);
	return 2.0 * std::acos(w);
}

/// Pointing error at knot k (radians).
double pointingError(
	const Eigen::Ref<const Eigen::MatrixXd>& X,
	const Eigen::Ref<const Eigen::MatrixXd>& attitude_target,
	const Eigen::Ref<const Eigen::MatrixXd>& boresight,
	int k
) {
	const Eigen::Vector4d target = attitude_target.col(k);
	const Eigen::Vector4d q = X.col(k).segment<4>(3);

	if (std::isnan(target(0))) {
		// Vector-pointing mode
		const Eigen::Vector3d target_vec = target.tail<3>().normalized();
		const Eigen::Matrix3d C = saltro::math::rotationMatrix(q);
		const Eigen::Vector3d boresight_eci = (C * boresight.col(k).head<3>()).normalized();
		return std::acos(std::clamp(boresight_eci.dot(target_vec), -1.0, 1.0));
	}

	// Quaternion mode
	return quatAngle(quatErrorShortest(target, q));
}

/// Find set of knot indices where goal changes.
std::set<int> findGoalTransitions(const Eigen::Ref<const Eigen::MatrixXd>& attitude_target) {
	const int N = static_cast<int>(attitude_target.cols());
	std::set<int> transitions;
	for (int k = 1; k < N; ++k) {
		if (!(attitude_target.col(k) - attitude_target.col(k - 1)).isZero(1e-9)) {
			transitions.insert(k);
		}
	}
	return transitions;
}

/// Check if any dominant control channel is saturated.
bool isSaturated(
	const Eigen::VectorXd& u,
	const Satellite& satellite,
	double threshold = 0.95
) {
	const int n_mtq = satellite.numMTQ();
	const int n_rw = satellite.numRW();
	for (int i = 0; i < n_mtq; ++i) {
		const double u_max = satellite.getMTQ(i).u_max();
		if (u_max > 0 && std::abs(u(i)) >= threshold * u_max) return true;
	}
	for (int i = 0; i < n_rw; ++i) {
		const double u_max = satellite.getRW(i).u_max();
		if (u_max > 0 && std::abs(u(n_mtq + i)) >= threshold * u_max) return true;
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
	const Eigen::Vector4d q_err = quatErrorShortest(target, q);
	const Eigen::Vector3d err_axis = q_err.tail<3>();
	const double err_norm = err_axis.norm();
	if (err_norm < 1e-8) return false;

	const Eigen::Vector3d n_err = err_axis / err_norm;
	const Eigen::Vector3d alpha = satellite.invInertiaNoRW() * tau_act;
	return alpha.dot(n_err) < 0.0;
}

// ---------------------------------------------------------------------------
// PD controller
// ---------------------------------------------------------------------------

Eigen::VectorXd buildPDControl(
	const Eigen::VectorXd& x,
	const Eigen::Vector4d& q_target,
	const Satellite& satellite,
	const Eigen::Vector3d& B_eci,
	double kp_q,
	double kd_w,
	double rw_scale
) {
	const Eigen::Vector4d q = x.segment<4>(3);
	const Eigen::Vector3d omega = x.head<3>();

	// Quaternion error
	Eigen::Vector4d q_err = quatErrorShortest(q_target, q);

	// Desired torque (body frame)
	const Eigen::Vector3d tau_des = -kp_q * q_err.tail<3>() - kd_w * omega;

	const int n_mtq = satellite.numMTQ();
	const int n_rw = satellite.numRW();
	const int nu = n_mtq + n_rw;
	Eigen::VectorXd u = Eigen::VectorXd::Zero(nu);

	// MTQ contribution
	const Eigen::Matrix3d C = saltro::math::rotationMatrix(q);
	const Eigen::Vector3d B_body = C.transpose() * B_eci;
	const double B_norm_sq = B_body.squaredNorm();
	Eigen::Vector3d tau_mtq_total = Eigen::Vector3d::Zero();

	if (B_norm_sq > 1e-20 && n_mtq > 0) {
		Eigen::MatrixXd A(3, n_mtq);
		for (int i = 0; i < n_mtq; ++i) {
			A.col(i) = satellite.getMTQ(i).axis().cross(B_body);
		}

		// Least-squares dipole: m = (A^T A)^{-1} A^T tau_des
		const Eigen::VectorXd m_raw =
			(A.transpose() * A).ldlt().solve(A.transpose() * tau_des);

		for (int i = 0; i < n_mtq; ++i) {
			const double u_max = satellite.getMTQ(i).u_max();
			u(i) = std::clamp(m_raw(i), -u_max, u_max);
			tau_mtq_total += u(i) * satellite.getMTQ(i).axis().cross(B_body);
		}
	}

	// RW contribution (scaled remainder)
	if (rw_scale > 0.0 && n_rw > 0) {
		const Eigen::Vector3d tau_remaining = tau_des - tau_mtq_total;
		for (int i = 0; i < n_rw; ++i) {
			const Eigen::Vector3d rw_axis = satellite.getRW(i).axis();
			const double u_max = satellite.getRW(i).u_max();
			const double tau_rw_i = tau_remaining.dot(rw_axis) * rw_scale;
			u(n_mtq + i) = std::clamp(tau_rw_i, -u_max, u_max);
		}
	}

	return u;
}

// ---------------------------------------------------------------------------
// RK4 step helper (matches forwardpass.cpp pattern)
// ---------------------------------------------------------------------------

Eigen::VectorXd rk4Step(
	const Satellite& satellite,
	const Eigen::VectorXd& x,
	const Eigen::VectorXd& u,
	double dt,
	const DisturbanceConfig& dist_cfg,
	const Eigen::Vector3d& R_k,
	const Eigen::Vector3d& B_k,
	const Eigen::Vector3d& S_k,
	const Eigen::Vector3d& V_k,
	int rho_k
) {
	Eigen::VectorXd x_next;
	rk4_step<Eigen::VectorXd>(
		[&](double, const Eigen::VectorXd& x_state, Eigen::VectorXd& dxdt) {
			dxdt = satellite.dynamics(x_state, u, dist_cfg, R_k, B_k, S_k, V_k, rho_k);
		},
		x, 0.0, dt, x_next
	);
	// Normalize quaternion
	Eigen::Vector4d q = x_next.segment<4>(3);
	const double qn = q.norm();
	if (qn > 1e-10) x_next.segment<4>(3) = q / qn;
	return x_next;
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

/// Reduced state error matching iLQR convention: [dw(3), mrp(3), dh(nRW)].
Eigen::VectorXd stateErrorReduced(
	const Eigen::VectorXd& x_current,
	const Eigen::VectorXd& x_nominal,
	const Satellite& satellite
) {
	const int nRW = satellite.numRW();
	const int nxr = satellite.reducedStateDim();
	Eigen::VectorXd dx = Eigen::VectorXd::Zero(nxr);

	dx.head<3>() = x_current.head<3>() - x_nominal.head<3>();

	const Eigen::Vector4d q_ref = x_nominal.segment<4>(3);
	const Eigen::Vector4d q_cur = x_current.segment<4>(3);
	const Eigen::Vector4d q_err = saltro::math::quatError(q_ref, q_cur);
	dx.segment<3>(3) = saltro::math::quatToMRP(q_err);

	for (int i = 0; i < nRW; ++i) {
		dx(6 + i) = x_current(7 + i) - x_nominal(7 + i);
	}
	return dx;
}

// ---------------------------------------------------------------------------
// PD segment simulation
// ---------------------------------------------------------------------------

/// Simulate PD-controlled trajectory over n_steps knots.
/// Returns (X_pd: nx × n_steps+1, U_pd: nu × n_steps).
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> simulatePDSegment(
	const Satellite& satellite,
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

		Eigen::VectorXd u_k = buildPDControl(x_k, q_target, satellite, B_k,
		                                     cfg.kp_q, cfg.kd_w, cfg.rw_scale);

		// Clamp to actuator limits
		const int n_mtq = satellite.numMTQ();
		const int n_rw = satellite.numRW();
		for (int i = 0; i < n_mtq; ++i) {
			u_k(i) = std::clamp(u_k(i), -satellite.getMTQ(i).u_max(), satellite.getMTQ(i).u_max());
		}
		for (int i = 0; i < n_rw; ++i) {
			u_k(n_mtq+i) = std::clamp(u_k(n_mtq+i), -satellite.getRW(i).u_max(), satellite.getRW(i).u_max());
		}
		U_pd.col(k) = u_k;

		Eigen::VectorXd x_next = rk4Step(satellite, x_k, u_k, dt, dist_cfg, R_k, B_k, S_k, V_k, rho_k);

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
// Cost comparison (currently disabled — detection criteria are sufficient)
// ---------------------------------------------------------------------------

#if 0  // Cost comparison disabled; kept for potential future re-enablement
bool pdIsCheaper(
	const Satellite& satellite,
	const Eigen::Ref<const Eigen::MatrixXd>& X_orig,
	const Eigen::Ref<const Eigen::MatrixXd>& U_orig,
	const Eigen::MatrixXd& X_pd,
	const Eigen::MatrixXd& U_pd,
	int t_enter, int t_exit,
	const Eigen::Ref<const Eigen::MatrixXd>& B,
	const Eigen::Ref<const Eigen::MatrixXd>& boresight,
	const Eigen::Ref<const Eigen::MatrixXd>& attitude_target,
	const CostConfig& cost_cfg,
	int N
) {
	double cost_orig = 0.0;
	double cost_pd = 0.0;

	for (int k = t_enter; k < t_exit; ++k) {
		const int idx = k - t_enter;

		const double c_orig = satellite.stageCost(
			k, N, X_orig.col(k),
			(k < U_orig.cols()) ? Eigen::VectorXd(U_orig.col(k)) : Eigen::VectorXd::Zero(satellite.controlDim()),
			boresight.col(k), attitude_target.col(k), B.col(k), cost_cfg);
		cost_orig += c_orig;

		const Eigen::VectorXd x_pd_k = (idx < X_pd.cols()) ? Eigen::VectorXd(X_pd.col(idx)) : X_orig.col(k);
		const Eigen::VectorXd u_pd_k = (idx < U_pd.cols()) ? Eigen::VectorXd(U_pd.col(idx)) : Eigen::VectorXd::Zero(satellite.controlDim());
		const double c_pd = satellite.stageCost(
			k, N, x_pd_k, u_pd_k,
			boresight.col(k), attitude_target.col(k), B.col(k), cost_cfg);
		cost_pd += c_pd;
	}

	return cost_pd < cost_orig;
}
#endif  // Cost comparison disabled

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

void substituteAndBlend(
	const Satellite& satellite,
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
	const int N = static_cast<int>(X.cols());
	const int nu = static_cast<int>(U.rows());
	const int n_pd = t_exit - t_enter;
	const int n_mtq = satellite.numMTQ();
	const int n_rw = satellite.numRW();

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

		// PD contribution
		Eigen::VectorXd u_pd_k = buildPDControl(X.col(k), q_exit_target, satellite, env.B,
		                                        cfg.kp_q, cfg.kd_w, cfg.rw_scale);
		// iLQR open-loop nominal
		const Eigen::VectorXd u_ilqr_k = (k < U_bar.cols())
			? Eigen::VectorXd(U_bar.col(k))
			: Eigen::VectorXd::Zero(nu);

		Eigen::VectorXd u_blend = (1.0 - lam) * u_pd_k + lam * u_ilqr_k;

		// Clamp
		for (int i = 0; i < n_mtq; ++i) {
			u_blend(i) = std::clamp(u_blend(i), -satellite.getMTQ(i).u_max(), satellite.getMTQ(i).u_max());
		}
		for (int i = 0; i < n_rw; ++i) {
			u_blend(n_mtq+i) = std::clamp(u_blend(n_mtq+i), -satellite.getRW(i).u_max(), satellite.getRW(i).u_max());
		}

		U.col(k) = u_blend;
		if (k + 1 < N) {
			X.col(k + 1) = rk4Step(satellite, X.col(k), u_blend, dt, dist_cfg,
			                       env.R, env.B, env.S, env.V, env.rho);
		}
	}

	// --- Tail re-rollout [blend_end, N-1) with gain correction ---
	for (int k = blend_end; k < N - 1; ++k) {
		const auto env = getEnv(R, B, S, V, rho, k);
		const Eigen::VectorXd u_bar_k = (k < U_bar.cols())
			? Eigen::VectorXd(U_bar.col(k))
			: Eigen::VectorXd::Zero(nu);

		Eigen::VectorXd u_k = u_bar_k;

		if (k < static_cast<int>(K.size()) && K[k].size() > 0) {
			const Eigen::VectorXd dx = stateErrorReduced(X.col(k), X_nominal_pre.col(k), satellite);
			Eigen::VectorXd du = K[k] * dx;

			// Clamp du per channel
			for (int i = 0; i < n_mtq; ++i) {
				du(i) = std::clamp(du(i), -satellite.getMTQ(i).u_max(), satellite.getMTQ(i).u_max());
			}
			for (int i = 0; i < n_rw; ++i) {
				du(n_mtq+i) = std::clamp(du(n_mtq+i), -satellite.getRW(i).u_max(), satellite.getRW(i).u_max());
			}
			u_k += du;

			// Clamp total
			for (int i = 0; i < n_mtq; ++i) {
				u_k(i) = std::clamp(u_k(i), -satellite.getMTQ(i).u_max(), satellite.getMTQ(i).u_max());
			}
			for (int i = 0; i < n_rw; ++i) {
				u_k(n_mtq+i) = std::clamp(u_k(n_mtq+i), -satellite.getRW(i).u_max(), satellite.getRW(i).u_max());
			}
		}

		U.col(k) = u_k;
		if (k + 1 < N) {
			X.col(k + 1) = rk4Step(satellite, X.col(k), u_k, dt, dist_cfg,
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
			if (isSaturated(u_k, satellite) &&
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
			if (isSaturated(U.col(ck), satellite, 0.5)) {  // 50% of any actuator limit
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
		const int m = std::max(cfg.min_consecutive, 5);  // context window
		const double converge_thresh = M_PI / 4.0;       // 45° — context must be below this
		const double deadband = 0.3;                      // min |q·q_prev| for neighbors

		for (int kk = m; kk < N - m; ++kk) {
			if (std::isnan(theta[kk])) continue;
			if (buffered.count(kk) != 0) continue;

			// Check for hemisphere flip: q_k · q_{k-1} changes sign around kk
			// Look for a region where consecutive q dots go through zero
			// Check neighbors are solidly on their hemisphere relative to each other
			bool left_solid = true;
			for (int j = kk - m; j < kk - 1; ++j) {
				if (j < 0) { left_solid = false; break; }
				const double qdot_j = X.col(j).segment<4>(3).dot(X.col(j + 1).segment<4>(3));
				if (std::abs(qdot_j) < deadband) { left_solid = false; break; }
			}
			if (!left_solid) continue;

			bool right_solid = true;
			for (int j = kk + 1; j < kk + m && j < N - 1; ++j) {
				const double qdot_j = X.col(j).segment<4>(3).dot(X.col(j + 1).segment<4>(3));
				if (std::abs(qdot_j) < deadband) { right_solid = false; break; }
			}
			if (!right_solid) continue;

			// Check if there's a hemisphere flip near kk:
			// The left side should be on a different hemisphere than the right side
			const Eigen::Vector4d q_left = X.col(std::max(0, kk - m)).segment<4>(3);
			const Eigen::Vector4d q_right = X.col(std::min(N - 1, kk + m)).segment<4>(3);
			const double left_right_dot = q_left.dot(q_right);

			// If left and right are on the same hemisphere (dot > 0), no flip
			// If on opposite hemispheres (dot < 0), there's a flip in between
			if (left_right_dot > -deadband) continue;

			// Filter 1: local context is well-converged
			const double pe_left = std::isnan(theta[kk - m]) ? 999.0 : theta[kk - m];
			const double pe_right = (kk + m < N && !std::isnan(theta[kk + m])) ? theta[kk + m] : 999.0;
			if (pe_left > converge_thresh || pe_right > converge_thresh) continue;

			// Filter 2: spike is symmetric — PE at center is higher than at edges
			if (theta[kk] <= pe_left || theta[kk] <= pe_right) continue;

			// Find spike window: expand around kk
			int t_enter_h = kk;
			while (t_enter_h > 0 && theta[t_enter_h - 1] > std::max(pe_left, pe_right)) {
				--t_enter_h;
			}
			int t_exit_h = kk;
			while (t_exit_h < N - 1 && theta[t_exit_h + 1] > std::max(pe_left, pe_right)) {
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
	const auto& cost_cfg = pass.cost;
	const auto& cnst_cfg = settings.constraints;
	const auto& dist_cfg = settings.disturbances;
	const int N = static_cast<int>(X.cols());

	// Compute dt
	double dt = pass.dt;
	if (jtime.size() >= 2) {
		dt = (jtime(1) - jtime(0)) * 36525.0 * 86400.0;
	}

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
			satellite, X.col(t_enter), q_target, n_steps,
			B_slice, S_slice, R_slice, V_slice, rho_slice,
			dist_cfg, dt, cfg
		);

		// Cost comparison disabled — detection criteria + keep-out are sufficient.
		// The stage-cost comparison was rejecting valid homotopy spikes where the
		// PD path is locally more expensive but globally beneficial (escapes the
		// wrong SO(3) homotopy class).
		(void)cost_cfg;  // suppress unused warning

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

		// Save pre-substitution nominal
		const Eigen::MatrixXd X_nominal_pre = X;

		substituteAndBlend(
			satellite, X, U, X_pd, U_pd,
			t_enter, t_exit, cfg.blend_len,
			X_nominal_pre, U_bar, K,
			dist_cfg, R, B, S, V, rho, dt, cfg
		);

		// Only substitute one spike per call
		return true;
	}

	return false;
}

} // namespace saltro::optimizer
