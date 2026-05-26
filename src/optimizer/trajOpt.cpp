#include <saltro/optimizer/trajOpt.h>
#include <saltro/optimizer/alilqr.h>
#include <saltro/optimizer/backwardpass.h>
#include <saltro/optimizer/warm_start.h>
#include <saltro/orbit_generation/generate_orbit.h>
#include <saltro/validation/validate_trajOpt.h>
#include <saltro/math/integrators/rk4.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace saltro::optimizer {

static std::vector<Eigen::MatrixXd> compute_gains_chunked(
	const PlannerSettings& settings,
	const Satellite& satellite,
	const Eigen::Ref<const Eigen::MatrixXd>& X,
	const Eigen::Ref<const Eigen::MatrixXd>& U,
	const Eigen::Ref<const Eigen::MatrixXd>& R,
	const Eigen::Ref<const Eigen::MatrixXd>& V,
	const Eigen::Ref<const Eigen::MatrixXd>& B,
	const Eigen::Ref<const Eigen::MatrixXd>& S,
	const Eigen::Ref<const Eigen::MatrixXd>& rho,
	const Eigen::Ref<const Eigen::MatrixXd>& boresight,
	const Eigen::Ref<const Eigen::MatrixXd>& q_goal,
	const int input_dim,
	const double dt_tvlqr,
	const double tvlqr_len,
	const double tvlqr_overlap
) {
	const int total_gain_steps = std::max(0, static_cast<int>(X.cols()) - 1);
	const int n_red = satellite.reducedStateDim();
	if (total_gain_steps == 0) {
		return {};
	}

	int len_steps = static_cast<int>(std::floor(tvlqr_len / dt_tvlqr));
	int overlap_steps = static_cast<int>(std::floor(tvlqr_overlap / dt_tvlqr));
	len_steps = std::max(1, len_steps);
	overlap_steps = std::max(0, overlap_steps);
	if (overlap_steps >= len_steps) {
		overlap_steps = len_steps - 1;
	}

	std::vector<Eigen::MatrixXd> K_stitched;
	int start_k = 0;

	while (start_k < total_gain_steps) {
		const int end_k = std::min(total_gain_steps - 1, start_k + len_steps - 1);
		const int chunk_gain_steps = end_k - start_k + 1;
		const int x_col_start = start_k;
		const int x_col_end = end_k + 1;
		const int x_cols = x_col_end - x_col_start + 1;

		std::vector<Eigen::MatrixXd> K_chunk(static_cast<size_t>(chunk_gain_steps));
		std::vector<Eigen::VectorXd> d_chunk(static_cast<size_t>(chunk_gain_steps));
		for (int k = 0; k < chunk_gain_steps; ++k) {
			K_chunk[static_cast<size_t>(k)] = Eigen::MatrixXd::Zero(input_dim, n_red);
			d_chunk[static_cast<size_t>(k)] = Eigen::VectorXd::Zero(input_dim);
		}

		Eigen::Vector2d deltaV = Eigen::Vector2d::Zero();
		const auto& reg_cfg = settings.passes[std::max(0, settings.num_passes - 1)].reg;
		double reg = std::max(reg_cfg.reg_min, reg_cfg.reg_init);
		bool bp_ok = false;
		while (reg <= reg_cfg.reg_max) {
			deltaV.setZero();
			bp_ok = backwardPass(
				satellite,
				X.middleCols(x_col_start, x_cols),
				U.middleCols(x_col_start, x_cols),
				R.middleCols(x_col_start, x_cols),
				V.middleCols(x_col_start, x_cols),
				B.middleCols(x_col_start, x_cols),
				S.middleCols(x_col_start, x_cols),
				rho.middleCols(x_col_start, x_cols),
				boresight.middleCols(x_col_start, x_cols),
				q_goal.middleCols(x_col_start, x_cols),
				settings,
				reg,
				K_chunk,
				d_chunk,
				deltaV
			);
			if (bp_ok) {
				break;
			}
			reg *= reg_cfg.reg_scale;
		}

		if (!bp_ok) {
			K_chunk.assign(static_cast<size_t>(chunk_gain_steps), Eigen::MatrixXd::Zero(input_dim, n_red));
		}

		if (K_stitched.empty()) {
			K_stitched.insert(K_stitched.end(), K_chunk.begin(), K_chunk.end());
		} else {
			const int drop = std::min(overlap_steps, static_cast<int>(K_stitched.size()));
			K_stitched.resize(static_cast<size_t>(static_cast<int>(K_stitched.size()) - drop));
			K_stitched.insert(K_stitched.end(), K_chunk.begin(), K_chunk.end());
		}

		int next_start = end_k + 1 - overlap_steps;
		if (next_start <= start_k) {
			next_start = start_k + 1;
		}
		start_k = next_start;
	}

	if (static_cast<int>(K_stitched.size()) > total_gain_steps) {
		K_stitched.resize(static_cast<size_t>(total_gain_steps));
	}
	if (static_cast<int>(K_stitched.size()) < total_gain_steps) {
		K_stitched.resize(static_cast<size_t>(total_gain_steps), Eigen::MatrixXd::Zero(input_dim, n_red));
	}

	return K_stitched;
}

static bool resample_zero_order_hold(
	const Eigen::Ref<const Eigen::VectorXd>& jtime_coarse,
	const Eigen::Ref<const Eigen::MatrixXd>& q_goal_coarse,
	const Eigen::Ref<const Eigen::MatrixXd>& boresight_coarse,
	double dt_seconds,
	Eigen::Ref<Eigen::Matrix<double, 1, Eigen::Dynamic>> jtime_fine,
	Eigen::Ref<Eigen::Matrix<double, 4, Eigen::Dynamic>> q_goal_fine,
	Eigen::Ref<Eigen::Matrix<double, 3, Eigen::Dynamic>> boresight_fine,
	int& N_fine
)
{
	N_fine = 0;
	if (jtime_coarse.size() <= 0 || dt_seconds <= 0.0) {
		return false;
	}

	const double t0 = jtime_coarse(0);
	const double tN = jtime_coarse(jtime_coarse.size() - 1);
	const double dt_centuries = dt_seconds / (36525.0 * 86400.0);
	const int max_samples = static_cast<int>(std::ceil((tN - t0) / dt_centuries)) + 1;
	if (max_samples <= 0 || max_samples > saltro::limits::MAX_LENGTH_TRAJ) {
		return false;
	}

	int idx_coarse = 0;
	double t = t0;
	for (int k = 0; k < max_samples; ++k) {
		if (t > tN + 1e-12) {
			break;
		}
		// Match numpy.searchsorted(..., side='right'): when t lands exactly on
		// a coarse knot, switch to the next segment's held value.
		while (
			idx_coarse + 1 < jtime_coarse.size()
			&& t >= jtime_coarse(idx_coarse + 1)
		) {
			++idx_coarse;
		}
		jtime_fine(0, k) = t;
		q_goal_fine.col(k) = q_goal_coarse.col(idx_coarse);
		boresight_fine.col(k) = boresight_coarse.col(idx_coarse);
		++N_fine;
		t += dt_centuries;
	}

	// Ensure last sample exactly matches tN for consistency
	if (N_fine > 0 && std::abs(jtime_fine(0, N_fine - 1) - tN) > 1e-12) {
		if (N_fine < saltro::limits::MAX_LENGTH_TRAJ) {
			jtime_fine(0, N_fine) = tN;
			q_goal_fine.col(N_fine) = q_goal_coarse.col(jtime_coarse.size() - 1);
			boresight_fine.col(N_fine) = boresight_coarse.col(jtime_coarse.size() - 1);
			++N_fine;
		} else {
			return false;
		}
	}

	return N_fine > 0;
}

bool trajOpt(
	const PlannerSettings& settings,
	const Satellite& satellite,
	const Satellite::VecX& x0,
	const Eigen::Vector3d& r0,
	const Eigen::Vector3d& v0,
	const Eigen::Ref<const Eigen::VectorXd>& jtime,
	const Eigen::Ref<const Eigen::MatrixXd>& q_goal,
	const Eigen::Ref<const Eigen::MatrixXd>& boresight,

	Eigen::Ref<Eigen::MatrixXd> X,
	Eigen::Ref<Eigen::MatrixXd> U,
	Eigen::Ref<Eigen::MatrixXd> K,

	int state_dim,
	int input_dim,
	int& N
) {
	PlannerSettings settings_local = settings;
	if (settings_local.constraints.u_max.size() == 0) {
		settings_local.constraints.u_max.resize(input_dim);
		for (int i = 0; i < satellite.numMTQ(); ++i) {
			settings_local.constraints.u_max(i) = std::abs(satellite.getMTQ(i).u_max());
		}
		for (int i = 0; i < satellite.numRW(); ++i) {
			settings_local.constraints.u_max(satellite.numMTQ() + i) = std::abs(satellite.getRW(i).u_max());
		}
		for (int i = 0; i < satellite.numMagic(); ++i) {
			settings_local.constraints.u_max(satellite.numMTQ() + satellite.numRW() + i) =
				std::abs(satellite.getMagic(i).u_max());
		}
	}

	std::string error_msg;
	if (!validation::validatetrajOpt(settings_local, satellite, x0, r0, v0, jtime, q_goal, boresight, state_dim, input_dim, N, error_msg)) {
		throw std::runtime_error("trajOpt input validation failed: " + error_msg);
	}

	const double dt_sec = settings_local.passes[0].dt;

	Eigen::Matrix<double, 1, Eigen::Dynamic> jtime_fixed(1, saltro::limits::MAX_LENGTH_TRAJ);
	Eigen::Matrix<double, 4, Eigen::Dynamic> q_goal_fixed(4, saltro::limits::MAX_LENGTH_TRAJ);
	Eigen::Matrix<double, 3, Eigen::Dynamic> boresight_fixed(3, saltro::limits::MAX_LENGTH_TRAJ);
	jtime_fixed.setZero();
	q_goal_fixed.setZero();
	boresight_fixed.setZero();
	int N_fixed = 0;

	if (!resample_zero_order_hold(jtime, q_goal, boresight, dt_sec, jtime_fixed, q_goal_fixed, boresight_fixed, N_fixed)) {
		throw std::runtime_error("trajOpt failed to resample time and goals");
	}

	if (!validation::validateTrajOptResampledContext(
		jtime_fixed,
		q_goal_fixed,
		boresight_fixed,
		N_fixed,
		static_cast<int>(X.cols()),
		static_cast<int>(U.cols()),
		error_msg
	)) {
		throw std::runtime_error("trajOpt post-resample validation failed: " + error_msg);
	}

	Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ> R;
	Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ> V;
	Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ> B;
	Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ> S;
	Eigen::Matrix<double, 1, saltro::limits::MAX_LENGTH_TRAJ> rho;

	const bool orbit_ok = orbits::generate_orbit(r0, v0, jtime_fixed, N_fixed, 1, 2, 0, 0, 0, R, V, B, S, rho);

	if (!orbit_ok) {
		throw std::runtime_error("trajOpt failed to generate orbit");
	}

	const bool warm_start_ok = warm_start(settings_local, satellite, x0, jtime_fixed.leftCols(N_fixed).transpose(), q_goal_fixed.leftCols(N_fixed), boresight_fixed.leftCols(N_fixed), N_fixed, R, V, B, S, rho, X, U);

	if (!warm_start_ok) {
		throw std::runtime_error("trajOpt failed to warm-start trajectory");
	}

	for (int pass_idx = 0; pass_idx < settings_local.num_passes; ++pass_idx) {
		ALILQRStatus al_status = ALILQRStatus::MaxOuterIterations;
		double max_c = 0.0;
		const bool ok = alilqr(
			settings_local,
			pass_idx,
			satellite,
			X.leftCols(N_fixed),
			U.leftCols(N_fixed),
			R.leftCols(N_fixed),
			V.leftCols(N_fixed),
			B.leftCols(N_fixed),
			S.leftCols(N_fixed),
			rho.leftCols(N_fixed),
			jtime_fixed.leftCols(N_fixed).transpose(),
			boresight_fixed.leftCols(N_fixed),
			q_goal_fixed.leftCols(N_fixed),
			al_status,
			max_c
		);
		(void)max_c;
		if (!ok) {
			// Return partial results (X and U contain the best trajectory found)
			// rather than throwing.  The caller can check the return value.
			N = N_fixed;
			return false;
		}

	}

	// Final pass for gains: compute K on converged trajectory.
	K.setZero();
	if (N_fixed > 1) {
		PlannerSettings tracking_settings = settings_local;
		tracking_settings.num_passes = 1;
		tracking_settings.passes[0] = settings_local.passes[std::max(0, settings_local.num_passes - 1)];

		const double dt_tracking = tracking_settings.passes[0].dt;
		if (dt_tracking > 0.0) {
			tracking_settings.passes[0].dt = dt_tracking;
		}

		int N_tracking = N_fixed;
		Eigen::MatrixXd X_tracking = X.leftCols(N_fixed);
		Eigen::MatrixXd U_tracking = U.leftCols(N_fixed);

		Eigen::Matrix<double, 1, Eigen::Dynamic> jtime_tracking(1, saltro::limits::MAX_LENGTH_TRAJ);
		Eigen::Matrix<double, 4, Eigen::Dynamic> q_goal_tracking(4, saltro::limits::MAX_LENGTH_TRAJ);
		Eigen::Matrix<double, 3, Eigen::Dynamic> boresight_tracking(3, saltro::limits::MAX_LENGTH_TRAJ);
		jtime_tracking.setZero();
		q_goal_tracking.setZero();
		boresight_tracking.setZero();

		Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ> R_tracking;
		Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ> V_tracking;
		Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ> B_tracking;
		Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ> S_tracking;
		Eigen::Matrix<double, 1, saltro::limits::MAX_LENGTH_TRAJ> rho_tracking;
		R_tracking.setZero();
		V_tracking.setZero();
		B_tracking.setZero();
		S_tracking.setZero();
		rho_tracking.setZero();

		const bool build_dense_tracking = (dt_tracking > 0.0 && dt_tracking + 1e-12 < dt_sec);
		if (build_dense_tracking) {
			const Eigen::VectorXd jtime_coarse = jtime_fixed.leftCols(N_fixed).transpose();
			if (!resample_zero_order_hold(
				jtime_coarse,
				q_goal_fixed.leftCols(N_fixed),
				boresight_fixed.leftCols(N_fixed),
				dt_tracking,
				jtime_tracking,
				q_goal_tracking,
				boresight_tracking,
				N_tracking
			)) {
				throw std::runtime_error("trajOpt failed to build dense tracking grid");
			}

			if (!orbits::generate_orbit(
				r0,
				v0,
				jtime_tracking,
				N_tracking,
				1,
				2,
				0,
				0,
				0,
				R_tracking,
				V_tracking,
				B_tracking,
				S_tracking,
				rho_tracking
			)) {
				throw std::runtime_error("trajOpt failed to generate dense tracking orbit");
			}

			X_tracking = Eigen::MatrixXd::Zero(state_dim, N_tracking);
			U_tracking = Eigen::MatrixXd::Zero(input_dim, N_tracking);
			X_tracking.col(0) = X.col(0);

			const int coarse_u_cols = std::max(0, N_fixed - 1);
			int coarse_idx = 0;
			for (int k = 0; k < N_tracking; ++k) {
				while (coarse_idx + 1 < N_fixed && jtime_tracking(0, k) >= jtime_fixed(0, coarse_idx + 1)) {
					++coarse_idx;
				}
				if (coarse_u_cols > 0) {
					const int u_idx = std::min(std::max(coarse_idx, 0), coarse_u_cols - 1);
					U_tracking.col(k) = U.col(u_idx);
				}
			}

			for (int k = 0; k < N_tracking - 1; ++k) {
				const double dt_centuries = jtime_tracking(0, k + 1) - jtime_tracking(0, k);
				const double dt_step = dt_centuries * 36525.0 * 86400.0;
				if (!std::isfinite(dt_step) || dt_step <= 0.0) {
					throw std::runtime_error("trajOpt invalid dense tracking timestep");
				}

				const Eigen::VectorXd u_k = U_tracking.col(k);
				Eigen::VectorXd x_next;
				saltro::math::rk4_step<Eigen::VectorXd>(
					[&](double, const Eigen::VectorXd& x_state, Eigen::VectorXd& dxdt) {
						dxdt = satellite.dynamics(
							x_state,
							u_k,
							tracking_settings.disturbances,
							R_tracking.col(k),
							B_tracking.col(k),
							S_tracking.col(k),
							V_tracking.col(k),
							static_cast<int>(std::max(0.0, std::round(rho_tracking(0, k))))
						);
					},
					X_tracking.col(k),
					0.0,
					dt_step,
					x_next
				);

				if (!x_next.allFinite()) {
					throw std::runtime_error("trajOpt dense tracking rollout produced invalid state");
				}
				if (x_next.size() >= 7) {
					Eigen::Vector4d q = x_next.segment<4>(3);
					const double qn = q.norm();
					if (!std::isfinite(qn) || qn <= 1e-10) {
						throw std::runtime_error("trajOpt dense tracking rollout quaternion invalid");
					}
					x_next.segment<4>(3) = q / qn;
				}
				X_tracking.col(k + 1) = x_next;
			}

			X.leftCols(N_tracking) = X_tracking;
			U.leftCols(N_tracking) = U_tracking;
		} else {
			jtime_tracking.leftCols(N_fixed) = jtime_fixed.leftCols(N_fixed);
			q_goal_tracking.leftCols(N_fixed) = q_goal_fixed.leftCols(N_fixed);
			boresight_tracking.leftCols(N_fixed) = boresight_fixed.leftCols(N_fixed);
			R_tracking.leftCols(N_fixed) = R.leftCols(N_fixed);
			V_tracking.leftCols(N_fixed) = V.leftCols(N_fixed);
			B_tracking.leftCols(N_fixed) = B.leftCols(N_fixed);
			S_tracking.leftCols(N_fixed) = S.leftCols(N_fixed);
			rho_tracking.leftCols(N_fixed) = rho.leftCols(N_fixed);
		}

		std::vector<Eigen::MatrixXd> K_gains = compute_gains_chunked(
			tracking_settings,
			satellite,
			X_tracking.leftCols(N_tracking),
			U_tracking.leftCols(N_tracking),
			R_tracking.leftCols(N_tracking),
			V_tracking.leftCols(N_tracking),
			B_tracking.leftCols(N_tracking),
			S_tracking.leftCols(N_tracking),
			rho_tracking.leftCols(N_tracking),
			boresight_tracking.leftCols(N_tracking),
			q_goal_tracking.leftCols(N_tracking),
			input_dim,
			dt_tracking,
			settings_local.tvlqr.tvlqr_len,
			settings_local.tvlqr.tvlqr_overlap
		);

		const int n_red = satellite.reducedStateDim();
		const int gain_count = std::min(N_tracking - 1, static_cast<int>(K_gains.size()));
		for (int k = 0; k < gain_count; ++k) {
			const int col0 = k * n_red;
			if (col0 + n_red > K.cols()) {
				break;
			}
			K.middleCols(col0, n_red) = K_gains[static_cast<size_t>(k)];
		}

		N_fixed = N_tracking;
	}

	N = N_fixed;

	return true;
}

}
