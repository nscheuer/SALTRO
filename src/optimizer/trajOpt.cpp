#include <saltro/optimizer/trajOpt.h>
#include <saltro/optimizer/warm_start.h>
#include <saltro/orbit_generation/generate_orbit.h>
#include <saltro/validation/validate_trajOpt.h>

#include <cmath>
#include <limits>

namespace saltro::optimizer {

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
		while (idx_coarse + 1 < jtime_coarse.size() && t > jtime_coarse(idx_coarse + 1)) {
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
	(void)K;

	std::string error_msg;
	if (!validation::validatetrajOpt(settings, satellite, x0, r0, v0, jtime, q_goal, boresight, state_dim, input_dim, N, error_msg)) {
		throw std::runtime_error("trajOpt input validation failed: " + error_msg);
	}

	double dt_sec = settings.num_passes > 0 ? settings.passes[0].dt : 0.0;
	if (dt_sec <= 0.0 && jtime.size() > 1) {
		// Fallback to minimum coarse spacing if dt not provided
		dt_sec = std::numeric_limits<double>::infinity();
		for (Eigen::Index i = 1; i < jtime.size(); ++i) {
			const double dcent = jtime(i) - jtime(i - 1);
			if (dcent > 0.0) {
				dt_sec = std::min(dt_sec, dcent * 36525.0 * 86400.0);
			}
		}
		if (!std::isfinite(dt_sec) || dt_sec <= 0.0) {
			throw std::runtime_error("trajOpt could not determine a valid dt");
		}
	}

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

	Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ> R;
	Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ> V;
	Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ> B;
	Eigen::Matrix<double, 3, saltro::limits::MAX_LENGTH_TRAJ> S;
	Eigen::Matrix<double, 1, saltro::limits::MAX_LENGTH_TRAJ> rho;

	const bool orbit_ok = orbits::generate_orbit(r0, v0, jtime_fixed, N_fixed, 0, 0, 0, 0, 0, R, V, B, S, rho);

	if (!orbit_ok) {
		throw std::runtime_error("trajOpt failed to generate orbit");
	}

	const bool warm_start_ok = warm_start(settings, satellite, x0, jtime_fixed.leftCols(N_fixed).transpose(), q_goal_fixed.leftCols(N_fixed), boresight_fixed.leftCols(N_fixed), N_fixed, R, V, B, S, rho, X, U);

	if (!warm_start_ok) {
		throw std::runtime_error("trajOpt failed to warm-start trajectory");
	}

	// Update N to reflect the resampled grid and ensure provided buffers are large enough
	if (N_fixed > X.cols() || N_fixed > U.cols()) {
		throw std::runtime_error("trajOpt resampled length exceeds provided X/U column capacity; allocate at least N_fixed columns");
	}
	N = N_fixed;

	return true;
}

}
