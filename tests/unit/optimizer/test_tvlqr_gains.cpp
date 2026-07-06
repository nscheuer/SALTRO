// C++ twin of tests/unit/optimizer/test_tvlqr_gains.py.
//
// Characterizes trajOpt's chunked TVLQR feedback gains (the K output of
// compute_gains_chunked): structural validity, the single-chunk no-op limit,
// that shorter windows change the gains while staying bounded, and the gains'
// purpose -- applying eq. 7.39 to a perturbed closed-loop rollout tracks the
// plan far better than open loop. See the .py twin for the full rationale.
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <vector>

#include <Eigen/Dense>

#include <saltro/limits.h>
#include <saltro/math/mrp.h>
#include <saltro/optimizer/trajOpt.h>
#include <saltro/orbit_generation/generate_orbit.h>
#include <saltro/pybind/satellite.h>

using namespace saltro;

namespace {

constexpr double SEC_PER_CENTURY = 36525.0 * 86400.0;

// A window longer than the (200 s) trajectory below with zero overlap is a
// single chunk spanning the whole horizon (see the .py twin for why overlap
// must be zero, and why the magnitude is kept modest).
constexpr double SINGLE_CHUNK_LEN = 1000.0;

PlannerSettings makeSettings(double dt_seconds, double tvlqr_len, double tvlqr_overlap) {
	PlannerSettings s;
	s.init_traj.initcontroller = 2;
	s.num_passes = 1;
	s.passes[0].dt = dt_seconds;
	s.passes[0].ilqr.cost_tol = 1e-5;
	s.passes[0].ilqr.max_iters = 20;
	s.passes[0].auglag.max_outer_iters = 10;
	s.passes[0].auglag.constraint_tol = 1e-3;

	auto& c = s.passes[0].cost;
	c.angle = 1.0;
	c.ang_vel = 1e1;
	c.control_mult = 1.0;
	c.mtq_control_weight = 1e-2;
	c.rw_control_weight = 1.0;
	c.ang_cost_func_type = 3;
	c.use_cost_hess = true;

	s.disturbances.plan_for_aero = false;
	s.disturbances.plan_for_gg = false;
	s.disturbances.plan_for_srp = false;
	s.disturbances.plan_for_prop = false;
	s.disturbances.plan_for_gendist = false;
	s.disturbances.plan_for_resdipole = false;

	s.passes[0].reg.reg_init = 1e-6;
	s.passes[0].reg.reg_max = 1e10;
	s.passes[0].reg.reg_scale = 10.0;
	s.passes[0].reg.use_dynamics_hess = false;
	s.passes[0].reg.use_constraint_hess = false;

	s.passes[0].linesearch.max_iters = 24;
	s.passes[0].linesearch.beta1 = 1e-10;
	s.passes[0].linesearch.beta2 = 5000.0;

	s.tvlqr.tvlqr_len = tvlqr_len;
	s.tvlqr.tvlqr_overlap = tvlqr_overlap;
	return s;
}

Eigen::Matrix3d rwInertia() {
	Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
	J(0, 0) = 0.067;
	J(1, 1) = 0.071;
	J(2, 2) = 0.069;
	return J;
}

// Satellite has a deleted copy/move ctor, so it must be built in place; this
// adds the three reaction wheels to an already-constructed satellite.
void configureRW(Satellite& satellite) {
	satellite.addRW(Eigen::Vector3d::UnitX(), 0.001, 1e-5, 0.0, 0.02);
	satellite.addRW(Eigen::Vector3d::UnitY(), 0.001, 1e-5, 0.0, 0.02);
	satellite.addRW(Eigen::Vector3d::UnitZ(), 0.001, 1e-5, 0.0, 0.02);
}

Satellite::VecX makeX0(const Satellite& satellite) {
	Satellite::VecX x0(satellite.stateDim());
	x0.setZero();
	x0.segment<3>(Satellite::AV_INDEX) = Eigen::Vector3d(-0.01, 0.02, 0.03);
	x0.segment<4>(Satellite::QUAT_INDEX) = Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
	return x0;
}

struct SolveResult {
	bool ok = false;
	int N = 0;
	Eigen::MatrixXd X;  // state_dim x N
	Eigen::MatrixXd U;  // input_dim x N
	Eigen::MatrixXd K;  // input_dim x (reduced_state_dim * N)
};

SolveResult solveRW(double dt_seconds, double tf_seconds, double tvlqr_len,
                    double tvlqr_overlap) {
	const PlannerSettings settings = makeSettings(dt_seconds, tvlqr_len, tvlqr_overlap);
	Satellite satellite(rwInertia(), settings);
	configureRW(satellite);
	const Satellite::VecX x0 = makeX0(satellite);

	const Eigen::Vector3d r0(7000e3, 0.0, 0.0);
	const Eigen::Vector3d v0(0.0, 7.5e3, 0.0);

	Eigen::VectorXd jtime(2);
	jtime(0) = 0.22;
	jtime(1) = 0.22 + tf_seconds / SEC_PER_CENTURY;

	Eigen::MatrixXd q_goal(4, 2);
	q_goal << std::sqrt(2.0) / 2.0, std::sqrt(2.0) / 2.0,
	          0.0, 0.0,
	          0.0, 0.0,
	          std::sqrt(2.0) / 2.0, std::sqrt(2.0) / 2.0;
	Eigen::MatrixXd boresight(3, 2);
	boresight << 1.0, 1.0, 0.0, 0.0, 0.0, 0.0;

	const int state_dim = satellite.stateDim();
	const int input_dim = satellite.controlDim();
	const int reduced_state_dim = satellite.reducedStateDim();

	Eigen::MatrixXd X = Eigen::MatrixXd::Zero(state_dim, limits::MAX_LENGTH_TRAJ);
	Eigen::MatrixXd U = Eigen::MatrixXd::Zero(input_dim, limits::MAX_LENGTH_TRAJ);
	Eigen::MatrixXd K =
	    Eigen::MatrixXd::Zero(input_dim, reduced_state_dim * limits::MAX_LENGTH_TRAJ);
	int N = static_cast<int>(jtime.size());

	SolveResult r;
	REQUIRE_NOTHROW(r.ok = optimizer::trajOpt(settings, satellite, x0, r0, v0, jtime,
	                                           q_goal, boresight,
	                                           Eigen::MatrixXd(0, 0),  // seed_X
	                                           Eigen::MatrixXd(0, 0),  // seed_U
	                                           X, U, K, state_dim,
	                                           input_dim, N));
	r.N = N;
	r.X = X.leftCols(N);
	r.U = U.leftCols(N);
	r.K = K.leftCols(reduced_state_dim * N);
	return r;
}

// Reduced-state error [dw, MRP(quatError), dh] -- the forward-pass convention.
Eigen::VectorXd reducedError(const Eigen::VectorXd& x_cur, const Eigen::VectorXd& x_plan,
                             int nRW) {
	Eigen::VectorXd dz = Eigen::VectorXd::Zero(6 + nRW);
	dz.head<3>() = x_cur.head<3>() - x_plan.head<3>();
	const Eigen::Vector4d q_err =
	    math::quatError(x_plan.segment<4>(3), x_cur.segment<4>(3));
	dz.segment<3>(3) = math::quatToMRP(q_err);
	for (int i = 0; i < nRW; ++i) {
		dz(6 + i) = x_cur(7 + i) - x_plan(7 + i);
	}
	return dz;
}

Eigen::VectorXd rk4Step(const Satellite& sat, const Eigen::VectorXd& x,
                        const Eigen::VectorXd& u, const DisturbanceConfig& dist,
                        const Eigen::Vector3d& R, const Eigen::Vector3d& B,
                        const Eigen::Vector3d& S, const Eigen::Vector3d& V, int rho,
                        double dt) {
	const auto f = [&](const Eigen::VectorXd& xx) {
		return sat.dynamics(xx, u, dist, R, B, S, V, rho);
	};
	const Eigen::VectorXd k1 = f(x);
	const Eigen::VectorXd k2 = f(x + 0.5 * dt * k1);
	const Eigen::VectorXd k3 = f(x + 0.5 * dt * k2);
	const Eigen::VectorXd k4 = f(x + dt * k3);
	Eigen::VectorXd xn = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
	xn.segment<4>(3).normalize();  // stay on the unit sphere
	return xn;
}

}  // namespace

TEST_CASE("chunked TVLQR gains have correct shape and are finite/nonzero",
          "[optimizer][tvlqr][gains]") {
	const SolveResult r = solveRW(10.0, 200.0, SINGLE_CHUNK_LEN, 0.0);
	REQUIRE(r.ok);
	Satellite satellite(rwInertia(), makeSettings(10.0, SINGLE_CHUNK_LEN, 0.0));
	configureRW(satellite);
	REQUIRE(r.K.rows() == satellite.controlDim());
	REQUIRE(r.K.cols() == satellite.reducedStateDim() * r.N);
	REQUIRE(r.K.allFinite());
	REQUIRE(r.K.cwiseAbs().maxCoeff() > 0.0);
}

TEST_CASE("chunked TVLQR gains are deterministic", "[optimizer][tvlqr][gains]") {
	const SolveResult a = solveRW(10.0, 200.0, SINGLE_CHUNK_LEN, 0.0);
	const SolveResult b = solveRW(10.0, 200.0, SINGLE_CHUNK_LEN, 0.0);
	REQUIRE(a.K.rows() == b.K.rows());
	REQUIRE(a.K.cols() == b.K.cols());
	REQUIRE((a.K - b.K).cwiseAbs().maxCoeff() == 0.0);
}

TEST_CASE("a window covering the horizon is a single-chunk no-op",
          "[optimizer][tvlqr][gains]") {
	const SolveResult a = solveRW(10.0, 200.0, 400.0, 0.0);
	const SolveResult b = solveRW(10.0, 200.0, 5000.0, 0.0);
	REQUIRE(a.K.cols() == b.K.cols());
	REQUIRE((a.K - b.K).cwiseAbs().maxCoeff() == 0.0);
}

TEST_CASE("shorter windows change the gains but stay bounded",
          "[optimizer][tvlqr][gains]") {
	const SolveResult full = solveRW(10.0, 200.0, SINGLE_CHUNK_LEN, 0.0);
	const SolveResult chunked = solveRW(10.0, 200.0, 30.0, 10.0);
	REQUIRE(full.K.cols() == chunked.K.cols());
	REQUIRE(chunked.K.allFinite());
	REQUIRE((chunked.K - full.K).cwiseAbs().maxCoeff() > 1e-6);
	REQUIRE(chunked.K.cwiseAbs().maxCoeff() <= 5.0 * full.K.cwiseAbs().maxCoeff());
}

TEST_CASE("feedback gains track the plan better than open loop (eq. 7.39)",
          "[optimizer][tvlqr][gains]") {
	const double dt = 10.0;
	const double tf = 200.0;
	const PlannerSettings settings = makeSettings(dt, SINGLE_CHUNK_LEN, 0.0);
	Satellite satellite(rwInertia(), settings);
	configureRW(satellite);
	const int nRW = satellite.stateDim() - 7;
	const int n_red = satellite.reducedStateDim();

	const SolveResult r = solveRW(dt, tf, SINGLE_CHUNK_LEN, 0.0);
	REQUIRE(r.ok);
	const int N = r.N;

	// Orbit environment along the fine trajectory.
	const Eigen::Vector3d r0(7000e3, 0.0, 0.0);
	const Eigen::Vector3d v0(0.0, 7.5e3, 0.0);
	Eigen::Matrix<double, 1, limits::MAX_LENGTH_TRAJ> jtime_full;
	jtime_full.setZero();
	const double dt_centuries = dt / SEC_PER_CENTURY;
	for (int k = 0; k < N; ++k) {
		jtime_full(0, k) = 0.22 + k * dt_centuries;
	}
	Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> R, V, B, S;
	Eigen::Matrix<double, 1, limits::MAX_LENGTH_TRAJ> rho;
	R.setZero();
	V.setZero();
	B.setZero();
	S.setZero();
	rho.setZero();
	REQUIRE(orbits::generate_orbit(r0, v0, jtime_full, N, 0, 0, 0, 0, 0, R, V, B, S, rho));

	// Perturb the initial state: ~1 deg about +z and a small rate error.
	const Satellite::VecX x0 = makeX0(satellite);
	Eigen::VectorXd x0p = x0;
	x0p.head<3>() += Eigen::Vector3d(0.002, -0.0015, 0.001);
	const double half = 0.5 * (1.0 * M_PI / 180.0);
	const Eigen::Vector4d dq(std::cos(half), 0.0, 0.0, std::sin(half));
	const Eigen::Vector4d q0 = x0.segment<4>(3);
	Eigen::Vector4d qp;
	qp(0) = q0(0) * dq(0) - q0(1) * dq(1) - q0(2) * dq(2) - q0(3) * dq(3);
	qp(1) = q0(0) * dq(1) + q0(1) * dq(0) + q0(2) * dq(3) - q0(3) * dq(2);
	qp(2) = q0(0) * dq(2) - q0(1) * dq(3) + q0(2) * dq(0) + q0(3) * dq(1);
	qp(3) = q0(0) * dq(3) + q0(1) * dq(2) - q0(2) * dq(1) + q0(3) * dq(0);
	x0p.segment<4>(3) = qp;

	const auto rollout = [&](bool use_feedback) {
		Eigen::VectorXd x = x0p;
		double final_err = 0.0;
		for (int k = 0; k < N - 1; ++k) {
			Eigen::VectorXd u = r.U.col(k);
			if (use_feedback) {
				const Eigen::MatrixXd K_k = r.K.block(0, k * n_red, r.K.rows(), n_red);
				u += K_k * reducedError(x, r.X.col(k), nRW);
			}
			x = rk4Step(satellite, x, u, settings.disturbances, R.col(k), B.col(k),
			            S.col(k), V.col(k), static_cast<int>(rho(0, k)), dt);
			final_err = reducedError(x, r.X.col(k + 1), nRW).norm();
		}
		return final_err;
	};

	const double err_open = rollout(false);
	const double err_closed = rollout(true);
	REQUIRE(std::isfinite(err_open));
	REQUIRE(std::isfinite(err_closed));
	REQUIRE(err_closed < 0.5 * err_open);
}
