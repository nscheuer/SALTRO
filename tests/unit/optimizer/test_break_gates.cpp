// Convergence/break-gate redesign tests (BREAK_GATE_DESIGN.md §5–§7).
//
// Covers the four symptom classes plus the new gate semantics:
//   1. Do-nothing guard (G15) + check-order (optimal warm start exits
//      Converged with zero accepted steps, never RegularizationExceeded).
//   2. Settle discipline (loose-tier exits while infeasible, strict
//      conjunctive exit before Converged).
//   3. Stall counter (z_count_lim, dlaZcount semantics) returns the best
//      trajectory as Stalled.
//   4. Budget/penalty statuses: MaxTotalIterations and PenaltyMaxReached are
//      reachable and carry trajectories.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <saltro/limits.h>
#include <saltro/optimizer/alilqr.h>
#include <saltro/optimizer/iLQR.h>
#include <saltro/optimizer/warm_start.h>
#include <saltro/orbit_generation/generate_orbit.h>
#include <saltro/pybind/satellite.h>

using namespace saltro;

namespace {

constexpr double SEC_PER_CENTURY = 36525.0 * 86400.0;

PlannerSettings createRWPlannerSettings(double dt_seconds) {
	PlannerSettings plannersettings;

	plannersettings.init_traj.initcontroller = 2;

	plannersettings.num_passes = 1;
	plannersettings.passes[0].dt = dt_seconds;
	plannersettings.passes[0].ilqr.cost_tol = 1e-5;
	plannersettings.passes[0].ilqr.max_iters = 20;

	plannersettings.passes[0].auglag.max_outer_iters = 30;
	plannersettings.passes[0].auglag.constraint_tol = 1e-3;

	auto& cost = plannersettings.passes[0].cost;
	cost.angle = 1.0;
	cost.ang_vel = 1e1;
	cost.control_mult = 1.0;
	cost.mtq_control_weight = 1e-2;
	cost.rw_control_weight = 1.0;
	cost.magic_control_weight = 0.0;
	cost.rw_AM_weight = 0.0;
	cost.rw_stic_weight = 0.0;
	cost.RWh_stiction_mult = 0.0;
	cost.angle_N = 0.0;
	cost.ang_vel_N = 0.0;
	cost.ang_cost_func_type = 3;
	cost.use_cost_hess = true;

	plannersettings.passes[0].reg.reg_init = 1e-6;
	plannersettings.passes[0].reg.reg_max = 1e10;
	plannersettings.passes[0].reg.reg_scale = 10.0;
	plannersettings.passes[0].reg.use_dynamics_hess = false;
	plannersettings.passes[0].reg.use_constraint_hess = false;

	plannersettings.passes[0].linesearch.max_iters = 24;
	plannersettings.passes[0].linesearch.beta1 = 1e-10;
	plannersettings.passes[0].linesearch.beta2 = 5000.0;

	return plannersettings;
}

Eigen::Matrix3d rwInertia() {
	Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
	J(0, 0) = 0.067;
	J(1, 1) = 0.071;
	J(2, 2) = 0.069;
	return J;
}

void addRWTriad(Satellite& satellite) {
	satellite.addRW(Eigen::Vector3d::UnitX(), 0.001, 1e-5, 0.0, 0.02);
	satellite.addRW(Eigen::Vector3d::UnitY(), 0.001, 1e-5, 0.0, 0.02);
	satellite.addRW(Eigen::Vector3d::UnitZ(), 0.001, 1e-5, 0.0, 0.02);
}

// Prepared problem data for direct alilqr()/iLQR() calls (no trajOpt
// convergence-or-throw policy in the way).
struct ProblemData {
	int N = 0;
	Eigen::MatrixXd X;
	Eigen::MatrixXd U;
	Eigen::MatrixXd q_goal;
	Eigen::MatrixXd boresight;
	Eigen::VectorXd jtime_vec;
	Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> R;
	Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> V;
	Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> B;
	Eigen::Matrix<double, 3, limits::MAX_LENGTH_TRAJ> S;
	Eigen::Matrix<double, 1, limits::MAX_LENGTH_TRAJ> rho;
};

ProblemData prepareRWProblem(
	const PlannerSettings& settings,
	const Satellite& satellite,
	double tf_seconds,
	double dt_seconds,
	double w0_scale = 1.0
) {
	ProblemData p;
	p.N = static_cast<int>(tf_seconds / dt_seconds) + 1;
	REQUIRE(p.N <= limits::MAX_LENGTH_TRAJ);

	Eigen::Matrix<double, 1, limits::MAX_LENGTH_TRAJ> jtime_full;
	jtime_full.setZero();
	for (int k = 0; k < p.N; ++k) {
		jtime_full(k) = 0.22 + k * dt_seconds / SEC_PER_CENTURY;
	}

	p.q_goal = Eigen::MatrixXd::Zero(4, p.N);
	p.boresight = Eigen::MatrixXd::Zero(3, p.N);
	for (int k = 0; k < p.N; ++k) {
		p.q_goal.col(k) = Eigen::Vector4d(std::sqrt(2.0) / 2.0, 0.0, 0.0, std::sqrt(2.0) / 2.0);
		p.boresight.col(k) = Eigen::Vector3d::UnitX();
	}

	const Eigen::Vector3d r0(7000e3, 0.0, 0.0);
	const Eigen::Vector3d v0(0.0, 7.5e3, 0.0);

	p.R.setZero();
	p.V.setZero();
	p.B.setZero();
	p.S.setZero();
	p.rho.setZero();
	REQUIRE(orbits::generate_orbit(r0, v0, jtime_full, p.N, 1, 2, 0, 0, 0, p.R, p.V, p.B, p.S, p.rho));

	Satellite::VecX x0(satellite.stateDim());
	x0.setZero();
	x0.segment<3>(Satellite::AV_INDEX) = w0_scale * Eigen::Vector3d(-0.01, 0.02, 0.03);
	x0.segment<4>(Satellite::QUAT_INDEX) = Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);

	Eigen::MatrixXd X_full = Eigen::MatrixXd::Zero(satellite.stateDim(), limits::MAX_LENGTH_TRAJ);
	Eigen::MatrixXd U_full = Eigen::MatrixXd::Zero(satellite.controlDim(), limits::MAX_LENGTH_TRAJ);
	p.jtime_vec = jtime_full.leftCols(p.N).transpose();
	REQUIRE(optimizer::warm_start(
		settings, satellite, x0, p.jtime_vec, p.q_goal, p.boresight,
		p.N, p.R, p.V, p.B, p.S, p.rho, X_full, U_full));

	p.X = X_full.leftCols(p.N);
	p.U = U_full.leftCols(p.N);
	return p;
}

bool runALILQR(
	const PlannerSettings& settings,
	const Satellite& satellite,
	ProblemData& p,
	optimizer::ALILQRStatus& status,
	double& max_c,
	optimizer::ALILQRTelemetry& telemetry
) {
	return optimizer::alilqr(
		settings, 0, satellite,
		p.X, p.U,
		p.R.leftCols(p.N), p.V.leftCols(p.N), p.B.leftCols(p.N),
		p.S.leftCols(p.N), p.rho.leftCols(p.N),
		p.jtime_vec, p.boresight, p.q_goal,
		status, max_c, telemetry);
}

bool runInner(
	const PlannerSettings& settings,
	const Satellite& satellite,
	ProblemData& p,
	bool settle,
	optimizer::ILQRStatus& status,
	double& J,
	optimizer::ILQRTelemetry& telemetry
) {
	return optimizer::iLQR(
		settings, satellite,
		p.X, p.U,
		p.R.leftCols(p.N), p.V.leftCols(p.N), p.B.leftCols(p.N),
		p.S.leftCols(p.N), p.rho.leftCols(p.N),
		p.jtime_vec, p.boresight, p.q_goal,
		0,
		std::vector<Eigen::VectorXd>{}, std::vector<Eigen::VectorXd>{},
		settle, status, J, telemetry);
}

void requireFinite(const Eigen::MatrixXd& M) {
	for (int i = 0; i < M.rows(); ++i) {
		for (int k = 0; k < M.cols(); ++k) {
			REQUIRE(std::isfinite(M(i, k)));
		}
	}
}

} // namespace

// ----------------------------------------------------------------------------
// 1. Do-nothing guard + check order
// ----------------------------------------------------------------------------

TEST_CASE("Break gates: feasible start with zero-progress inner is never blessed (G15)",
          "[optimizer][break_gates][do_nothing]") {
	// Constraints are loose (default wmax) so the warm start is feasible from
	// the very first iterate, and the strict fast path is armed. The inner
	// solve is sabotaged: line search gets zero attempts, so no step can ever
	// be accepted, and the gradient test is disabled so the loose grad branch
	// cannot legitimately fire. Old gate (max_c <= tol -> Converged) would
	// declare success at outer_iter 0 off the warm start. The new gate must
	// not: no progress, no blessing.
	PlannerSettings settings = createRWPlannerSettings(10.0);
	settings.passes[0].linesearch.max_iters = 0;   // forward pass can never accept
	settings.passes[0].ilqr.grad_tol = 0.0;        // gradient test disabled
	settings.passes[0].auglag.constraint_tol_strict =
		0.5 * settings.passes[0].auglag.constraint_tol;  // fast path armed
	settings.passes[0].auglag.max_outer_iters = 4;
	settings.passes[0].auglag.max_total_iters = 0; // don't trip the budget here

	settings.constraints.sun_limit_angle = 0.0;    // sun constraint vacuous
	Satellite satellite(rwInertia(), settings);
	addRWTriad(satellite);
	ProblemData p = prepareRWProblem(settings, satellite, 200.0, 10.0);

	// Engineered feasible start: at rest, zero controls — every constraint
	// (rate, control saturation, momentum) is strictly satisfied.
	p.X.setZero();
	for (int k = 0; k < p.N; ++k) {
		p.X(Satellite::QUAT_INDEX, k) = 1.0;  // identity quaternion
	}
	p.U.setZero();

	optimizer::ALILQRStatus status = optimizer::ALILQRStatus::MaxOuterIterations;
	double max_c = 0.0;
	optimizer::ALILQRTelemetry telemetry;
	const bool ok = runALILQR(settings, satellite, p, status, max_c, telemetry);

	// The starting trajectory IS feasible — that's the trap.
	REQUIRE(max_c <= settings.passes[0].auglag.constraint_tol);
	// ...but nothing was optimized, so Converged must NOT be declared.
	REQUIRE_FALSE(ok);
	REQUIRE(status != optimizer::ALILQRStatus::Converged);
	REQUIRE(telemetry.converged_via == optimizer::ALConvergedVia::None);
	for (const auto& rec : telemetry.outer) {
		REQUIRE(rec.accepted_steps == 0);
		REQUIRE(rec.inner_status != optimizer::ILQRStatus::Converged);
	}
}

TEST_CASE("Break gates: optimal warm start exits Converged with zero accepted steps",
          "[optimizer][break_gates][check_order]") {
	// Check-order test: solve once to (gradient) convergence, then re-run the
	// inner solve on its own answer. The second call must exit Converged off
	// the first backward pass — zero accepted steps — and must NOT die as
	// RegularizationExceeded (the historical failure mode when the
	// convergence check lived after the forward pass).
	PlannerSettings settings = createRWPlannerSettings(10.0);
	settings.passes[0].ilqr.max_iters = 300;
	settings.passes[0].ilqr.grad_tol = 1e-2;
	settings.passes[0].ilqr.z_count_lim = 0;  // no stall exit: run to the gradient
	Satellite satellite(rwInertia(), settings);
	addRWTriad(satellite);
	ProblemData p = prepareRWProblem(settings, satellite, 200.0, 10.0);

	optimizer::ILQRStatus status = optimizer::ILQRStatus::MaxIterations;
	double J = 0.0;
	optimizer::ILQRTelemetry telemetry;

	// First solve: strict tier so the exit certificate includes the gradient.
	REQUIRE(runInner(settings, satellite, p, true, status, J, telemetry));
	REQUIRE(status == optimizer::ILQRStatus::Converged);
	const double g_first = telemetry.final_grad;
	REQUIRE(g_first >= 0.0);
	REQUIRE(g_first <= settings.passes[0].ilqr.grad_tol);

	// Second solve from the optimum: must exit immediately, before any
	// forward pass.
	optimizer::ILQRStatus status2 = optimizer::ILQRStatus::MaxIterations;
	double J2 = 0.0;
	optimizer::ILQRTelemetry telemetry2;
	const Eigen::MatrixXd X_opt = p.X;
	const Eigen::MatrixXd U_opt = p.U;
	REQUIRE(runInner(settings, satellite, p, true, status2, J2, telemetry2));

	REQUIRE(status2 == optimizer::ILQRStatus::Converged);
	REQUIRE(status2 != optimizer::ILQRStatus::RegularizationExceeded);
	REQUIRE(telemetry2.accepted_steps == 0);
	REQUIRE(telemetry2.iterations == 1);
	REQUIRE(telemetry2.break_reason == optimizer::ILQRBreakReason::GradientStationary);
	// Zero accepted steps means the trajectory is untouched.
	REQUIRE((p.X - X_opt).cwiseAbs().maxCoeff() == 0.0);
	REQUIRE((p.U - U_opt).cwiseAbs().maxCoeff() == 0.0);

	// Same warm start through the LOOSE tier: the disjunctive gradient
	// branch must fire at zero accepted steps as well.
	optimizer::ILQRStatus status3 = optimizer::ILQRStatus::MaxIterations;
	double J3 = 0.0;
	optimizer::ILQRTelemetry telemetry3;
	REQUIRE(runInner(settings, satellite, p, false, status3, J3, telemetry3));
	REQUIRE(status3 == optimizer::ILQRStatus::Converged);
	REQUIRE(telemetry3.accepted_steps == 0);
	REQUIRE(telemetry3.break_reason == optimizer::ILQRBreakReason::GradientIntermediate);
}

// ----------------------------------------------------------------------------
// 2. Settle discipline
// ----------------------------------------------------------------------------

TEST_CASE("Break gates: loose-tier exits while infeasible, strict conjunctive exit before Converged",
          "[optimizer][break_gates][settle]") {
	// Active AngularVelocity constraint: w0 is small (feasible at the fixed
	// initial knot) but the warm-start slew exceeds wmax = 0.012 rad/s
	// mid-trajectory, while a feasible slower slew exists (the 90 deg slew in
	// 200 s only needs ~0.008 rad/s average). The solve starts infeasible,
	// works through loose-tier solves, and may only declare Converged off a
	// strict-tier (settle) solve that itself converged while feasible.
	PlannerSettings settings = createRWPlannerSettings(10.0);
	settings.constraints.wmax = 0.012;
	settings.passes[0].auglag.max_outer_iters = 60;
	settings.passes[0].auglag.max_total_iters = 2000;
	// Disable the stall exit so the Converged declaration must come from the
	// strict conjunctive certificate itself (a Stalled settle-tier solve is
	// also accepted by the outer gate in general — OldPlanner parity — but
	// this test pins the conjunction path).
	settings.passes[0].ilqr.z_count_lim = 0;
	Satellite satellite(rwInertia(), settings);
	addRWTriad(satellite);
	ProblemData p = prepareRWProblem(settings, satellite, 200.0, 10.0, 0.1);

	optimizer::ALILQRStatus status = optimizer::ALILQRStatus::MaxOuterIterations;
	double max_c = 0.0;
	optimizer::ALILQRTelemetry telemetry;
	const bool ok = runALILQR(settings, satellite, p, status, max_c, telemetry);

	REQUIRE(ok);
	REQUIRE(status == optimizer::ALILQRStatus::Converged);
	REQUIRE(telemetry.converged_via == optimizer::ALConvergedVia::Settled);
	REQUIRE(max_c <= settings.passes[0].auglag.constraint_tol);

	REQUIRE(telemetry.outer.size() >= 2);

	// While iterates were infeasible the solves must run the loose tier.
	bool saw_loose_infeasible = false;
	for (const auto& rec : telemetry.outer) {
		if (!rec.settle && rec.max_c > settings.passes[0].auglag.constraint_tol) {
			saw_loose_infeasible = true;
		}
		if (!rec.settle) {
			// Loose solves may only exit via intermediate (or non-converged)
			// reasons — never via the strict conjunction.
			REQUIRE(rec.inner_break_reason != optimizer::ILQRBreakReason::StrictConjunction);
		}
	}
	REQUIRE(saw_loose_infeasible);

	// The final record is the settling solve: strict tier, inner Converged
	// via a strict-tier certificate, feasible.
	const auto& last = telemetry.outer.back();
	REQUIRE(last.settle);
	REQUIRE(last.inner_status == optimizer::ILQRStatus::Converged);
	const bool strict_reason =
		last.inner_break_reason == optimizer::ILQRBreakReason::StrictConjunction
		|| last.inner_break_reason == optimizer::ILQRBreakReason::GradientStationary;
	REQUIRE(strict_reason);
	REQUIRE(last.max_c <= settings.passes[0].auglag.constraint_tol);

	// min_outer_iters: convergence may not be declared before 3 completed
	// outer iterations.
	REQUIRE(static_cast<int>(telemetry.outer.size()) >= settings.passes[0].auglag.min_outer_iters);
}

// ----------------------------------------------------------------------------
// 3. Stall counter
// ----------------------------------------------------------------------------

TEST_CASE("Break gates: engineered slow grind exits via z_count with best trajectory",
          "[optimizer][break_gates][stall]") {
	// Strict tier with an unreachable gradient tolerance: every accepted step
	// near the optimum makes negligible relative progress, so the stall
	// counter must trip and return the best trajectory as Stalled (a usable
	// non-failure exit, but NOT Converged).
	PlannerSettings settings = createRWPlannerSettings(10.0);
	settings.passes[0].ilqr.max_iters = 100;
	settings.passes[0].ilqr.grad_tol = 1e-13;   // unreachable: conjunction can't fire
	settings.passes[0].ilqr.cost_tol = 1e-13;
	settings.passes[0].ilqr.z_count_lim = 3;
	Satellite satellite(rwInertia(), settings);
	addRWTriad(satellite);
	ProblemData p = prepareRWProblem(settings, satellite, 200.0, 10.0);

	const Eigen::MatrixXd X_warm = p.X;

	optimizer::ILQRStatus status = optimizer::ILQRStatus::MaxIterations;
	double J = 0.0;
	optimizer::ILQRTelemetry telemetry;
	const bool ok = runInner(settings, satellite, p, true, status, J, telemetry);

	REQUIRE(ok);                                          // not a failure
	REQUIRE(status == optimizer::ILQRStatus::Stalled);    // but not Converged
	REQUIRE(telemetry.break_reason == optimizer::ILQRBreakReason::Stalled);
	REQUIRE(telemetry.accepted_steps >= settings.passes[0].ilqr.z_count_lim);
	REQUIRE(telemetry.iterations < settings.passes[0].ilqr.max_iters);

	// The best trajectory is returned: finite and actually optimized
	// (different from the warm start).
	requireFinite(p.X);
	requireFinite(p.U);
	REQUIRE((p.X - X_warm).cwiseAbs().maxCoeff() > 0.0);
}

// ----------------------------------------------------------------------------
// 4. Budget / penalty statuses
// ----------------------------------------------------------------------------

TEST_CASE("Break gates: MaxTotalIterations is reachable and carries a trajectory",
          "[optimizer][break_gates][budgets]") {
	PlannerSettings settings = createRWPlannerSettings(10.0);
	settings.constraints.wmax = 0.02;                     // active constraint
	settings.passes[0].auglag.max_total_iters = 1;        // trip after first inner solve
	Satellite satellite(rwInertia(), settings);
	addRWTriad(satellite);
	ProblemData p = prepareRWProblem(settings, satellite, 200.0, 10.0);

	optimizer::ALILQRStatus status = optimizer::ALILQRStatus::MaxOuterIterations;
	double max_c = 0.0;
	optimizer::ALILQRTelemetry telemetry;
	const bool ok = runALILQR(settings, satellite, p, status, max_c, telemetry);

	REQUIRE_FALSE(ok);
	REQUIRE(status == optimizer::ALILQRStatus::MaxTotalIterations);
	REQUIRE(telemetry.total_inner_iterations >= settings.passes[0].auglag.max_total_iters);
	REQUIRE(telemetry.outer.size() == 1);
	requireFinite(p.X);
	requireFinite(p.U);
	// Telemetry carries the cost decomposition on this exit path too.
	REQUIRE(std::isfinite(telemetry.nominal_cost));
	REQUIRE(std::isfinite(telemetry.penalty_cost));
}

TEST_CASE("Break gates: PenaltyMaxReached is reachable and carries a trajectory",
          "[optimizer][break_gates][budgets]") {
	// Unsatisfiable feasibility tolerance + penalties capped at their initial
	// value: every mu saturates immediately while max_c stays large, and the
	// lambda updates hit lag_mult_max so the dual sequence stalls. The solve
	// must end honestly as PenaltyMaxReached instead of grinding out the
	// remaining outer budget.
	PlannerSettings settings = createRWPlannerSettings(10.0);
	settings.constraints.wmax = 1e-4;                     // effectively unsatisfiable
	settings.passes[0].auglag.penalty_init = 1e-1;
	settings.passes[0].auglag.penalty_max = 1e-1;         // saturated from the start
	settings.passes[0].auglag.lag_mult_max = 1.0;         // lambda clamps -> stalls
	settings.passes[0].auglag.max_outer_iters = 30;
	settings.passes[0].auglag.max_total_iters = 0;        // isolate the penalty exit
	Satellite satellite(rwInertia(), settings);
	addRWTriad(satellite);
	ProblemData p = prepareRWProblem(settings, satellite, 200.0, 10.0);

	optimizer::ALILQRStatus status = optimizer::ALILQRStatus::MaxOuterIterations;
	double max_c = 0.0;
	optimizer::ALILQRTelemetry telemetry;
	const bool ok = runALILQR(settings, satellite, p, status, max_c, telemetry);

	REQUIRE_FALSE(ok);
	REQUIRE(status == optimizer::ALILQRStatus::PenaltyMaxReached);
	REQUIRE(max_c > settings.passes[0].auglag.constraint_tol);
	// It must give up well before the outer budget (patience default 5 plus
	// the 2-iteration lambda-stall window).
	REQUIRE(static_cast<int>(telemetry.outer.size())
	        <= settings.passes[0].auglag.penalty_max_patience + 2);
	requireFinite(p.X);
	requireFinite(p.U);
}

// ----------------------------------------------------------------------------
// 5. decide() truth table (BREAK_GATE_DESIGN.md §8)
// ----------------------------------------------------------------------------
//
// The outer-loop verdict logic is a pure function over named predicates; this
// table enumerates EVERY combination of the boolean GateInputs × representative
// values of the analog inputs (max_c against the two tolerances,
// outer_iters_done against min_outer_iters, total_inner_iters against the
// global budget) × two config variants (all gates armed / fast path + budget
// disabled). The table IS the documentation: expectedVerdict() restates the
// design rule by rule, and each row's name is captured so a failure reads as
// "which documented rule broke".

namespace {

using optimizer::ALConvergedVia;
using optimizer::GateConfig;
using optimizer::GateDecision;
using optimizer::GateInputs;
using optimizer::Verdict;

GateInputs gateRow(bool inner_settled, bool inner_progress, bool ran_strict_tier,
                   bool all_mu_saturated, bool lambda_stalled, bool patience_exhausted,
                   double max_c, int outer_iters_done, long total_inner_iters) {
	GateInputs in;
	in.inner_settled = inner_settled;
	in.inner_progress = inner_progress;
	in.ran_strict_tier = ran_strict_tier;
	in.all_mu_saturated = all_mu_saturated;
	in.lambda_stalled = lambda_stalled;
	in.patience_exhausted = patience_exhausted;
	in.max_c = max_c;
	in.outer_iters_done = outer_iters_done;
	in.total_inner_iters = total_inner_iters;
	return in;
}

std::string gateRowName(const GateInputs& in, const GateConfig& cfg) {
	std::string s;
	s += in.inner_settled ? "settled " : "unsettled ";
	s += in.inner_progress ? "progress " : "no-progress ";
	s += in.ran_strict_tier ? "strict-tier " : "loose-tier ";
	s += in.all_mu_saturated ? "mu-saturated " : "mu-headroom ";
	s += in.lambda_stalled ? "lambda-stalled " : "lambda-live ";
	s += in.patience_exhausted ? "patience-out " : "patience-left ";
	s += "max_c=" + std::to_string(in.max_c);
	s += " outers=" + std::to_string(in.outer_iters_done);
	s += " total=" + std::to_string(in.total_inner_iters);
	s += " | cfg: tol=" + std::to_string(cfg.constraint_tol);
	s += " strict=" + std::to_string(cfg.constraint_tol_strict);
	s += " min_outer=" + std::to_string(cfg.min_outer_iters);
	s += " budget=" + std::to_string(cfg.max_total_iters);
	return s;
}

// Doc-as-code oracle: BREAK_GATE_DESIGN.md §6/§8 restated as a priority list
// of named rules (deliberately phrased differently from decide() itself).
GateDecision expectedVerdict(const GateInputs& in, const GateConfig& cfg) {
	// Feasibility classes (the eta side of the two-sided test).
	const bool feasible = (in.max_c <= cfg.constraint_tol);
	const bool deeply_feasible =
		(cfg.constraint_tol_strict > 0.0) && (in.max_c <= cfg.constraint_tol_strict);
	const bool duals_matured = (in.outer_iters_done >= cfg.min_outer_iters);
	const bool budget_spent =
		(cfg.max_total_iters > 0) && (in.total_inner_iters >= cfg.max_total_iters);

	// Rule 1 — fast path: deep feasibility may bypass dual maturation and the
	// settle discipline, but NEVER the inner-progress floor (G15: a literal
	// do-nothing trajectory is not a solution, however feasible).
	if (deeply_feasible && in.inner_progress) {
		return {Verdict::Converged, ALConvergedVia::FastPath};
	}
	// Rule 2 — settled declaration (two-sided, omega*/eta*-style): a feasible
	// iterate (eta*) produced by a strict-tier solve that itself settled
	// (omega*), after enough outer iterations for the duals to mature.
	if (feasible && in.ran_strict_tier && in.inner_settled && duals_matured) {
		return {Verdict::Converged, ALConvergedVia::Settled};
	}
	// Rule 3 — honest saturation exit: every mu pegged at its cap while still
	// infeasible, and the leftover (slow) method of multipliers has stopped
	// working — duals stalled for 2 consecutive non-contracting iterations,
	// or the patience window is exhausted.
	if (in.all_mu_saturated && !feasible
	    && (in.lambda_stalled || in.patience_exhausted)) {
		return {Verdict::PenaltyMaxReached, ALConvergedVia::None};
	}
	// Rule 4 — global inner-iteration budget: return honestly, never throw.
	if (budget_spent) {
		return {Verdict::MaxTotalIterations, ALConvergedVia::None};
	}
	// Otherwise — keep iterating: the lambda/mu update is committed and the
	// outer loop continues (an exhausted OUTER budget becomes
	// MaxOuterIterations in the caller, not a decide() verdict).
	return {Verdict::Continue, ALConvergedVia::None};
}

} // namespace

TEST_CASE("Break gates: decide() truth table is exhaustive over the gate predicates",
          "[optimizer][break_gates][decide]") {
	// Config A: every gate armed.
	GateConfig armed;
	armed.constraint_tol = 1e-3;
	armed.constraint_tol_strict = 1e-4;
	armed.min_outer_iters = 3;
	armed.max_total_iters = 1000;
	// Config B: opt-in gates off (fast path disabled, no global budget) —
	// the shipping defaults for those two knobs.
	GateConfig plain = armed;
	plain.constraint_tol_strict = 0.0;
	plain.max_total_iters = 0;

	// Representative analog values: one per feasibility class / maturity
	// side / budget side.
	const double max_c_values[] = {
		5e-5,  // deeply feasible (<= constraint_tol_strict when armed)
		5e-4,  // feasible, but not deeply
		5e-2,  // infeasible
	};
	const int outer_values[] = {1, 3};        // immature / matured duals
	const long total_values[] = {500, 1000};  // budget remaining / spent

	long rows = 0;
	for (const GateConfig& cfg : {armed, plain}) {
		for (int bits = 0; bits < (1 << 6); ++bits) {
			for (const double max_c : max_c_values) {
				for (const int outers : outer_values) {
					for (const long total : total_values) {
						const GateInputs in = gateRow(
							(bits & 1) != 0,   // inner_settled
							(bits & 2) != 0,   // inner_progress
							(bits & 4) != 0,   // ran_strict_tier
							(bits & 8) != 0,   // all_mu_saturated
							(bits & 16) != 0,  // lambda_stalled
							(bits & 32) != 0,  // patience_exhausted
							max_c, outers, total);
						++rows;
						const std::string row_name = gateRowName(in, cfg);
						CAPTURE(row_name);
						const GateDecision got = optimizer::decide(in, cfg);
						const GateDecision want = expectedVerdict(in, cfg);
						REQUIRE(optimizer::verdictName(got.verdict)
						        == optimizer::verdictName(want.verdict));
						REQUIRE(static_cast<int>(got.via) == static_cast<int>(want.via));
					}
				}
			}
		}
	}
	// 2 configs x 64 boolean combinations x 3 max_c x 2 outer x 2 budget.
	REQUIRE(rows == 2L * 64L * 3L * 2L * 2L);
}

TEST_CASE("Break gates: decide() landmark rows (the symptom classes, by name)",
          "[optimizer][break_gates][decide]") {
	GateConfig cfg;
	cfg.constraint_tol = 1e-3;
	cfg.constraint_tol_strict = 1e-4;
	cfg.min_outer_iters = 3;
	cfg.max_total_iters = 1000;
	const double deep = 5e-5;        // <= constraint_tol_strict
	const double feas = 5e-4;        // <= constraint_tol only
	const double infeas = 5e-2;      // > constraint_tol

	SECTION("G15 do-nothing: feasible warm start, zero progress — never blessed") {
		const auto d = optimizer::decide(
			gateRow(false, false, false, false, false, false, deep, 1, 10), cfg);
		REQUIRE(d.verdict == Verdict::Continue);
	}
	SECTION("fast path: deep feasibility + the progress floor converges immediately") {
		const auto d = optimizer::decide(
			gateRow(false, true, false, false, false, false, deep, 1, 10), cfg);
		REQUIRE(d.verdict == Verdict::Converged);
		REQUIRE(d.via == ALConvergedVia::FastPath);
	}
	SECTION("settled declaration: feasible, strict-tier, settled, matured duals") {
		const auto d = optimizer::decide(
			gateRow(true, true, true, false, false, false, feas, 3, 10), cfg);
		REQUIRE(d.verdict == Verdict::Converged);
		REQUIRE(d.via == ALConvergedVia::Settled);
	}
	SECTION("settle discipline: a feasible LOOSE-tier solve must settle first") {
		const auto d = optimizer::decide(
			gateRow(true, true, false, false, false, false, feas, 3, 10), cfg);
		REQUIRE(d.verdict == Verdict::Continue);
	}
	SECTION("dual maturation: feasible + settled but outer_iters_done < min_outer_iters") {
		const auto d = optimizer::decide(
			gateRow(true, true, true, false, false, false, feas, 1, 10), cfg);
		REQUIRE(d.verdict == Verdict::Continue);
	}
	SECTION("unsettled strict-tier solve keeps iterating even when feasible") {
		const auto d = optimizer::decide(
			gateRow(false, true, true, false, false, false, feas, 3, 10), cfg);
		REQUIRE(d.verdict == Verdict::Continue);
	}
	SECTION("mu-saturation grind ends honestly once the duals stall") {
		const auto d = optimizer::decide(
			gateRow(false, true, false, true, true, false, infeas, 4, 10), cfg);
		REQUIRE(d.verdict == Verdict::PenaltyMaxReached);
	}
	SECTION("mu-saturation grind ends honestly when patience runs out") {
		const auto d = optimizer::decide(
			gateRow(false, true, false, true, false, true, infeas, 4, 10), cfg);
		REQUIRE(d.verdict == Verdict::PenaltyMaxReached);
	}
	SECTION("saturated but duals alive and patience left: the multipliers may still close it") {
		const auto d = optimizer::decide(
			gateRow(false, true, false, true, false, false, infeas, 4, 10), cfg);
		REQUIRE(d.verdict == Verdict::Continue);
	}
	SECTION("saturated while FEASIBLE is not a failure (settling may still succeed)") {
		const auto d = optimizer::decide(
			gateRow(false, true, false, true, true, true, feas, 4, 10), cfg);
		REQUIRE(d.verdict == Verdict::Continue);
	}
	SECTION("global budget exhaustion returns MaxTotalIterations, never throws") {
		const auto d = optimizer::decide(
			gateRow(false, true, false, false, false, false, infeas, 4, 1000), cfg);
		REQUIRE(d.verdict == Verdict::MaxTotalIterations);
	}
	SECTION("convergence outranks the budget on the same iteration") {
		const auto d = optimizer::decide(
			gateRow(true, true, true, false, false, false, feas, 3, 1000), cfg);
		REQUIRE(d.verdict == Verdict::Converged);
		REQUIRE(d.via == ALConvergedVia::Settled);
	}
	SECTION("the saturation exit outranks the budget on the same iteration") {
		const auto d = optimizer::decide(
			gateRow(false, true, false, true, true, false, infeas, 4, 1000), cfg);
		REQUIRE(d.verdict == Verdict::PenaltyMaxReached);
	}
}
