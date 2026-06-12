"""Python twin of test_break_gates.cpp — convergence/break-gate redesign
(BREAK_GATE_DESIGN.md sections 5-7), exercised through the saltro_py.alilqr
telemetry dict.

Covers the do-nothing guard (G15), settle-tier discipline, and the new
terminal statuses (MaxTotalIterations / PenaltyMaxReached). The inner-solver
check-order and stall-counter tests live in the C++ twin only (iLQR is not
bound to Python)."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(ROOT / "tests" / "debug" / "optimizer" / "configs"))

import saltro_py
from sat_0_3_rw import create_satellite as create_satellite_rw

SEC_PER_CENTURY = 36525.0 * 86400.0


def create_planner_settings(dt_seconds: float) -> saltro_py.PlannerSettings:
    ps = saltro_py.PlannerSettings()
    ps.init_traj.initcontroller = 2
    ps.num_passes = 1
    ps.passes[0].dt = dt_seconds
    ps.passes[0].ilqr.cost_tol = 1e-5
    ps.passes[0].ilqr.max_iters = 20
    ps.passes[0].auglag.max_outer_iters = 30
    ps.passes[0].auglag.constraint_tol = 1e-3

    cost = ps.passes[0].cost
    cost.angle = 1.0
    cost.ang_vel = 1e1
    cost.control_mult = 1.0
    cost.mtq_control_weight = 1e-2
    cost.rw_control_weight = 1.0
    cost.magic_control_weight = 0.0
    cost.rw_AM_weight = 0.0
    cost.rw_stic_weight = 0.0
    cost.RWh_max_mult = 0.0
    cost.RWh_stiction_mult = 0.0
    cost.RWh_ok_mult = 0.0
    cost.angle_N = 0.0
    cost.ang_vel_N = 0.0
    cost.ang_cost_func_type = 3
    cost.use_cost_hess = True

    ps.passes[0].reg.reg_init = 1e-6
    ps.passes[0].reg.reg_max = 1e10
    ps.passes[0].reg.reg_scale = 10.0
    ps.passes[0].linesearch.max_iters = 24
    ps.passes[0].linesearch.beta1 = 1e-10
    ps.passes[0].linesearch.beta2 = 5000.0
    return ps


def prepare_problem(ps, sat, tf_seconds, dt_seconds, w0_scale=1.0):
    n = int(round(tf_seconds / dt_seconds)) + 1
    jt = 0.22 + np.arange(n) * (dt_seconds / SEC_PER_CENTURY)
    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, 7.5e3, 0.0])
    ok, R, V, B, S, rho = saltro_py.generate_orbit(r0, v0, jt, 1, 2, 0, 0, 0)
    assert ok
    R = np.asarray(R)[:, :n]
    V = np.asarray(V)[:, :n]
    B = np.asarray(B)[:, :n]
    S = np.asarray(S)[:, :n]
    rho = np.asarray(rho).reshape(1, -1)[:, :n]
    qg = np.tile(np.array([[np.sqrt(2) / 2], [0.0], [0.0], [np.sqrt(2) / 2]]), (1, n))
    bs = np.tile(np.array([[1.0], [0.0], [0.0]]), (1, n))

    w0 = w0_scale * np.array([-0.01, 0.02, 0.03])
    x0 = np.hstack((w0, [1.0, 0.0, 0.0, 0.0], np.zeros(3)))
    okw, X, U = saltro_py.warm_start(ps, sat, x0, jt, qg, bs, R, V, B, S, rho)
    assert okw
    X = np.asarray(X)[:, :n]
    U = np.asarray(U)[:, :n]
    return dict(n=n, jt=jt, R=R, V=V, B=B, S=S, rho=rho, qg=qg, bs=bs, X=X, U=U)


def run_alilqr(ps, sat, p):
    return saltro_py.alilqr(
        ps, 0, sat, p["X"], p["U"], p["R"], p["V"], p["B"], p["S"], p["rho"],
        p["jt"], p["bs"], p["qg"],
    )


# ----------------------------------------------------------------------------
# 1. Do-nothing guard (G15)
# ----------------------------------------------------------------------------


def test_feasible_start_with_zero_progress_inner_is_never_blessed():
    """A feasible starting trajectory whose inner solve cannot take a single
    step (line search disabled) must NOT come back 'converged' — even with the
    constraint_tol_strict fast path armed."""
    ps = create_planner_settings(10.0)
    ps.passes[0].linesearch.max_iters = 0     # forward pass can never accept
    ps.passes[0].ilqr.grad_tol = 0.0          # gradient test disabled
    ps.passes[0].auglag.constraint_tol_strict = (
        0.5 * ps.passes[0].auglag.constraint_tol
    )
    ps.passes[0].auglag.max_outer_iters = 4
    ps.passes[0].auglag.max_total_iters = 0
    ps.constraints.sun_limit_angle = 0.0      # sun constraint vacuous
    sat = create_satellite_rw(ps)
    p = prepare_problem(ps, sat, 200.0, 10.0)

    # Engineered feasible start: at rest, zero controls.
    p["X"] = np.zeros_like(p["X"])
    p["X"][3, :] = 1.0                        # identity quaternion
    p["U"] = np.zeros_like(p["U"])

    ok, _X, _U, status, max_c, tel = run_alilqr(ps, sat, p)

    assert max_c <= ps.passes[0].auglag.constraint_tol  # the trap is real
    assert not ok
    assert status != "converged"
    assert tel["converged_via"] == "none"
    for rec in tel["outer"]:
        assert rec["accepted_steps"] == 0
        assert rec["inner_status"] != "converged"


# ----------------------------------------------------------------------------
# 2. Settle discipline
# ----------------------------------------------------------------------------


def test_settle_discipline_loose_then_strict_conjunctive_before_converged():
    """The solve must show loose-tier exits while infeasible and a strict
    conjunctive (settle-tier) inner exit on the iteration that declares
    Converged."""
    ps = create_planner_settings(10.0)
    ps.constraints.wmax = 0.012               # active mid-slew, feasible overall
    ps.passes[0].auglag.max_outer_iters = 60
    ps.passes[0].auglag.max_total_iters = 2000
    # Disable the stall exit so Converged must come from the strict
    # conjunctive certificate itself (Stalled settle-tier solves are also
    # accepted by the outer gate in general; this test pins the conjunction).
    ps.passes[0].ilqr.z_count_lim = 0
    sat = create_satellite_rw(ps)
    p = prepare_problem(ps, sat, 200.0, 10.0, w0_scale=0.1)

    ok, _X, _U, status, max_c, tel = run_alilqr(ps, sat, p)

    assert ok
    assert status == "converged"
    assert tel["converged_via"] == "settled"
    assert max_c <= ps.passes[0].auglag.constraint_tol

    outer = tel["outer"]
    assert len(outer) >= ps.passes[0].auglag.min_outer_iters

    saw_loose_infeasible = False
    for rec in outer:
        if not rec["settle"] and rec["max_c"] > ps.passes[0].auglag.constraint_tol:
            saw_loose_infeasible = True
        if not rec["settle"]:
            assert rec["break_reason"] != "strict_conjunction"
    assert saw_loose_infeasible

    last = outer[-1]
    assert last["settle"]
    assert last["inner_status"] == "converged"
    assert last["break_reason"] in ("strict_conjunction", "gradient_stationary")
    assert last["max_c"] <= ps.passes[0].auglag.constraint_tol


# ----------------------------------------------------------------------------
# 4. Budget / penalty statuses
# ----------------------------------------------------------------------------


def test_max_total_iterations_reachable_and_carries_trajectory():
    ps = create_planner_settings(10.0)
    ps.constraints.wmax = 0.02                # active constraint
    ps.passes[0].auglag.max_total_iters = 1   # trip after the first inner solve
    sat = create_satellite_rw(ps)
    p = prepare_problem(ps, sat, 200.0, 10.0)

    ok, X, U, status, _max_c, tel = run_alilqr(ps, sat, p)

    assert not ok
    assert status == "max_total_iterations"
    assert tel["total_inner_iterations"] >= 1
    assert len(tel["outer"]) == 1
    assert np.all(np.isfinite(X))
    assert np.all(np.isfinite(U))
    assert np.isfinite(tel["nominal_cost"])
    assert np.isfinite(tel["penalty_cost"])


def test_penalty_max_reached_reachable_and_carries_trajectory():
    ps = create_planner_settings(10.0)
    ps.constraints.wmax = 1e-4                # effectively unsatisfiable
    ps.passes[0].auglag.penalty_init = 1e-1
    ps.passes[0].auglag.penalty_max = 1e-1    # saturated from the start
    ps.passes[0].auglag.lag_mult_max = 1.0    # lambda clamps -> stalls
    ps.passes[0].auglag.max_outer_iters = 30
    ps.passes[0].auglag.max_total_iters = 0
    sat = create_satellite_rw(ps)
    p = prepare_problem(ps, sat, 200.0, 10.0)

    ok, X, U, status, max_c, tel = run_alilqr(ps, sat, p)

    assert not ok
    assert status == "penalty_max_reached"
    assert max_c > ps.passes[0].auglag.constraint_tol
    assert len(tel["outer"]) <= ps.passes[0].auglag.penalty_max_patience + 2
    assert np.all(np.isfinite(X))
    assert np.all(np.isfinite(U))


def test_telemetry_per_family_max_c_and_cost_share():
    """Telemetry must expose per-family max violation and the
    nominal-vs-penalty cost decomposition."""
    ps = create_planner_settings(10.0)
    ps.constraints.wmax = 0.02
    ps.passes[0].auglag.max_outer_iters = 3
    sat = create_satellite_rw(ps)
    p = prepare_problem(ps, sat, 200.0, 10.0)

    _ok, _X, _U, _status, max_c, tel = run_alilqr(ps, sat, p)

    fam = tel["max_c_family"]
    assert len(fam) == 7  # ConstraintFamily::NumFamilies
    # The overall max violation is one of the family maxima.
    assert abs(max(fam) - max_c) < 1e-12
    assert np.isfinite(tel["nominal_cost"])
    assert np.isfinite(tel["penalty_cost"])
    # Per-outer records carry the gate-debugging fields.
    for rec in tel["outer"]:
        for key in ("settle", "inner_status", "break_reason",
                    "inner_iterations", "accepted_steps", "max_c",
                    "final_grad", "last_delta_J", "min_delta_J"):
            assert key in rec
