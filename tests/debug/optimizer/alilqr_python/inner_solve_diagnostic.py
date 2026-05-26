"""Inner-solve diagnostic harness.

High-info debug test for diagnosing inner-solve failure modes
(ls_attempts_exceeded, reg_exceeded, slow convergence, etc.).

Two modes:
  * Driver (default): spawns subprocess with SALTRO_FP_VERBOSE=1, captures
    stderr (per-LS-trial detail), reads python telemetry via stdout JSON,
    merges into a per-iter table.
  * Inner (`--inner SCENARIO`): runs the named scenario through the synced
    python alilqr wrapper, dumps telemetry to stdout as JSON.

Usage:
  ./venv/bin/python tests/debug/optimizer/alilqr_python/inner_solve_diagnostic.py
  ./venv/bin/python tests/debug/optimizer/alilqr_python/inner_solve_diagnostic.py --scenario 12_omega_5x_ict1e3
  ./venv/bin/python tests/debug/optimizer/alilqr_python/inner_solve_diagnostic.py --list

Add scenarios by appending to SCENARIOS dict at the bottom.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np


# ============================================================================
# Scenario definitions
# ============================================================================

@dataclass
class Scenario:
    name: str
    sat_type: str = "3_1"   # 3_1, 3_0, 0_3, 3_3
    dt: float = 10.0
    time_s: float = 1000.0
    omega0: tuple = (0.01, 0.01, 0.01)
    goal_angle_deg: float = 90.0
    initcontroller: int = 1
    # Cost weights (raw, not via setBalanced — keeps existing tuning)
    angle: float = 1e4
    ang_vel: float = 1e2
    mtq_cw: float = 1e-1
    rw_cw: float = 1.0
    angle_N_mult: float = 1.0  # final angle_N = angle_N_mult * angle
    ang_cost_func_type: int = 3
    use_cost_hess: bool = True
    cost_hess_gauss_newton: bool = True
    crossterm: str = "newpath"  # "newpath" or "legacy"
    crossterm_beta: float = 0.3
    roll_ratio: float = 0.05  # vec mode: 0.05; quat: 1.0
    is_vec: bool = True
    # iLQR + AL
    cost_tol: float = 1e-6
    ilqr_cost_tol: float = 1.0
    grad_tol: float = 0.0
    max_iters: int = 200
    max_outer: int = 30
    constraint_tol: float = 1e-3
    # LS + reg
    ls_max_iters: int = 24
    reg_init: float = 0.0
    reg_max: float = 1e30
    reg_scale: float = 1.6
    # Eigen-modification regularization (relative-floor on λ_max)
    use_eigen_modification: bool = False
    eigen_reg_use_relative_floor: bool = False
    eigen_reg_condition_cap: float = 1e-6  # κ cap = 1/cap; only used if rel_floor on
    eigen_reg_use_abs: bool = False
    eigen_reg_mimic_uniform_trigger: bool = False
    eigen_reg_add_mode: bool = False
    # Second-order AL Hessian (μ·c·c_xx, μ·c·c_uu, μ·c·c_ux additions)
    use_constraint_hess: bool = False
    # Kink-aware dual update: λ_new = max(0, λ + μ·max(0, c))
    # Prevents λ from decaying when c < 0 → gate stays active via λ → no
    # C¹-not-C² kink in subsequent inner solves at constraint boundaries.
    kink_fix: bool = False
    # Frozen active set: snapshot active mask at outer start, hold during inner.
    # Eliminates mid-inner gate flipping (no C² kinks). Activation/deactivation
    # discovery deferred to outer boundary.
    frozen_active_set: bool = False
    # Lambda floor: keep λ ≥ ε·μ for previously-active constraints. Preserves
    # smooth cost surface across c=0 by preventing λ → 0 transitions.
    lambda_floor_eps: float = 0.0
    # Gradient check: at selected iters, FD-verify BP's deltaV(0) against
    # actual cost gradient via open-loop perturbation. Ratio close to 1 → BP
    # gradient matches the cost we're minimizing.
    grad_check: bool = False
    # Toggles
    use_spike: bool = False  # spike removal off for clean diagnostic
    description: str = ""


SCENARIOS: dict[str, Scenario] = {}


def _add(s: Scenario):
    SCENARIOS[s.name] = s


# ---- baseline + ict A/B ----------------------------------------------------
_add(Scenario(
    name="00_baseline_ict1",
    description="Vec t3 newpath baseline at ict=1.0 (synced sweep config)",
))
_add(Scenario(
    name="00_baseline_ict1e-3",
    ilqr_cost_tol=1e-3,
    description="Vec t3 newpath baseline at ict=1e-3 (val sweep — converges but slow)",
))

# ---- high-omega: regression scenarios -------------------------------------
_add(Scenario(
    name="12_omega_5x_ict1",
    omega0=(0.05, 0.05, 0.05),
    max_iters=200,  # diagnostic: cap to bound runtime; full convergence not needed
    description="ω0 = 5× nominal, ict=1.0 (synced — converged 9.0°)",
))
_add(Scenario(
    name="12_omega_5x_ict1e-3",
    omega0=(0.05, 0.05, 0.05),
    ilqr_cost_tol=1e-3,
    max_iters=200,
    description="ω0 = 5× nominal, ict=1e-3 (val — REGRESSED to 84.9° / ls_exc)",
))
_add(Scenario(
    name="13_omega_10x_ict1",
    omega0=(0.10, 0.10, 0.10),
    description="ω0 = 10× nominal, ict=1.0 (synced — ls_exc 9.3°)",
))
_add(Scenario(
    name="13_omega_10x_ict1e-3",
    omega0=(0.10, 0.10, 0.10),
    ilqr_cost_tol=1e-3,
    description="ω0 = 10× nominal, ict=1e-3 (val — converged 4.3° / 1265 iters)",
))

# ---- long duration --------------------------------------------------------
_add(Scenario(
    name="11_long_dt30_ict1",
    dt=30.0,
    time_s=3000.0,
    description="dt=30s, T=3000s, ict=1.0 (synced — 10° / fast)",
))
_add(Scenario(
    name="11_long_dt30_ict1e-3",
    dt=30.0,
    time_s=3000.0,
    ilqr_cost_tol=1e-3,
    description="dt=30s, T=3000s, ict=1e-3 (val — 3.4° / 781 iters)",
))

# ---- short duration -------------------------------------------------------
_add(Scenario(
    name="10_short_dt1_ict1",
    dt=1.0,
    time_s=100.0,
    description="dt=1s, T=100s, ict=1.0",
))

# ---- 4-way A/B for spike × ict interaction on 12_omega_5x ----------------
# This isolates whether the synced→val regression (3°→85°PE at ict=1.0→1e-3)
# is a Q_uu issue (would persist without spike) or a spike-removal × ict
# interaction (only emerges with spike on).
_add(Scenario(
    name="12ab_5x_ict1_nospike",
    omega0=(0.05, 0.05, 0.05),
    use_spike=False,
    max_iters=300, max_outer=6,
    description="A/B: 12_5x ict=1.0 NO spike (raw failure mode)",
))
_add(Scenario(
    name="12ab_5x_ict1e-3_nospike",
    omega0=(0.05, 0.05, 0.05),
    ilqr_cost_tol=1e-3,
    use_spike=False,
    max_iters=300, max_outer=6,
    description="A/B: 12_5x ict=1e-3 NO spike (raw failure mode)",
))
_add(Scenario(
    name="12ab_5x_ict1_spike",
    omega0=(0.05, 0.05, 0.05),
    use_spike=True,
    max_iters=300, max_outer=6,
    description="A/B: 12_5x ict=1.0 WITH spike (production config — synced 3°PE)",
))
_add(Scenario(
    name="12ab_5x_ict1e-3_spike",
    omega0=(0.05, 0.05, 0.05),
    ilqr_cost_tol=1e-3,
    use_spike=True,
    max_iters=300, max_outer=6,
    description="A/B: 12_5x ict=1e-3 WITH spike (production — val regressed 85°PE)",
))

# ---- relative-floor regularization curiosity test ------------------------
# Q_uu spectrum showed κ≈1e12 with λ_min≈+25, λ_max≈1e14. Mode A is
# ill-conditioning, not indefiniteness. Toggle the relative-floor eigen-mod
# regularization to bound κ and see if LS-rescue pattern collapses.
_add(Scenario(
    name="12cure_5x_ict1e-3_spike_relfloor",
    omega0=(0.05, 0.05, 0.05),
    ilqr_cost_tol=1e-3,
    use_spike=True,
    max_iters=300, max_outer=6,
    use_eigen_modification=True,
    eigen_reg_use_relative_floor=True,
    eigen_reg_condition_cap=1e-6,  # κ cap = 1e6
    description="CURE: 12_5x ict=1e-3 spike + relative-floor reg (κ cap 1e6)",
))
_add(Scenario(
    name="12cure_5x_ict1e-3_spike_chess",
    omega0=(0.05, 0.05, 0.05),
    ilqr_cost_tol=1e-3,
    use_spike=True,
    max_iters=300, max_outer=6,
    use_constraint_hess=True,
    description="CURE: 12_5x ict=1e-3 spike + use_constraint_hess (μ·c·c_xx)",
))
_add(Scenario(
    name="12cure_5x_ict1e-3_spike_kink",
    omega0=(0.05, 0.05, 0.05),
    ilqr_cost_tol=1e-3,
    use_spike=True,
    max_iters=300, max_outer=6,
    kink_fix=True,
    description="CURE: 12_5x ict=1e-3 spike + kink-aware dual (λ stays positive when c<0)",
))
_add(Scenario(
    name="12cure_5x_ict1e-3_spike_frozen",
    omega0=(0.05, 0.05, 0.05),
    ilqr_cost_tol=1e-3,
    use_spike=True,
    max_iters=300, max_outer=6,
    frozen_active_set=True,
    description="CURE: 12_5x ict=1e-3 spike + frozen active set (μ=0 for inactive during inner)",
))
_add(Scenario(
    name="12cure_5x_ict1e-3_spike_frozen_full",
    omega0=(0.05, 0.05, 0.05),
    ilqr_cost_tol=1e-3,
    use_spike=True,
    max_iters=200, max_outer=30,  # production max_outer
    frozen_active_set=True,
    description="CURE: frozen AS with production max_outer=30 — does AL catch up on constraints?",
))
_add(Scenario(
    name="12cure_5x_ict1e-3_spike_lamfloor",
    omega0=(0.05, 0.05, 0.05),
    ilqr_cost_tol=1e-3,
    use_spike=True,
    max_iters=300, max_outer=6,
    lambda_floor_eps=1e-4,
    description="CURE: 12_5x ict=1e-3 spike + λ floor at ε·μ (ε=1e-4) for ever-active constraints",
))
_add(Scenario(
    name="12gc_5x_ict1e-3_spike_gradcheck",
    omega0=(0.05, 0.05, 0.05),
    ilqr_cost_tol=1e-3,
    use_spike=True,
    max_iters=50, max_outer=3,  # fast — only need a few iters to see ratio
    grad_check=True,
    description="GRADCHECK: FD-verify BP deltaV(0) against actual cost gradient",
))


# ============================================================================
# Inner mode — runs scenario, dumps telemetry as JSON to stdout
# ============================================================================

def _build_planner(s: Scenario, saltro_py):
    ps = saltro_py.PlannerSettings()
    ps.init_traj.initcontroller = s.initcontroller
    ps.num_passes = 1
    p = ps.passes[0]
    p.dt = s.dt
    p.ilqr.cost_tol = s.cost_tol
    p.ilqr.ilqr_cost_tol = s.ilqr_cost_tol
    p.ilqr.max_iters = s.max_iters
    p.ilqr.grad_tol = s.grad_tol
    p.auglag.max_outer_iters = s.max_outer
    p.auglag.constraint_tol = s.constraint_tol

    c = p.cost
    c.angle = s.angle; c.ang_vel = s.ang_vel
    c.angle_N = s.angle * s.angle_N_mult; c.ang_vel_N = s.ang_vel
    c.control_mult = 1.0
    c.mtq_control_weight = s.mtq_cw; c.rw_control_weight = s.rw_cw
    c.ang_cost_func_type = s.ang_cost_func_type
    c.use_cost_hess = s.use_cost_hess
    c.cost_hess_gauss_newton = s.cost_hess_gauss_newton
    c.ang_vel_roll_ratio = s.roll_ratio
    if s.crossterm == "newpath":
        c.ang_vel_err_dir_ratio = s.crossterm_beta
        c.ang_vel_err_dir = 0.0
    else:
        c.ang_vel_err_dir = s.ang_vel
        c.ang_vel_err_dir_ratio = 0.0
    c.setTerminalEmphasis(100.0)

    p.reg.reg_init = s.reg_init; p.reg.reg_max = s.reg_max
    p.reg.reg_scale = s.reg_scale
    p.reg.use_eigen_modification = s.use_eigen_modification
    p.reg.eigen_reg_use_relative_floor = s.eigen_reg_use_relative_floor
    p.reg.eigen_reg_condition_cap = s.eigen_reg_condition_cap
    p.reg.eigen_reg_use_abs = s.eigen_reg_use_abs
    p.reg.eigen_reg_mimic_uniform_trigger = s.eigen_reg_mimic_uniform_trigger
    p.reg.eigen_reg_add_mode = s.eigen_reg_add_mode
    p.reg.use_constraint_hess = s.use_constraint_hess
    p.linesearch.max_iters = s.ls_max_iters
    p.linesearch.beta1 = 1e-10; p.linesearch.beta2 = 5000.0
    return ps


def run_inner(scenario_name: str):
    """Run the scenario through alilqr.py wrapper, dump telemetry to stdout."""
    ROOT = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(ROOT / "build"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))

    import saltro_py
    from trajOpt import trajOpt
    from sat_3_1_hybrid import create_satellite as create_3_1
    from sat_3_0_mtq import create_satellite as create_3_0
    from sat_0_3_rw import create_satellite as create_0_3
    from sat_3_3_hybrid import create_satellite as create_3_3

    SAT_FACTORIES = {"3_1": create_3_1, "3_0": create_3_0,
                     "0_3": create_0_3, "3_3": create_3_3}

    if scenario_name not in SCENARIOS:
        print(json.dumps({"error": f"unknown scenario: {scenario_name}"}))
        return 1
    s = SCENARIOS[scenario_name]

    ps = _build_planner(s, saltro_py)
    sat = SAT_FACTORIES[s.sat_type](ps)

    nRW = sat.numRW
    w0 = np.array(s.omega0)
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    h0 = np.zeros(nRW)
    x0 = np.hstack([w0, q0, h0])

    ang = np.radians(s.goal_angle_deg)
    qg = np.array([np.cos(ang/2), 0, 0, np.sin(ang/2)])
    qgoal = np.tile(qg[:, None], (1, 2))
    bs = np.array([[1, 1], [0, 0], [0, 0]], dtype=float)
    r0 = np.array([7e6, 0, 0]); v0 = np.array([0, 7.5e3, 0])
    jtime = np.array([0.22, 0.22 + s.time_s / (36525.0 * 86400.0)])

    spike_cfg = None
    if s.use_spike:
        # Match wide_test_runner production config
        spike_cfg = {
            "start_at_iter": 2, "max_intervention_iters": 10000,
            "blend_len": 30, "goal_switch_buffer": 15, "min_consecutive": 7,
            "exit_fudge": 2.0, "min_prior_decrease_knots": 5, "min_spike_ratio": 2.0,
            "kp_q": 0.3, "kd_w": 2.0, "rw_scale": -1.0, "omega_max": 0.30,
            "verbose": False, "constraint_gate_ratio": 0.0,
        }

    try:
        X, U, stop, snaps, trans, dt_val, ctol_cfg, elapsed = trajOpt(
            ps, sat, x0, r0, v0, jtime, qgoal, bs, debug=True, spike_removal_cfg=spike_cfg,
        )
    except Exception as e:
        print(json.dumps({"error": f"trajOpt exception: {type(e).__name__}: {e}"}))
        return 1

    # Compute PE_fin
    pe = np.array([2*np.degrees(np.arccos(min(abs(float(np.dot(X[3:7, k], qg))), 1)))
                   for k in range(X.shape[1])])

    # Filter inner transitions only (exclude AL outer-summary entries)
    inner_trans = [t for t in trans if "iter" in t]
    outer_trans = [t for t in trans if "outer_iter" in t and "max_constraint_violation" in t]

    # Drop heavy fields from snapshots — keep just J + cost components per iter
    # for plotting; X/U trajectories get too big across many iterations.
    snap_summary = []
    for snap in snaps:
        snap_summary.append({
            "outer_iter": snap.get("outer_iter", 0),
            "J": float(snap.get("J", 0.0)),
            "components_total": {
                k: float(np.sum(v)) for k, v in snap.get("components", {}).items()
            },
        })

    out = {
        "scenario": scenario_name,
        "stop": stop,
        "elapsed_s": float(elapsed),
        "iters": len(snaps),
        "PE_fin": float(pe[-1]),
        "PE_mean": float(pe.mean()),
        "inner_trans": inner_trans,
        "outer_trans": outer_trans,
        "snap_summary": snap_summary,
    }

    def _encode(o):
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"unhandled: {type(o)}")
    print(json.dumps(out, default=_encode))
    return 0


# ============================================================================
# Driver mode — spawns subprocess, parses FP verbose, merges
# ============================================================================

FP_LS_REJECT_RE = re.compile(
    r"\[FP\] ls_reject alpha=([-\deE.+]+) J_prev=([-\deE.+]+) J_new=([-\deE.+]+) "
    r"dJ=([-\deE.+]+) z=([-\deE.+nan]+)"
)
FP_ROLLOUT_FAIL_RE = re.compile(
    r"\[FP\] rollout_fail alpha=([-\deE.+]+) k=(\d+) reason=(\S+)"
)
FP_ALL_FAIL_RE = re.compile(r"\[FP\] ALL TRIALS FAILED")


def parse_fp_stderr(stderr_text: str) -> list[dict]:
    """Parse per-LS-trial events from FP_VERBOSE stderr.

    Returns a flat list. Each event records what happened on one LS trial.
    Events between successive 'ALL TRIALS FAILED' or rollouts represent one
    forward-pass call. Caller buckets into per-FP-call groups by counting
    ALL_TRIALS_FAILED (which marks end of a failed FP).
    """
    events = []
    for line in stderr_text.splitlines():
        m = FP_LS_REJECT_RE.search(line)
        if m:
            events.append({
                "event": "ls_reject",
                "alpha": float(m.group(1)),
                "J_prev": float(m.group(2)),
                "J_new": float(m.group(3)),
                "dJ": float(m.group(4)),
                "z": float("nan") if "nan" in m.group(5) else float(m.group(5)),
            })
            continue
        m = FP_ROLLOUT_FAIL_RE.search(line)
        if m:
            events.append({
                "event": "rollout_fail",
                "alpha": float(m.group(1)),
                "k": int(m.group(2)),
                "reason": m.group(3),
            })
            continue
        if FP_ALL_FAIL_RE.search(line):
            events.append({"event": "all_failed"})
    return events


def bucket_fp_events(events: list[dict]) -> list[dict]:
    """Group LS-trial events into per-FP-call buckets.

    A bucket ends on either:
      - an `all_failed` (failed FP — entire LS exhausted)
      - the next observed ls_accept (which we don't log) — handled by
        treating each new alpha=1.0 as the start of a new FP call.

    Since SALTRO_FP_VERBOSE only logs rejects + ALL_FAILED, accepted FPs
    leave gaps. We use the alpha=1.0 reset signal to detect new FP calls.
    """
    buckets = []
    cur: list[dict] = []
    for ev in events:
        if ev.get("event") == "all_failed":
            cur.append(ev)
            buckets.append(cur)
            cur = []
            continue
        # alpha=1.0 with non-empty cur means we started a new FP without a
        # terminal ALL_FAILED — previous FP must have accepted on a later trial.
        if cur and "alpha" in ev and ev["alpha"] == 1.0:
            buckets.append(cur)
            cur = []
        cur.append(ev)
    if cur:
        buckets.append(cur)
    return buckets


def merge(inner_trans: list[dict], fp_buckets: list[dict]) -> list[dict]:
    """Pair each inner-iter transition with its FP bucket.

    The python wrapper calls forward_pass once per accepted iter (or per
    rejected attempt loop). We pair sequentially: the i-th inner_trans
    corresponds to the i-th FP bucket. If counts mismatch (e.g., extra
    BP-only attempts that don't reach FP), we annotate but proceed.
    """
    rows = []
    for i, tr in enumerate(inner_trans):
        bucket = fp_buckets[i] if i < len(fp_buckets) else []
        ls_rejects = sum(1 for e in bucket if e.get("event") == "ls_reject")
        rollouts = sum(1 for e in bucket if e.get("event") == "rollout_fail")
        all_failed = any(e.get("event") == "all_failed" for e in bucket)
        last_z = None
        last_alpha = None
        for e in reversed(bucket):
            if e.get("event") == "ls_reject":
                last_z = e.get("z"); last_alpha = e.get("alpha")
                break
        rows.append({
            **tr,
            "ls_rejects": ls_rejects,
            "rollout_fails": rollouts,
            "fp_all_failed": all_failed,
            "ls_last_alpha": last_alpha,
            "ls_last_z": last_z,
        })
    return rows


def print_table(rows: list[dict], scenario_name: str, summary: dict):
    """Print compact per-iter table + summary."""
    print(f"\n=== {scenario_name} ===")
    print(f"stop:    {summary.get('stop', '?')}")
    print(f"elapsed: {summary.get('elapsed_s', 0):.1f}s   "
          f"iters={summary.get('iters', 0)}   "
          f"PE_fin={summary.get('PE_fin', float('nan')):.2f}°")
    print()

    has_quu = any('quu_lambda_min' in r for r in rows)
    if has_quu:
        hdr = ("it  reg_in    bp_at  bp_f  fp_f  ls_rej  ls_α      ls_z       "
               "J         dJ        λmin(Quu)  λmax(Quu)  indef%")
    else:
        hdr = ("it  reg_in    bp_at  bp_f  fp_f  ls_rej  rollouts  ls_α    "
               "ls_z       J         dJ        max_d     stag")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if "reason" in r:  # failure-row
            print(f"{r['iter']:3d}  {r['reg_at_entry']:.2e}  "
                  f"{r['bp_attempts']:5d}  {r['bp_failures']:4d}  "
                  f"{r['fp_failures']:4d}  ----    "
                  f"FAIL: {r['reason']}")
            continue
        ls_alpha = r.get('ls_last_alpha')
        ls_z = r.get('ls_last_z')
        ls_alpha_s = f"{ls_alpha:.2e}" if ls_alpha is not None else "  --    "
        ls_z_s     = f"{ls_z:+.2e}" if (ls_z is not None and not (isinstance(ls_z, float) and np.isnan(ls_z))) else "  --     "
        if has_quu and 'quu_lambda_min' in r:
            print(f"{r['iter']:3d}  {r['reg_at_entry']:.2e}  "
                  f"{r['bp_attempts']:5d}  {r['bp_failures']:4d}  "
                  f"{r['fp_failures']:4d}  {r['ls_rejects']:6d}  "
                  f"{ls_alpha_s}  {ls_z_s}  "
                  f"{r['J_new']:.2e}  {r['act_delta']:.2e}  "
                  f"{r['quu_lambda_min']:+.2e}  {r['quu_lambda_max']:+.2e}  "
                  f"{r['quu_indef_frac']*100:4.0f}%")
        else:
            print(f"{r['iter']:3d}  {r['reg_at_entry']:.2e}  "
                  f"{r['bp_attempts']:5d}  {r['bp_failures']:4d}  "
                  f"{r['fp_failures']:4d}  {r['ls_rejects']:6d}  "
                  f"{r.get('rollout_fails', 0):8d}  "
                  f"{ls_alpha_s}  {ls_z_s}  "
                  f"{r['J_new']:.2e}  {r['act_delta']:.2e}  "
                  f"{r['max_d']:.2e}  {r['stagnation_count']:4d}")


def aggregate_summary(rows: list[dict]) -> dict:
    """Failure-mode buckets + key signal counts."""
    if not rows:
        return {"empty": True}
    bp_fails = sum(r.get('bp_failures', 0) for r in rows)
    fp_fails = sum(r.get('fp_failures', 0) for r in rows)
    ls_rejects = sum(r.get('ls_rejects', 0) for r in rows if 'ls_rejects' in r)
    rollouts = sum(r.get('rollout_fails', 0) for r in rows if 'rollout_fails' in r)

    # z-distribution: how many LS rejects had negative z (ascent direction)?
    # We don't have per-trial z here without re-parsing; the merged row has
    # `ls_last_z` only. Negative count from last-z is a coarse proxy.
    last_z_neg = sum(1 for r in rows if r.get('ls_last_z') is not None
                     and r['ls_last_z'] < 0)
    last_z_huge = sum(1 for r in rows if r.get('ls_last_z') is not None
                      and r['ls_last_z'] > 5000)

    # Q_uu spectrum aggregates (only present when SALTRO_LOG_QUU=1)
    quu_rows = [r for r in rows if 'quu_lambda_min' in r]
    quu_summary = {}
    if quu_rows:
        lmins = np.array([r['quu_lambda_min'] for r in quu_rows])
        indefs = np.array([r['quu_indef_frac'] for r in quu_rows])
        quu_summary = {
            "iters_with_quu_data": len(quu_rows),
            "iters_with_indef_quu": int(np.sum(lmins < 0)),
            "min_lambda_min_observed": float(lmins.min()),
            "max_indef_frac_observed": float(indefs.max()),
            "mean_indef_frac": float(indefs.mean()),
        }

    final = rows[-1]
    return {
        "iters_recorded": len(rows),
        "bp_fails_total": bp_fails,
        "fp_fails_total": fp_fails,
        "ls_rejects_total": ls_rejects,
        "rollout_fails_total": rollouts,
        "iters_with_neg_last_z": last_z_neg,
        "iters_with_huge_last_z": last_z_huge,
        "final_reg": final.get('reg_at_entry'),
        "final_max_d": final.get('max_d'),
        "final_stagnation": final.get('stagnation_count'),
        **quu_summary,
    }


def run_driver(scenario_name: str, save_json: Path | None = None):
    """Spawn subprocess for the scenario, capture, merge, display."""
    cmd = [
        sys.executable, "-u", str(Path(__file__).resolve()),
        "--inner", scenario_name,
    ]
    env = os.environ.copy()
    env["SALTRO_FP_VERBOSE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["SALTRO_LOG_QUU"] = "1"
    if scenario_name in SCENARIOS and SCENARIOS[scenario_name].kink_fix:
        env["SALTRO_KINK_FIX"] = "1"
    if scenario_name in SCENARIOS and SCENARIOS[scenario_name].frozen_active_set:
        env["SALTRO_FROZEN_AS"] = "1"
    if scenario_name in SCENARIOS and SCENARIOS[scenario_name].lambda_floor_eps > 0.0:
        env["SALTRO_LAMBDA_FLOOR"] = str(SCENARIOS[scenario_name].lambda_floor_eps)
    if scenario_name in SCENARIOS and getattr(SCENARIOS[scenario_name], "grad_check", False):
        env["SALTRO_GRAD_CHECK"] = "1"
        env["SALTRO_GRAD_CHECK_ITERS"] = "0,3,5,10,20,40"
        env["SALTRO_GRAD_CHECK_EPS"] = "1e-7"

    print(f"[driver] running scenario {scenario_name} in subprocess...")
    res = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=2400)

    if res.returncode != 0:
        print(f"[driver] subprocess failed (rc={res.returncode}):")
        print(res.stderr[-2000:])
        return

    try:
        inner_data = json.loads(res.stdout)
    except json.JSONDecodeError as e:
        print(f"[driver] JSON parse failed: {e}")
        print("stdout (first 500):", res.stdout[:500])
        return

    if "error" in inner_data:
        print(f"[driver] inner error: {inner_data['error']}")
        return

    fp_events = parse_fp_stderr(res.stderr)
    fp_buckets = bucket_fp_events(fp_events)

    inner_trans = inner_data.get("inner_trans", [])
    rows = merge(inner_trans, fp_buckets)

    print_table(rows, scenario_name, inner_data)

    summary = aggregate_summary(rows)
    print()
    print("=== summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print()
    print(f"  outer_trans:")
    for ot in inner_data.get("outer_trans", []):
        print(f"    outer={ot['outer_iter']}  max_c={ot['max_constraint_violation']:.2e}  "
              f"λ_max={ot.get('lambda_max', 0):.2e}  μ_max={ot.get('mu_max', 0):.2e}")

    if save_json is not None:
        artifact = {
            "scenario_name": scenario_name,
            "scenario_def": asdict(SCENARIOS[scenario_name]),
            "inner_data": inner_data,
            "rows": rows,
            "summary": summary,
            "fp_events_total": len(fp_events),
        }
        save_json.parent.mkdir(parents=True, exist_ok=True)
        save_json.write_text(json.dumps(artifact, default=str, indent=1))
        print(f"\n[driver] saved JSON: {save_json}")


# ============================================================================
# Entry
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="list scenarios and exit")
    ap.add_argument("--scenario", "-s", default="12_omega_5x_ict1e-3",
                    help="scenario name (see --list)")
    ap.add_argument("--inner", help=argparse.SUPPRESS)  # internal: subprocess marker
    ap.add_argument("--save", type=Path, default=None,
                    help="save merged telemetry JSON to this path")
    args = ap.parse_args()

    if args.list:
        print("Available scenarios:")
        for name, s in SCENARIOS.items():
            print(f"  {name:30s}  {s.description}")
        return 0

    if args.inner:
        return run_inner(args.inner)

    return run_driver(args.scenario, save_json=args.save) or 0


if __name__ == "__main__":
    sys.exit(main() or 0)
