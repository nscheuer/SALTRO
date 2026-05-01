"""One-off: MTQ-only with all 2026-04-24 principled default changes active.

Tests the cumulative effect of:
  - reg_init = 0.0       (was 1e-2 / 1e-6 override)
  - conjunctive_convergence = false (reverted; disjunctive inner per literature)
  - ls_strict_decrease = true       (kept; Armijo-standard)
  - ilqr_cost_tol = 10× cost_tol    (new 2-tier)
  - min_outer_iters = 3             (kept)

Runs one 90° MTQ-only slew and reports PE_fin. Compares in-context to the
pre-change baseline (71.5° PE_fin) by running the same scenario with the
old-style config in the same script."""
import sys, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))

import saltro_py
from sat_3_0_mtq import create_satellite
from trajOpt import trajOpt

qg = np.array([np.sqrt(2)/2, 0, 0, np.sqrt(2)/2])
x0 = np.array([0.0, 0.0, 0.0, 1, 0, 0, 0])
r0 = np.array([7e6, 0, 0]); v0 = np.array([0, 7.5e3, 0])
jtime = np.array([0.22, 0.22 + 1000.0 / (36525.0 * 86400.0)])
qgoal = np.tile(qg[:, None], (1, 2))
bs = np.array([[1, 1], [0, 0], [0, 0]], dtype=float)


def build_ps(label):
    """Build a PlannerSettings with either 'old' (pre-change) or 'new'
    (principled defaults) config.

    Note: the defaults in plannersettings.h are now the 'new' ones. We
    explicitly set fields for old-style for A/B comparison within one run.
    """
    ps = saltro_py.PlannerSettings()
    ps.init_traj.initcontroller = 1
    ps.num_passes = 1
    ps.passes[0].dt = 10.0
    ps.passes[0].ilqr.cost_tol = 1e-6
    ps.passes[0].ilqr.max_iters = 200
    ps.passes[0].ilqr.grad_tol = 0.0
    ps.passes[0].auglag.max_outer_iters = 30
    ps.passes[0].auglag.constraint_tol = 1e-3

    c = ps.passes[0].cost
    c.angle = 1e4; c.ang_vel = 1e2
    c.control_mult = 1.0; c.mtq_control_weight = 0.1
    c.ang_cost_func_type = 3; c.use_cost_hess = True
    c.setTerminalEmphasis(100.0)
    for a in ["aero", "gg", "srp", "prop", "gendist", "resdipole"]:
        setattr(ps.disturbances, "plan_for_" + a, False)

    ps.passes[0].reg.reg_scale = 1.6
    ps.passes[0].reg.reg_max = 1e30
    ps.passes[0].linesearch.max_iters = 24
    ps.passes[0].linesearch.beta1 = 1e-10
    ps.passes[0].linesearch.beta2 = 5000.0

    if label == "old":
        ps.passes[0].reg.reg_init = 1e-6
        ps.passes[0].ilqr.ls_strict_decrease = False
        ps.passes[0].ilqr.conjunctive_convergence = False
        # no ilqr_cost_tol split in old config; set to match cost_tol
        ps.passes[0].ilqr.ilqr_cost_tol = ps.passes[0].ilqr.cost_tol
        ps.passes[0].auglag.min_outer_iters = 1  # old behavior: no min gate
    else:  # new
        ps.passes[0].reg.reg_init = 0.0
        ps.passes[0].ilqr.ls_strict_decrease = True
        ps.passes[0].ilqr.conjunctive_convergence = False
        ps.passes[0].ilqr.ilqr_cost_tol = ps.passes[0].ilqr.cost_tol * 10.0
        ps.passes[0].auglag.min_outer_iters = 3
    return ps


def pe_profile(X, qg):
    return np.array([2 * np.degrees(np.arccos(min(abs(float(np.dot(X[3:7, k], qg))), 1)))
                     for k in range(X.shape[1])])


for label in ("old", "new"):
    ps = build_ps(label)
    sat = create_satellite(ps)
    print(f"\n=== config: {label} ===")
    X, U, stop, snaps, trans, dt_val, ctol, elapsed = trajOpt(
        ps, sat, x0.copy(), r0, v0, jtime, qgoal, bs, debug=True, spike_removal_cfg=None,
    )
    pe = pe_profile(X, qg)
    omega_norm = np.linalg.norm(X[0:3, :], axis=0)
    print(f"  stop={stop}  it={len(snaps)}  PE_fin={pe[-1]:.3f}°  PE_mean={pe.mean():.1f}°  "
          f"|ω|_max={np.degrees(omega_norm.max()):.2f}°/s  t={elapsed:.1f}s")
