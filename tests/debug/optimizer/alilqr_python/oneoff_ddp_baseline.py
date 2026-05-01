"""Sanity check: wide_test 00_baseline scenario with DDP on/off.

Yesterday's DDP wide showed 00_baseline converged in 39 iter PE=41.4° — a
dramatic local-minimum collapse caused by the missing manifold-curvature
correction. This test reruns that exact scenario with the corrected BP
(2026-04-24 reduced-state curvature fix per Planning with Attitude eq 15)
to confirm DDP now behaves reasonably on the baseline 3+1 hybrid case.
"""
import sys, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))

import saltro_py
from sat_3_1_hybrid import create_satellite
from trajOpt import trajOpt

qg = np.array([np.sqrt(2)/2, 0, 0, np.sqrt(2)/2])
r0 = np.array([7e6, 0, 0]); v0 = np.array([0, 7.5e3, 0])
jtime = np.array([0.22, 0.22 + 1000.0 / (36525.0 * 86400.0)])
qgoal = np.tile(qg[:, None], (1, 2))
bs = np.array([[1, 1], [0, 0], [0, 0]], dtype=float)
x0 = np.array([0.01, 0.01, 0.01, 1, 0, 0, 0, 0.0])  # 3+1 hybrid: +1 RW momentum


def build_ps(use_dyn_hess: bool):
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
    c.control_mult = 1.0; c.mtq_control_weight = 0.1; c.rw_control_weight = 1.0
    c.ang_cost_func_type = 3; c.use_cost_hess = True
    c.setTerminalEmphasis(100.0)
    for a in ["aero", "gg", "srp", "prop", "gendist", "resdipole"]:
        setattr(ps.disturbances, "plan_for_" + a, False)
    ps.passes[0].reg.reg_max = 1e30
    ps.passes[0].reg.reg_scale = 1.6
    ps.passes[0].reg.use_dynamics_hess = use_dyn_hess
    ps.passes[0].linesearch.max_iters = 24
    ps.passes[0].linesearch.beta1 = 1e-10
    ps.passes[0].linesearch.beta2 = 5000.0
    return ps


def pe_profile(X, qg):
    return np.array([2 * np.degrees(np.arccos(min(abs(float(np.dot(X[3:7, k], qg))), 1)))
                     for k in range(X.shape[1])])


for label, flag in [("iLQR (DDP off)", False), ("DDP on", True)]:
    ps = build_ps(flag)
    sat = create_satellite(ps)
    X, U, stop, snaps, trans, dt_val, ctol, elapsed = trajOpt(
        ps, sat, x0.copy(), r0, v0, jtime, qgoal, bs,
        debug=True, spike_removal_cfg=None,
    )
    pe = pe_profile(X, qg)
    print(f"{label:20s} stop={stop}  it={len(snaps)}  "
          f"PE_fin={pe[-1]:7.3f}°  PE_mean={pe.mean():6.1f}°  t={elapsed:5.1f}s")
