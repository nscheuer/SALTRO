"""Sweep linesearch tightness on high-ω cases to see if tighter beta2
helps the reg cascade.

beta2 = cost-ratio upper bound.  z = (J_prev - J_new) / (-delta_V_alpha).
Currently 5000 (very permissive).  ALTRO default is 500.  Very tight
linesearch rejects "barely working" steps earlier, so reg bumps less.
"""
import sys, time, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))

import saltro_py
from trajOpt import trajOpt
from sat_3_1_hybrid import create_satellite


def build(omega0=0.05, time_s=1000.0, beta2=5000.0, strict_decrease=False,
          ls_attempts_lim=30):
    ps = saltro_py.PlannerSettings()
    ps.init_traj.initcontroller = 1; ps.num_passes = 1
    ps.passes[0].dt = 10.0
    ps.passes[0].ilqr.cost_tol = 1e-6
    ps.passes[0].ilqr.max_iters = 300
    ps.passes[0].ilqr.grad_tol = 0.0
    ps.passes[0].ilqr.ls_attempts_lim = ls_attempts_lim
    ps.passes[0].ilqr.ls_strict_decrease = strict_decrease
    ps.passes[0].auglag.max_outer_iters = 30
    ps.passes[0].auglag.constraint_tol = 1e-3
    c = ps.passes[0].cost
    c.angle = 1e4; c.ang_vel = 1e2
    c.control_mult = 1.0
    c.mtq_control_weight = 1e-1; c.rw_control_weight = 1.0
    c.angle_N = 1e4; c.ang_vel_N = 1e2
    c.ang_cost_func_type = 3; c.use_cost_hess = True
    ps.passes[0].reg.reg_init = 1e-6; ps.passes[0].reg.reg_max = 1e30
    ps.passes[0].reg.reg_scale = 1.6
    ps.passes[0].linesearch.max_iters = 24
    ps.passes[0].linesearch.beta1 = 1e-10
    ps.passes[0].linesearch.beta2 = beta2
    return ps


def run(name, omega0, time_s, beta2, strict_decrease=False, use_spike=True):
    ps = build(omega0=omega0, time_s=time_s, beta2=beta2, strict_decrease=strict_decrease)
    sat = create_satellite(ps)
    x0 = np.hstack([[omega0, omega0, omega0], [1, 0, 0, 0], np.zeros(sat.numRW)])
    r0 = np.array([7e6, 0, 0]); v0 = np.array([0, 7.5e3, 0])
    jtime = np.array([0.22, 0.22 + time_s/(36525*86400)])
    qg = np.array([np.sqrt(2)/2, 0, 0, np.sqrt(2)/2])
    cfg = None
    if use_spike:
        cfg = {"start_at_iter": 2, "max_intervention_iters": 20, "blend_len": 30,
               "goal_switch_buffer": 15, "min_consecutive": 7, "exit_fudge": 2.0,
               "min_prior_decrease_knots": 5, "min_spike_ratio": 2.0,
               "kp_q": 0.3, "kd_w": 2.0, "rw_scale": 0.0,
               "omega_max": 0.30, "verbose": False}
    t0 = time.time()
    try:
        X, U, stop, snaps, *_ = trajOpt(
            ps, sat, x0, r0, v0, jtime,
            np.tile(qg[:, None], (1, 2)),
            np.array([[1,1],[0,0],[0,0]], dtype=float),
            debug=True, spike_removal_cfg=cfg)
        wall = time.time() - t0
        pe = np.array([2*np.degrees(np.arccos(min(abs(float(np.dot(X[3:7,k], qg))), 1)))
                       for k in range(X.shape[1])])
        w_max = float(np.max(np.abs(X[:3, :])) * 180 / np.pi)
        print(f"  {name:<35} it={len(snaps):>4} t={wall:>5.1f}s  "
              f"PE_fin={pe[-1]:>6.1f}°  PE_mean={pe.mean():>5.1f}°  "
              f"ωmax={w_max:>5.1f}°/s  stop={stop.split(':')[0]}", flush=True)
    except Exception as e:
        print(f"  {name:<35} FAIL: {type(e).__name__}", flush=True)


for (omega, tname) in [(0.05, "ω=5× (0.05 rad/s)"),
                        (0.10, "ω=10× (0.10 rad/s)")]:
    for time_s in [1000.0, 2000.0]:
        print("="*110)
        print(f"{tname}, {time_s:.0f}s")
        print("="*110)
        print("  --- no spike ---")
        for b2 in [5000.0, 1000.0, 500.0, 100.0]:
            run(f"β2={b2:.0f}  strict=F",
                omega, time_s, b2, strict_decrease=False, use_spike=False)
        print("  --- with spike ---")
        for b2 in [5000.0, 1000.0, 500.0, 100.0]:
            run(f"β2={b2:.0f}  strict=F",
                omega, time_s, b2, strict_decrease=False, use_spike=True)
        print()
