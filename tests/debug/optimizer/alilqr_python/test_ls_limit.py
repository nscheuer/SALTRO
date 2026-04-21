"""Verify ls_attempts_lim safety net on cases that previously hit reg_exceeded."""
import sys, time, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))

import saltro_py
from trajOpt import trajOpt
from sat_3_1_hybrid import create_satellite as create_3_1
from sat_0_3_rw     import create_satellite as create_0_3
from sat_3_3_hybrid import create_satellite as create_3_3


def build(dt=10.0, angle=1e4, max_iters=200, max_outer=30, ls_attempts_lim=10):
    ps = saltro_py.PlannerSettings()
    ps.init_traj.initcontroller = 1; ps.num_passes = 1
    ps.passes[0].dt = dt
    ps.passes[0].ilqr.cost_tol = 1e-6
    ps.passes[0].ilqr.max_iters = max_iters
    ps.passes[0].ilqr.grad_tol = 0.0
    ps.passes[0].ilqr.ls_attempts_lim = ls_attempts_lim
    ps.passes[0].auglag.max_outer_iters = max_outer
    ps.passes[0].auglag.constraint_tol = 1e-3
    c = ps.passes[0].cost
    c.angle = angle; c.ang_vel = angle/100
    c.control_mult = 1.0
    c.mtq_control_weight = 1e-1; c.rw_control_weight = 1.0
    c.angle_N = angle; c.ang_vel_N = angle/100
    c.ang_cost_func_type = 3; c.use_cost_hess = True
    ps.passes[0].reg.reg_init = 1e-6; ps.passes[0].reg.reg_max = 1e30
    ps.passes[0].reg.reg_scale = 1.6
    ps.passes[0].linesearch.max_iters = 24
    ps.passes[0].linesearch.beta1 = 1e-10; ps.passes[0].linesearch.beta2 = 5000.0
    return ps


def run(name, sat_fn, time_s=1000.0, dt=10.0, omega0=0.01,
        use_spike=True, rw_scale=0.0, ls_attempts_lim=10):
    ps = build(dt=dt, ls_attempts_lim=ls_attempts_lim)
    sat = sat_fn(ps)
    x0 = np.hstack([[omega0, omega0, omega0], [1, 0, 0, 0], np.zeros(sat.numRW)])
    r0 = np.array([7e6, 0, 0]); v0 = np.array([0, 7.5e3, 0])
    jtime = np.array([0.22, 0.22 + time_s/(36525*86400)])
    qg = np.array([np.sqrt(2)/2, 0, 0, np.sqrt(2)/2])
    cfg = None
    if use_spike:
        cfg = {"start_at_iter": 2, "max_intervention_iters": 20, "blend_len": 30,
               "goal_switch_buffer": 15, "min_consecutive": 7, "exit_fudge": 2.0,
               "min_prior_decrease_knots": 5, "min_spike_ratio": 2.0,
               "kp_q": 0.3, "kd_w": 2.0, "rw_scale": rw_scale,
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
        print(f"  {name:<42} lsl={ls_attempts_lim:>3}  it={len(snaps):>4}  "
              f"t={wall:>5.1f}s  PE_fin={pe[-1]:>6.1f}°  PE_mean={pe.mean():>5.1f}°  "
              f"stop={stop.split(':')[0]}", flush=True)
    except Exception as e:
        print(f"  {name:<42} lsl={ls_attempts_lim:>3}  FAIL: {type(e).__name__}", flush=True)


print("=" * 115)
print("LS-attempts-limit safety net: test previously reg_exceeded cases")
print("=" * 115)

# Cases that reg_exceeded in bug_probe
print("\n--- Compare lsl=10 (default) vs lsl=1000 (effectively unlimited) ---")
for lsl in [1000, 10]:
    print(f"\n  ls_attempts_lim = {lsl}:")
    run("3+3 + spike(rw=0)  (was reg_exceeded)",    create_3_3, use_spike=True, rw_scale=0.0, ls_attempts_lim=lsl)
    run("3+3 + spike(rw=1)  (was converged)",       create_3_3, use_spike=True, rw_scale=1.0, ls_attempts_lim=lsl)
    run("3+1 dt=30 3000s + spike  (was reg_exceeded)", create_3_1, time_s=3000.0, dt=30.0, use_spike=True, ls_attempts_lim=lsl)
    run("3+1 ω=5× 2000s + spike  (was reg_exceeded)",  create_3_1, omega0=0.05, time_s=2000.0, use_spike=True, ls_attempts_lim=lsl)
    run("3+1 ω=10× 2000s + spike (was reg_exceeded)",  create_3_1, omega0=0.10, time_s=2000.0, use_spike=True, ls_attempts_lim=lsl)

print("\n--- Ensure normal cases still converge with lsl=10 ---")
run("3+1 baseline + spike",   create_3_1, use_spike=True, ls_attempts_lim=10)
run("3+1 baseline NO spike",  create_3_1, use_spike=False, ls_attempts_lim=10)
run("0+3 RW NO spike",        create_0_3, use_spike=False, ls_attempts_lim=10)
run("0+3 RW + spike",         create_0_3, use_spike=True, ls_attempts_lim=10)
