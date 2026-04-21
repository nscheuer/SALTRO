"""Targeted probe: test user's 3 requested scenarios + diagnose 3+3 + dt=1/1000s."""
import sys, time, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))

import saltro_py
from trajOpt import trajOpt
from sat_3_1_hybrid import create_satellite as create_3_1
from sat_3_0_mtq    import create_satellite as create_3_0
from sat_0_3_rw     import create_satellite as create_0_3
from sat_3_3_hybrid import create_satellite as create_3_3

def build(angle=1e4, dt=10.0, max_iters=200, max_outer=30):
    ps = saltro_py.PlannerSettings()
    ps.init_traj.initcontroller = 1
    ps.num_passes = 1
    ps.passes[0].dt = dt
    ps.passes[0].ilqr.cost_tol = 1e-6
    ps.passes[0].ilqr.max_iters = max_iters
    ps.passes[0].ilqr.grad_tol = 0.0
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


def try_run(name, sat_fn, time_s=1000.0, dt=10.0, omega0=0.01, angle=1e4,
            use_spike=True, rw_scale=0.0):
    ps = build(angle=angle, dt=dt)
    sat = sat_fn(ps)
    x0 = np.hstack([[omega0, omega0, omega0], [1, 0, 0, 0], np.zeros(sat.numRW)])
    r0 = np.array([7e6, 0, 0]); v0 = np.array([0, 7.5e3, 0])
    jtime = np.array([0.22, 0.22 + time_s/(36525*86400)])
    qg = np.array([np.sqrt(2)/2, 0, 0, np.sqrt(2)/2])
    qgoal = np.tile(qg[:, None], (1, 2))
    bs = np.array([[1,1],[0,0],[0,0]], dtype=float)
    cfg = None
    if use_spike:
        cfg = {"start_at_iter": 2, "max_intervention_iters": 20, "blend_len": 30,
               "goal_switch_buffer": 15, "min_consecutive": 7, "exit_fudge": 2.0,
               "min_prior_decrease_knots": 5, "min_spike_ratio": 2.0,
               "kp_q": 0.3, "kd_w": 2.0, "rw_scale": rw_scale,
               "omega_max": 0.30, "verbose": False}
    t0 = time.time()
    try:
        X, U, stop, snaps, trans, *_ = trajOpt(ps, sat, x0, r0, v0, jtime, qgoal, bs,
                                                debug=True, spike_removal_cfg=cfg)
        wall = time.time() - t0
        pe = np.array([2*np.degrees(np.arccos(min(abs(float(np.dot(X[3:7,k], qg))), 1)))
                       for k in range(X.shape[1])])
        print(f"  {name:<45} it={len(snaps):>5}  PE_fin={pe[-1]:>6.1f}°  "
              f"PE_mean={pe.mean():>5.1f}°  wall={wall:>5.1f}s  stop={stop.split(':')[0]}", flush=True)
        return True
    except Exception as e:
        wall = time.time() - t0
        print(f"  {name:<45} FAIL at t={wall:.1f}s  {type(e).__name__}: {str(e)[:60]}", flush=True)
        return False


print("="*110)
print("1. 3+3 hybrid — isolate when it fails")
print("="*110)
try_run("3+3  with spike (rw_scale=0)", create_3_3, use_spike=True, rw_scale=0.0)
try_run("3+3  with spike (rw_scale=1)", create_3_3, use_spike=True, rw_scale=1.0)
try_run("3+3  NO spike",                create_3_3, use_spike=False)

print("\n" + "="*110)
print("2. 0+3 RW-only — with/without spike")
print("="*110)
try_run("0+3  with spike (rw_scale=0)", create_0_3, use_spike=True, rw_scale=0.0)
try_run("0+3  with spike (rw_scale=1)", create_0_3, use_spike=True, rw_scale=1.0)
try_run("0+3  NO spike",                create_0_3, use_spike=False)

print("\n" + "="*110)
print("3. dt=1, 1000s (same total time, 10× knots)")
print("="*110)
try_run("3+1  dt=1 1000s w/spike",      create_3_1, time_s=1000.0, dt=1.0, use_spike=True)
try_run("3+1  dt=1 1000s NO spike",     create_3_1, time_s=1000.0, dt=1.0, use_spike=False)

print("\n" + "="*110)
print("4. dt=30, 3000s (replicating failure)")
print("="*110)
try_run("3+1  dt=30 3000s w/spike",     create_3_1, time_s=3000.0, dt=30.0, use_spike=True)
try_run("3+1  dt=30 3000s NO spike",    create_3_1, time_s=3000.0, dt=30.0, use_spike=False)
try_run("3+1  dt=20 2000s NO spike",    create_3_1, time_s=2000.0, dt=20.0, use_spike=False)

print("\n" + "="*110)
print("5. MTQ-only with 2000s horizon")
print("="*110)
try_run("3+0 MTQ  1000s",               create_3_0, time_s=1000.0, use_spike=True, rw_scale=0.0)
try_run("3+0 MTQ  2000s",               create_3_0, time_s=2000.0, use_spike=True, rw_scale=0.0)
try_run("3+0 MTQ  3000s",               create_3_0, time_s=3000.0, use_spike=True, rw_scale=0.0)

print("\n" + "="*110)
print("6. High ω0 with longer horizons (give it time to bleed off)")
print("="*110)
try_run("3+1  ω=5x  1000s",             create_3_1, omega0=0.05, time_s=1000.0)
try_run("3+1  ω=5x  2000s",             create_3_1, omega0=0.05, time_s=2000.0)
try_run("3+1  ω=5x  3000s",             create_3_1, omega0=0.05, time_s=3000.0)
try_run("3+1  ω=10x 2000s",             create_3_1, omega0=0.1,  time_s=2000.0)
try_run("3+1  ω=10x 5000s",             create_3_1, omega0=0.1,  time_s=5000.0)
