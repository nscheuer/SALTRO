"""Diagnose convergence brittleness: warm-start × tighter-exit × suspicious configs."""
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

INITCTRL_NAMES = {0: "Zero", 1: "Exc", 2: "Bdot", 3: "PD"}


def build(initctrl=1, dt=10.0, angle=1e4, max_iters=300, max_outer=40,
          cost_tol=1e-6, total_cost_tol=1e-2, beta2=5000.0, z_count_lim=10):
    ps = saltro_py.PlannerSettings()
    ps.init_traj.initcontroller = initctrl
    ps.num_passes = 1
    ps.passes[0].dt = dt
    ps.passes[0].ilqr.cost_tol = cost_tol
    ps.passes[0].ilqr.max_iters = max_iters
    ps.passes[0].ilqr.grad_tol = 0.0
    ps.passes[0].ilqr.z_count_lim = z_count_lim
    ps.passes[0].auglag.max_outer_iters = max_outer
    ps.passes[0].auglag.constraint_tol = 1e-3
    ps.passes[0].auglag.total_cost_tol = total_cost_tol
    c = ps.passes[0].cost
    c.angle = angle; c.ang_vel = angle/100
    c.control_mult = 1.0
    c.mtq_control_weight = 1e-1; c.rw_control_weight = 1.0
    c.ang_cost_func_type = 3; c.use_cost_hess = True
    c.setTerminalEmphasis(100.0)
    ps.passes[0].reg.reg_init = 1e-6; ps.passes[0].reg.reg_max = 1e30
    ps.passes[0].reg.reg_scale = 1.6
    ps.passes[0].linesearch.max_iters = 24
    ps.passes[0].linesearch.beta1 = 1e-10; ps.passes[0].linesearch.beta2 = beta2
    return ps


def run(label, sat_fn, time_s=1000.0, dt=10.0, **build_kw):
    ps = build(dt=dt, **build_kw)
    sat = sat_fn(ps)
    x0 = np.hstack([[0.01, 0.01, 0.01], [1, 0, 0, 0], np.zeros(sat.numRW)])
    r0 = np.array([7e6, 0, 0]); v0 = np.array([0, 7.5e3, 0])
    jtime = np.array([0.22, 0.22 + time_s/(36525*86400)])
    qg = np.array([np.sqrt(2)/2, 0, 0, np.sqrt(2)/2])
    t0 = time.time()
    try:
        X, U, stop, snaps, *_ = trajOpt(
            ps, sat, x0, r0, v0, jtime,
            np.tile(qg[:, None], (1, 2)),
            np.array([[1,1],[0,0],[0,0]], dtype=float),
            debug=True, spike_removal_cfg=None,
        )
        pe = np.array([2*np.degrees(np.arccos(min(abs(float(np.dot(X[3:7,k], qg))), 1)))
                       for k in range(X.shape[1])])
        wall = time.time() - t0
        print(f"  {label:<38} it={len(snaps):>4}  t={wall:>5.1f}s  "
              f"PE_fin={pe[-1]:>6.1f}°  PE_mean={pe.mean():>5.1f}°  "
              f"stop={stop[:50]}", flush=True)
        return len(snaps), pe[-1]
    except Exception as e:
        print(f"  {label:<38} FAIL: {type(e).__name__}: {str(e)[:50]}", flush=True)
        return -1, 999.0


# ============================================================================
# Part A: warm-start controller for each suspicious config
# ============================================================================
print("="*115)
print("PART A: warm-start controller sweep (no spike removal) — which initctrl unlocks convergence?")
print("="*115)
for config_name, sat_fn, time_s, dt in [
    ("3+1 hybrid 1000s",  create_3_1, 1000.0, 10.0),
    ("0+3 RW only  1000s", create_0_3, 1000.0, 10.0),
    ("3+0 MTQ only 1000s", create_3_0, 1000.0, 10.0),
    ("3+3 hybrid  1000s",  create_3_3, 1000.0, 10.0),
    ("3+1 dt=1    1000s",  create_3_1, 1000.0,  1.0),
    ("3+1 dt=30   3000s",  create_3_1, 3000.0, 30.0),
]:
    print(f"\n— {config_name} —")
    for ic in [0, 1, 2, 3]:
        run(f"initctrl={ic} ({INITCTRL_NAMES[ic]})", sat_fn, time_s=time_s, dt=dt, initctrl=ic)


# ============================================================================
# Part B: tightening exit criteria on the default baseline (3+1, initctrl=1)
# ============================================================================
print("\n")
print("="*115)
print("PART B: does tightening exit criteria help the baseline converge tighter?")
print("="*115)
print("\n— 3+1 baseline 1000s, initctrl=1 —")
run("DEFAULT (beta2=5000, tcostol=1e-2)",    create_3_1)
run("tighter tcostol=1e-4",                   create_3_1, total_cost_tol=1e-4)
run("tighter cost_tol=1e-10",                 create_3_1, cost_tol=1e-10)
run("beta2=500 (default ALTRO)",              create_3_1, beta2=500.0)
run("z_count_lim=50 (very patient)",          create_3_1, z_count_lim=50)
run("ALL tight: beta2=500, tcostol=1e-4, zcl=50", create_3_1,
    beta2=500.0, total_cost_tol=1e-4, z_count_lim=50)


# ============================================================================
# Part C: best warm-start + tight exit on 0+3 (the one that SHOULD converge to ~0)
# ============================================================================
print("\n")
print("="*115)
print("PART C: 0+3 RW-only with best warm-start + tight exit")
print("="*115)
print("\n— 0+3 1000s, trying combinations —")
for ic in [2, 3]:
    run(f"initctrl={INITCTRL_NAMES[ic]}, default", create_0_3, initctrl=ic)
    run(f"initctrl={INITCTRL_NAMES[ic]}, tight",   create_0_3, initctrl=ic,
        beta2=500.0, total_cost_tol=1e-4, z_count_lim=50, max_iters=500, max_outer=60)
