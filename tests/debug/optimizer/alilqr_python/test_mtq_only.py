"""MTQ-only (sat_3_0_mtq) deep dive.

Check whether the terminal-emphasis fix rescues this config, probe
alternative horizons, initial ω, angle weight.  Report actuator usage
(mtq % of u_max) and peak ω per run.

MTQ is fundamentally underactuated: it can only produce torque
perpendicular to B_body, and the B direction in body frame changes
slowly as the sat rotates.  Over a full orbit (~90 min) you get
omnidirectional authority on average, but over a 1000s arc you're
limited to whatever directions B gives you.
"""
import sys, time, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))

import saltro_py
from trajOpt import trajOpt
from sat_3_0_mtq import create_satellite as create_mtq


def build(dt=10.0, angle=1e4, ang_vel_mult=0.01, terminal_k=100.0,
          max_iters=300, max_outer=30, ls_attempts_lim=30):
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
    c.angle = angle; c.ang_vel = angle * ang_vel_mult
    c.control_mult = 1.0
    c.mtq_control_weight = 1e-1; c.rw_control_weight = 1.0
    c.ang_cost_func_type = 3; c.use_cost_hess = True
    c.setTerminalEmphasis(terminal_k)
    ps.passes[0].reg.reg_init = 1e-6; ps.passes[0].reg.reg_max = 1e30
    ps.passes[0].reg.reg_scale = 1.6
    ps.passes[0].linesearch.max_iters = 24
    ps.passes[0].linesearch.beta1 = 1e-10; ps.passes[0].linesearch.beta2 = 5000.0
    return ps


def run(name, time_s=1000.0, dt=10.0, omega0=0.01, use_spike=True,
        rw_scale=0.0, **build_kw):
    ps = build(dt=dt, **build_kw)
    sat = create_mtq(ps)
    nRW = sat.numRW
    x0 = np.hstack([[omega0, omega0, omega0], [1, 0, 0, 0], np.zeros(nRW)])
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
    except Exception as e:
        print(f"  {name:<42} FAIL: {type(e).__name__}", flush=True)
        return

    wall = time.time() - t0
    pe = np.array([2*np.degrees(np.arccos(min(abs(float(np.dot(X[3:7,k], qg))), 1)))
                   for k in range(X.shape[1])])
    w_peak = float(np.max(np.abs(X[:3, :])) * 180 / np.pi)
    mtq_max = sat.getMTQ(0).u_max
    mtq_util = float(np.max(np.abs(U[:sat.numMTQ, :])) / mtq_max * 100)
    print(f"  {name:<42} it={len(snaps):>4}  t={wall:>4.1f}s  "
          f"PE_fin={pe[-1]:>6.1f}°  PE_mean={pe.mean():>5.1f}°  "
          f"ω={w_peak:>5.1f}°/s  mtq={mtq_util:>5.1f}%  stop={stop.split(':')[0]}",
          flush=True)


def section(title):
    print("\n" + "="*120 + f"\n{title}\n" + "="*120)


# -----------------------------------------------------------------------
# 1. BASELINE: does 100× terminal emphasis rescue MTQ-only?
# -----------------------------------------------------------------------
section("1. Compare old (k=1) vs new (k=100) terminal emphasis")
run("k=1    1000s no spike",  time_s=1000.0, use_spike=False, terminal_k=1.0)
run("k=100  1000s no spike",  time_s=1000.0, use_spike=False, terminal_k=100.0)
run("k=1    1000s + spike",   time_s=1000.0, use_spike=True,  terminal_k=1.0)
run("k=100  1000s + spike",   time_s=1000.0, use_spike=True,  terminal_k=100.0)

# -----------------------------------------------------------------------
# 2. HORIZON sweep (does more time help?  orbit period ~5400s)
# -----------------------------------------------------------------------
section("2. Horizon sweep — MTQ effectiveness is B-geometry dependent")
run("500s  no spike",  time_s=500.0,  use_spike=False)
run("1000s no spike",  time_s=1000.0, use_spike=False)
run("2000s no spike",  time_s=2000.0, use_spike=False)
run("3000s no spike",  time_s=3000.0, use_spike=False)
run("5400s no spike (~1 orbit)", time_s=5400.0, use_spike=False)
run("5400s + spike",   time_s=5400.0, use_spike=True)

# -----------------------------------------------------------------------
# 3. ANGLE weight sweep (higher stage angle, preserve ratio)
# -----------------------------------------------------------------------
section("3. Angle weight sweep (k=100 preserved)")
run("angle=1e3",  time_s=2000.0, use_spike=False, angle=1e3)
run("angle=1e4",  time_s=2000.0, use_spike=False, angle=1e4)
run("angle=1e5",  time_s=2000.0, use_spike=False, angle=1e5)
run("angle=1e6",  time_s=2000.0, use_spike=False, angle=1e6)

# -----------------------------------------------------------------------
# 4. ω_vel ratio (rebalance: maybe MTQ needs less damping emphasis)
# -----------------------------------------------------------------------
section("4. ang_vel ratio sweep")
run("ang_vel = 0.01 × angle",  time_s=2000.0, use_spike=False, ang_vel_mult=0.01)
run("ang_vel = 0.1  × angle",  time_s=2000.0, use_spike=False, ang_vel_mult=0.1)
run("ang_vel = 1    × angle",  time_s=2000.0, use_spike=False, ang_vel_mult=1.0)
run("ang_vel = 10   × angle",  time_s=2000.0, use_spike=False, ang_vel_mult=10.0)

# -----------------------------------------------------------------------
# 5. INITIAL ω (test if lower initial makes it easier)
# -----------------------------------------------------------------------
section("5. Initial ω — is there a sweet spot?")
run("ω0=0   ",  time_s=2000.0, use_spike=False, omega0=0.0)
run("ω0=0.01",  time_s=2000.0, use_spike=False, omega0=0.01)
run("ω0=0.05",  time_s=2000.0, use_spike=False, omega0=0.05)
