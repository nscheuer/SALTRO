"""Reproduce the dt=30 RuntimeError at lsl=10, with full stack + state dump."""
import sys, traceback, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))

import saltro_py
from sat_3_1_hybrid import create_satellite
from trajOpt import trajOpt

ps = saltro_py.PlannerSettings()
ps.init_traj.initcontroller = 1; ps.num_passes = 1
ps.passes[0].dt = 30.0
ps.passes[0].ilqr.cost_tol = 1e-6
ps.passes[0].ilqr.max_iters = 200
ps.passes[0].ilqr.grad_tol = 0.0
ps.passes[0].ilqr.ls_attempts_lim = 10
ps.passes[0].auglag.max_outer_iters = 30
ps.passes[0].auglag.constraint_tol = 1e-3
c = ps.passes[0].cost
c.angle = 1e4; c.ang_vel = 1e2
c.control_mult = 1.0
c.mtq_control_weight = 1e-1; c.rw_control_weight = 1.0
c.ang_cost_func_type = 3; c.use_cost_hess = True
c.setTerminalEmphasis(100.0)
ps.passes[0].reg.reg_init = 1e-6; ps.passes[0].reg.reg_max = 1e30
ps.passes[0].reg.reg_scale = 1.6
ps.passes[0].linesearch.max_iters = 24
ps.passes[0].linesearch.beta1 = 1e-10; ps.passes[0].linesearch.beta2 = 5000.0

sat = create_satellite(ps)
x0 = np.array([0.01, 0.01, 0.01, 1, 0, 0, 0, 0.0])
r0 = np.array([7e6, 0, 0]); v0 = np.array([0, 7.5e3, 0])
jtime = np.array([0.22, 0.22 + 3000.0/(36525*86400)])
qg = np.array([np.sqrt(2)/2, 0, 0, np.sqrt(2)/2])
qgoal = np.tile(qg[:, None], (1, 2))
bs = np.array([[1,1],[0,0],[0,0]], dtype=float)

cfg = {"start_at_iter": 2, "max_intervention_iters": 20, "blend_len": 30,
       "goal_switch_buffer": 15, "min_consecutive": 7, "exit_fudge": 2.0,
       "min_prior_decrease_knots": 5, "min_spike_ratio": 2.0,
       "kp_q": 0.3, "kd_w": 2.0, "rw_scale": 0.0, "omega_max": 0.30,
       "verbose": True}

print("Running dt=30 + spike + lsl=10, expecting crash…")
try:
    X, U, stop, snaps, *_ = trajOpt(
        ps, sat, x0, r0, v0, jtime, qgoal, bs,
        debug=True, spike_removal_cfg=cfg)
    print(f"OK: stop={stop}, iters={len(snaps)}")
except Exception as e:
    print(f"\n### CAUGHT: {type(e).__name__}: {e}")
    print("\n### Full traceback:")
    traceback.print_exc()
