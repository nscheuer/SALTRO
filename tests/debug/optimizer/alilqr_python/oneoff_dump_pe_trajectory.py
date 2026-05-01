"""Dump the PE trajectory at a mid-optimization snapshot of `00_baseline`
so we can see what shape the spike detector is actually being asked to
find. Looks at iter 100 (well past initial transient, well before
convergence) of a baseline run."""
import sys, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))

import saltro_py
from sat_3_1_hybrid import create_satellite
from trajOpt import trajOpt
from spike_removal import _pointing_error

QG = np.array([np.sqrt(2)/2, 0, 0, np.sqrt(2)/2])
R0 = np.array([7e6, 0, 0]); V0 = np.array([0, 7.5e3, 0])
JTIME = np.array([0.22, 0.22 + 1000.0 / (36525.0 * 86400.0)])
QGOAL = np.tile(QG[:, None], (1, 2))
BS = np.array([[1, 1], [0, 0], [0, 0]], dtype=float)
X0 = np.array([0.01, 0.01, 0.01, 1, 0, 0, 0, 0.0])

ps = saltro_py.PlannerSettings()
ps.init_traj.initcontroller = 1
ps.num_passes = 1
ps.passes[0].dt = 10.0
ps.passes[0].ilqr.cost_tol = 1e-6
ps.passes[0].ilqr.max_iters = 200
ps.passes[0].auglag.max_outer_iters = 30
ps.passes[0].auglag.constraint_tol = 1e-3
c = ps.passes[0].cost
c.angle = 1e4; c.ang_vel = 1e2
c.control_mult = 1.0; c.mtq_control_weight = 0.1; c.rw_control_weight = 1.0
c.ang_cost_func_type = 3; c.use_cost_hess = True
c.setTerminalEmphasis(100.0)
for a in ["aero", "gg", "srp", "prop", "gendist", "resdipole"]:
    setattr(ps.disturbances, "plan_for_" + a, False)
ps.passes[0].reg.reg_init = 0.0; ps.passes[0].reg.reg_max = 1e30
ps.passes[0].reg.reg_scale = 1.6
ps.passes[0].linesearch.max_iters = 24

sat = create_satellite(ps)
import os
SR_ON = os.environ.get("SR", "0") == "1"
sr_cfg = None
if SR_ON:
    sr_cfg = {
        "start_at_iter": 2, "max_intervention_iters": 10000,
        "blend_len": 30, "goal_switch_buffer": 15, "min_consecutive": 7,
        "exit_fudge": 2.0, "min_prior_decrease_knots": 5, "min_spike_ratio": 2.0,
        "kp_q": 0.3, "kd_w": 2.0, "rw_scale": -1.0, "omega_max": 0.30, "verbose": False,
        "constraint_gate_ratio": 0.0,
    }
print(f"SR_ON={SR_ON}")
X, U, stop, snaps, trans, dt_val, ctol, elapsed = trajOpt(
    ps, sat, X0.copy(), R0, V0, JTIME, QGOAL, BS,
    debug=True, spike_removal_cfg=sr_cfg,
)
print(f"Stopped: {stop[:50]}, {len(snaps)} iters, PE_fin computed below.\n")

# Pick a mid-iteration snapshot.
mid = len(snaps) // 2
snap = snaps[mid]
X_mid = snap["X"]
N = X_mid.shape[1]

# We need the dense q_goal/boresight.  Build them like trajOpt does.
from trajOpt import _resample_zero_order_hold
_, qgoal_dense, bs_dense = _resample_zero_order_hold(JTIME, QGOAL, BS, ps.passes[0].dt)

# Compute PE at every knot.
theta = np.array([
    np.degrees(_pointing_error(X_mid, qgoal_dense, bs_dense, k))
    for k in range(N)
])

print(f"=== PE trajectory at iter {mid} of {len(snaps)} ===")
print(f"min={theta.min():.2f}°  max={theta.max():.2f}°  mean={theta.mean():.2f}°")
print()
print("idx |   PE°  |  Δ from prev")
print("----+--------+--------------")
for k in range(N):
    delta = (theta[k] - theta[k-1]) if k > 0 else 0.0
    marker = " ← peak" if theta[k] == theta.max() else ""
    print(f"{k:3d} | {theta[k]:6.2f} | {delta:+6.2f}{marker}")
