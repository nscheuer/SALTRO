"""Inspect the vector-mode 00_baseline trajectory for NaN locations."""
import sys, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))

import saltro_py
from sat_3_1_hybrid import create_satellite
from trajOpt import trajOpt

# Mirror wide_test_runner_vec.py 00_baseline.
ang = np.radians(90.0)
eci_target = np.array([np.cos(ang), np.sin(ang), 0.0])
qg = np.array([np.nan, eci_target[0], eci_target[1], eci_target[2]])
qgoal = np.tile(qg[:, None], (1, 2))
bs = np.array([[1, 1], [0, 0], [0, 0]], dtype=float)

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
c.ang_vel_err_dir = 1.0 * c.ang_vel  # legacy
ps.passes[0].reg.reg_init = 1e-12; ps.passes[0].reg.reg_max = 1e30
ps.passes[0].reg.reg_scale = 1.6
ps.passes[0].linesearch.max_iters = 24

sat = create_satellite(ps)
x0 = np.array([0.01, 0.01, 0.01, 1, 0, 0, 0, 0.0])
r0 = np.array([7e6, 0, 0]); v0 = np.array([0, 7.5e3, 0])
jtime = np.array([0.22, 0.22 + 1000.0 / (36525.0 * 86400.0)])

X, U, stop, snaps, trans, dt_val, ctol, elapsed = trajOpt(
    ps, sat, x0, r0, v0, jtime, qgoal, bs, debug=True, spike_removal_cfg=None,
)

print(f"Stop: {stop[:60]}")
print(f"X.shape = {X.shape}, U.shape = {U.shape}")
print()

# Find where NaN starts
nan_per_knot = np.isnan(X).any(axis=0)
if nan_per_knot.any():
    first_nan = int(np.where(nan_per_knot)[0][0])
    print(f"FIRST NaN at knot {first_nan} of {X.shape[1]}")
    print(f"X[:, {max(0,first_nan-2)}:{first_nan+2}] = ")
    for k in range(max(0, first_nan-2), min(X.shape[1], first_nan+2)):
        print(f"  k={k}: {X[:, k]}")
else:
    print("No NaN in X")

print()
print(f"Final state X[:, -1] = {X[:, -1]}")
print(f"Penultimate X[:, -2] = {X[:, -2]}")
print(f"q-norm at last 5 knots: {[float(np.linalg.norm(X[3:7, k])) for k in range(X.shape[1]-5, X.shape[1])]}")

# Check snapshots progression
print(f"\n{len(snaps)} snapshots; final-snap X NaN count: {int(np.isnan(snaps[-1]['X']).sum())}")
