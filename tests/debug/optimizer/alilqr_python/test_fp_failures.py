"""Count how many FP failures vs successes at angle=1e6."""
import os, sys, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))

import saltro_py
from sat_3_1_hybrid import create_satellite
from trajOpt import trajOpt

qg = np.array([np.sqrt(2)/2, 0, 0, np.sqrt(2)/2])
x0 = np.array([0.01, 0.01, 0.01, 1, 0, 0, 0, 0.0])
r0 = np.array([7e6, 0, 0]); v0 = np.array([0, 7.5e3, 0])
jtime = np.array([0.22, 0.22 + 1000.0 / (36525.0 * 86400.0)])
qgoal = np.tile(qg[:, None], (1, 2))
bs = np.array([[1,1],[0,0],[0,0]], dtype=float)

ps = saltro_py.PlannerSettings()
ps.init_traj.initcontroller = 1; ps.num_passes = 1
ps.passes[0].dt = 10.0; ps.passes[0].ilqr.cost_tol = 1e-6
ps.passes[0].ilqr.max_iters = 20; ps.passes[0].ilqr.grad_tol = 0.0
ps.passes[0].auglag.max_outer_iters = 3
ps.passes[0].auglag.constraint_tol = 1e-3
c = ps.passes[0].cost
c.angle = 1e6; c.ang_vel = 1e4
c.control_mult = 1.0; c.mtq_control_weight = 1e-1; c.rw_control_weight = 1.0
c.angle_N = 1e6; c.ang_vel_N = 1e4
c.ang_cost_func_type = 3; c.use_cost_hess = True
c.rw_AM_weight = 0.0; c.rw_stic_weight = 0.0
c.RWh_max_mult = 0.0; c.RWh_stiction_mult = 0.0; c.RWh_ok_mult = 0.0
for a in ["aero","gg","srp","prop","gendist","resdipole"]:
    setattr(ps.disturbances, "plan_for_"+a, False)
ps.passes[0].reg.reg_init = 1e-6; ps.passes[0].reg.reg_max = 1e30
ps.passes[0].reg.reg_scale = 10.0
ps.passes[0].linesearch.max_iters = 24
ps.passes[0].linesearch.beta1 = 1e-10; ps.passes[0].linesearch.beta2 = 5000.0

sat = create_satellite(ps)
X, U, stop, snaps, trans, dt, ctol, elapsed = trajOpt(
    ps, sat, x0, r0, v0, jtime, qgoal, bs, debug=True
)

print(f"Stop: {stop}  {len(snaps)} iterations")
# Each snapshot is an accepted step. Count how many steps were accepted
# vs how many iterations ran (max_iters * max_outer = 60)
print(f"Expected max iterations: {ps.passes[0].ilqr.max_iters * ps.passes[0].auglag.max_outer_iters}")
print(f"Accepted steps: {len(snaps)}")

# Show cost and PE at each accepted step
for i, snap in enumerate(snaps):
    X_s = snap['X']
    pe = np.array([2*np.degrees(np.arccos(min(abs(float(np.dot(X_s[3:7,k], qg))), 1)))
                    for k in range(X_s.shape[1])])
    max_u = np.max(np.abs(snap['U']))
    print(f"  step {i:3d}: J={snap['J']:14.4e}  max_pe={pe.max():6.1f}  final_pe={pe[-1]:6.2f}  max|u|={max_u:.2f}")
