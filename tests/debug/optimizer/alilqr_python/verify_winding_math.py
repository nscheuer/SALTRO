"""Sanity-check the winding excess computation.

Tests three reference trajectories:
 1. Pure SLERP from q0 to q_goal (should give excess = 0)
 2. Optimized converged trajectory (what the optimizer actually produces)
 3. Peek at q(t) components to see if off-axis motion is real
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


def step_angle(q1, q2):
    return 2 * np.arccos(min(abs(float(np.dot(q1, q2))), 1.0))


def excess_over(q_trace, t1, t2):
    steps = [step_angle(q_trace[:, k], q_trace[:, k+1]) for k in range(t1, t2)]
    traveled = sum(steps)
    direct = step_angle(q_trace[:, t1], q_trace[:, t2])
    return traveled, direct, traveled - direct


# -----------------------------------------------------------------------------
# Test 1: SLERP from identity to 90° about z — should have zero excess.
# -----------------------------------------------------------------------------
q0 = np.array([1.0, 0.0, 0.0, 0.0])
q_goal = np.array([np.sqrt(2)/2, 0, 0, np.sqrt(2)/2])

def slerp(q0, q1, t):
    d = float(np.dot(q0, q1))
    if d < 0:
        q1 = -q1
        d = -d
    d = min(d, 1.0)
    omega = np.arccos(d)
    if omega < 1e-9:
        return q0
    return (np.sin((1-t)*omega)*q0 + np.sin(t*omega)*q1) / np.sin(omega)

N = 100
q_slerp = np.zeros((4, N))
for k in range(N):
    t = k / (N - 1)
    q_slerp[:, k] = slerp(q0, q_goal, t)

trav, direct, exc = excess_over(q_slerp, 0, N-1)
print(f"SLERP (ideal):   traveled={np.degrees(trav):.2f}°  direct={np.degrees(direct):.2f}°  excess={np.degrees(exc):.4f}°")

# -----------------------------------------------------------------------------
# Test 2: Run the converged trajectory, look at step-angle profile.
# -----------------------------------------------------------------------------
x0 = np.array([0.01, 0.01, 0.01, 1, 0, 0, 0, 0.0])
r0 = np.array([7e6, 0, 0]); v0 = np.array([0, 7.5e3, 0])
jtime = np.array([0.22, 0.22 + 1000.0 / (36525.0 * 86400.0)])
qgoal = np.tile(q_goal[:, None], (1, 2))
bs = np.array([[1,1],[0,0],[0,0]], dtype=float)

ps = saltro_py.PlannerSettings()
ps.init_traj.initcontroller = 1; ps.num_passes = 1
ps.passes[0].dt = 10.0; ps.passes[0].ilqr.cost_tol = 1e-6
ps.passes[0].ilqr.max_iters = 200; ps.passes[0].ilqr.grad_tol = 0.0
ps.passes[0].auglag.max_outer_iters = 30
ps.passes[0].auglag.constraint_tol = 1e-3
c = ps.passes[0].cost
c.angle = 1e4; c.ang_vel = 1e2
c.control_mult = 1.0; c.mtq_control_weight = 1e-1; c.rw_control_weight = 1.0
c.angle_N = 1e4; c.ang_vel_N = 1e2
c.ang_cost_func_type = 3; c.use_cost_hess = True
for a in ["aero","gg","srp","prop","gendist","resdipole"]:
    setattr(ps.disturbances, "plan_for_"+a, False)
ps.passes[0].reg.reg_init = 1e-6; ps.passes[0].reg.reg_max = 1e30
ps.passes[0].reg.reg_scale = 1.6
ps.passes[0].linesearch.max_iters = 24
ps.passes[0].linesearch.beta1 = 1e-10; ps.passes[0].linesearch.beta2 = 5000.0

sat = create_satellite(ps)
X, U, stop = trajOpt(
    ps, sat, x0, r0, v0, jtime, qgoal, bs, debug=False, spike_removal_cfg=None,
)

q_opt = X[3:7, :]
N_opt = q_opt.shape[1]
trav, direct, exc = excess_over(q_opt, 0, N_opt-1)
print(f"Optimized:       traveled={np.degrees(trav):.2f}°  direct={np.degrees(direct):.2f}°  excess={np.degrees(exc):.2f}°")

# Step angle profile: how much does each knot rotate?
steps = [step_angle(q_opt[:, k], q_opt[:, k+1]) for k in range(N_opt-1)]
print(f"Per-step rotation (deg): min={np.degrees(min(steps)):.3f}  max={np.degrees(max(steps)):.3f}  mean={np.degrees(np.mean(steps)):.3f}")
print(f"Sum of per-step rotations = traveled = {np.degrees(sum(steps)):.2f}°")

# Angular velocity check — should match step angle / dt
omega_mag = np.linalg.norm(X[0:3, :], axis=0)
print(f"‖ω‖ (rad/s): min={omega_mag.min():.4f}  max={omega_mag.max():.4f}  mean={omega_mag.mean():.4f}")
print(f"∫‖ω‖dt over trajectory = {np.degrees(omega_mag.sum() * 10.0):.2f}°  (approximation of traveled)")

# Quaternion components over time
print(f"q_w (scalar) range: [{q_opt[0, :].min():.3f}, {q_opt[0, :].max():.3f}]")
print(f"q_x range: [{q_opt[1, :].min():.3f}, {q_opt[1, :].max():.3f}]")
print(f"q_y range: [{q_opt[2, :].min():.3f}, {q_opt[2, :].max():.3f}]")
print(f"q_z range: [{q_opt[3, :].min():.3f}, {q_opt[3, :].max():.3f}]")
