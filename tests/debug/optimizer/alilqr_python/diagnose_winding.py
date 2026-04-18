"""Run three-pass to convergence, then inspect winding excess over the trajectory.

Tells us whether the winding detector's persistent flag is:
 (a) catching a real residual spike that three-pass missed, or
 (b) over-flagging natural trajectory curvature.
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

qg = np.array([np.sqrt(2)/2, 0, 0, np.sqrt(2)/2])
x0 = np.array([0.01, 0.01, 0.01, 1, 0, 0, 0, 0.0])
r0 = np.array([7e6, 0, 0]); v0 = np.array([0, 7.5e3, 0])
jtime = np.array([0.22, 0.22 + 1000.0 / (36525.0 * 86400.0)])
qgoal = np.tile(qg[:, None], (1, 2))
bs = np.array([[1,1],[0,0],[0,0]], dtype=float)

def run(winding):
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
    cfg = {
        "start_at_iter": 2, "max_intervention_iters": 20,
        "blend_len": 30, "goal_switch_buffer": 15, "min_consecutive": 7,
        "exit_fudge": 2.0, "min_prior_decrease_knots": 5, "min_spike_ratio": 2.0,
        "kp_q": 0.3, "kd_w": 2.0, "rw_scale": 0.0, "omega_max": 0.30, "verbose": False,
    }
    if winding:
        cfg["winding_detector"] = True
        cfg["winding_excess_threshold"] = np.pi

    X, U, stop, snaps, trans, dt_val, ctol, elapsed = trajOpt(
        ps, sat, x0, r0, v0, jtime, qgoal, bs, debug=True, spike_removal_cfg=cfg,
    )
    return X, U, stop, len(snaps), elapsed

def excess_profile(X, t1, t2):
    step = [2*np.arccos(min(abs(float(np.dot(X[3:7,k], X[3:7,k+1]))), 1.0))
            for k in range(t1, t2)]
    traveled = sum(step)
    direct = 2*np.arccos(min(abs(float(np.dot(X[3:7,t1], X[3:7,t2]))), 1.0))
    return traveled, direct, traveled - direct

def pe_profile(X):
    return np.array([2*np.degrees(np.arccos(min(abs(float(np.dot(X[3:7,k], qg))), 1)))
                     for k in range(X.shape[1])])

# Run three-pass to convergence
print("=== THREE-PASS ===")
X_tp, U_tp, stop_tp, n_tp, elapsed_tp = run(winding=False)
print(f"stop: {stop_tp}")
print(f"iters: {n_tp}, elapsed: {elapsed_tp:.1f}s")
N = X_tp.shape[1]
trav, direct, exc = excess_profile(X_tp, 0, N-1)
pe = pe_profile(X_tp)
print(f"final PE: mean={pe.mean():.1f}° max={pe.max():.1f}° (knot {pe.argmax()})")
print(f"winding over (0,{N-1}): traveled={np.degrees(trav):.1f}°  direct={np.degrees(direct):.1f}°  excess={np.degrees(exc):.1f}°")
# Sliding window to find the biggest excess window
best = (0, 0, 0)  # (t1, t2, excess)
for t1 in range(0, N-5):
    for w in (10, 20, 30, 50, 80):
        if t1+w >= N: continue
        _, _, e = excess_profile(X_tp, t1, t1+w)
        if e > best[2]:
            best = (t1, t1+w, e)
print(f"max excess window: ({best[0]}, {best[1]}) excess={np.degrees(best[2]):.1f}°")

print("\n=== WINDING (threshold π) ===")
X_w, U_w, stop_w, n_w, elapsed_w = run(winding=True)
print(f"stop: {stop_w}")
print(f"iters: {n_w}, elapsed: {elapsed_w:.1f}s")
trav, direct, exc = excess_profile(X_w, 0, N-1)
pe = pe_profile(X_w)
print(f"final PE: mean={pe.mean():.1f}° max={pe.max():.1f}° (knot {pe.argmax()})")
print(f"winding over (0,{N-1}): traveled={np.degrees(trav):.1f}°  direct={np.degrees(direct):.1f}°  excess={np.degrees(exc):.1f}°")
