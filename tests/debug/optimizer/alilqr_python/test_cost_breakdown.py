"""Cost breakdown: what dominates the cost at each iteration?"""
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

def make_ps(angle_w, av_w):
    ps = saltro_py.PlannerSettings()
    ps.init_traj.initcontroller = 1
    ps.num_passes = 1
    ps.passes[0].dt = 10.0
    ps.passes[0].ilqr.cost_tol = 1e-6
    ps.passes[0].ilqr.max_iters = 10
    ps.passes[0].auglag.max_outer_iters = 3
    ps.passes[0].auglag.constraint_tol = 1e-3
    c = ps.passes[0].cost
    c.angle = angle_w; c.ang_vel = av_w
    c.control_mult = 1.0; c.mtq_control_weight = 1e-1; c.rw_control_weight = 1.0
    c.angle_N = angle_w; c.ang_vel_N = av_w
    c.ang_cost_func_type = 3; c.use_cost_hess = True
    c.rw_AM_weight = 0.0; c.rw_stic_weight = 0.0
    c.RWh_max_mult = 0.0; c.RWh_stiction_mult = 0.0; c.RWh_ok_mult = 0.0
    for a in ["aero","gg","srp","prop","gendist","resdipole"]:
        setattr(ps.disturbances, "plan_for_"+a, False)
    ps.passes[0].reg.reg_init = 1e-6; ps.passes[0].reg.reg_max = 1e30
    ps.passes[0].reg.reg_scale = 10.0
    ps.passes[0].linesearch.max_iters = 24
    ps.passes[0].linesearch.beta1 = 1e-10; ps.passes[0].linesearch.beta2 = 5000.0
    return ps

def analyze(snap, sat, B, dt):
    """Break down cost into angle, ang_vel, control contributions."""
    X = snap['X']; U = snap['U']
    N = X.shape[1]

    sum_angle = 0.0
    sum_av = 0.0
    sum_ctrl = 0.0
    max_u = 0.0
    max_w = 0.0

    for k in range(N):
        q = X[3:7, k]
        w = X[0:3, k]
        wnorm = np.linalg.norm(w)
        if wnorm > max_w:
            max_w = wnorm

        # Angle cost: 0.5 * w_angle * phi^2
        dot = min(abs(float(np.dot(q, qg))), 1.0)
        phi = np.arccos(dot)
        sum_angle += 0.5 * phi * phi

        # Angular velocity cost: 0.5 * w_av * ||w||^2
        sum_av += 0.5 * np.dot(w, w)

        # Control cost (normalized)
        if k < U.shape[1]:
            u = U[:, k]
            if np.max(np.abs(u)) > max_u:
                max_u = np.max(np.abs(u))
            for i in range(int(sat.numMTQ)):
                lim = max(1e-9, abs(sat.getMTQ(i).u_max))
                sum_ctrl += 0.5 * 1e-1 * (u[i] / lim) ** 2
            for i in range(int(sat.numRW)):
                lim = max(1e-9, abs(sat.getRW(i).u_max))
                sum_ctrl += 0.5 * 1.0 * (u[sat.numMTQ + i] / lim) ** 2

    return sum_angle, sum_av, sum_ctrl, max_u, np.degrees(max_w)

for angle_w, av_w, label in [
    (1e2, 1e0, "baseline"),
    (1e6, 1e4, "high angle"),
    (1e6, 1e6, "high angle+av"),
]:
    ps = make_ps(angle_w, av_w)
    sat = create_satellite(ps)
    print(f"\n{'='*90}")
    print(f"{label}: angle={angle_w:.0e}  ang_vel={av_w:.0e}")
    print(f"{'='*90}")

    X, U, stop_reason, snapshots, transitions, dt_val, cost_tol, elapsed = trajOpt(
        ps, sat, x0, r0, v0, jtime, qgoal, bs, debug=True
    )

    ok_orbit, R, V, B, S, rho = saltro_py.generate_orbit(r0, v0,
        np.array([jtime[0] + i * dt_val / (36525*86400) for i in range(X.shape[1])]), 0,0,0,0,0)

    print(f"Stop: {stop_reason}  {len(snapshots)} snapshots")
    print(f"{'iter':>4s} {'sum_phi2':>12s} {'sum_w2':>12s} {'sum_ctrl':>12s} {'max|u|':>10s} {'max||w||':>10s}")
    print("-" * 70)
    if not snapshots:
        # No snapshots — just analyze final
        snapshots = [{'X': X, 'U': U}]
    for i, snap in enumerate(snapshots):
        try:
            s_ang, s_av, s_ctrl, mu, mw = analyze(snap, sat, B, dt_val)
            print(f"{i:4d} {s_ang:12.4f} {s_av:12.6f} {s_ctrl:12.2f} {mu:10.4f} {mw:8.1f} deg/s")
        except Exception as e:
            print(f"{i:4d} ERROR: {e}")
            break

    print(f"\nWeighted costs at final iter:")
    s_ang, s_av, s_ctrl, mu, mw = analyze(snapshots[-1], sat, B, dt_val)
    print(f"  angle_cost  = {angle_w} * {s_ang:.4f} = {angle_w * s_ang:.4e}")
    print(f"  av_cost     = {av_w} * {s_av:.6f} = {av_w * s_av:.4e}")
    print(f"  ctrl_cost   = 1.0 * {s_ctrl:.2f} = {s_ctrl:.4e}")
    print(f"  max|u| = {mu:.4f}  (MTQ lim={sat.getMTQ(0).u_max:.4f}  RW lim={sat.getRW(0).u_max:.6f})")
    print(f"  max||w|| = {mw:.1f} deg/s  (wmax={np.degrees(ps.constraints.wmax):.1f} deg/s)")
