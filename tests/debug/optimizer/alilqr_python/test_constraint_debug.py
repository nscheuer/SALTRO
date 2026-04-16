"""Trace which constraints blow up at angle=1e6."""
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

def make_ps(angle_w):
    ps = saltro_py.PlannerSettings()
    ps.init_traj.initcontroller = 1
    ps.num_passes = 1
    ps.passes[0].dt = 10.0
    ps.passes[0].ilqr.cost_tol = 1e-6
    ps.passes[0].ilqr.max_iters = 20
    ps.passes[0].auglag.max_outer_iters = 10
    ps.passes[0].auglag.constraint_tol = 1e-3
    c = ps.passes[0].cost
    c.angle = angle_w; c.ang_vel = angle_w / 100
    c.control_mult = 1.0; c.mtq_control_weight = 1e-1; c.rw_control_weight = 1.0
    c.angle_N = angle_w; c.ang_vel_N = angle_w / 100
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

def analyze_constraints(sat, ps, X, U, S):
    """Print per-knot constraint violations."""
    N = X.shape[1]
    cnst_cfg = ps.constraints
    nu = sat.controlDim

    # Get constraint names from first knot
    u0 = U[:, 0] if U.shape[1] > 0 else np.zeros(nu)
    c0 = np.asarray(sat.constraints(0, N, X[:, 0], u0, S[:, 0], cnst_cfg))
    nc = len(c0)

    # Constraint ordering for 3MTQ+1RW:
    # 0: angular velocity  (||w||^2 - wmax^2) / wmax^2
    # 1: sun avoidance     (s^T R b_cam - cos(limit))
    # 2..2+2*nMTQ-1: MTQ bounds (+/- for each)
    # then RW bounds, then RW stiction
    names = ["ang_vel", "sun_avoid"]
    for i in range(sat.numMTQ):
        names.extend([f"mtq{i}_hi", f"mtq{i}_lo"])
    for i in range(sat.numRW):
        names.extend([f"rw{i}_tau_hi", f"rw{i}_tau_lo", f"rw{i}_h_hi", f"rw{i}_h_lo", f"rw{i}_stic"])

    # Find max violation per constraint type across all knots
    max_per_type = np.full(nc, -np.inf)
    worst_k_per_type = np.zeros(nc, dtype=int)
    for k in range(N):
        uk = U[:, k] if k < U.shape[1] else np.zeros(nu)
        ck = np.asarray(sat.constraints(k, N, X[:, k], uk, S[:, k], cnst_cfg))
        for i in range(min(nc, len(ck))):
            if ck[i] > max_per_type[i]:
                max_per_type[i] = ck[i]
                worst_k_per_type[i] = k

    print(f"\n  Constraint violations (max over all knots):")
    for i in range(min(nc, len(names))):
        name = names[i] if i < len(names) else f"c{i}"
        v = max_per_type[i]
        k = worst_k_per_type[i]
        if v > 0:
            print(f"    {name:>15s}: {v:10.4f}  (worst at k={k})")
        else:
            print(f"    {name:>15s}: {v:10.4f}  (satisfied)")

    # Also print angular velocity profile
    wmax = ps.constraints.wmax
    print(f"\n  Angular velocity profile (wmax={np.degrees(wmax):.1f} deg/s = {wmax:.4f} rad/s):")
    for k in range(0, N, 10):
        w = X[0:3, k]
        wnorm = np.linalg.norm(w)
        ratio = wnorm / wmax
        bar = "#" * min(int(ratio * 10), 80)
        print(f"    k={k:3d}  ||w||={np.degrees(wnorm):7.2f} deg/s  ({ratio:.1f}x wmax) {bar}")

for angle_w, pen_init in [(1e6, 0.1), (1e6, 1.0), (1e6, 1e2), (1e6, 1e3)]:
    ps = make_ps(angle_w)
    ps.passes[0].auglag.penalty_init = pen_init
    sat = create_satellite(ps)
    print(f"\n{'='*80}")
    print(f"angle={angle_w:.0e}  penalty_init={pen_init:.0e}  ExcCtrl  3MTQ+1RW")
    print(f"{'='*80}")

    X, U, stop_reason, snapshots, transitions, dt, cost_tol, elapsed = trajOpt(
        ps, sat, x0, r0, v0, jtime, qgoal, bs, debug=True
    )

    # Generate orbit for S
    ok_orbit, R, V, B, S, rho = saltro_py.generate_orbit(r0, v0,
        np.array([jtime[0] + i * dt / (36525*86400) for i in range(X.shape[1])]), 0,0,0,0,0)

    print(f"Stop: {stop_reason}")
    analyze_constraints(sat, ps, X, U, S)

    # Also check the warm-start trajectory
    print(f"\n  --- Warm-start trajectory ---")
    if len(snapshots) > 0:
        X0 = snapshots[0]['X']
        U0 = snapshots[0]['U']
        analyze_constraints(sat, ps, X0, U0, S)
