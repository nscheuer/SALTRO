"""Iteration-level diagnostics for ExcCtrl at different angle weights."""
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

def pe_profile(X):
    N = X.shape[1]
    return np.array([2*np.degrees(np.arccos(min(abs(float(np.dot(X[3:7,k], qg))), 1)))
                      for k in range(N)])

def make_ps(angle_w):
    ps = saltro_py.PlannerSettings()
    ps.init_traj.initcontroller = 1  # ExcCtrl
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

for angle_w in [1e2, 1e4, 1e6]:
    ps = make_ps(angle_w)
    sat = create_satellite(ps)
    print(f"\n{'='*80}")
    print(f"angle={angle_w:.0e}  ExcCtrl  3MTQ+1RW  (max_iters=20, max_outer=10)")
    print(f"{'='*80}")

    X, U, stop_reason, snapshots, transitions, dt, cost_tol, elapsed = trajOpt(
        ps, sat, x0, r0, v0, jtime, qgoal, bs, debug=True
    )

    print(f"Stop: {stop_reason}  elapsed: {elapsed:.2f}s")
    print(f"{'#':>4s} {'outer':>5s} {'J':>14s} {'dJ':>14s} {'max_pe':>8s} {'final_pe':>8s} {'stop':>20s}")
    print("-" * 80)

    prev_J = None
    for i, snap in enumerate(snapshots):
        J = snap['J']
        dJ = J - prev_J if prev_J is not None else 0
        prev_J = J
        pe = pe_profile(snap['X'])
        outer = snap.get('outer_iter', '?')
        stop = snap.get('stop_reason', '')
        print(f"{i:4d} {outer:>5} {J:14.4e} {dJ:+14.4e} {pe.max():8.1f} {pe[-1]:8.2f} {stop:>20s}")

    # Print AL outer loop transitions
    print(f"\nAL outer loop transitions:")
    for t in transitions:
        if isinstance(t, dict):
            print(f"  outer={t.get('outer_iter','?')}  max_c={t.get('max_constraint_violation',0):.4e}  "
                  f"mu_max={t.get('mu_max',0):.4e}  lambda_max={t.get('lambda_max',0):.4e}")

    pe_final = pe_profile(X)
    print(f"\nFinal: max_pe={pe_final.max():.1f}  final_pe={pe_final[-1]:.2f}")
