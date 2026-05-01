"""A/B: AL tolerances on vec 00_baseline (type=3 + GN + new path).

Profile showed inner iLQR converges in 28 iters but total run is 2439 iters
because AL outer keeps iterating. Test:

  A. baseline (current)        constraint_tol=1e-3, max_outer=30, cost_tol=1e-6
  B. loose constraint           constraint_tol=1e-2, max_outer=30, cost_tol=1e-6
  C. cap outer iters            constraint_tol=1e-3, max_outer=3,  cost_tol=1e-6
  D. tight inner (cost_tol)     constraint_tol=1e-3, max_outer=30, cost_tol=1e-9
  E. combined relax             constraint_tol=1e-2, max_outer=5,  cost_tol=1e-6
  F. combined tight inner+relax constraint_tol=1e-2, max_outer=5,  cost_tol=1e-9

Hypothesis: B/C/E save iters by exiting outer earlier with same PE_fin.
Hypothesis: D may reduce outer count by converging inner tighter per outer.
"""
import sys, time, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))

import saltro_py
from sat_3_1_hybrid import create_satellite
from trajOpt import trajOpt


def _quat_rotmat(q):
    q0, qv = q[0], q[1:]
    qx, qy, qz = qv
    skew = np.array([[0, -qz, qy], [qz, 0, -qx], [-qy, qx, 0]])
    return (q0*q0 - qv.dot(qv)) * np.eye(3) + 2*np.outer(qv, qv) + 2*q0*skew


def pe_fin_at(X, r_eci, bs=np.array([1.0, 0.0, 0.0])):
    q = X[3:7, -1]; qn = q / max(np.linalg.norm(q), 1e-12)
    bs_eci = _quat_rotmat(qn) @ bs
    return np.degrees(np.arccos(min(max(float(bs_eci.dot(r_eci)), -1.0), 1.0)))


def run(label, constraint_tol, max_outer_iters, cost_tol):
    ps = saltro_py.PlannerSettings()
    ps.init_traj.initcontroller = 1
    ps.num_passes = 1
    ps.passes[0].dt = 10.0
    ps.passes[0].ilqr.cost_tol = cost_tol
    ps.passes[0].ilqr.max_iters = 5000
    ps.passes[0].ilqr.grad_tol = 0.0
    ps.passes[0].auglag.max_outer_iters = max_outer_iters
    ps.passes[0].auglag.constraint_tol = constraint_tol
    c = ps.passes[0].cost
    c.angle = 1e4; c.ang_vel = 100.0
    c.control_mult = 1.0; c.mtq_control_weight = 0.1; c.rw_control_weight = 1.0
    c.ang_cost_func_type = 3
    c.use_cost_hess = True
    c.cost_hess_gauss_newton = True
    c.setTerminalEmphasis(100.0)
    c.ang_vel_err_dir = 0.0
    c.ang_vel_err_dir_ratio = 0.3
    c.ang_vel_roll_ratio = 0.05
    ps.passes[0].reg.reg_init = 1e-12; ps.passes[0].reg.reg_max = 1e30
    ps.passes[0].reg.reg_scale = 1.6
    ps.passes[0].linesearch.max_iters = 24
    ps.passes[0].linesearch.beta1 = 1e-10; ps.passes[0].linesearch.beta2 = 5000.0

    sat = create_satellite(ps)
    nRW = sat.numRW
    x0 = np.hstack([np.array([0.01, 0.01, 0.01]),
                    np.array([1.0, 0.0, 0.0, 0.0]),
                    np.zeros(nRW)])
    ang = np.radians(90.0)
    eci_target = np.array([np.cos(ang), np.sin(ang), 0.0])
    qg = np.array([np.nan, eci_target[0], eci_target[1], eci_target[2]])
    qgoal = np.tile(qg[:, None], (1, 2))
    bs = np.array([[1, 1], [0, 0], [0, 0]], dtype=float)
    r0 = np.array([7e6, 0, 0]); v0 = np.array([0, 7.5e3, 0])
    jtime = np.array([0.22, 0.22 + 1000.0 / (36525.0 * 86400.0)])

    cfg = {
        "start_at_iter": 2, "max_intervention_iters": 10000,
        "blend_len": 30, "goal_switch_buffer": 15, "min_consecutive": 7,
        "exit_fudge": 2.0, "min_prior_decrease_knots": 5, "min_spike_ratio": 2.0,
        "kp_q": 0.3, "kd_w": 2.0, "rw_scale": -1.0, "omega_max": 0.30,
        "verbose": False, "constraint_gate_ratio": 0.0,
    }

    t0 = time.time()
    X, U, stop, snaps, trans, dt_val, ctol_cfg, elapsed = trajOpt(
        ps, sat, x0, r0, v0, jtime, qgoal, bs, debug=True, spike_removal_cfg=cfg,
    )
    wall = time.time() - t0
    pe_fin = pe_fin_at(X, eci_target)
    pe_at_iter = np.array([pe_fin_at(s["X"], eci_target) for s in snaps])
    n_outer = int(snaps[-1].get("outer_iter", 0)) + 1 if snaps else 0
    return {
        "label": label,
        "ctol": constraint_tol,
        "max_outer": max_outer_iters,
        "cost_tol": cost_tol,
        "pe_fin": float(pe_fin),
        "iters": len(snaps),
        "outer_iters": n_outer,
        "wall_s": wall,
        "stop": stop[:60],
        "pe_at_50": float(pe_at_iter[49]) if len(pe_at_iter) > 49 else None,
        "pe_at_100": float(pe_at_iter[99]) if len(pe_at_iter) > 99 else None,
    }


configs = [
    # (label,        ctol,  max_outer, cost_tol)
    ("A_baseline",   1e-3,  30,        1e-6),
    ("B_loose_ctol", 1e-2,  30,        1e-6),
    ("C_cap_outer",  1e-3,   3,        1e-6),
    ("D_tight_inner",1e-3,  30,        1e-9),
    ("E_combined",   1e-2,   5,        1e-6),
    ("F_tight+relax",1e-2,   5,        1e-9),
]
rows = []
for cfg in configs:
    print(f"Running {cfg[0]} ...")
    rows.append(run(*cfg))
    print(f"  → iters={rows[-1]['iters']}, outer={rows[-1]['outer_iters']}, "
          f"PE_fin={rows[-1]['pe_fin']:.3f}°, wall={rows[-1]['wall_s']:.1f}s")

print()
print(f"{'cfg':<18} {'ctol':>8} {'mo':>4} {'ct':>7} {'iters':>6} {'outer':>5} "
      f"{'PE_fin':>8} {'PE@50':>7} {'PE@100':>8} {'wall':>7}")
print("-" * 92)
for r in rows:
    pe50 = f"{r['pe_at_50']:.3f}" if r['pe_at_50'] is not None else "n/a"
    pe100 = f"{r['pe_at_100']:.3f}" if r['pe_at_100'] is not None else "n/a"
    print(f"{r['label']:<18} {r['ctol']:>8.0e} {r['max_outer']:>4d} {r['cost_tol']:>7.0e} "
          f"{r['iters']:>6d} {r['outer_iters']:>5d} {r['pe_fin']:>8.3f} "
          f"{pe50:>7} {pe100:>8} {r['wall_s']:>6.1f}s")
