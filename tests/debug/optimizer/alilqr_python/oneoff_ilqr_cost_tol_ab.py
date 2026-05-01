"""A/B: ilqr_cost_tol on vec 00_baseline (synced wrapper, GN on, type 3 + newpath).

Default ilqr_cost_tol = 1e0 is the loose inner-exit threshold from C++. With cost
magnitudes ~1e6, that's relative 1e-6 — already loose. Sweep tighter values to
see if cleaner inner subproblem solves let outer exit faster (better λ/μ
updates → fewer total outer iters → net iter savings AND deeper convergence).

Hypothesis: 1e-3 hits the sweet spot — tight enough to recover the regressed
hard cases (e.g., 11_long_3000s_dt30, 17_slew_180) without ballooning iter
count back to unsynced levels.

Configs (all else identical):
  A: ilqr_cost_tol = 1e0  (current default after sync)
  B: ilqr_cost_tol = 1e-2
  C: ilqr_cost_tol = 1e-3
  D: ilqr_cost_tol = 1e-4
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


def run(label, ilqr_cost_tol):
    ps = saltro_py.PlannerSettings()
    ps.init_traj.initcontroller = 1
    ps.num_passes = 1
    ps.passes[0].dt = 10.0
    ps.passes[0].ilqr.cost_tol = 1e-6
    ps.passes[0].ilqr.ilqr_cost_tol = ilqr_cost_tol
    ps.passes[0].ilqr.max_iters = 200
    ps.passes[0].ilqr.grad_tol = 0.0
    ps.passes[0].ilqr.z_count_lim = 10
    ps.passes[0].auglag.max_outer_iters = 30
    ps.passes[0].auglag.constraint_tol = 1e-3
    ps.passes[0].auglag.min_outer_iters = 3
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
    n_outer = int(snaps[-1].get("outer_iter", 0)) + 1 if snaps else 0
    return {
        "label": label,
        "ilqr_cost_tol": ilqr_cost_tol,
        "iters": len(snaps),
        "outer_iters": n_outer,
        "inner_per_outer_avg": len(snaps) / max(1, n_outer),
        "pe_fin": float(pe_fin),
        "wall_s": wall,
        "stop": stop[:60],
    }


configs = [
    ("A_1e0",   1e0),
    ("B_1em2",  1e-2),
    ("C_1em3",  1e-3),
    ("D_1em4",  1e-4),
]
rows = []
for cfg in configs:
    print(f"Running {cfg[0]} (ilqr_cost_tol={cfg[1]:.0e}) ...")
    rows.append(run(*cfg))
    r = rows[-1]
    print(f"  → iters={r['iters']}, outer={r['outer_iters']}, "
          f"in/out={r['inner_per_outer_avg']:.1f}, "
          f"PE_fin={r['pe_fin']:.4f}°, wall={r['wall_s']:.1f}s")

print()
print(f"{'cfg':<10} {'ict':>8} {'iters':>6} {'outer':>5} {'in/out':>7} "
      f"{'PE_fin':>9} {'wall':>7}  stop")
print("-" * 80)
for r in rows:
    print(f"{r['label']:<10} {r['ilqr_cost_tol']:>8.0e} {r['iters']:>6d} "
          f"{r['outer_iters']:>5d} {r['inner_per_outer_avg']:>7.1f} "
          f"{r['pe_fin']:>9.4f} {r['wall_s']:>6.1f}s  {r['stop']}")
