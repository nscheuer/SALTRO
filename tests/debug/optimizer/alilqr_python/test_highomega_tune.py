"""Tune cost/AL for high-ω cases.

Report actuator utilization (mtq %, rw %, h_rw % of sat), ω peak,
PE_mean, PE_fin.  Sweep angle_N, total_cost_tol, penalty_max.
"""
import sys, time, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))

import saltro_py
from trajOpt import trajOpt
from sat_3_1_hybrid import create_satellite


def build(angle=1e4, angle_N_mult=1.0, ang_vel=1e2, ang_vel_N_mult=1.0,
          mtq_cw=1e-1, rw_cw=1.0,
          total_cost_tol=1e-2, penalty_max=1e16, penalty_scale=10.0,
          ls_attempts_lim=30):
    ps = saltro_py.PlannerSettings()
    ps.init_traj.initcontroller = 1; ps.num_passes = 1
    ps.passes[0].dt = 10.0
    ps.passes[0].ilqr.cost_tol = 1e-6
    ps.passes[0].ilqr.max_iters = 300
    ps.passes[0].ilqr.grad_tol = 0.0
    ps.passes[0].ilqr.ls_attempts_lim = ls_attempts_lim
    ps.passes[0].auglag.max_outer_iters = 30
    ps.passes[0].auglag.constraint_tol = 1e-3
    ps.passes[0].auglag.total_cost_tol = total_cost_tol
    ps.passes[0].auglag.penalty_max = penalty_max
    ps.passes[0].auglag.penalty_scale = penalty_scale
    c = ps.passes[0].cost
    c.angle = angle; c.ang_vel = ang_vel
    c.control_mult = 1.0
    c.mtq_control_weight = mtq_cw; c.rw_control_weight = rw_cw
    c.angle_N = angle * angle_N_mult; c.ang_vel_N = ang_vel * ang_vel_N_mult
    c.ang_cost_func_type = 3; c.use_cost_hess = True
    ps.passes[0].reg.reg_init = 1e-6; ps.passes[0].reg.reg_max = 1e30
    ps.passes[0].reg.reg_scale = 1.6
    ps.passes[0].linesearch.max_iters = 24
    ps.passes[0].linesearch.beta1 = 1e-10; ps.passes[0].linesearch.beta2 = 5000.0
    return ps


def run(name, omega0, time_s, use_spike=True, **build_kw):
    ps = build(**build_kw)
    sat = create_satellite(ps)
    x0 = np.hstack([[omega0, omega0, omega0], [1, 0, 0, 0], np.zeros(sat.numRW)])
    r0 = np.array([7e6, 0, 0]); v0 = np.array([0, 7.5e3, 0])
    jtime = np.array([0.22, 0.22 + time_s/(36525*86400)])
    qg = np.array([np.sqrt(2)/2, 0, 0, np.sqrt(2)/2])
    cfg = None
    if use_spike:
        cfg = {"start_at_iter": 2, "max_intervention_iters": 20, "blend_len": 30,
               "goal_switch_buffer": 15, "min_consecutive": 7, "exit_fudge": 2.0,
               "min_prior_decrease_knots": 5, "min_spike_ratio": 2.0,
               "kp_q": 0.3, "kd_w": 2.0, "rw_scale": 0.0,
               "omega_max": 0.30, "verbose": False}
    t0 = time.time()
    try:
        X, U, stop, snaps, *_ = trajOpt(
            ps, sat, x0, r0, v0, jtime,
            np.tile(qg[:, None], (1, 2)),
            np.array([[1,1],[0,0],[0,0]], dtype=float),
            debug=True, spike_removal_cfg=cfg)
    except Exception as e:
        print(f"  {name:<30} FAIL: {type(e).__name__}", flush=True)
        return
    wall = time.time() - t0

    pe = np.array([2*np.degrees(np.arccos(min(abs(float(np.dot(X[3:7,k], qg))), 1)))
                   for k in range(X.shape[1])])
    w_peak = float(np.max(np.abs(X[:3, :])) * 180 / np.pi)

    # Actuator utilization
    n_mtq = sat.numMTQ; n_rw = sat.numRW
    mtq_util = 0.0
    rw_util = 0.0
    h_util = 0.0
    if n_mtq > 0:
        mtq_max = sat.getMTQ(0).u_max
        mtq_util = float(np.max(np.abs(U[:n_mtq, :])) / mtq_max * 100)
    if n_rw > 0:
        rw_max = sat.getRW(0).u_max
        h_max = sat.getRW(0).momentumMax
        rw_util = float(np.max(np.abs(U[n_mtq:n_mtq+n_rw, :])) / rw_max * 100)
        # h_rw in x[7:7+n_rw]
        h_peak = float(np.max(np.abs(X[7:7+n_rw, :])))
        h_util = h_peak / h_max * 100

    print(f"  {name:<30} it={len(snaps):>4}  t={wall:>4.1f}s  "
          f"PE_fin={pe[-1]:>6.1f}°  PE_mean={pe.mean():>5.1f}°  "
          f"ω={w_peak:>5.1f}°/s  mtq={mtq_util:>5.1f}%  rw={rw_util:>5.1f}%  "
          f"h={h_util:>5.1f}%  stop={stop.split(':')[0]}", flush=True)


def section(title):
    print("\n" + "=" * 130)
    print(title)
    print("=" * 130)


# --------------------- ω=5× baseline probe ---------------------
section("ω=5× 1000s no spike — baseline to see actuator usage")
run("DEFAULT (ang_N = ang)",   0.05, 1000.0, use_spike=False)

section("ω=5× 1000s no spike — angle_N sweep")
run("ang_N = 10× ang",          0.05, 1000.0, use_spike=False, angle_N_mult=10)
run("ang_N = 100× ang",         0.05, 1000.0, use_spike=False, angle_N_mult=100)
run("ang_N = 1000× ang",        0.05, 1000.0, use_spike=False, angle_N_mult=1000)

section("ω=5× 1000s no spike — cost tolerance sweep (total_cost_tol)")
run("tcostol=1e-2 (default)",   0.05, 1000.0, use_spike=False, total_cost_tol=1e-2)
run("tcostol=1e-4",             0.05, 1000.0, use_spike=False, total_cost_tol=1e-4)
run("tcostol=1e-6",             0.05, 1000.0, use_spike=False, total_cost_tol=1e-6)

section("ω=5× 1000s no spike — penalty_max sweep")
run("pen_max=1e16 (default)",   0.05, 1000.0, use_spike=False, penalty_max=1e16)
run("pen_max=1e20",             0.05, 1000.0, use_spike=False, penalty_max=1e20)
run("pen_max=1e24",             0.05, 1000.0, use_spike=False, penalty_max=1e24)

section("ω=5× 1000s no spike — combined tweaks")
run("ang_N=100×  tcostol=1e-4",  0.05, 1000.0, use_spike=False,
    angle_N_mult=100, total_cost_tol=1e-4)
run("ang_N=100×  tcostol=1e-4  pen_max=1e20", 0.05, 1000.0, use_spike=False,
    angle_N_mult=100, total_cost_tol=1e-4, penalty_max=1e20)

section("ω=10× 1000s no spike — combined tweaks")
run("DEFAULT",                   0.10, 1000.0, use_spike=False)
run("ang_N=100× tcostol=1e-4 pen20", 0.10, 1000.0, use_spike=False,
    angle_N_mult=100, total_cost_tol=1e-4, penalty_max=1e20)

section("ω=10× 2000s no spike — combined tweaks")
run("DEFAULT",                   0.10, 2000.0, use_spike=False)
run("ang_N=100× tcostol=1e-4 pen20", 0.10, 2000.0, use_spike=False,
    angle_N_mult=100, total_cost_tol=1e-4, penalty_max=1e20)

section("Best settings with spike removal on")
run("ω=5× 1000s  combined + spike",  0.05, 1000.0, use_spike=True,
    angle_N_mult=100, total_cost_tol=1e-4, penalty_max=1e20)
run("ω=10× 1000s combined + spike",  0.10, 1000.0, use_spike=True,
    angle_N_mult=100, total_cost_tol=1e-4, penalty_max=1e20)
run("ω=10× 2000s combined + spike",  0.10, 2000.0, use_spike=True,
    angle_N_mult=100, total_cost_tol=1e-4, penalty_max=1e20)
