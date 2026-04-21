"""Fine sweep of angle_N/angle ratio to see where the bad band is.

Also includes the ang_vel terminal multiplier to check interactions.
Only tests the ω=5× 1000s no-spike case (known to depend on tuning).
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


def build(ang_N_mult=1.0, av_N_mult=1.0):
    ps = saltro_py.PlannerSettings()
    ps.init_traj.initcontroller = 1; ps.num_passes = 1
    ps.passes[0].dt = 10.0
    ps.passes[0].ilqr.cost_tol = 1e-6
    ps.passes[0].ilqr.max_iters = 300
    ps.passes[0].ilqr.grad_tol = 0.0
    ps.passes[0].ilqr.ls_attempts_lim = 30
    ps.passes[0].auglag.max_outer_iters = 30
    ps.passes[0].auglag.constraint_tol = 1e-3
    c = ps.passes[0].cost
    c.angle = 1e4; c.ang_vel = 1e2
    c.control_mult = 1.0
    c.mtq_control_weight = 1e-1; c.rw_control_weight = 1.0
    c.angle_N = 1e4 * ang_N_mult
    c.ang_vel_N = 1e2 * av_N_mult
    c.ang_cost_func_type = 3; c.use_cost_hess = True
    ps.passes[0].reg.reg_init = 1e-6; ps.passes[0].reg.reg_max = 1e30
    ps.passes[0].reg.reg_scale = 1.6
    ps.passes[0].linesearch.max_iters = 24
    ps.passes[0].linesearch.beta1 = 1e-10; ps.passes[0].linesearch.beta2 = 5000.0
    return ps


def run(ang_N_mult, av_N_mult=1.0, omega0=0.05, time_s=1000.0):
    ps = build(ang_N_mult=ang_N_mult, av_N_mult=av_N_mult)
    sat = create_satellite(ps)
    x0 = np.hstack([[omega0, omega0, omega0], [1, 0, 0, 0], np.zeros(sat.numRW)])
    r0 = np.array([7e6, 0, 0]); v0 = np.array([0, 7.5e3, 0])
    jtime = np.array([0.22, 0.22 + time_s/(36525*86400)])
    qg = np.array([np.sqrt(2)/2, 0, 0, np.sqrt(2)/2])
    t0 = time.time()
    try:
        X, U, stop, snaps, *_ = trajOpt(
            ps, sat, x0, r0, v0, jtime,
            np.tile(qg[:, None], (1, 2)),
            np.array([[1,1],[0,0],[0,0]], dtype=float),
            debug=True, spike_removal_cfg=None)
    except Exception as e:
        print(f"  angN={ang_N_mult:>5.1f}×  avN={av_N_mult:>4.1f}×  FAIL", flush=True)
        return
    wall = time.time() - t0
    pe = np.array([2*np.degrees(np.arccos(min(abs(float(np.dot(X[3:7,k], qg))), 1)))
                   for k in range(X.shape[1])])
    w_peak = float(np.max(np.abs(X[:3, :])) * 180 / np.pi)
    n_mtq = sat.numMTQ; n_rw = sat.numRW
    mtq_max = sat.getMTQ(0).u_max if n_mtq > 0 else 1
    rw_max = sat.getRW(0).u_max if n_rw > 0 else 1
    mtq_util = float(np.max(np.abs(U[:n_mtq, :])) / mtq_max * 100) if n_mtq else 0
    rw_util = float(np.max(np.abs(U[n_mtq:n_mtq+n_rw, :])) / rw_max * 100) if n_rw else 0
    print(f"  angN={ang_N_mult:>5.1f}×  avN={av_N_mult:>4.1f}×  it={len(snaps):>4}  t={wall:>4.1f}s  "
          f"PE_fin={pe[-1]:>6.1f}°  PE_mean={pe.mean():>5.1f}°  ω={w_peak:>5.1f}°/s  "
          f"mtq={mtq_util:>5.1f}%  rw={rw_util:>5.1f}%  stop={stop.split(':')[0]}",
          flush=True)


print("="*130)
print("Fine angle_N sweep — ω=5× 1000s no spike")
print("="*130)
print("\n  av_N_mult = 1.0  (default)")
for m in [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0, 70.0, 100.0, 150.0, 200.0, 300.0, 500.0, 1000.0]:
    run(m, 1.0)

print("\n  av_N_mult = 10.0  (also scale terminal ang_vel)")
for m in [1.0, 10.0, 100.0, 1000.0]:
    run(m, 10.0)

print("\n  av_N_mult = 100.0  (matching terminal weights)")
for m in [1.0, 10.0, 100.0, 1000.0]:
    run(m, 100.0)

print("\n  Repeat key angle_N values on ω=10× 1000s no spike:")
for m in [1.0, 30.0, 100.0, 300.0]:
    run(m, 1.0, omega0=0.10, time_s=1000.0)

print("\n  Repeat on baseline (ω=0.01) — see if 100× regresses what works:")
for m in [1.0, 30.0, 100.0, 300.0]:
    run(m, 1.0, omega0=0.01, time_s=1000.0)
