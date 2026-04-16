"""Sweep angle weight from 1e2 to 1e6 with ExcCtrl to find where it breaks."""
import os, sys, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))

import saltro_py
from sat_3_1_hybrid import create_satellite

x0 = np.array([0.01, 0.01, 0.01, 1, 0, 0, 0, 0.0])
r0 = np.array([7e6, 0, 0]); v0 = np.array([0, 7.5e3, 0])
jtime = np.array([0.22, 0.22 + 1000.0 / (36525.0 * 86400.0)])
qgoal = np.array([[np.sqrt(2)/2, np.sqrt(2)/2],[0,0],[0,0],[np.sqrt(2)/2, np.sqrt(2)/2]])
bs = np.array([[1,1],[0,0],[0,0]], dtype=float)
qg = qgoal[:, 0]

def run(angle_w, ic):
    ps = saltro_py.PlannerSettings()
    ps.init_traj.initcontroller = ic; ps.num_passes = 1
    ps.passes[0].dt = 10.0; ps.passes[0].ilqr.cost_tol = 1e-6
    ps.passes[0].ilqr.max_iters = 1000; ps.passes[0].auglag.max_outer_iters = 30
    ps.passes[0].auglag.constraint_tol = 1e-3
    c = ps.passes[0].cost
    c.angle = angle_w; c.ang_vel = angle_w / 100
    c.control_mult = 1.0; c.mtq_control_weight = 1e-1; c.rw_control_weight = 1.0
    c.angle_N = angle_w; c.ang_vel_N = angle_w / 100
    c.ang_cost_func_type = 3; c.use_cost_hess = True
    for a in ["aero","gg","srp","prop","gendist","resdipole"]:
        setattr(ps.disturbances, "plan_for_"+a, False)
    ps.passes[0].reg.reg_init = 1e-6; ps.passes[0].reg.reg_max = 1e30
    ps.passes[0].reg.reg_scale = 1.6
    ps.passes[0].linesearch.max_iters = 24
    ps.passes[0].linesearch.beta1 = 1e-10; ps.passes[0].linesearch.beta2 = 5000.0
    sat = create_satellite(ps)
    log_path = f"/tmp/sweep_{ic}_{angle_w:.0e}.log"
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC); old = os.dup(1)
    os.dup2(fd, 1); sys.stdout.flush()
    try:
        ok, X, U, K = saltro_py.trajOpt(ps, sat, x0, r0, v0, jtime, qgoal, bs)
    except Exception as e:
        os.dup2(old, 1); os.close(fd); os.close(old)
        eprint(f"    Exception: {e}")
        return False, None
    os.dup2(old, 1); os.close(fd); os.close(old)
    if X is None:
        return False, None
    pe = np.array([2*np.degrees(np.arccos(min(abs(float(np.dot(X[3:7,k], qg))), 1)))
                    for k in range(X.shape[1])])
    return ok, pe

import sys as _sys
def eprint(*args, **kwargs):
    print(*args, file=_sys.stderr, **kwargs)

import time as _time
eprint(f"{'IC':>10s} {'angle':>8s} {'ok':>6s} {'max_pe':>8s} {'final_pe':>8s} {'time':>6s}")
eprint("-" * 58)
for ic, ic_name in [(1, "ExcCtrl"), (2, "BdotCtrl")]:
    for angle_w in [1e4, 1e5, 1e6]:
        t0 = _time.time()
        ok, pe = run(angle_w, ic)
        elapsed = _time.time() - t0
        if pe is not None:
            eprint(f"{ic_name:>10s} {angle_w:8.0e} {str(ok):>6s} {pe.max():8.1f} {pe[-1]:8.1f} {elapsed:6.1f}s")
        else:
            eprint(f"{ic_name:>10s} {angle_w:8.0e}  THREW                  {elapsed:6.1f}s")
