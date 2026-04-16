"""Test impact of each ALTRO fix flag individually and in combination."""
import os, sys, time, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))

import saltro_py
from sat_3_1_hybrid import create_satellite

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

qg = np.array([np.sqrt(2)/2, 0, 0, np.sqrt(2)/2])
x0 = np.array([0.01, 0.01, 0.01, 1, 0, 0, 0, 0.0])
r0 = np.array([7e6, 0, 0]); v0 = np.array([0, 7.5e3, 0])
jtime = np.array([0.22, 0.22 + 1000.0 / (36525.0 * 86400.0)])
qgoal = np.array([[np.sqrt(2)/2, np.sqrt(2)/2],[0,0],[0,0],[np.sqrt(2)/2, np.sqrt(2)/2]])
bs = np.array([[1,1],[0,0],[0,0]], dtype=float)

def run(angle_w, ls_strict, conj_conv, persist_reg, ic=1):
    ps = saltro_py.PlannerSettings()
    ps.init_traj.initcontroller = ic; ps.num_passes = 1
    ps.passes[0].dt = 10.0; ps.passes[0].ilqr.cost_tol = 1e-6
    ps.passes[0].ilqr.max_iters = 200; ps.passes[0].ilqr.grad_tol = 0.0
    ps.passes[0].auglag.max_outer_iters = 30
    ps.passes[0].auglag.constraint_tol = 1e-3
    # Flags
    ps.passes[0].ilqr.ls_strict_decrease = ls_strict
    ps.passes[0].ilqr.conjunctive_convergence = conj_conv
    ps.passes[0].ilqr.persistent_reg = persist_reg

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
    sat = create_satellite(ps)
    fd = os.open(os.devnull, os.O_WRONLY); old = os.dup(1)
    os.dup2(fd, 1); sys.stdout.flush()
    try:
        ok, X, U, K = saltro_py.trajOpt(ps, sat, x0, r0, v0, jtime, qgoal, bs)
    except Exception:
        ok = False; X = None
    os.dup2(old, 1); os.close(fd); os.close(old)
    if X is None:
        return False, None, None
    pe = np.array([2*np.degrees(np.arccos(min(abs(float(np.dot(X[3:7,k], qg))), 1)))
                    for k in range(X.shape[1])])
    max_u = float(np.max(np.abs(U)))
    return ok, pe, max_u

eprint(f"{'flags':>20s} {'angle':>8s} {'ok':>5s} {'max_pe':>8s} {'final':>8s} {'max|u|':>8s} {'time':>6s}")
eprint("-" * 75)

for angle_w in [1e4, 1e6]:
    configs = [
        ("none",           False, False, False),
        ("strict_LS",      True,  False, False),
        ("conj_conv",      False, True,  False),
        ("persist_reg",    False, False, True),
        ("strict+conj",    True,  True,  False),
        ("strict+persist", True,  False, True),
        ("ALL",            True,  True,  True),
    ]
    for label, ls, cc, pr in configs:
        t0 = time.time()
        ok, pe, mu = run(angle_w, ls, cc, pr)
        elapsed = time.time() - t0
        if pe is not None:
            eprint(f"{label:>20s} {angle_w:8.0e} {str(ok):>5s} {pe.max():8.1f} {pe[-1]:8.2f} {mu:8.2f} {elapsed:6.1f}s")
        else:
            eprint(f"{label:>20s} {angle_w:8.0e} THREW                          {elapsed:6.1f}s")
