"""Compare Python iLQR vs C++ trajOpt on exact same scenario."""
import os
import sys
import time
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))

import saltro_py
from sat_3_1_hybrid import create_satellite
from trajOpt import trajOpt as py_trajOpt


def make_settings():
    ps = saltro_py.PlannerSettings()
    ps.init_traj.initcontroller = 1
    ps.num_passes = 1
    ps.passes[0].dt = 10.0
    ps.passes[0].ilqr.cost_tol = 1e-3
    ps.passes[0].ilqr.max_iters = 20
    ps.passes[0].auglag.max_outer_iters = 10
    ps.passes[0].auglag.constraint_tol = 1e-3

    cost = ps.passes[0].cost
    cost.angle = 1e2
    cost.ang_vel = 1e1
    cost.control_mult = 1.0
    cost.mtq_control_weight = 1e-1
    cost.rw_control_weight = 1.0
    cost.angle_N = 1e2
    cost.ang_vel_N = 1e1
    cost.ang_cost_func_type = 3
    cost.use_cost_hess = True

    for a in ['aero', 'gg', 'srp', 'prop', 'gendist', 'resdipole']:
        setattr(ps.disturbances, f'plan_for_{a}', False)

    ps.passes[0].reg.reg_init = 1e-6
    ps.passes[0].reg.reg_max = 1e10
    ps.passes[0].reg.reg_scale = 10.0

    ps.passes[0].linesearch.max_iters = 24
    ps.passes[0].linesearch.beta1 = 1e-10
    ps.passes[0].linesearch.beta2 = 5000.0

    return ps


def pe_deg(X, qg):
    N = X.shape[1]
    errs = np.zeros(N)
    for k in range(N):
        q = X[3:7, k]
        dot = np.clip(abs(float(np.dot(q, qg))), 0, 1)
        errs[k] = 2.0 * np.degrees(np.arccos(dot))
    return errs


def main():
    jtime = np.array([0.22, 0.22 + 1000.0 / (36525.0 * 86400.0)])
    qgoal = np.array([
        [np.sqrt(2)/2, np.sqrt(2)/2],
        [0.0, 0.0],
        [0.0, 0.0],
        [np.sqrt(2)/2, np.sqrt(2)/2],
    ])
    boresight = np.array([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    x0 = np.array([0.01, 0.01, 0.01, 1.0, 0.0, 0.0, 0.0, 0.0])
    r0 = np.array([7e6, 0.0, 0.0])
    v0 = np.array([0.0, 7.5e3, 0.0])
    qg = qgoal[:, 0]

    # --- Python iLQR ---
    print("Running Python iLQR (20x10)...")
    ps_py = make_settings()
    sat_py = create_satellite(ps_py)
    t0 = time.time()
    X_py, U_py, stop_py = py_trajOpt(ps_py, sat_py, x0, r0, v0, jtime, qgoal, boresight, debug=False)
    t_py = time.time() - t0
    pe_py = pe_deg(X_py, qg)
    print(f"  Python: stop={stop_py}  max_pe={pe_py.max():.1f}  final_pe={pe_py[-1]:.2f}  time={t_py:.1f}s")

    # --- C++ trajOpt ---
    print("Running C++ trajOpt (20x10)...")
    ps_cpp = make_settings()
    sat_cpp = create_satellite(ps_cpp)
    old_stdout = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    sys.stdout.flush()
    t0 = time.time()
    ok_cpp, X_cpp, U_cpp, _K = saltro_py.trajOpt(ps_cpp, sat_cpp, x0, r0, v0, jtime, qgoal, boresight)
    t_cpp = time.time() - t0
    os.dup2(old_stdout, 1)
    os.close(devnull)
    os.close(old_stdout)
    pe_cpp = pe_deg(X_cpp, qg)
    conv = "CONVERGED" if ok_cpp else "not converged"
    print(f"  C++:    {conv}  max_pe={pe_cpp.max():.1f}  final_pe={pe_cpp[-1]:.2f}  time={t_cpp:.1f}s")

    # --- Side by side final 20 knots ---
    print("\nFinal 20 knots comparison:")
    print(f"{'k':>3s}  {'Python':>8s}  {'C++':>8s}")
    for k in range(max(0, X_py.shape[1]-20), X_py.shape[1]):
        print(f"{k:3d}  {pe_py[k]:8.2f}  {pe_cpp[k]:8.2f}")


if __name__ == "__main__":
    main()
