"""Run the EXACT same settings as debug_3_1_slew90_dt10.py and print PE profile."""
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))

import saltro_py
from sat_3_1_hybrid import create_satellite
from trajOpt import trajOpt

def create_planner_settings():
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
    cost.ang_vel_mag = 0.0
    cost.ang_vel_err_dir = 0.0
    cost.control_mult = 1.0
    cost.mtq_control_weight = 1e-1
    cost.rw_control_weight = 1.0
    cost.magic_control_weight = 0.0
    cost.rw_AM_weight = 0.0
    cost.rw_stic_weight = 0.0
    cost.RWh_max_mult = 0.0
    cost.RWh_stiction_mult = 0.0
    cost.RWh_ok_mult = 0.0
    cost.angle_N = 1e2
    cost.ang_vel_N = 1e1
    cost.ang_vel_mag_N = 0.0
    cost.ang_vel_err_dir_N = 0.0
    cost.ang_cost_func_type = 3
    cost.use_cost_hess = True

    ps.disturbances.plan_for_aero = False
    ps.disturbances.plan_for_gg = False
    ps.disturbances.plan_for_srp = False
    ps.disturbances.plan_for_prop = False
    ps.disturbances.plan_for_gendist = False
    ps.disturbances.plan_for_resdipole = False

    ps.passes[0].reg.reg_init = 1e-6
    ps.passes[0].reg.reg_max = 1e10
    ps.passes[0].reg.reg_scale = 10.0
    ps.passes[0].reg.use_dynamics_hess = False
    ps.passes[0].reg.use_constraint_hess = False

    ps.passes[0].linesearch.max_iters = 24
    ps.passes[0].linesearch.beta1 = 1e-10
    ps.passes[0].linesearch.beta2 = 5000.0

    return ps


def main():
    ps = create_planner_settings()
    sat = create_satellite(ps)

    jtime = np.array([0.22, 0.22 + 1000 / (36525 * 86400)])
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

    # Run C++ first (with stdout suppressed), then print both profiles
    import os

    # --- C++ run ---
    old_stdout = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    sys.stdout.flush()
    try:
        ok_cpp, X_cpp, U_cpp, K_cpp = saltro_py.trajOpt(
            ps, sat, x0, r0, v0, jtime, qgoal, boresight
        )
        cpp_err = None
    except Exception as e:
        ok_cpp, X_cpp = False, None
        cpp_err = str(e)
    os.dup2(old_stdout, 1)
    os.close(devnull)
    os.close(old_stdout)

    # --- Python run ---
    X, U, stop, snaps, trans, dt_val, ctol, elapsed = trajOpt(
        ps, sat, x0, r0, v0, jtime, qgoal, boresight, debug=True
    )

    # --- Print Python results ---
    print(f"\n=== PYTHON iLQR ===")
    print(f"Stop reason: {stop}")
    print(f"Final cost: {snaps[-1]['J']:.6e}")
    print(f"Elapsed: {elapsed:.2f}s")
    N = X.shape[1]
    print(f"N={N}")
    print(f"\nPointing error profile:")
    for k in range(N):
        q = X[3:7, k]
        pe = 2 * np.degrees(np.arccos(min(abs(float(np.dot(q, qg))), 1)))
        marker = " ***SPIKE***" if pe > 90 else ""
        print(f"  k={k:3d}  t={k*10:5.0f}s  pe={pe:6.1f}{marker}")

    # --- Print C++ results ---
    print(f"\n=== C++ trajOpt ===")
    if cpp_err:
        print(f"Exception: {cpp_err}")
    elif X_cpp is not None:
        N_cpp = X_cpp.shape[1]
        print(f"Converged: {ok_cpp}, N={N_cpp}")
        print(f"\nPointing error profile:")
        for k in range(N_cpp):
            q = X_cpp[3:7, k]
            pe = 2 * np.degrees(np.arccos(min(abs(float(np.dot(q, qg))), 1)))
            marker = " ***SPIKE***" if pe > 90 else ""
            print(f"  k={k:3d}  t={k*10:5.0f}s  pe={pe:6.1f}{marker}")


if __name__ == "__main__":
    main()
