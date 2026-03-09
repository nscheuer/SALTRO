"""Direct state and control comparison for slew180 test case."""
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(ROOT / "tests" / "debug" / "optimizer" / "alilqr_python"))
sys.path.insert(0, str(ROOT / "tests" / "debug" / "optimizer" / "configs"))

import saltro_py
from sat_0_3_rw import create_satellite
from trajOpt import trajOpt


def create_planner_settings():
    plannersettings = saltro_py.PlannerSettings()
    plannersettings.init_traj.initcontroller = 2
    
    plannersettings.num_passes = 1
    plannersettings.passes[0].dt = 10.0
    plannersettings.passes[0].ilqr.cost_tol = 1e-5
    plannersettings.passes[0].ilqr.max_iters = 20

    plannersettings.passes[0].auglag.max_outer_iters = 10
    plannersettings.passes[0].auglag.constraint_tol = 1e-3

    cost = plannersettings.passes[0].cost
    cost.angle = 1.0
    cost.ang_vel = 1e1
    cost.ang_vel_mag = 0.0
    cost.ang_vel_err_dir = 0.0
    cost.control_mult = 1.0
    cost.mtq_control_weight = 1e-2
    cost.rw_control_weight = 1.0
    cost.magic_control_weight = 0.0
    cost.rw_AM_weight = 0.0
    cost.rw_stic_weight = 0.0
    cost.RWh_max_mult = 0.0
    cost.RWh_stiction_mult = 0.0
    cost.RWh_ok_mult = 0.0
    cost.angle_N = 0.0
    cost.ang_vel_N = 0.0
    cost.ang_vel_mag_N = 0.0
    cost.ang_vel_err_dir_N = 0.0
    cost.ang_cost_func_type = 3
    cost.use_cost_hess = True

    plannersettings.disturbances.plan_for_aero = False
    plannersettings.disturbances.plan_for_gg = False
    plannersettings.disturbances.plan_for_srp = False
    plannersettings.disturbances.plan_for_prop = False
    plannersettings.disturbances.plan_for_gendist = False
    plannersettings.disturbances.plan_for_resdipole = False

    plannersettings.passes[0].reg.reg_init = 1e-6
    plannersettings.passes[0].reg.reg_max = 1e10
    plannersettings.passes[0].reg.reg_scale = 10.0
    plannersettings.passes[0].reg.use_dynamics_hess = False
    plannersettings.passes[0].reg.use_constraint_hess = False

    plannersettings.passes[0].linesearch.max_iters = 24
    plannersettings.passes[0].linesearch.beta1 = 1e-10
    plannersettings.passes[0].linesearch.beta2 = 5000.0
    
    return plannersettings


def main():
    plannersettings = create_planner_settings()
    satellite = create_satellite(plannersettings)

    jtime = np.array([0.22, 0.22 + 100/(36525 * 86400), 0.22 + 200/(36525 * 86400)])
    qgoal = np.array([
        [np.sqrt(2)/2, 0.0, 0.0],
        [0.0, 0.0, 0.0],           
        [0.0, 0.0, 0.0],            
        [np.sqrt(2)/2, 1.0, 1.0]
    ])
    boresight = np.array([
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ])

    w0 = np.array([0.01, 0.01, 0.01])
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    h0 = np.array([0.0, 0.0, 0.0])
    x0 = np.hstack((w0, q0, h0))

    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, 7.5e3, 0.0])

    # Run Python trajOpt
    print("Running Python trajOpt...")
    X_py, U_py, stop_py, _, _, _, _, _ = trajOpt(
        plannersettings, satellite, x0, r0, v0, jtime, qgoal, boresight, debug=True
    )

    # Run C++ trajOpt
    print("Running C++ trajOpt...")
    ok, X_cpp, U_cpp, _K = saltro_py.trajOpt(
        plannersettings, satellite, x0, r0, v0, jtime, qgoal, boresight
    )
    
    if not ok:
        print("ERROR: C++ trajOpt failed")
        return

    # Compare outputs
    print("\n" + "="*60)
    print("TRAJECTORY COMPARISON")
    print("="*60)
    
    print(f"\nPython stop reason: {stop_py}")
    print(f"C++ returned: {'success' if ok else 'failure'}")
    
    print(f"\nShape comparison:")
    print(f"  Python: X={X_py.shape}, U={U_py.shape}")
    print(f"  C++:    X={X_cpp.shape}, U={U_cpp.shape}")
    
    # Element-wise comparison
    diff_X = np.abs(X_py - X_cpp)
    diff_U = np.abs(U_py - U_cpp)
    
    max_diff_X = np.max(diff_X)
    max_diff_U = np.max(diff_U)
    
    print(f"\nMaximum absolute differences:")
    print(f"  States (X):   {max_diff_X:.16e}")
    print(f"  Controls (U): {max_diff_U:.16e}")
    
    # Check at specific timesteps
    dt = 10.0
    n = X_py.shape[1]
    t_vec = np.arange(n) * dt
    
    print(f"\nState differences at key times:")
    for t_check in [0, 50, 100, 150, 200]:
        if t_check / dt >= n:
            continue
        k = int(t_check / dt)
        diff_k = np.max(np.abs(X_py[:, k] - X_cpp[:, k]))
        print(f"  t={t_check:3d}s (k={k:2d}): max_diff = {diff_k:.6e}")
    
    # Detailed breakdown at t=100s (where user sees issue)
    k_100 = int(100 / dt)
    if k_100 < n:
        print(f"\nDetailed state comparison at t=100s (k={k_100}):")
        print(f"  {'Component':<15} {'Python':<20} {'C++':<20} {'Diff':<15}")
        print(f"  {'-'*15} {'-'*20} {'-'*20} {'-'*15}")
        for i, name in enumerate(['wx', 'wy', 'wz', 'q0', 'q1', 'q2', 'q3', 'h0', 'h1', 'h2']):
            py_val = X_py[i, k_100]
            cpp_val = X_cpp[i, k_100]
            diff = abs(py_val - cpp_val)
            print(f"  {name:<15} {py_val:<20.12e} {cpp_val:<20.12e} {diff:<15.6e}")
    
    # Tolerance check
    tol = 1e-12
    match = (max_diff_X < tol) and (max_diff_U < tol)
    
    print(f"\n{'='*60}")
    if match:
        print(f"✓ MATCH: All differences < {tol:.0e}")
    else:
        print(f"✗ MISMATCH: Differences exceed {tol:.0e}")
        print(f"  Largest X diff: {max_diff_X:.6e} at index {np.unravel_index(np.argmax(diff_X), diff_X.shape)}")
        print(f"  Largest U diff: {max_diff_U:.6e} at index {np.unravel_index(np.argmax(diff_U), diff_U.shape)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
