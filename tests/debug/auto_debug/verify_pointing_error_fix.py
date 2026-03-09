"""Verify pointing error calculations match after fix."""
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(ROOT / "tests" / "debug" / "optimizer" / "alilqr_python"))
sys.path.insert(0, str(ROOT / "tests" / "debug" / "optimizer" / "alilqr_cpp"))
sys.path.insert(0, str(ROOT / "tests" / "debug" / "optimizer" / "configs"))

import saltro_py
from sat_0_3_rw import create_satellite
from trajOpt import trajOpt
from final_viewer import _expand_q_goal


def quat_inverse(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def compute_pointing_error_deg(q, q_goal):
    """Compute pointing error in degrees."""
    n = q.shape[1]
    err_deg = np.zeros(n)
    for k in range(n):
        q_err = quat_multiply(quat_inverse(q_goal[:, k]), q[:, k])
        err_deg[k] = 2.0 * np.arctan2(np.linalg.norm(q_err[1:]), abs(q_err[0])) * 180.0 / np.pi
    return err_deg


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
    qgoal_compact = np.array([
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
    
    dt = plannersettings.passes[0].dt

    # Run Python trajOpt
    print("Running Python trajOpt...")
    X_py, U_py, stop_py, snapshots, _, _, _, _ = trajOpt(
        plannersettings, satellite, x0, r0, v0, jtime, qgoal_compact, boresight, debug=True
    )
    q_goal_py = snapshots[-1]['q_goal']

    # Run C++ trajOpt
    print("Running C++ trajOpt...")
    ok, X_cpp, U_cpp, _K = saltro_py.trajOpt(
        plannersettings, satellite, x0, r0, v0, jtime, qgoal_compact, boresight
    )
    
    if not ok:
        print("ERROR: C++ trajOpt failed")
        return

    # Expand goal using FIXED viewer logic (now with jtime)
    n = X_cpp.shape[1]
    q_goal_cpp_viewer = _expand_q_goal(qgoal_compact, n, jtime, dt)
    
    print("\n" + "="*70)
    print("GOAL QUATERNION EXPANSION VERIFICATION")
    print("="*70)
    
    # Compare expanded goals
    print(f"\nPython internal q_goal shape: {q_goal_py.shape}")
    print(f"C++ viewer expanded q_goal shape: {q_goal_cpp_viewer.shape}")
    
    print(f"\nGoal quaternion comparison at key times:")
    print(f"  {'Time':<10} {'Python q_goal':<35} {'C++ viewer q_goal':<35} {'Match':<8}")
    print(f"  {'-'*10} {'-'*35} {'-'*35} {'-'*8}")
    for t_check in [0, 50, 100, 150, 200]:
        if t_check / dt >= n:
            continue
        k = int(t_check / dt)
        q_py = q_goal_py[:, k]
        q_cpp = q_goal_cpp_viewer[:, k]
        diff = np.linalg.norm(q_py - q_cpp)
        match = '✓' if diff < 1e-10 else '✗'
        print(f"  {t_check:3d}s (k={k:2d})  [{q_py[0]:7.5f},{q_py[1]:7.5f},{q_py[2]:7.5f},{q_py[3]:7.5f}]  " +
              f"[{q_cpp[0]:7.5f},{q_cpp[1]:7.5f},{q_cpp[2]:7.5f},{q_cpp[3]:7.5f}]  {match:<8}")
    
    max_diff_goal = np.max(np.abs(q_goal_py - q_goal_cpp_viewer))
    print(f"\nMax difference in expanded goals: {max_diff_goal:.6e}")
    
    # Compute pointing errors
    q_py_traj = X_py[3:7, :]
    q_cpp_traj = X_cpp[3:7, :]
    
    pe_py = compute_pointing_error_deg(q_py_traj, q_goal_py)
    pe_cpp_viewer = compute_pointing_error_deg(q_cpp_traj, q_goal_cpp_viewer)
    
    print("\n" + "="*70)
    print("POINTING ERROR COMPARISON")
    print("="*70)
    
    print(f"\nPointing errors at key times:")
    print(f"  {'Time':<10} {'Python':<15} {'C++ viewer':<15} {'Diff':<15} {'Match':<8}")
    print(f"  {'-'*10} {'-'*15} {'-'*15} {'-'*15} {'-'*8}")
    for t_check in [0, 50, 100, 150, 200]:
        if t_check / dt >= n:
            continue
        k = int(t_check / dt)
        diff = abs(pe_py[k] - pe_cpp_viewer[k])
        match = '✓' if diff < 0.001 else '✗'  # 0.001 degree tolerance
        print(f"  {t_check:3d}s (k={k:2d})  {pe_py[k]:12.6f}°  {pe_cpp_viewer[k]:12.6f}°  {diff:12.6f}°  {match:<8}")
    
    max_pe_diff = np.max(np.abs(pe_py - pe_cpp_viewer))
    print(f"\nMax pointing error difference: {max_pe_diff:.6e}°")
    
    print("\n" + "="*70)
    if max_diff_goal < 1e-10 and max_pe_diff < 0.001:
        print("✓ SUCCESS: Goal quaternions and pointing errors now match!")
    else:
        print("✗ FAILURE: Discrepancies still exist")
        if max_diff_goal >= 1e-10:
            print(f"  - Goal quaternions differ by {max_diff_goal:.6e}")
        if max_pe_diff >= 0.001:
            print(f"  - Pointing errors differ by {max_pe_diff:.6f}°")
    print("="*70)


if __name__ == "__main__":
    main()
