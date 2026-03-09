"""Compare pointing error calculation between Python and C++ for slew180."""
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

    # Run Python trajOpt (returns expanded q_goal in snapshots)
    print("Running Python trajOpt...")
    X_py, U_py, stop_py, snapshots, _, dt, _, _ = trajOpt(
        plannersettings, satellite, x0, r0, v0, jtime, qgoal_compact, boresight, debug=True
    )
    q_goal_py = snapshots[-1]['q_goal']  # This is the expanded goal used internally

    # Run C++ trajOpt
    print("Running C++ trajOpt...")
    ok, X_cpp, U_cpp, _K = saltro_py.trajOpt(
        plannersettings, satellite, x0, r0, v0, jtime, qgoal_compact, boresight
    )
    
    if not ok:
        print("ERROR: C++ trajOpt failed")
        return

    # Compare goal quaternions
    print("\n" + "="*60)
    print("GOAL QUATERNION COMPARISON")
    print("="*60)
    
    print(f"\nCompact qgoal shape: {qgoal_compact.shape}")
    print(f"Compact qgoal (normalized):")
    for i in range(3):
        q_norm = qgoal_compact[:, i] / np.linalg.norm(qgoal_compact[:, i])
        angle = 2.0 * np.arctan2(np.linalg.norm(q_norm[1:]), abs(q_norm[0])) * 180.0 / np.pi
        print(f"  Column {i}: {q_norm} → {angle:.2f}°")
    
    print(f"\nPython expanded q_goal shape: {q_goal_py.shape}")
    print(f"Expected expanded shape: (4, {X_py.shape[1]})")
    
    # Show what Python actually used at key timesteps
    n = X_py.shape[1]
    t_vec = np.arange(n) * dt
    
    print(f"\nPython q_goal values at key times:")
    for t_check in [0, 50, 100, 150, 200]:
        if t_check / dt >= n:
            continue
        k = int(t_check / dt)
        q_g = q_goal_py[:, k]
        angle = 2.0 * np.arctan2(np.linalg.norm(q_g[1:]), abs(q_g[0])) * 180.0 / np.pi
        print(f"  t={t_check:3d}s (k={k:2d}): {q_g} → {angle:.2f}°")
    
    # Compute pointing errors for Python trajectory
    q_py = X_py[3:7, :]
    pe_py = compute_pointing_error_deg(q_py, q_goal_py)
    
    # For C++, we need to manually expand qgoal_compact (simulating what viewer does)
    # This might be the source of the discrepancy!
    print(f"\n" + "="*60)
    print("POINTING ERROR COMPARISON (C++ using viewer expansion logic)")
    print("="*60)
    
    # Expand using zero-order hold matching final_viewer.py logic
    m = qgoal_compact.shape[1]
    q_goal_cpp_expanded = np.zeros((4, n))
    for k in range(n):
        s = k * (m - 1) / float(n - 1)
        seg = int(np.floor(s))
        if seg >= m - 1:
            seg = m - 1
        q_goal_cpp_expanded[:, k] = qgoal_compact[:, seg] / np.linalg.norm(qgoal_compact[:, seg])
    
    print(f"\nC++ expanded q_goal (via viewer logic) at key times:")
    for t_check in [0, 50, 100, 150, 200]:
        if t_check / dt >= n:
            continue
        k = int(t_check / dt)
        q_g = q_goal_cpp_expanded[:, k]
        angle = 2.0 * np.arctan2(np.linalg.norm(q_g[1:]), abs(q_g[0])) * 180.0 / np.pi
        print(f"  t={t_check:3d}s (k={k:2d}): {q_g} → {angle:.2f}°")
    
    # Compute pointing error for C++
    q_cpp = X_cpp[3:7, :]
    pe_cpp_viewer = compute_pointing_error_deg(q_cpp, q_goal_cpp_expanded)
    pe_py_with_cpp_goal = compute_pointing_error_deg(q_py, q_goal_cpp_expanded)
    
    # Compare pointing errors at key times
    print(f"\nPointing errors at key times:")
    print(f"  {'Time':<10} {'Python':<12} {'C++(viewer)':<12} {'Py(cpp_goal)':<12} {'Diff':<12}")
    print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    for t_check in [0, 50, 100, 150, 200]:
        if t_check / dt >= n:
            continue
        k = int(t_check / dt)
        print(f"  {t_check:3d}s (k={k:2d})  {pe_py[k]:10.4f}°  {pe_cpp_viewer[k]:10.4f}°  {pe_py_with_cpp_goal[k]:10.4f}°  {abs(pe_py[k] - pe_cpp_viewer[k]):10.4f}°")
    
    # Check if goal quaternions match
    diff_q_goal = np.max(np.abs(q_goal_py - q_goal_cpp_expanded))
    print(f"\nMax difference in expanded goal quaternions: {diff_q_goal:.6e}")
    
    if diff_q_goal > 1e-10:
        print("\n⚠ WARNING: Expanded goal quaternions differ!")
        print("This explains the pointing error discrepancy.")
        print("\nDifferences at key times:")
        for t_check in [0, 50, 100, 150, 200]:
            if t_check / dt >= n:
                continue
            k = int(t_check / dt)
            diff = np.linalg.norm(q_goal_py[:, k] - q_goal_cpp_expanded[:, k])
            print(f"  t={t_check:3d}s: ||q_goal_py - q_goal_cpp|| = {diff:.6e}")


if __name__ == "__main__":
    main()
