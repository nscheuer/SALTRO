"""
Investigate the 360-degree flip bug in long trajectories.

The bug: Sometimes the trajectory tries to do a 360-degree rotation to reach
the target, and gets stuck trying to make it faster rather than taking the
shorter path.

This is likely a quaternion double-cover issue: q and -q represent the same
rotation, but the optimizer can get trapped in a local minimum where it
tries to rotate the "long way" around.
"""
import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "build"))
ILQR_DIR = str(Path(__file__).resolve().parents[0].parent / "optimizer" / "ilqr")
sys.path.insert(0, ILQR_DIR)

import saltro_py
from create_3rw_sat import create_3rw_satellite
from trajOpt import trajOpt


def create_planner_settings():
    ps = saltro_py.PlannerSettings()
    ps.init_traj.initcontroller = 2
    ps.num_passes = 1
    ps.passes[0].dt = 10.0
    ps.passes[0].ilqr.cost_tol = 1e-5
    ps.passes[0].ilqr.max_iters = 40
    cost = ps.passes[0].cost
    cost.angle = 1.0
    cost.ang_vel = 1e2
    cost.ang_vel_mag = 0.0
    cost.ang_vel_err_dir = 0.0
    cost.control_mult = 1.0
    cost.mtq_control_weight = 1.0
    cost.rw_control_weight = 1e1
    cost.magic_control_weight = 0.0
    cost.rw_AM_weight = 0.0
    cost.rw_stic_weight = 0.0
    cost.RWh_max_mult = 0.0
    cost.RWh_stiction_mult = 0.0
    cost.RWh_ok_mult = 0.0
    cost.angle_N = 0.0
    cost.ang_vel_N = 1.0
    cost.ang_vel_mag_N = 0.0
    cost.ang_vel_err_dir_N = 0.0
    cost.ang_cost_func_type = 4
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
    ps.passes[0].reg.use_dynamics_hess = True
    ps.passes[0].reg.use_constraint_hess = False
    ps.passes[0].linesearch.max_iters = 24
    ps.passes[0].linesearch.beta1 = 1e-10
    ps.passes[0].linesearch.beta2 = 5000.0
    return ps


def quat_dot(q1, q2):
    """Compute dot product of two quaternions."""
    return q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3]


def quat_error_angle(q, q_goal):
    """
    Compute rotation angle error between q and q_goal.
    Uses the quaternion double-cover aware formula.
    """
    qdot = quat_dot(q, q_goal)
    # Clamp to [-1, 1] for numerical safety
    qdot = np.clip(qdot, -1.0, 1.0)
    # Angle = 2 * acos(|q·q_goal|)
    angle = 2.0 * np.arccos(np.abs(qdot))
    return np.degrees(angle)


def analyze_trajectory(X, q_goal_traj, dt):
    """
    Analyze the quaternion trajectory for flips and high-angle errors.
    
    Returns:
    - angle_errors: angle error at each timestep (degrees)
    - angular_velocities: magnitude of angular velocity at each timestep (deg/s)
    - flip_indicators: locations where trajectory might be flipping
    """
    N = X.shape[1]
    angle_errors = np.zeros(N)
    angular_velocities = np.zeros(N)
    qdot_signs = np.zeros(N)
    
    for i in range(N):
        q = X[3:7, i]
        q_goal = q_goal_traj[:, i]
        w = X[0:3, i]
        
        angle_errors[i] = quat_error_angle(q, q_goal)
        angular_velocities[i] = np.linalg.norm(w) * 180.0 / np.pi  # rad/s to deg/s
        qdot_signs[i] = np.sign(quat_dot(q, q_goal))
    
    # Detect sign flips in q·q_goal (indicates crossing between q and -q hemispheres)
    sign_changes = np.diff(qdot_signs)
    flip_indices = np.where(np.abs(sign_changes) > 0.5)[0] + 1
    
    return angle_errors, angular_velocities, flip_indices, qdot_signs


def run_test(total_time_sec, initial_offset_deg=0):
    """
    Run trajectory optimization and analyze for flips.
    
    Args:
        total_time_sec: Total trajectory time in seconds
        initial_offset_deg: Initial attitude offset from identity (degrees)
    """
    ps = create_planner_settings()
    satellite = create_3rw_satellite(ps)
    dt = ps.passes[0].dt

    jtime = np.array([0.22, 0.22 + total_time_sec / (36525 * 86400)])
    
    # Target: 90 degree rotation about x-axis
    qgoal = np.array([
        [np.sqrt(2)/2, np.sqrt(2)/2],
        [0.0, 0.0],
        [0.0, 0.0],
        [np.sqrt(2)/2, np.sqrt(2)/2]
    ])
    boresight = np.array([
        [1.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0]
    ])

    # Initial state with optional offset
    if initial_offset_deg > 0:
        offset_rad = np.radians(initial_offset_deg)
        q0 = np.array([np.cos(offset_rad/2), np.sin(offset_rad/2), 0.0, 0.0])
    else:
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
    
    w0 = np.array([0.0, 0.0, 0.0])
    h0 = np.array([0.0, 0.0, 0.0])
    x0 = np.hstack((w0, q0, h0))

    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, 7.5e3, 0.0])

    dt_cent = dt / (36525.0 * 86400.0)
    jtime_flat = np.arange(jtime[0], jtime[1] + dt_cent/2, dt_cent)
    if abs(jtime_flat[-1] - jtime[1]) > 1e-12:
        jtime_flat = np.append(jtime_flat, jtime[1])
    N = len(jtime_flat)

    idx = np.searchsorted(jtime[1:], jtime_flat, side='right')
    q_goal_traj = qgoal[:, idx]
    boresight_traj = boresight[:, idx]

    ok, R, V_orbit, B_field, S_sun, rho = saltro_py.generate_orbit(r0, v0, jtime_flat, 0, 0, 0, 0, 0)
    rho = rho.reshape(1, -1) if rho.ndim == 1 else rho

    # Warm start
    ok, X, U = saltro_py.warm_start(ps, satellite, x0, jtime_flat,
                                     np.asfortranarray(q_goal_traj),
                                     np.asfortranarray(boresight_traj),
                                     np.asfortranarray(R), np.asfortranarray(V_orbit),
                                     np.asfortranarray(B_field), np.asfortranarray(S_sun), rho)

    print(f"\n{'='*80}")
    print(f"Test: total_time={total_time_sec}s, dt={dt}s, N={N}, initial_offset={initial_offset_deg}°")
    print(f"{'='*80}")
    
    # Analyze warm start
    print("\n--- Warm Start Analysis ---")
    angle_errors_ws, ang_vel_ws, flips_ws, qdot_signs_ws = analyze_trajectory(X, q_goal_traj, dt)
    print(f"Max angle error: {np.max(angle_errors_ws):.2f}°")
    print(f"Mean angle error: {np.mean(angle_errors_ws):.2f}°")
    print(f"Max angular velocity: {np.max(ang_vel_ws):.2f} deg/s")
    print(f"Number of hemisphere crossings: {len(flips_ws)}")
    if len(flips_ws) > 0:
        print(f"  At timesteps: {flips_ws}")
    
    # Check for the "360 flip" signature: high angle error that doesn't decrease
    high_error_indices = np.where(angle_errors_ws > 170)[0]
    if len(high_error_indices) > 0:
        print(f"\n⚠️  WARNING: Detected near-180° errors at {len(high_error_indices)} timesteps!")
        print(f"  This indicates the trajectory is trying to rotate the 'long way'")
        print(f"  Indices: {high_error_indices[:10]}..." if len(high_error_indices) > 10 else f"  Indices: {high_error_indices}")
    
    # Run iLQR
    print("\n--- Running iLQR ---")
    X_opt, U_opt, stop_reason, snapshots, transitions, _, _ = trajOpt(
        ps, satellite, x0, r0, v0, jtime, qgoal, boresight, debug=False
    )
    
    print(f"Stop reason: {stop_reason}")
    if "converged" in stop_reason.lower():
        print("✓ iLQR converged")
        X = X_opt  # Use optimized trajectory
        ok = True
    else:
        print("✗ iLQR failed to converge")
        ok = False
    
    # Analyze optimized trajectory
    print("\n--- Optimized Trajectory Analysis ---")
    if ok:
        angle_errors_opt, ang_vel_opt, flips_opt, qdot_signs_opt = analyze_trajectory(X, q_goal_traj, dt)
    else:
        # If failed, just use warm start for comparison
        angle_errors_opt, ang_vel_opt, flips_opt, qdot_signs_opt = angle_errors_ws, ang_vel_ws, flips_ws, qdot_signs_ws
    print(f"Max angle error: {np.max(angle_errors_opt):.2f}°")
    print(f"Mean angle error: {np.mean(angle_errors_opt):.2f}°")
    print(f"Final angle error: {angle_errors_opt[-1]:.2f}°")
    print(f"Max angular velocity: {np.max(ang_vel_opt):.2f} deg/s")
    print(f"Number of hemisphere crossings: {len(flips_opt)}")
    if len(flips_opt) > 0:
        print(f"  At timesteps: {flips_opt}")
    
    # Check for persistent high errors (the flip bug)
    high_error_indices_opt = np.where(angle_errors_opt > 170)[0]
    if len(high_error_indices_opt) > 0:
        print(f"\n🐛 BUG DETECTED: Near-180° errors persist after optimization!")
        print(f"  Found at {len(high_error_indices_opt)} timesteps")
        print(f"  The optimizer is stuck trying to rotate the 'long way' (360° instead of 0°)")
        print(f"  Indices: {high_error_indices_opt[:10]}..." if len(high_error_indices_opt) > 10 else f"  Indices: {high_error_indices_opt}")
    
    return X, q_goal_traj, angle_errors_ws, angle_errors_opt, ang_vel_opt, qdot_signs_opt


def main():
    # Test 1: Standard 1000s case (reported to show the bug)
    print("\n" + "="*80)
    print("TEST 1: Standard 1000s trajectory")
    print("="*80)
    X1, q_goal1, ae_ws1, ae_opt1, av1, qs1 = run_test(1000)
    
    # Test 2: Different initial conditions
    print("\n" + "="*80)
    print("TEST 2: 1000s with 45° initial offset")
    print("="*80)
    X2, q_goal2, ae_ws2, ae_opt2, av2, qs2 = run_test(1000, initial_offset_deg=45)
    
    # Test 3: Shorter trajectory
    print("\n" + "="*80)
    print("TEST 3: Shorter 300s trajectory")
    print("="*80)
    X3, q_goal3, ae_ws3, ae_opt3, av3, qs3 = run_test(300)
    
    # Plot results
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('360-Degree Flip Investigation', fontsize=16)
    
    tests = [
        ("1000s standard", ae_ws1, ae_opt1, av1, qs1),
        ("1000s w/ 45° offset", ae_ws2, ae_opt2, av2, qs2),
        ("300s standard", ae_ws3, ae_opt3, av3, qs3),
    ]
    
    for i, (label, ae_ws, ae_opt, av, qs) in enumerate(tests):
        # Angle errors
        axes[i, 0].plot(ae_ws, 'b--', alpha=0.5, label='Warm start')
        axes[i, 0].plot(ae_opt, 'r-', linewidth=2, label='Optimized')
        axes[i, 0].axhline(180, color='k', linestyle=':', alpha=0.3, label='180°')
        axes[i, 0].set_ylabel('Angle Error (deg)')
        axes[i, 0].set_title(f'{label}: Angle Error')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        
        # q·q_goal sign
        axes[i, 1].plot(qs, 'g-', linewidth=2)
        axes[i, 1].axhline(0, color='k', linestyle='-', alpha=0.3)
        axes[i, 1].set_ylabel('sign(q·q_goal)')
        axes[i, 1].set_title(f'{label}: Quaternion Hemisphere')
        axes[i, 1].set_ylim(-1.5, 1.5)
        axes[i, 1].grid(True, alpha=0.3)
    
    axes[2, 0].set_xlabel('Timestep')
    axes[2, 1].set_xlabel('Timestep')
    
    plt.tight_layout()
    plt.savefig(ROOT / 'tests/debug/auto_debug/flip_investigation.png', dpi=150)
    print(f"\n✓ Plots saved to tests/debug/auto_debug/flip_investigation.png")
    plt.show()


if __name__ == "__main__":
    main()
