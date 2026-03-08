"""
Quick diagnostic: Check warm start for 360-degree flip issues.

The flip bug likely originates in the warm start initialization, where
the trajectory planner decides to rotate "the long way" around.
"""
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "build"))
ILQR_DIR = str(Path(__file__).resolve().parents[0].parent / "optimizer" / "ilqr")
sys.path.insert(0, ILQR_DIR)

import saltro_py
from create_3rw_sat import create_3rw_satellite


def create_planner_settings():
    ps = saltro_py.PlannerSettings()
    ps.init_traj.initcontroller = 2
    ps.num_passes = 1
    ps.passes[0].dt = 10.0
    return ps


def quat_dot(q1, q2):
    """Compute dot product of two quaternions."""
    return q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3]


def quat_error_angle(q, q_goal):
    """Compute rotation angle error between q and q_goal."""
    qdot = quat_dot(q, q_goal)
    qdot = np.clip(qdot, -1.0, 1.0)
    angle = 2.0 * np.arccos(np.abs(qdot))
    return np.degrees(angle)


def check_trajectory(total_time_sec):
    """Check if warm start trajectory has flip issues."""
    ps = create_planner_settings()
    satellite = create_3rw_satellite(ps)
    dt = ps.passes[0].dt

    jtime = np.array([0.22, 0.22 + total_time_sec / (36525 * 86400)])
    
    # Target: 90 degree rotation
    qgoal = np.array([
        [np.sqrt(2)/2, np.sqrt(2)/2],
        [0.0, 0.0],
        [0.0, 0.0],
        [np.sqrt(2)/2, np.sqrt(2)/2]
    ])
    boresight = np.array([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])

    w0 = np.array([0.0, 0.0, 0.0])
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
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

    # Generate warm start
    ok, X, U = saltro_py.warm_start(ps, satellite, x0, jtime_flat,
                                     np.asfortranarray(q_goal_traj),
                                     np.asfortranarray(boresight_traj),
                                     np.asfortranarray(R), np.asfortranarray(V_orbit),
                                     np.asfortranarray(B_field), np.asfortranarray(S_sun), rho)

    print(f"\n{'='*70}")
    print(f"Trajectory: total_time={total_time_sec}s, dt={dt}s, N={N} timesteps")
    print(f"{'='*70}")
    
    # Analyze each timestep
    angle_errors = []
    qdot_values = []
    bad_indices = []
    
    for i in range(N):
        q = X[3:7, i]
        q_goal = q_goal_traj[:, i]
        
        angle_err = quat_error_angle(q, q_goal)
        qdot = quat_dot(q, q_goal)
        
        angle_errors.append(angle_err)
        qdot_values.append(qdot)
        
        # Flag if angle error > 170° (close to 180° = wrong direction)
        if angle_err > 170:
            bad_indices.append(i)
    
    angle_errors = np.array(angle_errors)
    qdot_values = np.array(qdot_values)
    
    print(f"\nAngle Error Statistics:")
    print(f"  Min:  {np.min(angle_errors):.2f}°")
    print(f"  Max:  {np.max(angle_errors):.2f}°")
    print(f"  Mean: {np.mean(angle_errors):.2f}°")
    print(f"  Final: {angle_errors[-1]:.2f}°")
    
    print(f"\nq·q_goal Statistics:")
    print(f"  Min:  {np.min(qdot_values):.4f}")
    print(f"  Max:  {np.max(qdot_values):.4f}")
    print(f"  Mean: {np.mean(qdot_values):.4f}")
    
    if len(bad_indices) > 0:
        print(f"\n🐛 FLIP BUG DETECTED!")
        print(f"  Found {len(bad_indices)} timesteps with >170° error")
        print(f"  This means the warm start is initialized to rotate the 'long way'")
        print(f"  Bad timesteps: {bad_indices[:10]}..." if len(bad_indices) > 10 else f"  Bad timesteps: {bad_indices}")
        
        # Show details for first few bad timesteps
        print(f"\n  Details:")
        for idx in bad_indices[:5]:
            print(f"    t={idx}: angle_err={angle_errors[idx]:.2f}°, q·q_goal={qdot_values[idx]:.4f}")
        
        return True  # Bug detected
    else:
        print(f"\n✓ No flip issues detected in warm start")
        return False


def main():
    print("\n" + "="*70)
    print("QUICK FLIP BUG DIAGNOSTIC")
    print("="*70)
    
    test_cases = [100, 200, 300, 500, 1000, 2000]
    
    results = {}
    for t in test_cases:
        has_flip = check_trajectory(t)
        results[t] = has_flip
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for t, has_flip in results.items():
        status = "🐛 FLIP BUG" if has_flip else "✓ OK"
        print(f"  {t:4d}s: {status}")
    
    if any(results.values()):
        print("\n⚠️  The flip bug appears in the WARM START initialization.")
        print("   This is likely due to the warm start controller picking")
        print("   the wrong quaternion hemisphere (q vs -q) for interpolation.")
        print("\n   Suggested fixes:")
        print("   1. Modify warm start to always pick shortest rotation path")
        print("   2. Add quaternion sign correction in warm start interpolation")
        print("   3. Use a different cost function that's quaternion-agnostic")


if __name__ == "__main__":
    main()
