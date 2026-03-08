"""
Trace iLQR iterations to see when/how the 360-degree flip develops.

Since warm start looks fine, the flip must develop during optimization
as the optimizer gets attracted to a bad local minimum.
"""
import sys
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
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
    cost.ang_cost_func_type = 4  # 1 - |q·q_goal|²
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
    return q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3]


def quat_error_angle(q, q_goal):
    qdot = quat_dot(q, q_goal)
    qdot = np.clip(qdot, -1.0, 1.0)
    angle = 2.0 * np.arccos(np.abs(qdot))
    return np.degrees(angle)


def analyze_snapshots(snapshots, q_goal_traj):
    """Analyze iLQR iteration snapshots for flip development."""
    
    print(f"\n{'='*80}")
    print(f"iLQR Iteration Analysis")
    print(f"{'='*80}")
    print(f"{'Iter':>4} | {'Cost':>12} | {'Max Err':>10} | {'Mean Err':>10} | {'Final Err':>10} | {'Flips?':>10}")
    print(f"{'-'*80}")
    
    flip_development = []
    
    for iter_num, snap in enumerate(snapshots):
        cost = snap['J']
        X = snap['X']
        N = X.shape[1]
        
        angle_errors = []
        for i in range(N):
            q = X[3:7, i]
            q_goal = q_goal_traj[:, i]
            angle_err = quat_error_angle(q, q_goal)
            angle_errors.append(angle_err)
        
        angle_errors = np.array(angle_errors)
        max_err = np.max(angle_errors)
        mean_err = np.mean(angle_errors)
        final_err = angle_errors[-1]
        
        # Check for flips (>170° errors)
        num_flips = np.sum(angle_errors > 170)
        flip_str = f"{num_flips} pts" if num_flips > 0 else "-"
        
        print(f"{iter_num:4d} | {cost:12.4e} | {max_err:10.2f}° | {mean_err:10.2f}° | {final_err:10.2f}° | {flip_str:>10}")
        
        flip_development.append({
            'iter': iter_num,
            'max_err': max_err,
            'mean_err': mean_err,
            'num_flips': num_flips,
            'angle_errors': angle_errors
        })
    
    return flip_development


def main():
    ps = create_planner_settings()
    satellite = create_3rw_satellite(ps)
    
    total_time = 1000  # The problematic case
    dt = ps.passes[0].dt
    
    jtime = np.array([0.22, 0.22 + total_time / (36525 * 86400)])
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
    
    w0 = np.array([0.0, 0.0, 0.0])
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    h0 = np.array([0.0, 0.0, 0.0])
    x0 = np.hstack((w0, q0, h0))
    
    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, 7.5e3, 0.0])
    
    print(f"\nRunning iLQR for {total_time}s trajectory...")
    print(f"This will take a moment, tracking all iterations...\n")
    
    X, U, stop_reason, snapshots, transitions, _, _ = trajOpt(
        ps, satellite, x0, r0, v0, jtime, qgoal, boresight, debug=True
    )
    
    print(f"\nStop reason: {stop_reason}")
    print(f"Total iterations: {len(snapshots)}")
    
    # Get q_goal trajectory
    dt_cent = dt / (36525.0 * 86400.0)
    jtime_flat = np.arange(jtime[0], jtime[1] + dt_cent/2, dt_cent)
    if abs(jtime_flat[-1] - jtime[1]) > 1e-12:
        jtime_flat = np.append(jtime_flat, jtime[1])
    idx = np.searchsorted(jtime[1:], jtime_flat, side='right')
    q_goal_traj = qgoal[:, idx]
    
    # Analyze all snapshots
    flip_dev = analyze_snapshots(snapshots, q_goal_traj)
    
    # Plot the evolution
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    iters = [d['iter'] for d in flip_dev]
    max_errs = [d['max_err'] for d in flip_dev]
    mean_errs = [d['mean_err'] for d in flip_dev]
    num_flips = [d['num_flips'] for d in flip_dev]
    
    axes[0].plot(iters, max_errs, 'r-o', label='Max error', linewidth=2)
    axes[0].plot(iters, mean_errs, 'b-o', label='Mean error', linewidth=2)
    axes[0].axhline(170, color='k', linestyle='--', alpha=0.5, label='Flip threshold (170°)')
    axes[0].set_xlabel('iLQR Iteration')
    axes[0].set_ylabel('Angle Error (degrees)')
    axes[0].set_title('Angle Error Evolution During iLQR')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(iters, num_flips, 'g-o', linewidth=2, markersize=8)
    axes[1].set_xlabel('iLQR Iteration')
    axes[1].set_ylabel('Number of Timesteps with >170° Error')
    axes[1].set_title('Flip Development During iLQR')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(ROOT / 'tests/debug/auto_debug/flip_evolution.png', dpi=150)
    print(f"\n✓ Plot saved to tests/debug/auto_debug/flip_evolution.png")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    
    first_flip_iter = None
    for d in flip_dev:
        if d['num_flips'] > 0:
            first_flip_iter = d['iter']
            break
    
    if first_flip_iter is not None:
        print(f"🐛 FLIP BUG CONFIRMED!")
        print(f"   First appeared at iteration {first_flip_iter}")
        print(f"   Final state: {num_flips[-1]} timesteps with >170° error")
        print(f"\nThis indicates the optimizer is converging to a LOCAL MINIMUM")
        print(f"where it rotates ~270° (the 'long way') instead of 90°.")
        print(f"\nROOT CAUSE: Quaternion double-cover + non-convex cost landscape")
        print(f"   • Quaternions q and -q represent the same rotation")
        print(f"   • Cost function ang_cost = 1 - |q·q_goal|² is non-convex")
        print(f"   • Has TWO local minima: one at q·q_goal ≈ 1, one at q·q_goal ≈ -1")
        print(f"   • The optimizer can get stuck in the wrong minimum")
    else:
        print(f"✓ No flip bug detected in this run")
        print(f"   Final mean error: {mean_errs[-1]:.2f}°")


if __name__ == "__main__":
    main()
