"""
Diagnostic script: Investigate converge-then-drift behavior in SALTRO trajectories.

Tests:
1. processAttitudeTarget correctness for vector goals
2. Cost function vs true geodesic error consistency
3. Dynamics forward integration consistency
4. B-field controllability analysis
5. RW usage analysis
"""
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

np.set_printoptions(precision=6, suppress=True, linewidth=120)


def quat_angle(q1, q2):
    """Geodesic angle between two quaternions (degrees)."""
    dot = np.abs(np.dot(q1, q2))
    dot = min(dot, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def quat_mult(p, q):
    """Hamilton product p ⊗ q, scalar-first."""
    p0, p1, p2, p3 = p
    q0, q1, q2, q3 = q
    return np.array([
        p0*q0 - p1*q1 - p2*q2 - p3*q3,
        p0*q1 + p1*q0 + p2*q3 - p3*q2,
        p0*q2 - p1*q3 + p2*q0 + p3*q1,
        p0*q3 + p1*q2 - p2*q1 + p3*q0,
    ])


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def rotmat(q):
    """Body-to-ECI rotation matrix from scalar-first quaternion."""
    q0, q1, q2, q3 = q / np.linalg.norm(q)
    R = np.array([
        [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)],
    ])
    return R


def run_diagnostic():
    # =========================================================================
    # Setup: Same as debug_3_1_slew90_dt10.py but with vector goal
    # =========================================================================
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

    satellite = create_satellite(plannersettings)

    # =========================================================================
    # TEST 1: Quaternion goal (should work correctly)
    # =========================================================================
    print("=" * 70)
    print("TEST 1: Quaternion goal (baseline)")
    print("=" * 70)

    jtime = np.array([0.22, 0.22 + 2000 / (36525 * 86400)])
    q_target = np.array([np.sqrt(2)/2, 0.0, 0.0, np.sqrt(2)/2])  # 90° about z
    qgoal_quat = np.column_stack([q_target, q_target])
    boresight = np.array([[1, 1], [0, 0], [0, 0.0]])

    w0 = np.zeros(3)
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    x0 = np.hstack((w0, q0, [0.0]))  # Include RW momentum

    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, 7.5e3, 0.0])

    X_q, U_q, reason_q = trajOpt(
        plannersettings, satellite, x0, r0, v0, jtime, qgoal_quat, boresight, debug=False
    )
    print(f"  Stop reason: {reason_q}")
    N_q = X_q.shape[1]
    dt = 10.0

    # Compute attitude error at each timestep
    errors_q = []
    for k in range(N_q):
        q_k = X_q[3:7, k]
        q_k = q_k / np.linalg.norm(q_k)
        err = quat_angle(q_k, q_target)
        errors_q.append(err)

    print(f"  Error profile: start={errors_q[0]:.2f}°, min={min(errors_q):.4f}° at t={errors_q.index(min(errors_q))*dt:.0f}s, end={errors_q[-1]:.2f}°")
    print(f"  Final ω = {X_q[:3, -1]} rad/s")
    print(f"  Max |u_rw| = {np.max(np.abs(U_q[3, :])):.6f} N·m")

    # =========================================================================
    # TEST 2: Vector goal (ECI direction) — [NaN, x, y, z] format
    # =========================================================================
    print()
    print("=" * 70)
    print("TEST 2: Vector goal (ECI direction)")
    print("=" * 70)

    # Goal: align body x-axis (boresight) with ECI z-axis
    eci_dir = np.array([0.0, 0.0, 1.0])
    qgoal_vec = np.column_stack([
        np.array([np.nan, eci_dir[0], eci_dir[1], eci_dir[2]]),
        np.array([np.nan, eci_dir[0], eci_dir[1], eci_dir[2]])
    ])

    X_v, U_v, reason_v = trajOpt(
        plannersettings, satellite, x0, r0, v0, jtime, qgoal_vec, boresight, debug=False
    )
    print(f"  Stop reason: {reason_v}")
    N_v = X_v.shape[1]

    # For vector goal, the error is the angle between boresight_eci and eci_dir
    boresight_body = np.array([1.0, 0.0, 0.0])
    errors_v_boresight = []
    errors_v_quat = []
    for k in range(N_v):
        q_k = X_v[3:7, k]
        q_k = q_k / np.linalg.norm(q_k)
        R_k = rotmat(q_k)
        boresight_eci = R_k @ boresight_body
        cos_angle = np.clip(np.dot(boresight_eci, eci_dir), -1, 1)
        err_boresight = np.degrees(np.arccos(cos_angle))
        errors_v_boresight.append(err_boresight)

        # Also compute what processAttitudeTarget would give as q_goal
        # Minimum rotation from boresight_body to eci_dir
        axis = np.cross(boresight_body, eci_dir)
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-8:
            axis = axis / axis_norm
            angle = np.arccos(np.clip(np.dot(boresight_body, eci_dir), -1, 1))
            q_goal_vec = np.array([
                np.cos(angle / 2),
                np.sin(angle / 2) * axis[0],
                np.sin(angle / 2) * axis[1],
                np.sin(angle / 2) * axis[2],
            ])
        else:
            q_goal_vec = np.array([1, 0, 0, 0.0])

        err_quat = quat_angle(q_k, q_goal_vec)
        errors_v_quat.append(err_quat)

    min_bore_idx = errors_v_boresight.index(min(errors_v_boresight))
    min_quat_idx = errors_v_quat.index(min(errors_v_quat))
    print(f"  Boresight error: start={errors_v_boresight[0]:.2f}°, min={min(errors_v_boresight):.4f}° at t={min_bore_idx*dt:.0f}s, end={errors_v_boresight[-1]:.2f}°")
    print(f"  Quat error:      start={errors_v_quat[0]:.2f}°, min={min(errors_v_quat):.4f}° at t={min_quat_idx*dt:.0f}s, end={errors_v_quat[-1]:.2f}°")
    print(f"  Final ω = {X_v[:3, -1]} rad/s")
    print(f"  Max |u_rw| = {np.max(np.abs(U_v[3, :])):.6f} N·m")

    # Check: boresight error vs quaternion error at a few timesteps
    print("\n  Boresight vs Quaternion error comparison:")
    for k in [0, min_bore_idx, N_v // 2, N_v - 1]:
        print(f"    t={k*dt:6.0f}s: boresight={errors_v_boresight[k]:7.3f}°, quat={errors_v_quat[k]:7.3f}°, diff={abs(errors_v_boresight[k]-errors_v_quat[k]):7.3f}°")

    # =========================================================================
    # TEST 3: processAttitudeTarget verification
    # =========================================================================
    print()
    print("=" * 70)
    print("TEST 3: processAttitudeTarget verification")
    print("=" * 70)

    # Verify that the q_goal from processAttitudeTarget correctly aligns boresight
    test_cases = [
        ([1, 0, 0], [0, 0, 1]),  # x → z (90°)
        ([1, 0, 0], [0, 1, 0]),  # x → y (90°)
        ([0, 0, 1], [1, 0, 0]),  # z → x (90°)
        ([1, 0, 0], [-1, 0, 0]),  # x → -x (180°)
        ([1, 0, 0], [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]),  # x → [1,1,1]/√3
    ]
    for bore, eci in test_cases:
        bore = np.array(bore, dtype=float)
        eci = np.array(eci, dtype=float)
        axis = np.cross(bore, eci)
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-8:
            axis_n = axis / axis_norm
            angle = np.arccos(np.clip(np.dot(bore, eci), -1, 1))
            q_goal = np.array([np.cos(angle/2), *(np.sin(angle/2) * axis_n)])
        elif np.dot(bore, eci) > 0:
            q_goal = np.array([1, 0, 0, 0.0])
        else:
            perp = np.array([0, 1, 0.0]) if abs(bore[0]) < 0.9 else np.array([1, 0, 0.0])
            perp = perp - np.dot(perp, bore) * bore
            perp = perp / np.linalg.norm(perp)
            q_goal = np.array([0, *perp])

        R_goal = rotmat(q_goal)
        bore_eci = R_goal @ bore
        alignment_error = np.degrees(np.arccos(np.clip(np.dot(bore_eci, eci), -1, 1)))
        print(f"  bore={bore} → eci={eci}: q_goal={q_goal}, R*bore={bore_eci}, error={alignment_error:.6f}°")

    # =========================================================================
    # TEST 4: Dynamics consistency check
    # =========================================================================
    print()
    print("=" * 70)
    print("TEST 4: Dynamics consistency (optimizer X matches forward integration)")
    print("=" * 70)

    # Use the quaternion trajectory (TEST 1) to check dynamics consistency
    # Re-run orbit generation to get B, R_pos, etc.
    dt_cent = dt / (36525.0 * 86400.0)
    t0 = jtime[0]
    tN = jtime[1]
    jtime_fine = np.arange(t0, tN + dt_cent/2, dt_cent)
    if abs(jtime_fine[-1] - tN) > 1e-12:
        jtime_fine = np.append(jtime_fine, tN)

    ok, R_pos, V_vel, B_field, S_sun, rho = saltro_py.generate_orbit(
        r0, v0, jtime_fine, 0, 0, 0, 0, 0
    )
    B_field = np.asfortranarray(B_field)

    max_dyn_error = 0.0
    for k in range(min(N_q - 1, B_field.shape[1] - 1)):
        x_k = X_q[:, k]
        u_k = U_q[:, k]
        dist = saltro_py.DisturbanceConfig()
        x_next_model = X_q[:, k + 1]

        # Forward step with RK4
        dxdt = satellite.dynamics(x_k, u_k, dist, R_pos[:, k], B_field[:, k], S_sun[:, k], V_vel[:, k], 0)
        # Simple Euler check (not exact, but shows order of magnitude)
        x_next_euler = x_k + dt * dxdt
        q_next = x_next_euler[3:7]
        x_next_euler[3:7] = q_next / np.linalg.norm(q_next)

        dyn_error = np.linalg.norm(x_next_model - x_next_euler)
        max_dyn_error = max(max_dyn_error, dyn_error)

    print(f"  Max Euler step deviation: {max_dyn_error:.6e}")
    print(f"  (This should be O(dt²) ≈ {dt**2:.0f} for Euler vs RK4)")

    # =========================================================================
    # TEST 5: B-field and controllability analysis
    # =========================================================================
    print()
    print("=" * 70)
    print("TEST 5: B-field and controllability")
    print("=" * 70)

    # At each timestep, compute the MTQ torque authority
    # MTQ can produce torque perpendicular to B (in body frame)
    # The null direction is along B (in body frame)
    for k_check in [0, N_q // 4, N_q // 2, 3 * N_q // 4, N_q - 2]:
        q_k = X_q[3:7, k_check]
        q_k = q_k / np.linalg.norm(q_k)
        R_k = rotmat(q_k)
        B_eci_k = B_field[:, k_check]
        B_body_k = R_k.T @ B_eci_k
        B_body_hat = B_body_k / np.linalg.norm(B_body_k)
        print(f"  t={k_check*dt:6.0f}s: B_body_hat={B_body_hat}, |B|={np.linalg.norm(B_eci_k):.3e} T")

    # =========================================================================
    # TEST 6: Cost function at each timestep
    # =========================================================================
    print()
    print("=" * 70)
    print("TEST 6: Stage cost profile (quaternion goal)")
    print("=" * 70)

    # Check stage cost at a few timesteps
    cost_cfg = plannersettings.passes[0].cost
    for k_check in [0, N_q // 4, N_q // 2, 3 * N_q // 4, N_q - 2, N_q - 1]:
        x_k = X_q[:, k_check]
        u_k = U_q[:, k_check] if k_check < U_q.shape[1] else np.zeros(satellite.controlDim)
        bore_k = boresight[:, 0]  # constant
        att_k = qgoal_quat[:, 0]  # constant
        B_k = B_field[:, k_check]
        c = satellite.stageCost(k_check, N_q, x_k, u_k, bore_k, att_k, B_k, cost_cfg)
        err = errors_q[k_check]
        w_k = x_k[:3]
        print(f"  t={k_check*dt:6.0f}s: cost={c:.6e}, err={err:.3f}°, |ω|={np.linalg.norm(w_k):.6e} rad/s")

    # =========================================================================
    # TEST 7: Compare quaternion vs vector goal behavior
    # =========================================================================
    print()
    print("=" * 70)
    print("TEST 7: Drift analysis")
    print("=" * 70)

    # For quaternion goal: check if drift is present
    drift_q = errors_q[-1] - min(errors_q)
    drift_v = errors_v_boresight[-1] - min(errors_v_boresight)
    print(f"  Quaternion goal: min_error={min(errors_q):.4f}°, final_error={errors_q[-1]:.2f}°, drift={drift_q:.2f}°")
    print(f"  Vector goal:     min_error={min(errors_v_boresight):.4f}°, final_error={errors_v_boresight[-1]:.2f}°, drift={drift_v:.2f}°")

    # Check angular velocity at min-error time and final time
    min_k_q = errors_q.index(min(errors_q))
    min_k_v = errors_v_boresight.index(min(errors_v_boresight))
    print(f"\n  Quat goal: ω at min error (t={min_k_q*dt:.0f}s) = {X_q[:3, min_k_q]}")
    print(f"  Quat goal: ω at final     (t={(N_q-1)*dt:.0f}s) = {X_q[:3, -1]}")
    print(f"  Vec  goal: ω at min error (t={min_k_v*dt:.0f}s) = {X_v[:3, min_k_v]}")
    print(f"  Vec  goal: ω at final     (t={(N_v-1)*dt:.0f}s) = {X_v[:3, -1]}")

    # =========================================================================
    # TEST 8: Terminal cost analysis
    # =========================================================================
    print()
    print("=" * 70)
    print("TEST 8: Terminal cost weights")
    print("=" * 70)
    print(f"  angle_N = {cost.angle_N}")
    print(f"  ang_vel_N = {cost.ang_vel_N}")
    print(f"  >> Terminal costs are ZERO — optimizer has no incentive to minimize final error!")
    print(f"  >> This could cause convergence-then-drift behavior.")

    # =========================================================================
    # TEST 9: Check if non-zero terminal costs fix the drift
    # =========================================================================
    print()
    print("=" * 70)
    print("TEST 9: With terminal costs enabled")
    print("=" * 70)

    cost.angle_N = 10.0  # Terminal attitude cost
    cost.ang_vel_N = 100.0  # Terminal angular velocity cost

    X_t, U_t, reason_t = trajOpt(
        plannersettings, satellite, x0, r0, v0, jtime, qgoal_quat, boresight, debug=False
    )
    N_t = X_t.shape[1]

    errors_t = []
    for k in range(N_t):
        q_k = X_t[3:7, k]
        q_k = q_k / np.linalg.norm(q_k)
        err = quat_angle(q_k, q_target)
        errors_t.append(err)

    min_err_t = min(errors_t)
    drift_t = errors_t[-1] - min_err_t
    print(f"  Stop reason: {reason_t}")
    print(f"  min_error={min_err_t:.4f}°, final_error={errors_t[-1]:.2f}°, drift={drift_t:.2f}°")
    print(f"  Final ω = {X_t[:3, -1]} rad/s")

    # Reset terminal costs
    cost.angle_N = 0.0
    cost.ang_vel_N = 0.0

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Without terminal cost: drift = {drift_q:.2f}° (quat), {drift_v:.2f}° (vec)")
    print(f"  With terminal cost:    drift = {drift_t:.2f}°")
    if drift_t < drift_q * 0.5:
        print(f"  >> Terminal cost significantly reduces drift — root cause is likely")
        print(f"     missing terminal cost, not a frame issue.")
    else:
        print(f"  >> Terminal cost does NOT fix drift — investigate frame/dynamics issues.")


if __name__ == "__main__":
    run_diagnostic()
