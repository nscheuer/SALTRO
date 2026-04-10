"""Standalone test for substitute_and_blend without running the full optimizer.

Builds a synthetic spiked trajectory (pointing error drops, spikes up, returns),
calls detect_spikes → simulate_pd_segment → substitute_and_blend directly, and
plots before/after to verify the substitution is geometrically sensible.

Run from WSL:
    cd /path/to/saltro
    python tests/debug/optimizer/alilqr_python/test_substitute_blend.py

No optimizer (saltro_py forward/backward passes) is needed.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))

import saltro_py
from sat_3_1_hybrid import create_satellite
from spike_removal import (
    detect_spikes,
    simulate_pd_segment,
    substitute_and_blend,
    _rk4_step,
    _pointing_error,
)


# -----------------------------------------------------------------------
# Build synthetic spiked trajectory
# -----------------------------------------------------------------------

def _quat_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _axis_angle_quat(axis, angle_rad):
    axis = np.array(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    return np.array([np.cos(angle_rad / 2), *(np.sin(angle_rad / 2) * axis)])


def build_spiked_trajectory(satellite, dist_cfg, B_field, S_sun, R_pos, V_vel, rho, dt, N,
                             spike_enter, spike_exit, spike_angle_deg=80.0):
    """
    Build a synthetic trajectory that:
      - starts at identity q=[1,0,0,0]
      - slews smoothly toward q_goal (90° about Z) for knots 0..spike_enter
      - inserts an artificial detour (spike) from spike_enter..spike_exit
      - resumes toward q_goal after spike_exit
      - is dynamically feasible (integrated with zero control, so ω≈0 throughout)

    The spike is purely kinematic: we directly set quaternion states.
    Angular velocity is set to the finite-difference of the quaternion path.
    """
    nx = 8  # [ω(3), q(4), h(1)]
    nu = satellite.controlDim

    q_start = np.array([1.0, 0.0, 0.0, 0.0])
    q_goal  = _axis_angle_quat([0, 0, 1], np.pi / 2)  # 90° about Z

    # Detour quaternion: go the "wrong way" (spike = extra 80° opposite)
    q_detour = _axis_angle_quat([0, 0, 1], -(spike_angle_deg * np.pi / 180.0))

    X = np.zeros((nx, N))
    U = np.zeros((nu, N - 1))

    def slerp(q0, q1, t):
        dot = np.clip(np.dot(q0, q1), -1, 1)
        if dot < 0:
            q1 = -q1
            dot = -dot
        if dot > 0.9999:
            return q0 + t * (q1 - q0)
        theta = np.arccos(dot)
        s = np.sin(theta)
        return np.sin((1 - t) * theta) / s * q0 + np.sin(t * theta) / s * q1

    for k in range(N):
        if k <= spike_enter:
            t = k / spike_enter if spike_enter > 0 else 0.0
            q = slerp(q_start, q_goal, t * 0.6)  # only 60% of the way by spike entry
        elif k <= spike_exit:
            t = (k - spike_enter) / (spike_exit - spike_enter)
            # Interpolate from current to detour and back
            # Up phase: spike_enter → mid
            mid = (spike_enter + spike_exit) // 2
            if k <= mid:
                t2 = (k - spike_enter) / (mid - spike_enter)
                q_at_enter = slerp(q_start, q_goal, 0.6)
                q = slerp(q_at_enter, q_detour, t2)
            else:
                t2 = (k - mid) / (spike_exit - mid)
                q_at_enter = slerp(q_start, q_goal, 0.6)
                q = slerp(q_detour, q_at_enter, t2)
        else:
            t = (k - spike_exit) / (N - 1 - spike_exit)
            q_at_exit = slerp(q_start, q_goal, 0.6)
            q = slerp(q_at_exit, q_goal, t)

        q /= np.linalg.norm(q)
        X[3:7, k] = q
        X[7, k] = 0.0  # RW momentum = 0

    # Compute angular velocities from quaternion differences
    for k in range(N - 1):
        q0 = X[3:7, k]
        q1 = X[3:7, k + 1]
        # dq/dt ≈ (q1 - q0) / dt
        # omega_body ≈ 2 * quat_mult(conj(q0), dq)  [vector part only]
        dq = (q1 - q0) / dt
        # conj(q0) ⊗ dq → vector part * 2 = omega
        # q = [w, x, y, z], conj = [w, -x, -y, -z]
        q0c = np.array([q0[0], -q0[1], -q0[2], -q0[3]])
        # Manual mult
        w1, x1, y1, z1 = q0c
        w2, x2, y2, z2 = dq
        prod = np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])
        omega = 2.0 * prod[1:4]
        X[0:3, k] = omega

    X[0:3, N - 1] = X[0:3, N - 2]  # copy last omega

    return X, U, q_goal


def compute_pointing_errors(X, q_goal, N):
    """Compute geodesic angle error at each knot (degrees)."""
    errs = []
    for k in range(N):
        q = X[3:7, k]
        q /= np.linalg.norm(q)
        dot = abs(np.dot(q, q_goal))
        dot = min(dot, 1.0)
        errs.append(np.degrees(2.0 * np.arccos(dot)))
    return errs


def main():
    # -----------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------
    plannersettings = saltro_py.PlannerSettings()
    plannersettings.init_traj.initcontroller = 1
    plannersettings.num_passes = 1
    plannersettings.passes[0].dt = 10.0
    plannersettings.passes[0].ilqr.cost_tol = 1e-3
    plannersettings.passes[0].ilqr.max_iters = 20
    plannersettings.passes[0].auglag.max_outer_iters = 10
    plannersettings.passes[0].auglag.constraint_tol = 1e-3
    cost = plannersettings.passes[0].cost
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
    cost.use_cost_hess = False
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
    dist_cfg = plannersettings.disturbances
    cnst_cfg = plannersettings.constraints

    dt = 10.0
    N = 100  # small enough to run fast

    # Orbit environment at a single location (LEO)
    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, 7.5e3, 0.0])
    t0 = 0.22
    dt_cent = dt / (36525.0 * 86400.0)
    jtime_fine = np.array([t0 + i * dt_cent for i in range(N)])

    ok, R_pos, V_vel, B_field, S_sun, rho_arr = saltro_py.generate_orbit(
        r0, v0, jtime_fine, 0, 0, 0, 0, 0
    )
    if not ok:
        raise RuntimeError("generate_orbit failed")

    B_field = np.asfortranarray(B_field)
    S_sun   = np.asfortranarray(S_sun)
    R_pos   = np.asfortranarray(R_pos)
    V_vel   = np.asfortranarray(V_vel)
    rho_arr = rho_arr.reshape(1, -1) if rho_arr.ndim == 1 else rho_arr

    # -----------------------------------------------------------------------
    # Build spiked trajectory
    # -----------------------------------------------------------------------
    spike_enter = 25
    spike_exit  = 55

    X, U, q_goal = build_spiked_trajectory(
        satellite, dist_cfg, B_field, S_sun, R_pos, V_vel, rho_arr, dt, N,
        spike_enter, spike_exit, spike_angle_deg=80.0
    )
    U_bar = U.copy()  # treat synthetic U as "nominal"

    # Attitude target: constant q_goal for all knots
    attitude_target = np.tile(q_goal[:, None], (1, N))
    boresight       = np.tile(np.array([1.0, 0.0, 0.0])[:, None], (1, N))

    errs_before = compute_pointing_errors(X, q_goal, N)

    print("=" * 60)
    print("Spiked trajectory summary:")
    print(f"  N={N}, spike=[{spike_enter},{spike_exit}], dt={dt}s")
    print(f"  Max error before: {max(errs_before):.1f}° at k={errs_before.index(max(errs_before))}")
    print(f"  Error at spike_enter: {errs_before[spike_enter]:.1f}°")
    print(f"  Error at spike_exit: {errs_before[spike_exit]:.1f}°")

    # -----------------------------------------------------------------------
    # Run detect_spikes
    # -----------------------------------------------------------------------
    candidates = detect_spikes(
        X, U, attitude_target, boresight, B_field, satellite, cnst_cfg,
        goal_switch_buffer=5,
        min_consecutive=5,
        exit_fudge=2.0,
    )
    print(f"\ndetect_spikes found: {candidates}")
    if not candidates:
        print("WARNING: No spikes detected — check min_consecutive / exit_fudge parameters")
        # Still run the substitution manually for testing
        candidates = [(spike_enter, spike_exit)]
        print(f"  Forcing manual candidate: {candidates}")

    # -----------------------------------------------------------------------
    # Run PD sim on first candidate
    # -----------------------------------------------------------------------
    t_enter, t_exit = candidates[0]
    n_steps = t_exit - t_enter

    t_end_pd = min(t_exit + 1, N)
    B_slice   = B_field[:, t_enter:t_end_pd]
    S_slice   = S_sun[:, t_enter:t_end_pd]
    R_slice   = R_pos[:, t_enter:t_end_pd]
    V_slice   = V_vel[:, t_enter:t_end_pd]
    rho_slice = rho_arr[:, t_enter:t_end_pd]

    x_target = X[:, t_exit].copy()

    print(f"\nsimulate_pd_segment: [{t_enter}, {t_exit}), n_steps={n_steps}")
    X_pd, U_pd = simulate_pd_segment(
        x_start=X[:, t_enter].copy(),
        x_target=x_target,
        n_steps=n_steps,
        B_cols=B_slice,
        S_cols=S_slice,
        R_cols=R_slice,
        V_cols=V_slice,
        rho_cols=rho_slice,
        satellite=satellite,
        dist_cfg=dist_cfg,
        dt=dt,
        kp_q=2.0,
        kd_w=5.0,
    )

    # PD pointing errors within the window
    pd_errs = []
    for i in range(X_pd.shape[1]):
        q = X_pd[3:7, i]
        q /= np.linalg.norm(q)
        dot = abs(np.dot(q, q_goal))
        pd_errs.append(np.degrees(2.0 * np.arccos(min(dot, 1.0))))

    print(f"  PD max error: {max(pd_errs):.1f}°, PD final error: {pd_errs[-1]:.1f}°")
    print(f"  Original max error in window: {max(errs_before[t_enter:t_exit+1]):.1f}°")

    # -----------------------------------------------------------------------
    # Run substitute_and_blend
    # -----------------------------------------------------------------------
    X_pre = X.copy()
    K_list = None  # open-loop only (no iLQR gains in this synthetic test)

    X_after, U_after = substitute_and_blend(
        X.copy(), U.copy(),
        X_pd, U_pd,
        t_enter, t_exit,
        blend_len=20,
        X_nominal_pre=X_pre,
        U_bar=U_bar,
        K_list=K_list,
        satellite=satellite,
        dist_cfg=dist_cfg,
        B=B_field, S=S_sun, R=R_pos, V=V_vel, rho=rho_arr,
        dt=dt,
        kp_q=2.0,
        kd_w=5.0,
    )

    errs_after = compute_pointing_errors(X_after, q_goal, N)

    print("\nAfter substitution:")
    print(f"  Max error: {max(errs_after):.1f}° at k={errs_after.index(max(errs_after))}")
    print(f"  Error at t_enter={t_enter}: before={errs_before[t_enter]:.1f}°, after={errs_after[t_enter]:.1f}°")
    print(f"  Error at t_exit={t_exit}:  before={errs_before[t_exit]:.1f}°, after={errs_after[t_exit]:.1f}°")
    print(f"  Error at final:            before={errs_before[-1]:.1f}°, after={errs_after[-1]:.1f}°")

    # Sanity checks
    max_err_before_spike = max(errs_before[t_enter:t_exit+1])
    max_err_after_spike  = max(errs_after[t_enter:t_exit+1])
    print(f"\nSanity: max error in spike window reduced: {max_err_before_spike:.1f}° → {max_err_after_spike:.1f}°", end=" ")
    print("PASS" if max_err_after_spike < max_err_before_spike else "FAIL")

    # Check quaternion normalization throughout
    q_norms = [np.linalg.norm(X_after[3:7, k]) for k in range(N)]
    max_qnorm_dev = max(abs(n - 1.0) for n in q_norms)
    print(f"Sanity: max quaternion norm deviation: {max_qnorm_dev:.2e}", end=" ")
    print("PASS" if max_qnorm_dev < 1e-8 else "FAIL")

    # Check state is finite
    is_finite = np.all(np.isfinite(X_after))
    print(f"Sanity: all states finite: {is_finite}", end=" ")
    print("PASS" if is_finite else "FAIL")

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    t_axis = np.arange(N) * dt

    ax = axes[0]
    ax.plot(t_axis, errs_before, "b-", label="Before substitution", lw=2)
    ax.plot(t_axis, errs_after, "r-", label="After substitution", lw=2)
    ax.axvspan(t_enter * dt, t_exit * dt, alpha=0.15, color="orange", label="Spike window")
    ax.axvspan(t_exit * dt, min((t_exit + 20) * dt, (N-1)*dt), alpha=0.1, color="green", label="Blend zone")
    ax.set_ylabel("Pointing error (deg)")
    ax.set_title("Substitute-and-Blend: Pointing Error")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)

    ax = axes[1]
    omega_before = np.linalg.norm(X_pre[0:3, :], axis=0)
    omega_after  = np.linalg.norm(X_after[0:3, :], axis=0)
    ax.plot(t_axis, omega_before, "b-", label="Before", lw=2)
    ax.plot(t_axis, omega_after,  "r-", label="After", lw=2)
    ax.axvspan(t_enter * dt, t_exit * dt, alpha=0.15, color="orange")
    ax.set_ylabel("|ω| (rad/s)")
    ax.set_title("Angular velocity magnitude")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)

    ax = axes[2]
    # PD errors in the window
    pd_t = np.arange(t_enter, t_enter + len(pd_errs)) * dt
    ax.plot(t_axis, errs_before, "b--", alpha=0.5, label="Original trajectory", lw=1.5)
    ax.plot(pd_t,  pd_errs,      "g-",  label="PD segment (standalone)", lw=2)
    ax.plot(t_axis, errs_after,  "r-",  label="Blended+tail trajectory", lw=2)
    ax.axvspan(t_enter * dt, t_exit * dt, alpha=0.15, color="orange", label="Spike window")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pointing error (deg)")
    ax.set_title("PD segment vs blended result")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    out_path = Path.home() / "test_substitute_blend.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {out_path}")


if __name__ == "__main__":
    main()
