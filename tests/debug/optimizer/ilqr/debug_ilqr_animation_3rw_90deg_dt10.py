"""
Interactive iLQR debugger with per-iteration visualization.

This script runs iLQR in Python (loop control), while using C++ bindings for
backward pass and forward pass. It opens a single interactive window with
forward/back buttons to inspect warm-start and each iLQR iteration.

REACTION WHEELS ONLY VERSION - No magnetorquers
"""

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.ticker import ScalarFormatter

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
import saltro_py

SEC_PER_CENTURY = 36525.0 * 86400.0


def _configure_sci_y(ax):
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(formatter)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))


def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def _is_eci_format(attitude_target_k):
    return np.isnan(attitude_target_k[0])


def _quat_from_two_vectors(v_from, v_to):
    a = _normalize(v_from)
    b = _normalize(v_to)
    dot = np.clip(np.dot(a, b), -1.0, 1.0)

    if dot > 1.0 - 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])

    if dot < -1.0 + 1e-10:
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-10:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        axis = _normalize(axis)
        return np.array([0.0, axis[0], axis[1], axis[2]])

    c = np.cross(a, b)
    q = np.array([1.0 + dot, c[0], c[1], c[2]])
    return _normalize(q)


def _goal_quaternion_for_plot(attitude_target_k, boresight_k, q_current):
    if _is_eci_format(attitude_target_k):
        target_vec = attitude_target_k[1:4]
        if np.linalg.norm(target_vec) < 1e-9:
            return q_current.copy()
        return _quat_from_two_vectors(boresight_k, target_vec)
    return _normalize(attitude_target_k)


def _angle_error_deg(q, q_goal):
    qd = np.clip(np.abs(np.dot(_normalize(q), _normalize(q_goal))), -1.0, 1.0)
    return np.degrees(2.0 * np.arccos(qd))


def _sanitize_quaternion_state_inplace(X, quat_start=3, eps=1e-10):
    """Ensure quaternion columns are normalizable before calling C++ passes."""
    if X.shape[0] < quat_start + 4:
        return
    for k in range(X.shape[1]):
        q = X[quat_start:quat_start + 4, k]
        n = np.linalg.norm(q)
        if n < eps or not np.isfinite(n):
            X[quat_start:quat_start + 4, k] = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            X[quat_start:quat_start + 4, k] = q / n


def setup_satellite():
    # Quick-tune optimization knobs
    line_search_max_iters = 24
    line_search_beta1 = 1e-10
    line_search_beta2 = 5000.0
    # Angle cost function selection:
    # 0: 1-|q·q_goal|, 1: 0.5*(1-|q·q_goal|)^2, 2: acos(|q·q_goal|),
    # 3: 0.5*acos(|q·q_goal|)^2, 4: 1-|q·q_goal|^2
    ang_cost_func_type = 4

    settings = saltro_py.PlannerSettings()
    settings.num_passes = 1
    settings.passes[0].dt = 10.0
    settings.passes[0].ilqr.cost_tol = 1e-5
    settings.init_traj.initcontroller = 2
    settings.passes[0].linesearch.max_iters = line_search_max_iters
    settings.passes[0].linesearch.beta1 = line_search_beta1
    settings.passes[0].linesearch.beta2 = line_search_beta2

    # ---------------------------------------------------------------------
    # Manual objective/constraint baseline: start from effectively-zero setup
    # ---------------------------------------------------------------------
    # Constraints (neutralized so they do not affect optimization)
    settings.constraints.control_limit_scale = 0.0
    settings.constraints.wmax = 1e9
    settings.constraints.sun_limit_angle = 0.0

    # Costs (all zero; user can manually increase desired terms)
    cost = settings.passes[0].cost
    cost.angle = 1e5
    cost.ang_vel = 1e4
    cost.ang_vel_mag = 0.0
    cost.ang_vel_err_dir = 0.0
    cost.control_mult = 1.0
    cost.mtq_control_weight = 1.0
    cost.rw_control_weight = 1e7
    cost.magic_control_weight = 0.0
    cost.rw_AM_weight = 0.0
    cost.rw_stic_weight = 0.0
    cost.RWh_max_mult = 0.0
    cost.RWh_stiction_mult = 0.0
    cost.RWh_ok_mult = 0.0
    cost.angle_N = 1e6
    cost.ang_vel_N = 1e7
    cost.ang_vel_mag_N = 0.0
    cost.ang_vel_err_dir_N = 0.0
    cost.ang_cost_func_type = ang_cost_func_type
    cost.use_cost_hess = True

    settings.disturbances.plan_for_aero = False
    settings.disturbances.plan_for_gg = False
    settings.disturbances.plan_for_srp = False
    settings.disturbances.plan_for_prop = False
    settings.disturbances.plan_for_gendist = False
    settings.disturbances.plan_for_resdipole = False

    J = np.diag([0.067, 0.071, 0.069])
    satellite = saltro_py.Satellite(J, settings)

    # Reaction wheels only - no magnetorquers
    satellite.addRW(np.array([1.0, 0.0, 0.0]), 0.001, 1e-5, 0.0, 0.02)
    satellite.addRW(np.array([0.0, 1.0, 0.0]), 0.001, 1e-5, 0.0, 0.02)
    satellite.addRW(np.array([0.0, 0.0, 1.0]), 0.001, 1e-5, 0.0, 0.02)

    return satellite, settings


def make_time_grid(N, dt):
    jtime = np.zeros(N)
    dt_centuries = dt / SEC_PER_CENTURY
    for k in range(N):
        jtime[k] = 0.25 + k * dt_centuries
    return jtime


def make_targets(N):
    # 90-degree rotation about z-axis: q = [cos(45°), 0, 0, sin(45°)]
    attitude_target_traj = np.zeros((4, N))
    attitude_target_traj[0, :] = np.sqrt(2) / 2  # cos(45°) ≈ 0.7071
    attitude_target_traj[3, :] = np.sqrt(2) / 2  # sin(45°) ≈ 0.7071
    boresight = np.zeros((3, N))
    boresight[0, :] = 1.0
    return attitude_target_traj, boresight


def make_environment(jtime):
    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, 7500.0, 0.0])
    ok, R, V, B, S, rho = saltro_py.generate_orbit(r0, v0, jtime, 0, 0, 0, 0, 0)
    if not ok:
        raise RuntimeError("generate_orbit failed")
    return R, V, B, S, rho.reshape(1, -1)


def make_initial_state(satellite):
    x0 = np.zeros(satellite.stateDim)
    x0[0:3] = np.array([0.0, 0.0, 0.0])  # Zero initial angular velocity
    x0[3:7] = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion (no initial rotation)
    return x0


def generate_warmstart(satellite, settings, x0, jtime, attitude_target_traj, boresight, R, V, B, S, rho):
    t0 = time.time()
    ok, X, U = saltro_py.warm_start(
        settings, satellite, x0, jtime,
        attitude_target_traj, boresight,
        R, V, B, S, rho,
    )
    if not ok:
        raise RuntimeError("warm_start failed")
    return X, U, (time.time() - t0)


def compute_cost_components(X, U, satellite, attitude_target_traj, boresight, B, cost_cfg):
    N = X.shape[1]
    components = {
        "attitude": np.zeros(N),
        "angular_velocity": np.zeros(N),
        "control": np.zeros(N),
        "rw_momentum": np.zeros(N),
    }

    for comp_name in components.keys():
        cfg = saltro_py.CostConfig()
        cfg.angle = cost_cfg.angle if comp_name == "attitude" else 0.0
        cfg.angle_N = cost_cfg.angle_N if comp_name == "attitude" else 0.0
        cfg.ang_vel = cost_cfg.ang_vel if comp_name == "angular_velocity" else 0.0
        cfg.ang_vel_N = cost_cfg.ang_vel_N if comp_name == "angular_velocity" else 0.0
        cfg.ang_vel_mag = cost_cfg.ang_vel_mag if comp_name == "angular_velocity" else 0.0
        cfg.ang_vel_mag_N = cost_cfg.ang_vel_mag_N if comp_name == "angular_velocity" else 0.0
        cfg.ang_vel_err_dir = cost_cfg.ang_vel_err_dir if comp_name == "angular_velocity" else 0.0
        cfg.ang_vel_err_dir_N = cost_cfg.ang_vel_err_dir_N if comp_name == "angular_velocity" else 0.0
        cfg.control_mult = cost_cfg.control_mult if comp_name == "control" else 0.0
        cfg.mtq_control_weight = cost_cfg.mtq_control_weight if comp_name == "control" else 0.0
        cfg.rw_control_weight = cost_cfg.rw_control_weight if comp_name == "control" else 0.0
        cfg.magic_control_weight = cost_cfg.magic_control_weight if comp_name == "control" else 0.0
        cfg.rw_AM_weight = cost_cfg.rw_AM_weight if comp_name == "rw_momentum" else 0.0
        cfg.rw_stic_weight = cost_cfg.rw_stic_weight if comp_name == "rw_momentum" else 0.0
        cfg.RWh_max_mult = cost_cfg.RWh_max_mult if comp_name == "rw_momentum" else 0.0
        cfg.RWh_stiction_mult = cost_cfg.RWh_stiction_mult if comp_name == "rw_momentum" else 0.0
        cfg.RWh_ok_mult = cost_cfg.RWh_ok_mult if comp_name == "rw_momentum" else 0.0
        cfg.ang_cost_func_type = cost_cfg.ang_cost_func_type
        cfg.use_cost_hess = False

        for k in range(N):
            u_k = U[:, k] if k < U.shape[1] else np.zeros(satellite.controlDim)
            c = satellite.stageCost(
                k, N,
                X[:, k],
                u_k,
                boresight[:, k],
                attitude_target_traj[:, k],
                B[:, k],
                cfg,
            )
            components[comp_name][k] = max(0.0, c)

    return components


def collect_snapshot(X, U, satellite, B, boresight, attitude_target_traj, cost_cfg):
    N = X.shape[1]
    U_trim = U[:, :N - 1]
    J = satellite.totalCost(X, U_trim, B, boresight, attitude_target_traj, cost_cfg)
    components = compute_cost_components(X, U, satellite, attitude_target_traj, boresight, B, cost_cfg)

    q = X[3:7, :]
    q_goal = np.zeros_like(q)
    pointing_err_deg = np.zeros(N)
    for k in range(N):
        q_goal[:, k] = _goal_quaternion_for_plot(attitude_target_traj[:, k], boresight[:, k], q[:, k])
        pointing_err_deg[k] = _angle_error_deg(q[:, k], q_goal[:, k])

    return {
        "X": X.copy(),
        "U": U.copy(),
        "J": float(J),
        "components": components,
        "q_goal": q_goal,
        "pointing_err_deg": pointing_err_deg,
    }


def run_ilqr_python_loop(satellite, settings, X0, U0, jtime, R, V, B, S, rho, boresight, attitude_target_traj):
    cost_cfg = settings.passes[0].cost
    ilqr_cfg = settings.passes[0].ilqr
    reg_cfg = settings.passes[0].reg

    X = X0.copy()
    U = U0.copy()
    _sanitize_quaternion_state_inplace(X)
    U_trim = U[:, :X.shape[1] - 1]

    snapshots = [collect_snapshot(X, U, satellite, B, boresight, attitude_target_traj, cost_cfg)]
    transitions = []
    stop_reason = "max_iters reached"

    for _ in range(ilqr_cfg.max_iters):
        _sanitize_quaternion_state_inplace(X)
        
        # Initialize regularization for this iteration
        reg_current = reg_cfg.reg_init
        ok_fp = False
        
        # Retry loop: if line search fails, increase regularization and retry
        while reg_current <= reg_cfg.reg_max:
            settings.passes[0].reg.reg_init = reg_current
            
            try:
                ok_bp, K_arr, d_arr, deltaV = saltro_py.backward_pass(
                    satellite, X, U_trim, R, V, B, S, rho, boresight, attitude_target_traj, settings
                )
            except RuntimeError as exc:
                stop_reason = f"backward pass exception: {exc}"
                return snapshots, transitions, stop_reason
            if not ok_bp:
                stop_reason = "backward pass failed"
                return snapshots, transitions, stop_reason

            K_list = [K_arr[k] for k in range(K_arr.shape[0])]
            d_list = [d_arr[:, k] for k in range(d_arr.shape[1])]

            J_prev = snapshots[-1]["J"]
            try:
                ok_fp, X_new, U_new, J_new = saltro_py.forward_pass(
                    satellite,
                    X,
                    U,
                    K_list,
                    d_list,
                    deltaV,
                    B,
                    R,
                    V,
                    S,
                    rho,
                    boresight,
                    attitude_target_traj,
                    settings,
                    jtime,
                    J_prev,
                )
            except RuntimeError as exc:
                stop_reason = f"forward pass exception: {exc}"
                return snapshots, transitions, stop_reason
            
            if ok_fp:
                # Forward pass succeeded, exit retry loop
                break
            else:
                # Forward pass failed, increase regularization and retry
                reg_current *= reg_cfg.reg_scale
        
        # Check if we exhausted regularization retries
        if not ok_fp:
            stop_reason = "forward pass failed: regularization exceeded max"
            break

        _sanitize_quaternion_state_inplace(X_new)

        pred_delta = max(0.0, -(deltaV[0] + deltaV[1]))
        act_delta = J_prev - J_new
        tol_ok = abs(act_delta) <= ilqr_cfg.cost_tol

        transition = {
            "bp_ok": bool(ok_bp),
            "fp_ok": bool(ok_fp),
            "pred_delta": float(pred_delta),
            "act_delta": float(act_delta),
            "cost_decrease_ok": bool(J_new <= J_prev + 1e-12),
            "delta_tol_ok": bool(tol_ok),
        }
        transitions.append(transition)

        X = X_new
        U = U_new
        U_trim = U[:, :X.shape[1] - 1]
        snapshots.append(collect_snapshot(X, U, satellite, B, boresight, attitude_target_traj, cost_cfg))

        if tol_ok:
            stop_reason = "converged: |ΔJ| <= cost_tol"
            break

    return snapshots, transitions, stop_reason


def launch_iteration_viewer(snapshots, transitions, stop_reason, dt, cost_tol):
    N = snapshots[0]["X"].shape[1]
    t_state = np.arange(N) * dt
    component_order = ["attitude", "angular_velocity", "control", "rw_momentum"]
    component_colors = ["#FF6B6B", "#2A9D8F", "#6D597A", "#FFA07A"]

    fig = plt.figure(figsize=(18, 14), constrained_layout=True)
    gs = fig.add_gridspec(5, 2, hspace=0.35, wspace=0.25)

    ax_q = fig.add_subplot(gs[0, 0])
    ax_w = fig.add_subplot(gs[0, 1])
    ax_h = fig.add_subplot(gs[1, 0])
    ax_pe = fig.add_subplot(gs[1, 1])
    ax_u_rw = fig.add_subplot(gs[2, 0])
    ax_cat = fig.add_subplot(gs[2, 1])
    ax_ind = fig.add_subplot(gs[3, 0])
    ax_J = fig.add_subplot(gs[3, 1])
    ax_txt = fig.add_subplot(gs[4, :])

    idx = {"value": 0}

    def set_index(new_idx):
        idx["value"] = int(np.clip(new_idx, 0, len(snapshots) - 1))
        update_view(idx["value"])

    def update_view(i):
        snap = snapshots[i]
        X = snap["X"]
        U = snap["U"]
        t_control = (np.arange(N - 1) * dt if U.shape[1] == N - 1 else np.arange(N) * dt)
        q = X[3:7, :]
        w = X[0:3, :]
        h = X[7:10, :]
        q_goal = snap["q_goal"]
        pe = snap["pointing_err_deg"]
        components = snap["components"]

        ax_q.clear()
        ax_q.plot(t_state, q[0, :], label="q0")
        ax_q.plot(t_state, q[1, :], label="q1")
        ax_q.plot(t_state, q[2, :], label="q2")
        ax_q.plot(t_state, q[3, :], label="q3")
        ax_q.plot(t_state, q_goal[0, :], "--", alpha=0.7, label="q0 goal")
        ax_q.plot(t_state, q_goal[1, :], "--", alpha=0.7, label="q1 goal")
        ax_q.plot(t_state, q_goal[2, :], "--", alpha=0.7, label="q2 goal")
        ax_q.plot(t_state, q_goal[3, :], "--", alpha=0.7, label="q3 goal")
        ax_q.set_title("Quaternion Time Series")
        ax_q.set_xlabel("Time [s]")
        ax_q.set_ylabel("q")
        ax_q.grid(True, alpha=0.3)
        ax_q.legend(fontsize=7, ncol=2)

        ax_w.clear()
        ax_w.plot(t_state, w[0, :], label="ωx")
        ax_w.plot(t_state, w[1, :], label="ωy")
        ax_w.plot(t_state, w[2, :], label="ωz")
        ax_w.plot(t_state, np.linalg.norm(w, axis=0), "k--", label="‖ω‖")
        ax_w.set_title("Angular Velocity Time Series")
        ax_w.set_xlabel("Time [s]")
        ax_w.set_ylabel("rad/s")
        ax_w.grid(True, alpha=0.3)
        ax_w.legend(fontsize=8)

        ax_h.clear()
        ax_h.plot(t_state, h[0, :], label="h_rw0")
        ax_h.plot(t_state, h[1, :], label="h_rw1")
        ax_h.plot(t_state, h[2, :], label="h_rw2")
        ax_h.set_title("Wheel Momentum Time Series")
        ax_h.set_xlabel("Time [s]")
        ax_h.set_ylabel("N·m·s")
        ax_h.grid(True, alpha=0.3)
        ax_h.legend(fontsize=8)

        ax_pe.clear()
        ax_pe.plot(t_state, pe, "o-", color="C3", markersize=3)
        ax_pe.set_title("Pointing Error Time Series")
        ax_pe.set_xlabel("Time [s]")
        ax_pe.set_ylabel("deg")
        ax_pe.grid(True, alpha=0.3)

        ax_u_rw.clear()
        ax_u_rw.plot(t_control, U[0, :], label="τ_rw0")
        ax_u_rw.plot(t_control, U[1, :], label="τ_rw1")
        ax_u_rw.plot(t_control, U[2, :], label="τ_rw2")
        ax_u_rw.set_title("RW Input Time Series")
        ax_u_rw.set_xlabel("Time [s]")
        ax_u_rw.set_ylabel("Torque [N·m]")
        ax_u_rw.grid(True, alpha=0.3)
        ax_u_rw.legend(fontsize=8)

        ax_cat.clear()
        bottom = np.zeros_like(t_state)
        for name, color in zip(component_order, component_colors):
            vals = components[name]
            ax_cat.bar(t_state, vals, width=dt * 0.8, bottom=bottom, color=color, alpha=0.85, label=name)
            bottom += vals
        ax_cat.set_title("Cost Time Series (Broken into Categories)")
        ax_cat.set_xlabel("Time [s]")
        ax_cat.set_ylabel("Cost")
        ax_cat.grid(True, alpha=0.3, axis="y")
        _configure_sci_y(ax_cat)
        ax_cat.legend(fontsize=8)

        ax_ind.clear()
        for name, color in zip(component_order, component_colors):
            ax_ind.plot(t_state, components[name], "o-", linewidth=1.8, markersize=3, color=color, label=name)
        ax_ind.set_title("Individual Costs Time Series")
        ax_ind.set_xlabel("Time [s]")
        ax_ind.set_ylabel("Cost")
        ax_ind.grid(True, alpha=0.3)
        _configure_sci_y(ax_ind)
        ax_ind.legend(fontsize=8)

        ax_J.clear()
        J_hist = [s["J"] for s in snapshots]
        it = np.arange(len(J_hist))
        ax_J.plot(it, J_hist, "o-", linewidth=2.0, color="C0")
        ax_J.plot(i, J_hist[i], "ro", markersize=8)
        ax_J.set_title("Total Cost ITERATION Series")
        ax_J.set_xlabel("Iteration")
        ax_J.set_ylabel("Total Cost")
        ax_J.grid(True, alpha=0.3)
        _configure_sci_y(ax_J)

        ax_txt.clear()
        ax_txt.axis("off")
        lines = []
        lines.append(f"Selected iteration: {i} / {len(snapshots)-1}")
        lines.append(f"Total cost J: {snap['J']:.6e}")
        lines.append(f"Stop reason: {stop_reason}")
        lines.append(f"Convergence tolerance: {cost_tol:.3e}")
        lines.append("")
        
        # Compress iteration table for long trajectories
        n_iters = len(snapshots)
        if n_iters > 20:
            # Show first 5, last 5, and current iteration
            lines.append("Convergence (showing first 5, last 5, current):")
            lines.append("it |bp fp Δ≤ tol")
            
            shown_rows = set()
            # First 5 iterations
            for k in range(1, min(6, n_iters)):
                shown_rows.add(k)
            # Last 5 iterations
            for k in range(max(1, n_iters - 5), n_iters):
                shown_rows.add(k)
            # Current iteration
            shown_rows.add(i)
            
            prev_shown = 0
            for k in sorted(shown_rows):
                if k - prev_shown > 1:
                    lines.append("  |  ...")
                
                tr = transitions[k - 1]
                marker = "<" if k == i else " "
                lines.append(
                    f"{marker}{k:2d} |"
                    f"{int(tr['bp_ok'])} {int(tr['fp_ok'])} "
                    f"{int(tr['delta_tol_ok'])} {int(tr['cost_decrease_ok'])}"
                )
                prev_shown = k
        else:
            # Show all for short trajectories
            lines.append("Convergence conditions (explicit per iteration):")
            lines.append("iter | bp_ok fp_ok J_new<=J_prev |ΔJ|<=tol")
            for k in range(1, n_iters):
                tr = transitions[k - 1]
                marker = "<" if k == i else " "
                lines.append(
                    f"{marker}{k:>3d} |"
                    f"   {int(tr['bp_ok'])}     {int(tr['fp_ok'])}"
                    f"       {int(tr['cost_decrease_ok'])}"
                    f"          {int(tr['delta_tol_ok'])}"
                )

        if i > 0 and i - 1 < len(transitions):
            tr = transitions[i - 1]
            lines.append("")
            lines.append("Current selected-step:")
            lines.append(f"ΔJ_pred: {tr['pred_delta']:.3e}  ΔJ_act: {tr['act_delta']:.3e}")
            lines.append(f"bp_ok: {tr['bp_ok']} fp_ok: {tr['fp_ok']} dec: {tr['cost_decrease_ok']} tol: {tr['delta_tol_ok']}")

        ax_txt.text(0.01, 0.99, "\n".join(lines), va="top", ha="left", fontsize=8, family="monospace")

        fig.suptitle("iLQR Iteration Debugger - RW Only (Warm-start + Iterations)", fontsize=14, fontweight="bold")
        fig.canvas.draw_idle()

    def on_prev(_event):
        set_index(idx["value"] - 1)

    def on_next(_event):
        set_index(idx["value"] + 1)

    btn_prev_ax = fig.add_axes([0.42, 0.01, 0.08, 0.04])
    btn_next_ax = fig.add_axes([0.51, 0.01, 0.08, 0.04])
    btn_prev = Button(btn_prev_ax, "◀ Back")
    btn_next = Button(btn_next_ax, "Forward ▶")
    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)

    def on_key(event):
        if event.key in ("left", "a"):
            on_prev(None)
        elif event.key in ("right", "d"):
            on_next(None)

    def on_cost_click(event):
        if event.inaxes is not ax_J or event.xdata is None:
            return
        if event.button != 1:
            return
        set_index(int(np.rint(event.xdata)))

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("button_press_event", on_cost_click)

    set_index(0)
    plt.show()


def main():
    N = 20
    dt = 10.0
    ilqr_max_iters = 20

    satellite, settings = setup_satellite()
    jtime = make_time_grid(N, dt)
    attitude_target_traj, boresight = make_targets(N)
    R, V, B, S, rho = make_environment(jtime)
    x0 = make_initial_state(satellite)

    # Warm-start only
    X0, U0, ws_time = generate_warmstart(
        satellite, settings, x0, jtime,
        attitude_target_traj, boresight,
        R, V, B, S, rho,
    )

    # Python-controlled iLQR loop with C++ passes
    settings.passes[0].ilqr.max_iters = ilqr_max_iters
    ilqr_t0 = time.time()
    snapshots, transitions, stop_reason = run_ilqr_python_loop(
        satellite,
        settings,
        X0,
        U0,
        jtime,
        R,
        V,
        B,
        S,
        rho,
        boresight,
        attitude_target_traj,
    )
    ilqr_time = time.time() - ilqr_t0

    X_final = snapshots[-1]["X"]
    print(f"Warm-start calculation time: {ws_time*1000:.2f} ms")
    print(f"iLQR sequence calculation time: {ilqr_time*1000:.2f} ms")
    print(f"Angular rates final: {X_final[0:3, -1]}")
    print(f"iLQR snapshots available: {len(snapshots)} (warm-start + iterations)")

    launch_iteration_viewer(
        snapshots,
        transitions,
        stop_reason,
        dt,
        settings.passes[0].ilqr.cost_tol,
    )


if __name__ == "__main__":
    main()
