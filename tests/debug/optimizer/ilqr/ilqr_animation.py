"""
Interactive iLQR iteration viewer with matplotlib.

Provides a single-window interface with forward/back navigation buttons
to inspect warm-start and each iLQR iteration's state, control, and cost breakdown.
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.ticker import ScalarFormatter


def _configure_sci_y(ax):
    """Configure scientific notation for y-axis."""
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(formatter)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))


def launch_iteration_viewer(snapshots, transitions, stop_reason, dt, cost_tol):
    """
    Launch interactive iLQR iteration viewer.
    
    Parameters
    ----------
    snapshots : list of dict
        Each snapshot contains: X, U, J, components, q_goal, pointing_err_deg
    transitions : list of dict
        Each transition contains: bp_ok, fp_ok, pred_delta, act_delta, cost_decrease_ok, delta_tol_ok
    stop_reason : str
        Why iLQR stopped (converged, max_iters, failed, etc.)
    dt : float
        Timestep in seconds
    cost_tol : float
        Cost tolerance for convergence
    """
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
