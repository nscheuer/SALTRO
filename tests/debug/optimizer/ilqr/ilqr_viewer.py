"""Interactive iLQR iteration viewer with matplotlib."""

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


def _compute_pointing_error_deg(q, q_goal):
    """Compute pointing error in degrees for each timestep."""
    N = q.shape[1]
    err_deg = np.zeros(N)
    for k in range(N):
        q_err = _quat_multiply(_quat_inverse(q_goal[:, k]), q[:, k])
        err_deg[k] = 2.0 * np.arctan2(np.linalg.norm(q_err[1:]), abs(q_err[0])) * 180.0 / np.pi
    return err_deg


def _quat_inverse(q):
    """Quaternion inverse: [w, -x, -y, -z] for unit quaternion."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _quat_multiply(q1, q2):
    """Quaternion multiplication."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def launch_viewer(snapshots, transitions, stop_reason, dt, cost_tol):
    """
    Launch interactive iLQR iteration viewer.
    
    Parameters
    ----------
    snapshots : list of dict
        Each: X, U, J, q_goal, components
    transitions : list of dict
        Each: bp_ok, fp_ok, act_delta, delta_tol_ok
    stop_reason : str
    dt : float (seconds)
    cost_tol : float
    """
    N = snapshots[0]["X"].shape[1]
    t_state = np.arange(N) * dt
    component_order = ["attitude", "angular_velocity", "control", "rw_momentum"]
    component_colors = ["#FF6B6B", "#2A9D8F", "#6D597A", "#FFA07A"]

    fig = plt.figure(figsize=(18, 16), constrained_layout=True)
    gs = fig.add_gridspec(5, 2, hspace=0.35, wspace=0.25)

    ax_q = fig.add_subplot(gs[0, 0])
    ax_w = fig.add_subplot(gs[0, 1])
    ax_h = fig.add_subplot(gs[1, 0])
    ax_pe = fig.add_subplot(gs[1, 1])
    ax_mtq_u = fig.add_subplot(gs[2, 0])
    ax_rw_u = fig.add_subplot(gs[2, 1])
    ax_cat = fig.add_subplot(gs[3, 0])
    ax_ind = fig.add_subplot(gs[3, 1])
    ax_J = fig.add_subplot(gs[4, 0])
    ax_txt = fig.add_subplot(gs[4, 1])

    idx = {"value": 0}

    def set_index(new_idx):
        idx["value"] = int(np.clip(new_idx, 0, len(snapshots) - 1))
        update_view(idx["value"])

    def update_view(i):
        snap = snapshots[i]
        X = snap["X"]
        U = snap["U"]
        q_goal = snap["q_goal"]
        components = snap.get("components", None)
        
        q = X[3:7, :]
        w = X[0:3, :]
        # RW momentum states are present only on RW-enabled satellites.
        h = X[7:, :]
        has_rw_state = h.shape[0] > 0
        
        N_u = U.shape[1]
        t_control = np.arange(N_u) * dt

        # Control ordering is [MTQ..., RW...]. RW momentum state count equals num_rw.
        num_rw = h.shape[0] if has_rw_state else 0
        num_rw = min(num_rw, U.shape[0])
        num_mtq = max(0, U.shape[0] - num_rw)
        mtq_u = U[0:num_mtq, :] if num_mtq > 0 else np.zeros((0, N_u))
        rw_u = U[num_mtq:num_mtq + num_rw, :] if num_rw > 0 else np.zeros((0, N_u))
        
        pe = _compute_pointing_error_deg(q, q_goal)

        # Quaternion
        ax_q.clear()
        ax_q.plot(t_state, q[0, :], label="q0")
        ax_q.plot(t_state, q[1, :], label="q1")
        ax_q.plot(t_state, q[2, :], label="q2")
        ax_q.plot(t_state, q[3, :], label="q3")
        ax_q.plot(t_state, q_goal[0, :], "--", alpha=0.6, label="q0 goal")
        ax_q.plot(t_state, q_goal[1, :], "--", alpha=0.6, label="q1 goal")
        ax_q.plot(t_state, q_goal[2, :], "--", alpha=0.6, label="q2 goal")
        ax_q.plot(t_state, q_goal[3, :], "--", alpha=0.6, label="q3 goal")
        ax_q.set_title("Quaternion")
        ax_q.set_xlabel("Time [s]")
        ax_q.set_ylabel("q")
        ax_q.grid(True, alpha=0.3)
        ax_q.legend(fontsize=7, ncol=2)

        # Angular velocity
        ax_w.clear()
        ax_w.plot(t_state, w[0, :], label="ωx")
        ax_w.plot(t_state, w[1, :], label="ωy")
        ax_w.plot(t_state, w[2, :], label="ωz")
        ax_w.plot(t_state, np.linalg.norm(w, axis=0), "k--", label="‖ω‖")
        ax_w.set_title("Angular Velocity")
        ax_w.set_xlabel("Time [s]")
        ax_w.set_ylabel("rad/s")
        ax_w.grid(True, alpha=0.3)
        ax_w.legend(fontsize=8)

        # Wheel momentum
        ax_h.clear()
        if has_rw_state:
            for idx_h in range(h.shape[0]):
                ax_h.plot(t_state, h[idx_h, :], label=f"h_rw{idx_h}")
            ax_h.legend(fontsize=8)
            ax_h.set_title("Wheel Momentum")
        else:
            ax_h.text(0.5, 0.5, "No RW momentum states for this satellite", 
                      ha="center", va="center", transform=ax_h.transAxes)
            ax_h.set_title("Actuator Internal States")
        ax_h.set_xlabel("Time [s]")
        ax_h.set_ylabel("N·m·s")
        ax_h.grid(True, alpha=0.3)

        # Pointing error
        ax_pe.clear()
        ax_pe.plot(t_state, pe, "o-", color="C3", markersize=3)
        ax_pe.set_title("Pointing Error")
        ax_pe.set_xlabel("Time [s]")
        ax_pe.set_ylabel("deg")
        ax_pe.grid(True, alpha=0.3)

        # MTQ control
        ax_mtq_u.clear()
        if num_mtq > 0:
            for idx_u in range(mtq_u.shape[0]):
                ax_mtq_u.plot(t_control, mtq_u[idx_u, :], label=f"m_mtq{idx_u}")
            ax_mtq_u.legend(fontsize=8)
            ax_mtq_u.set_ylabel("A m^2")
        else:
            ax_mtq_u.text(0.5, 0.5, "No MTQ controls", ha="center", va="center", transform=ax_mtq_u.transAxes)
            ax_mtq_u.set_ylabel("A m^2")
        ax_mtq_u.set_title("MTQ Control Inputs")
        ax_mtq_u.set_xlabel("Time [s]")
        ax_mtq_u.grid(True, alpha=0.3)

        # RW control
        ax_rw_u.clear()
        if num_rw > 0:
            for idx_u in range(rw_u.shape[0]):
                ax_rw_u.plot(t_control, rw_u[idx_u, :], label=f"tau_rw{idx_u}")
            ax_rw_u.legend(fontsize=8)
            ax_rw_u.set_ylabel("N m")
        else:
            ax_rw_u.text(0.5, 0.5, "No RW controls", ha="center", va="center", transform=ax_rw_u.transAxes)
            ax_rw_u.set_ylabel("N m")
        ax_rw_u.set_title("RW Control Inputs")
        ax_rw_u.set_xlabel("Time [s]")
        ax_rw_u.grid(True, alpha=0.3)

        # Cost breakdown - stacked bar chart
        ax_cat.clear()
        if components is not None:
            bottom = np.zeros_like(t_state)
            for name, color in zip(component_order, component_colors):
                vals = components[name]
                ax_cat.bar(t_state, vals, width=dt * 0.8, bottom=bottom, 
                          color=color, alpha=0.85, label=name)
                bottom += vals
            ax_cat.set_title("Cost Time Series (Stacked by Category)")
            ax_cat.set_xlabel("Time [s]")
            ax_cat.set_ylabel("Cost")
            ax_cat.grid(True, alpha=0.3, axis="y")
            _configure_sci_y(ax_cat)
            ax_cat.legend(fontsize=8)
        else:
            ax_cat.text(0.5, 0.5, "Cost breakdown not available", 
                       ha="center", va="center", transform=ax_cat.transAxes)
            ax_cat.set_title("Cost Time Series (Stacked by Category)")

        # Cost breakdown - individual components
        ax_ind.clear()
        if components is not None:
            for name, color in zip(component_order, component_colors):
                ax_ind.plot(t_state, components[name], "o-", linewidth=1.8, 
                           markersize=3, color=color, label=name)
            ax_ind.set_title("Individual Cost Components")
            ax_ind.set_xlabel("Time [s]")
            ax_ind.set_ylabel("Cost")
            ax_ind.grid(True, alpha=0.3)
            _configure_sci_y(ax_ind)
            ax_ind.legend(fontsize=8)
        else:
            ax_ind.text(0.5, 0.5, "Cost breakdown not available", 
                       ha="center", va="center", transform=ax_ind.transAxes)
            ax_ind.set_title("Individual Cost Components")

        # Total cost
        ax_J.clear()
        J_hist = [s["J"] for s in snapshots]
        it = np.arange(len(J_hist))
        ax_J.plot(it, J_hist, "o-", linewidth=2.0, color="C0")
        ax_J.plot(i, J_hist[i], "ro", markersize=8)
        ax_J.set_title("Total Cost vs Iteration")
        ax_J.set_xlabel("Iteration")
        ax_J.set_ylabel("Total Cost")
        ax_J.grid(True, alpha=0.3)
        _configure_sci_y(ax_J)

        # Text info
        ax_txt.clear()
        ax_txt.axis("off")
        lines = [
            f"Iteration: {i} / {len(snapshots)-1}",
            f"Total cost J: {snap['J']:.6e}",
            f"Stop reason: {stop_reason}",
            f"Cost tolerance: {cost_tol:.3e}",
            ""
        ]
        
        n_iters = len(snapshots)
        if n_iters > 1:
            lines.append("Convergence history:")
            lines.append("it | bp fp |ΔJ|≤tol")
            for k in range(1, min(n_iters, 15)):
                tr = transitions[k - 1]
                marker = ">" if k == i else " "
                lines.append(
                    f"{marker}{k:2d} | "
                    f"{int(tr['bp_ok'])}  {int(tr['fp_ok'])}    "
                    f"{int(tr['delta_tol_ok'])}"
                )
            if n_iters > 15:
                lines.append("  ... (showing first 14)")

        if i > 0 and i - 1 < len(transitions):
            tr = transitions[i - 1]
            lines.append("")
            lines.append(f"Transition {i-1}→{i}:")
            lines.append(f"ΔJ: {tr['act_delta']:.3e}")
            lines.append(f"bp_ok={tr['bp_ok']} fp_ok={tr['fp_ok']} tol_ok={tr['delta_tol_ok']}")

        ax_txt.text(0.01, 0.99, "\n".join(lines), va="top", ha="left", 
               fontsize=8.5, family="monospace")

        fig.suptitle("iLQR Iteration Viewer with Cost Breakdown", fontsize=14, fontweight="bold")
        fig.canvas.draw_idle()

    def on_prev(_event):
        set_index(idx["value"] - 1)

    def on_next(_event):
        set_index(idx["value"] + 1)

    btn_prev_ax = fig.add_axes([0.40, 0.01, 0.08, 0.03])
    btn_next_ax = fig.add_axes([0.52, 0.01, 0.08, 0.03])
    btn_prev = Button(btn_prev_ax, "◀ Back")
    btn_next = Button(btn_next_ax, "Next ▶")
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
        set_index(int(np.rint(event.xdata)))

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("button_press_event", on_cost_click)

    set_index(0)
    plt.show()
