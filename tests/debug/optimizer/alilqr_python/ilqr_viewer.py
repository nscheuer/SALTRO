"""Interactive AL-iLQR iteration viewer with matplotlib."""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.ticker import ScalarFormatter


def _configure_sci_y(ax):
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(formatter)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))


def _compute_pointing_error_deg(q, q_goal):
    n = q.shape[1]
    err_deg = np.zeros(n)
    for k in range(n):
        q_err = _quat_multiply(_quat_inverse(q_goal[:, k]), q[:, k])
        err_deg[k] = 2.0 * np.arctan2(np.linalg.norm(q_err[1:]), abs(q_err[0])) * 180.0 / np.pi
    return err_deg


def _boresight_eci(q_b2i: np.ndarray, bore_body_unit: np.ndarray) -> np.ndarray:
    q0, q1, q2, q3 = q_b2i
    r = np.array(
        [
            [q0**2 + q1**2 - q2**2 - q3**2, 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
            [2 * (q1 * q2 + q0 * q3), q0**2 - q1**2 + q2**2 - q3**2, 2 * (q2 * q3 - q0 * q1)],
            [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), q0**2 - q1**2 - q2**2 + q3**2],
        ],
        dtype=float,
    )
    out = r @ bore_body_unit
    nrm = np.linalg.norm(out)
    return out / max(nrm, 1e-15)


def _vec_angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    u = np.asarray(u, dtype=float).reshape(3)
    v = np.asarray(v, dtype=float).reshape(3)
    u = u / max(np.linalg.norm(u), 1e-15)
    v = v / max(np.linalg.norm(v), 1e-15)
    d = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return float(np.degrees(np.arccos(d)))


def _compute_pointing_error_deg_mixed(q, q_goal, boresight=None):
    n = q.shape[1]
    err_deg = np.full(n, np.nan, dtype=float)
    for k in range(n):
        row = q_goal[:, k]
        if not np.isnan(row[0]):
            q_err = _quat_multiply(_quat_inverse(row), q[:, k])
            err_deg[k] = 2.0 * np.arctan2(np.linalg.norm(q_err[1:]), abs(q_err[0])) * 180.0 / np.pi
            continue

        if boresight is None:
            continue

        target_vec = row[1:4]
        if np.linalg.norm(target_vec) <= 0.0:
            continue

        bore = boresight[:, k] if boresight.ndim == 2 else boresight
        if np.linalg.norm(bore) <= 0.0:
            continue

        bore_unit = bore / np.linalg.norm(bore)
        target_unit = target_vec / np.linalg.norm(target_vec)
        bore_i = _boresight_eci(q[:, k], bore_unit)
        err_deg[k] = _vec_angle_deg(bore_i, target_unit)

    return err_deg


def _quat_inverse(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _outer_start_indices(snapshots):
    starts = []
    seen = set()
    for i, s in enumerate(snapshots):
        oi = s.get("outer_iter", None)
        if oi is None:
            continue
        if oi not in seen:
            starts.append(i)
            seen.add(oi)
    return starts


def _save_snapshots(snapshots, transitions, stop_reason, dt, cost_tol, out_dir=None):
    """Save first, last, and ~3 intermediate snapshots as PNGs, plus raw data as npz."""
    import os, pathlib
    if out_dir is None:
        out_dir = pathlib.Path.home() / "ilqr_viewer_out"
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_snaps = len(snapshots)
    # Indices: first, ~25%, ~50%, ~75%, last
    candidates = sorted(set([
        0,
        max(0, n_snaps // 4),
        max(0, n_snaps // 2),
        max(0, 3 * n_snaps // 4),
        n_snaps - 1,
    ]))

    n = snapshots[0]["X"].shape[1]
    t_state = np.arange(n) * dt

    for snap_idx in candidates:
        snap = snapshots[snap_idx]
        x = snap["X"]
        q = x[3:7, :]
        q_goal = snap["q_goal"]

        pe = np.array([
            2.0 * np.arctan2(
                np.linalg.norm(
                    _quat_multiply(_quat_inverse(q_goal[:, k]), q[:, k])[1:]
                ),
                abs(_quat_multiply(_quat_inverse(q_goal[:, k]), q[:, k])[0])
            ) * 180.0 / np.pi
            for k in range(q.shape[1])
        ])

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"Snapshot {snap_idx}/{n_snaps-1}  |  J={snap['J']:.4e}  |  {stop_reason}", fontsize=11)

        axes[0].plot(t_state, pe, "o-", color="C3", markersize=3)
        axes[0].set_title("Pointing Error [deg]")
        axes[0].set_xlabel("Time [s]"); axes[0].grid(True, alpha=0.3)

        axes[1].plot(t_state, q[0, :], label="q0")
        axes[1].plot(t_state, q[1, :], label="q1")
        axes[1].plot(t_state, q[2, :], label="q2")
        axes[1].plot(t_state, q[3, :], label="q3")
        axes[1].set_title("Quaternion"); axes[1].set_xlabel("Time [s]")
        axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.3)

        w = x[0:3, :]
        axes[2].plot(t_state, np.linalg.norm(w, axis=0), "k-", label="||ω||")
        axes[2].plot(t_state, w[0, :], label="wx", alpha=0.7)
        axes[2].plot(t_state, w[1, :], label="wy", alpha=0.7)
        axes[2].plot(t_state, w[2, :], label="wz", alpha=0.7)
        axes[2].set_title("Angular Velocity [rad/s]"); axes[2].set_xlabel("Time [s]")
        axes[2].legend(fontsize=7); axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        fname = out_dir / f"snap_{snap_idx:04d}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[viewer] saved {fname}")

    # Save raw data
    npz_path = out_dir / "snapshots.npz"
    np.savez_compressed(
        npz_path,
        n_snapshots=n_snaps,
        dt=dt,
        stop_reason=stop_reason,
        **{f"X_{i}": snapshots[i]["X"] for i in range(n_snaps)},
        **{f"U_{i}": snapshots[i]["U"] for i in range(n_snaps)},
        **{f"J_{i}": np.array([snapshots[i]["J"]]) for i in range(n_snaps)},
        **{f"q_goal_{i}": snapshots[i]["q_goal"] for i in range(n_snaps)},
    )
    print(f"[viewer] saved raw data → {npz_path}")


def launch_viewer(snapshots, transitions, stop_reason, dt, cost_tol):
    if not snapshots:
        raise ValueError("No snapshots available for viewer")

    _save_snapshots(snapshots, transitions, stop_reason, dt, cost_tol)

    n = snapshots[0]["X"].shape[1]
    t_state = np.arange(n) * dt
    component_order = ["attitude", "angular_velocity", "control", "rw_momentum"]
    component_colors = ["#FF6B6B", "#2A9D8F", "#6D597A", "#FFA07A"]
    cviol_component_order = [
        "angular_velocity",
        "sun_avoidance",
        "mtq_limits",
        "rw_torque_limits",
        "rw_momentum_limits",
        "rw_stiction",
        "other",
    ]
    cviol_component_labels = {
        "angular_velocity": "ang vel",
        "sun_avoidance": "sun avoid",
        "mtq_limits": "mtq lim",
        "rw_torque_limits": "rw torque lim",
        "rw_momentum_limits": "rw momentum lim",
        "rw_stiction": "rw stiction",
        "other": "other",
    }

    fig = plt.figure(figsize=(20, 11), constrained_layout=True)
    gs = fig.add_gridspec(3, 4, hspace=0.40, wspace=0.30)

    ax_q     = fig.add_subplot(gs[0, 0])
    ax_w     = fig.add_subplot(gs[0, 1])
    ax_pe    = fig.add_subplot(gs[0, 2])
    ax_h     = fig.add_subplot(gs[0, 3])
    ax_mtq_u = fig.add_subplot(gs[1, 0])
    ax_rw_u  = fig.add_subplot(gs[1, 1])
    ax_cat   = fig.add_subplot(gs[1, 2])
    ax_ind   = fig.add_subplot(gs[1, 3])
    ax_cviol = fig.add_subplot(gs[2, 0])
    ax_J     = fig.add_subplot(gs[2, 1])
    ax_txt   = fig.add_subplot(gs[2, 2:])

    idx = {"value": 0}
    outer_starts = _outer_start_indices(snapshots)

    def set_index(new_idx):
        idx["value"] = int(np.clip(new_idx, 0, len(snapshots) - 1))
        update_view(idx["value"])

    def update_view(i):
        snap = snapshots[i]
        x = snap["X"]
        u = snap["U"]
        q_goal = snap["q_goal"]
        boresight = snap.get("boresight", None)
        components = snap.get("components", None)

        q = x[3:7, :]
        w = x[0:3, :]
        h = x[7:, :]
        has_rw_state = h.shape[0] > 0

        n_u = u.shape[1]
        t_control = np.arange(n_u) * dt

        num_rw = h.shape[0] if has_rw_state else 0
        num_rw = min(num_rw, u.shape[0])
        num_mtq = max(0, u.shape[0] - num_rw)
        mtq_u = u[0:num_mtq, :] if num_mtq > 0 else np.zeros((0, n_u))
        rw_u = u[num_mtq:num_mtq + num_rw, :] if num_rw > 0 else np.zeros((0, n_u))

        pe = _compute_pointing_error_deg_mixed(q, q_goal, boresight=boresight)

        ax_q.clear()
        ax_q.plot(t_state, q[0, :], label="q0")
        ax_q.plot(t_state, q[1, :], label="q1")
        ax_q.plot(t_state, q[2, :], label="q2")
        ax_q.plot(t_state, q[3, :], label="q3")
        if np.any(np.isnan(q_goal[0, :])):
            ax_q.plot(t_state, q_goal[1, :], "--", alpha=0.6, label="target x")
            ax_q.plot(t_state, q_goal[2, :], "--", alpha=0.6, label="target y")
            ax_q.plot(t_state, q_goal[3, :], "--", alpha=0.6, label="target z")
        else:
            ax_q.plot(t_state, q_goal[0, :], "--", alpha=0.6, label="q0 goal")
            ax_q.plot(t_state, q_goal[1, :], "--", alpha=0.6, label="q1 goal")
            ax_q.plot(t_state, q_goal[2, :], "--", alpha=0.6, label="q2 goal")
            ax_q.plot(t_state, q_goal[3, :], "--", alpha=0.6, label="q3 goal")
        ax_q.set_title("Quaternion")
        ax_q.set_xlabel("Time [s]")
        ax_q.set_ylabel("q")
        ax_q.grid(True, alpha=0.3)
        ax_q.legend(fontsize=7, ncol=2)

        ax_w.clear()
        ax_w.plot(t_state, w[0, :], label="wx")
        ax_w.plot(t_state, w[1, :], label="wy")
        ax_w.plot(t_state, w[2, :], label="wz")
        ax_w.plot(t_state, np.linalg.norm(w, axis=0), "k--", label="||w||")
        ax_w.set_title("Angular Velocity")
        ax_w.set_xlabel("Time [s]")
        ax_w.set_ylabel("rad/s")
        ax_w.grid(True, alpha=0.3)
        ax_w.legend(fontsize=8)

        ax_h.clear()
        if has_rw_state:
            for j in range(h.shape[0]):
                ax_h.plot(t_state, h[j, :], label=f"h_rw{j}")
            ax_h.legend(fontsize=8)
            ax_h.set_title("Wheel Momentum")
        else:
            ax_h.text(0.5, 0.5, "No RW momentum states", ha="center", va="center", transform=ax_h.transAxes)
            ax_h.set_title("Actuator Internal States")
        ax_h.set_xlabel("Time [s]")
        ax_h.set_ylabel("N m s")
        ax_h.grid(True, alpha=0.3)

        ax_pe.clear()
        if np.all(np.isnan(pe)):
            ax_pe.text(0.5, 0.5, "Pointing error unavailable", ha="center", va="center", transform=ax_pe.transAxes)
        else:
            ax_pe.plot(t_state, pe, "o-", color="C3", markersize=3)
        ax_pe.set_title("Pointing Error")
        ax_pe.set_xlabel("Time [s]")
        ax_pe.set_ylabel("deg")
        ax_pe.grid(True, alpha=0.3)

        ax_mtq_u.clear()
        if num_mtq > 0:
            for j in range(mtq_u.shape[0]):
                ax_mtq_u.plot(t_control, mtq_u[j, :], label=f"m_mtq{j}")
            ax_mtq_u.legend(fontsize=8)
        else:
            ax_mtq_u.text(0.5, 0.5, "No MTQ controls", ha="center", va="center", transform=ax_mtq_u.transAxes)
        ax_mtq_u.set_title("MTQ Control Inputs")
        ax_mtq_u.set_xlabel("Time [s]")
        ax_mtq_u.set_ylabel("A m^2")
        ax_mtq_u.grid(True, alpha=0.3)

        ax_rw_u.clear()
        if num_rw > 0:
            for j in range(rw_u.shape[0]):
                ax_rw_u.plot(t_control, rw_u[j, :], label=f"tau_rw{j}")
            ax_rw_u.legend(fontsize=8)
        else:
            ax_rw_u.text(0.5, 0.5, "No RW controls", ha="center", va="center", transform=ax_rw_u.transAxes)
        ax_rw_u.set_title("RW Control Inputs")
        ax_rw_u.set_xlabel("Time [s]")
        ax_rw_u.set_ylabel("N m")
        ax_rw_u.grid(True, alpha=0.3)

        ax_cat.clear()
        if components is not None:
            bottom = np.zeros_like(t_state)
            for name, color in zip(component_order, component_colors):
                vals = components[name]
                ax_cat.bar(t_state, vals, width=dt * 0.8, bottom=bottom, color=color, alpha=0.85, label=name)
                bottom += vals
            ax_cat.legend(fontsize=8)
            _configure_sci_y(ax_cat)
        else:
            ax_cat.text(0.5, 0.5, "Cost breakdown not available", ha="center", va="center", transform=ax_cat.transAxes)
        ax_cat.set_title("Cost Time Series (Stacked)")
        ax_cat.set_xlabel("Time [s]")
        ax_cat.set_ylabel("Cost")
        ax_cat.grid(True, alpha=0.3, axis="y")

        ax_ind.clear()
        if components is not None:
            for name, color in zip(component_order, component_colors):
                ax_ind.plot(t_state, components[name], "o-", linewidth=1.8, markersize=3, color=color, label=name)
            ax_ind.legend(fontsize=8)
            _configure_sci_y(ax_ind)
        else:
            ax_ind.text(0.5, 0.5, "Cost breakdown not available", ha="center", va="center", transform=ax_ind.transAxes)
        ax_ind.set_title("Individual Cost Components")
        ax_ind.set_xlabel("Time [s]")
        ax_ind.set_ylabel("Cost")
        ax_ind.grid(True, alpha=0.3)

        ax_cviol.clear()
        cviol_components = snap.get("constraint_components_t", None)
        cviol_t = snap.get("constraint_violation_t", None)
        if cviol_components is not None:
            plotted_any = False
            for i_comp, name in enumerate(cviol_component_order):
                vals = cviol_components.get(name, None)
                if vals is None:
                    continue
                if np.max(vals) <= 0.0:
                    continue
                ax_cviol.plot(
                    t_state,
                    vals,
                    "o-",
                    linewidth=1.6,
                    markersize=3,
                    color=f"C{i_comp}",
                    label=cviol_component_labels.get(name, name),
                )
                plotted_any = True

            if cviol_t is not None:
                ax_cviol.plot(
                    t_state,
                    cviol_t,
                    "k--",
                    linewidth=1.4,
                    alpha=0.9,
                    label="max positive",
                )
                plotted_any = True

            if plotted_any:
                ax_cviol.legend(fontsize=7, ncol=2)
            else:
                ax_cviol.text(
                    0.5,
                    0.5,
                    "No active constraint violations",
                    ha="center",
                    va="center",
                    transform=ax_cviol.transAxes,
                )
            _configure_sci_y(ax_cviol)
        elif cviol_t is not None:
            ax_cviol.plot(t_state, cviol_t, "o-", color="C1", markersize=3, label="max positive violation")
            ax_cviol.legend(fontsize=8)
            _configure_sci_y(ax_cviol)
        else:
            ax_cviol.text(0.5, 0.5, "Constraint violation trace not available", ha="center", va="center", transform=ax_cviol.transAxes)
        ax_cviol.set_title("Constraint Violations by Type")
        ax_cviol.set_xlabel("Time [s]")
        ax_cviol.set_ylabel("Violation")
        ax_cviol.grid(True, alpha=0.3)

        ax_J.clear()
        j_hist = [s["J"] for s in snapshots]
        it = np.arange(len(j_hist))
        ax_J.plot(it, j_hist, "o-", linewidth=2.0, color="C0")
        ax_J.plot(i, j_hist[i], "ro", markersize=8)
        for xline in outer_starts:
            ax_J.axvline(xline, color="k", linestyle="--", alpha=0.35, linewidth=1.2)
        ax_J.set_title("Total Cost vs Iteration (dashed=outer-loop start)")
        ax_J.set_xlabel("Snapshot index")
        ax_J.set_ylabel("Total Cost")
        ax_J.grid(True, alpha=0.3)
        _configure_sci_y(ax_J)

        ax_txt.clear()
        ax_txt.axis("off")
        outer_iter = snap.get("outer_iter", "?")
        lines = [
            f"Snapshot: {i} / {len(snapshots) - 1}",
            f"Outer iteration: {outer_iter}",
            f"Total cost J: {snap['J']:.6e}",
            f"Stop reason: {stop_reason}",
            f"Cost tolerance: {cost_tol:.3e}",
            "",
        ]

        outer_rows = [t for t in transitions if "outer_iter" in t]
        if outer_rows:
            lines.append("Outer-loop summary:")
            lines.append("it | max_c      lambda_max   mu_max")
            for row in outer_rows[:20]:
                lines.append(
                    f"{row['outer_iter']:2d} | "
                    f"{row['max_constraint_violation']:.3e}  "
                    f"{row['lambda_max']:.3e}  "
                    f"{row['mu_max']:.3e}"
                )

        ax_txt.text(0.01, 0.98, "\n".join(lines), va="top", ha="left", fontsize=9, family="monospace")

        fig.suptitle("AL-iLQR Iteration Viewer", fontsize=14, fontweight="bold")
        fig.canvas.draw_idle()

    def on_prev(_event):
        set_index(idx["value"] - 1)

    def on_next(_event):
        set_index(idx["value"] + 1)

    btn_prev_ax = fig.add_axes([0.40, 0.01, 0.08, 0.03])
    btn_next_ax = fig.add_axes([0.52, 0.01, 0.08, 0.03])
    btn_prev = Button(btn_prev_ax, "< Back")
    btn_next = Button(btn_next_ax, "Next >")
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
