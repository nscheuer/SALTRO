"""Animated AL-iLQR progress viewer and GIF exporter."""

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import matplotlib


def _has_display() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


if _has_display():
    matplotlib.use("TkAgg")
else:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.widgets import Button
from matplotlib.ticker import ScalarFormatter


def _configure_sci_y(ax):
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(formatter)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))


def _quat_inverse(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def _quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def _compute_pointing_error_deg(q, q_goal):
    n = q.shape[1]
    err_deg = np.zeros(n, dtype=float)
    for k in range(n):
        q_err = _quat_multiply(_quat_inverse(q_goal[:, k]), q[:, k])
        err_deg[k] = 2.0 * np.arctan2(np.linalg.norm(q_err[1:]), abs(q_err[0])) * 180.0 / np.pi
    return err_deg


def _derive_mtq_limits(mtq_limits, snapshots):
    if mtq_limits is not None:
        vals = np.asarray(mtq_limits, dtype=float).reshape(-1)
        return np.abs(vals)

    if not snapshots:
        return np.array([], dtype=float)

    n_u = snapshots[0]["U"].shape[0]
    inferred = np.zeros(n_u, dtype=float)
    for s in snapshots:
        U = s["U"]
        if U.size == 0:
            continue
        this_max = np.max(np.abs(U), axis=1)
        inferred[: min(n_u, this_max.size)] = np.maximum(
            inferred[: min(n_u, this_max.size)], this_max[: min(n_u, this_max.size)]
        )
    return inferred


def launch_animator(
    snapshots,
    transitions,
    stop_reason,
    dt,
    cost_tol,
    mtq_limits=None,
    rw_limits=5.7e-6,
    gif_path=None,
    fps=6,
):
    if not snapshots:
        raise ValueError("No snapshots available for animator")

    n_frames = len(snapshots)
    n = snapshots[0]["X"].shape[1]
    t_state = np.arange(n, dtype=float) * dt

    first_u = snapshots[0]["U"]
    n_u = first_u.shape[0]

    first_h = snapshots[0]["X"][7:, :]
    first_num_rw = min(first_h.shape[0], n_u)
    n_mtq = max(0, n_u - first_num_rw)

    if mtq_limits is None:
        lims_mtq = np.full(n_mtq, 0.20, dtype=float)
    else:
        lims_all = np.abs(np.asarray(mtq_limits, dtype=float).reshape(-1))
        if lims_all.size == 1:
            lims_mtq = np.full(n_mtq, float(lims_all[0]), dtype=float)
        elif lims_all.size >= n_mtq:
            lims_mtq = lims_all[:n_mtq]
        else:
            lims_mtq = np.zeros(n_mtq, dtype=float)
            if lims_all.size > 0:
                lims_mtq[: lims_all.size] = lims_all

    lims_rw_all = np.abs(np.asarray(rw_limits, dtype=float).reshape(-1))

    max_abs_mtq = 0.0
    max_abs_rw = 0.0
    max_u_len = 0
    for s in snapshots:
        U_s = s["U"]
        X_s = s["X"]
        max_u_len = max(max_u_len, U_s.shape[1])
        h_s = X_s[7:, :]
        num_rw_s = min(h_s.shape[0], U_s.shape[0])
        num_mtq_s = max(0, U_s.shape[0] - num_rw_s)
        if num_mtq_s > 0:
            max_abs_mtq = max(max_abs_mtq, float(np.max(np.abs(U_s[0:num_mtq_s, :]))))
        if num_rw_s > 0:
            max_abs_rw = max(max_abs_rw, float(np.max(np.abs(U_s[num_mtq_s:num_mtq_s + num_rw_s, :]))))

    lim_mtq_max = float(np.max(lims_mtq)) if lims_mtq.size > 0 else 0.20
    lim_rw_max = float(np.max(lims_rw_all)) if lims_rw_all.size > 0 else 5.7e-6
    y_mtq_max = max(max_abs_mtq, lim_mtq_max)
    y_rw_max = max(max_abs_rw, lim_rw_max)
    y_mtq_lim = 1.15 * y_mtq_max if y_mtq_max > 0.0 else 1.0
    y_rw_lim = 1.15 * y_rw_max if y_rw_max > 0.0 else 1.0

    if lims_rw_all.size == 1:
        lims_rw = np.full(max(1, first_num_rw), float(lims_rw_all[0]), dtype=float)
    elif lims_rw_all.size > 1:
        lims_rw = lims_rw_all
    else:
        lims_rw = np.full(max(1, first_num_rw), 5.7e-6, dtype=float)

    t_u_full = np.arange(max_u_len, dtype=float) * dt

    j_hist = np.asarray([float(s["J"]) for s in snapshots], dtype=float)

    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, hspace=0.18, wspace=0.12)

    ax_mtq = fig.add_subplot(gs[0, 0])
    ax_rw = ax_mtq.twinx()
    ax_pe = fig.add_subplot(gs[0, 1])
    ax_cost_t = fig.add_subplot(gs[1, 0])
    ax_cnst_t = fig.add_subplot(gs[1, 1])
    ax_cost_iter = fig.add_subplot(gs[2, :])

    component_order = ["attitude", "angular_velocity", "control", "rw_momentum"]
    component_colors = ["#FF6B6B", "#2A9D8F", "#6D597A", "#FFA07A"]

    def _draw_frame(frame_idx):
        snap = snapshots[frame_idx]
        X = snap["X"]
        U = snap["U"]
        q_goal = snap["q_goal"]
        components = snap.get("components", None)
        cviol_t = snap.get("constraint_violation_t", None)

        n_u_local = U.shape[1]
        t_u = np.arange(n_u_local, dtype=float) * dt

        h = X[7:, :]
        num_rw = min(h.shape[0], U.shape[0])
        num_mtq = max(0, U.shape[0] - num_rw)
        mtq_u = U[0:num_mtq, :] if num_mtq > 0 else np.zeros((0, n_u_local))
        rw_u = U[num_mtq:num_mtq + num_rw, :] if num_rw > 0 else np.zeros((0, n_u_local))

        q = X[3:7, :]
        pe = _compute_pointing_error_deg(q, q_goal)

        ax_mtq.clear()
        ax_rw.clear()

        mtq_handles = []
        mtq_labels = []
        rw_handles = []
        rw_labels = []

        if num_mtq > 0:
            for i in range(mtq_u.shape[0]):
                line, = ax_mtq.plot(t_u, mtq_u[i, :], linewidth=1.8, label=f"mtq{i}")
                mtq_handles.append(line)
                mtq_labels.append(f"mtq{i}")
                lim = lims_mtq[i] if i < lims_mtq.size else 0.0
                if lim > 0.0:
                    ax_mtq.axhline(lim, color="tab:red", linestyle="--", linewidth=1.0, alpha=0.9, zorder=1)
                    ax_mtq.axhline(-lim, color="tab:red", linestyle="--", linewidth=1.0, alpha=0.9, zorder=1)

        if num_rw > 0:
            for i in range(rw_u.shape[0]):
                line, = ax_rw.plot(t_u, rw_u[i, :], "-", color="purple", linewidth=1.8, label=f"rw{i}")
                rw_handles.append(line)
                rw_labels.append(f"rw{i}")
                rw_lim = lims_rw[i] if i < lims_rw.size else lim_rw_max
                if rw_lim > 0.0:
                    ax_rw.axhline(rw_lim, color="tab:orange", linestyle="--", linewidth=1.0, alpha=0.9, zorder=1)
                    ax_rw.axhline(-rw_lim, color="tab:orange", linestyle="--", linewidth=1.0, alpha=0.9, zorder=1)

        if not mtq_handles and not rw_handles:
            ax_mtq.text(0.5, 0.5, "No controls", transform=ax_mtq.transAxes, ha="center", va="center")

        if mtq_handles or rw_handles:
            ax_mtq.legend(mtq_handles + rw_handles, mtq_labels + rw_labels, fontsize=8, ncol=2)

        ax_mtq.set_title("Control Inputs (MTQ + RW)")
        ax_mtq.set_xlabel("Time [s]")
        ax_mtq.set_ylabel("MTQ [A m^2]", color="C0")
        ax_rw.set_ylabel("RW [N m]", color="C1")
        ax_mtq.tick_params(axis="y", colors="C0")
        ax_rw.tick_params(axis="y", colors="C1")
        ax_mtq.grid(True, alpha=0.3)

        if t_u_full.size > 0:
            ax_mtq.set_xlim(t_u_full[0], t_u_full[-1])
            ax_rw.set_xlim(t_u_full[0], t_u_full[-1])
        mtq_zoom_lim = max(y_mtq_lim / 4.0, 1e-12)
        ax_mtq.set_ylim(-mtq_zoom_lim, mtq_zoom_lim)
        ax_rw.set_ylim(-y_rw_lim, y_rw_lim)

        ax_pe.clear()
        ax_pe.plot(t_state, pe, "o-", markersize=2.5, linewidth=1.8, color="C3")
        ax_pe.set_title("Pointing Error")
        ax_pe.set_xlabel("Time [s]")
        ax_pe.set_ylabel("deg")
        ax_pe.grid(True, alpha=0.3)

        ax_cost_t.clear()
        if components is not None:
            bottom = np.zeros_like(t_state)
            for name, color in zip(component_order, component_colors):
                vals = np.asarray(components.get(name, np.zeros_like(t_state)), dtype=float)
                ax_cost_t.bar(t_state, vals, width=max(dt * 0.8, 1e-12), bottom=bottom, color=color, alpha=0.85, label=name)
                bottom += vals
            ax_cost_t.legend(fontsize=8, ncol=2)
            _configure_sci_y(ax_cost_t)
        else:
            ax_cost_t.text(0.5, 0.5, "Cost components unavailable", transform=ax_cost_t.transAxes, ha="center", va="center")
        ax_cost_t.set_title("Cost Time Series")
        ax_cost_t.set_xlabel("Time [s]")
        ax_cost_t.set_ylabel("Cost")
        ax_cost_t.grid(True, alpha=0.3, axis="y")

        ax_cnst_t.clear()
        if cviol_t is not None:
            cviol_vals = np.asarray(cviol_t, dtype=float)
            ax_cnst_t.plot(t_state, cviol_vals, "o-", markersize=2.5, linewidth=1.8, color="C1", label="max positive")
            ax_cnst_t.legend(fontsize=8)
            _configure_sci_y(ax_cnst_t)
        else:
            ax_cnst_t.text(0.5, 0.5, "Constraint trace unavailable", transform=ax_cnst_t.transAxes, ha="center", va="center")
        ax_cnst_t.set_title("Constraint Time Series")
        ax_cnst_t.set_xlabel("Time [s]")
        ax_cnst_t.set_ylabel("Violation")
        ax_cnst_t.grid(True, alpha=0.3)

        ax_cost_iter.clear()
        ax_cost_iter.plot(np.arange(n_frames), j_hist, "o-", linewidth=2.0, markersize=4, color="C0")
        ax_cost_iter.plot(frame_idx, j_hist[frame_idx], "ro", markersize=8)
        ax_cost_iter.axhline(cost_tol, color="k", linestyle=":", linewidth=1.0, alpha=0.6)
        ax_cost_iter.set_title("Cost Iteration Series")
        ax_cost_iter.set_xlabel("Snapshot index")
        ax_cost_iter.set_ylabel("Total cost")
        ax_cost_iter.grid(True, alpha=0.3)
        _configure_sci_y(ax_cost_iter)

        outer_iter = snap.get("outer_iter", "?")
        fig.suptitle(
            f"AL-iLQR Animation | frame {frame_idx + 1}/{n_frames} | outer {outer_iter} | stop: {stop_reason}",
            fontsize=12,
            fontweight="bold",
        )

    final_hold_seconds = 5.0
    final_hold_frames = max(1, int(round(final_hold_seconds * max(1, fps))))
    frame_sequence = list(range(n_frames)) + [n_frames - 1] * final_hold_frames

    anim = FuncAnimation(
        fig,
        _draw_frame,
        frames=frame_sequence,
        interval=max(20, int(1000 / max(1, fps))),
        repeat=True,
    )

    if gif_path is None:
        gif_path = Path(__file__).resolve().parent / "alilqr_progress.gif"
    else:
        gif_path = Path(gif_path)

    writer = PillowWriter(fps=fps)
    anim.save(str(gif_path), writer=writer)
    print(f"GIF written to: {gif_path}")

    def _open_gif(path: Path):
        candidates = ["wslview", "xdg-open", "explorer.exe"]
        for cmd in candidates:
            exe = shutil.which(cmd)
            if exe is None:
                continue
            try:
                subprocess.Popen([exe, str(path)])
                return True
            except Exception:
                continue
        return False

    def _save_as(_event):
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            out = filedialog.asksaveasfilename(
                title="Save AL-iLQR GIF",
                defaultextension=".gif",
                filetypes=[("GIF", "*.gif")],
                initialfile=gif_path.name,
            )
            root.destroy()
            if out:
                anim.save(out, writer=PillowWriter(fps=fps))
                print(f"Saved GIF to: {out}")
        except Exception as exc:
            print(f"Save dialog failed: {exc}")

    if _has_display():
        btn_ax = fig.add_axes([0.84, 0.01, 0.13, 0.045])
        btn = Button(btn_ax, "Save GIF As...")
        btn.on_clicked(_save_as)
        plt.show()
    else:
        opened = _open_gif(gif_path)
        if not opened:
            print("No GUI display detected and no opener found. Open the GIF manually from the path above.")

    return str(gif_path)
