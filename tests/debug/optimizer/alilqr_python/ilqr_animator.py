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
from matplotlib.lines import Line2D
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


def _safe_normalize(v):
    arr = np.asarray(v, dtype=float).reshape(-1)
    if arr.size == 0:
        return np.zeros(3, dtype=float)
    if arr.size < 3:
        out = np.zeros(3, dtype=float)
        out[: arr.size] = arr
        arr = out
    elif arr.size > 3:
        arr = arr[:3]
    nrm = np.linalg.norm(arr)
    if nrm <= 1e-14:
        return np.zeros(3, dtype=float)
    return arr / nrm


def _quat_to_rotmat(q):
    q = np.asarray(q, dtype=float).reshape(4)
    nrm = np.linalg.norm(q)
    if nrm <= 1e-14:
        return np.eye(3)
    q = q / nrm
    w, x, y, z = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _state_index_for_frame(frame_idx, n_frames, n_state):
    if n_state <= 1:
        return 0
    if n_frames <= 1:
        return n_state - 1
    frac = frame_idx / float(max(1, n_frames - 1))
    return int(round(frac * (n_state - 1)))


def _control_index_for_state(state_idx, n_u):
    if n_u <= 0:
        return 0
    return min(state_idx, n_u - 1)


def _to_rgb_image(img):
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    arr = arr.astype(float)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _build_earth_surface(re_km, texture_path=None, nu=96, nv=48):
    u = np.linspace(0.0, 2.0 * np.pi, nu)
    v = np.linspace(0.0, np.pi, nv)
    uu, vv = np.meshgrid(u, v, indexing="ij")

    xs = re_km * np.cos(uu) * np.sin(vv)
    ys = re_km * np.sin(uu) * np.sin(vv)
    zs = re_km * np.cos(vv)

    facecolors = None
    texture_loaded = False
    if texture_path:
        texture_file = Path(texture_path)
        if texture_file.exists():
            try:
                tex = _to_rgb_image(plt.imread(str(texture_file)))
                h, w, _ = tex.shape
                tx = np.mod(uu / (2.0 * np.pi), 1.0) * (w - 1)
                ty = np.clip(vv / np.pi, 0.0, 1.0) * (h - 1)
                txi = tx.astype(int)
                tyi = ty.astype(int)
                facecolors = tex[tyi, txi, :]
                texture_loaded = True
            except Exception:
                facecolors = None

    return {
        "x": xs,
        "y": ys,
        "z": zs,
        "facecolors": facecolors,
        "texture_loaded": texture_loaded,
    }


def _build_dipole_lines(re_km):
    lines = []
    thetas = np.linspace(0.20, np.pi - 0.20, 160)
    longitudes = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    l_shells = [1.25, 1.65, 2.10, 2.80]

    for l_shell in l_shells:
        r = l_shell * re_km * (np.sin(thetas) ** 2)
        for lon in longitudes:
            x0 = r * np.sin(thetas)
            y0 = np.zeros_like(x0)
            z0 = r * np.cos(thetas)

            x = x0 * np.cos(lon) - y0 * np.sin(lon)
            y = x0 * np.sin(lon) + y0 * np.cos(lon)
            z = z0
            lines.append((x, y, z))

    return lines


def _set_equal_3d_axes(ax, lim):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def _rw_to_body_vector(rw_cmd):
    rw_cmd = np.asarray(rw_cmd, dtype=float).reshape(-1)
    out = np.zeros(3, dtype=float)
    if rw_cmd.size >= 3:
        out[:] = rw_cmd[:3]
    elif rw_cmd.size == 2:
        out[0:2] = rw_cmd
    elif rw_cmd.size == 1:
        out[2] = rw_cmd[0]
    return out


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
    earth_texture_path=None,
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

    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, hspace=0.18, wspace=0.12)

    ax_mtq = fig.add_subplot(gs[0, 0])
    ax_rw = ax_mtq.twinx()
    ax_pe = fig.add_subplot(gs[0, 1])
    ax_cost_t = fig.add_subplot(gs[1, 0])
    ax_cnst_t = fig.add_subplot(gs[1, 1])
    ax_cost_iter = fig.add_subplot(gs[2, 0:2])
    ax_earth = fig.add_subplot(gs[0:2, 2], projection="3d")
    ax_sat = fig.add_subplot(gs[2, 2], projection="3d")

    component_order = ["attitude", "angular_velocity", "control", "rw_momentum"]
    component_colors = ["#FF6B6B", "#2A9D8F", "#6D597A", "#FFA07A"]

    earth_radius_km = 6371.0
    earth_surface = _build_earth_surface(earth_radius_km, texture_path=earth_texture_path)
    dipole_lines = _build_dipole_lines(earth_radius_km)

    def _draw_earth_animation(snap, state_idx):
        ax_earth.clear()

        R = np.asarray(snap.get("R", np.zeros((3, n))), dtype=float)
        if R.ndim != 2 or R.shape[0] < 3:
            R = np.zeros((3, n), dtype=float)
        if R.shape[1] == 0:
            R = np.zeros((3, 1), dtype=float)

        B = np.asarray(snap.get("B", np.zeros_like(R)), dtype=float)
        if B.ndim != 2 or B.shape[0] < 3:
            B = np.zeros_like(R)
        if B.shape[1] < R.shape[1]:
            B_pad = np.zeros((3, R.shape[1]), dtype=float)
            B_pad[:, : B.shape[1]] = B[:, : B.shape[1]]
            B = B_pad

        r_idx = min(state_idx, R.shape[1] - 1)
        R_km = R[0:3, :] / 1e3
        sat_pos = R_km[:, r_idx]

        if earth_surface["facecolors"] is not None:
            ax_earth.plot_surface(
                earth_surface["x"],
                earth_surface["y"],
                earth_surface["z"],
                facecolors=earth_surface["facecolors"],
                rstride=1,
                cstride=1,
                linewidth=0,
                antialiased=False,
                shade=False,
                alpha=1.0,
            )
        else:
            ax_earth.plot_surface(
                earth_surface["x"],
                earth_surface["y"],
                earth_surface["z"],
                color="#6baed6",
                linewidth=0,
                antialiased=True,
                alpha=0.75,
            )

        for x_line, y_line, z_line in dipole_lines:
            ax_earth.plot(x_line, y_line, z_line, color="#1f77b4", linewidth=0.7, alpha=0.35)

        ax_earth.plot(R_km[0, :], R_km[1, :], R_km[2, :], color="#f28e2b", linewidth=1.8, alpha=0.95)
        ax_earth.scatter(sat_pos[0], sat_pos[1], sat_pos[2], color="red", s=30)

        b_vec = B[0:3, r_idx]
        b_hat = _safe_normalize(b_vec)
        if np.linalg.norm(b_hat) > 0.0:
            b_scale = 0.22 * earth_radius_km
            ax_earth.quiver(
                sat_pos[0],
                sat_pos[1],
                sat_pos[2],
                b_hat[0],
                b_hat[1],
                b_hat[2],
                length=b_scale,
                normalize=True,
                color="#e15759",
                linewidth=2.0,
            )

        orb_norm = np.linalg.norm(R_km, axis=0)
        lim = max(1.25 * earth_radius_km, 1.05 * float(np.max(orb_norm)))
        _set_equal_3d_axes(ax_earth, lim)

        ax_earth.view_init(elev=24, azim=36)
        ax_earth.set_xlabel("x [km]")
        ax_earth.set_ylabel("y [km]")
        ax_earth.set_zlabel("z [km]")
        title = "Earth Orbit + Field"
        if earth_texture_path and not earth_surface["texture_loaded"]:
            title += " (texture fallback)"
        ax_earth.set_title(title)

    def _draw_sat_animation(q_curr, q_goal_curr, b_inertial, tau_body):
        ax_sat.clear()

        C_ib_curr = _quat_to_rotmat(q_curr)
        C_bi_curr = C_ib_curr.T
        axis_len = 1.0
        body_axes = np.eye(3)

        colors = ["#e41a1c", "#4daf4a", "#377eb8"]
        for i in range(3):
            b_axis = body_axes[:, i]
            ax_sat.quiver(0.0, 0.0, 0.0, b_axis[0], b_axis[1], b_axis[2], color=colors[i], length=axis_len, linewidth=2.0)

        has_quat_goal = not np.isnan(q_goal_curr[0])
        if has_quat_goal:
            C_ib_goal = _quat_to_rotmat(q_goal_curr)
            C_bg = C_bi_curr @ C_ib_goal
            for i in range(3):
                g_axis = C_bg[:, i]
                ax_sat.quiver(
                    0.0,
                    0.0,
                    0.0,
                    g_axis[0],
                    g_axis[1],
                    g_axis[2],
                    color=colors[i],
                    length=axis_len,
                    linewidth=1.6,
                    linestyle="--",
                    alpha=0.85,
                )
        else:
            target_i = _safe_normalize(q_goal_curr[1:4])
            target_b = C_bi_curr @ target_i
            target_b = _safe_normalize(target_b)
            if np.linalg.norm(target_b) > 0.0:
                ax_sat.quiver(0.0, 0.0, 0.0, target_b[0], target_b[1], target_b[2], color="#ffd166", length=1.2, linewidth=2.2)

        b_body = C_bi_curr @ np.asarray(b_inertial, dtype=float).reshape(3)
        b_hat = _safe_normalize(b_body)
        tau_hat = _safe_normalize(tau_body)

        if np.linalg.norm(b_hat) > 0.0:
            ax_sat.quiver(0.0, 0.0, 0.0, b_hat[0], b_hat[1], b_hat[2], color="#17becf", length=1.2, linewidth=2.4)
        if np.linalg.norm(tau_hat) > 0.0:
            ax_sat.quiver(0.0, 0.0, 0.0, tau_hat[0], tau_hat[1], tau_hat[2], color="#ff7f0e", length=1.2, linewidth=2.4)

        legend_handles = [
            Line2D([0], [0], color="k", linewidth=2.0, label="sat frame (solid xyz)"),
            Line2D([0], [0], color="k", linewidth=1.6, linestyle="--", label="goal frame (dashed xyz)"),
            Line2D([0], [0], color="#ffd166", linewidth=2.2, label="target dir (vector goal)"),
            Line2D([0], [0], color="#17becf", linewidth=2.4, label="B direction (normalized)"),
            Line2D([0], [0], color="#ff7f0e", linewidth=2.4, label="torque direction (normalized)"),
        ]
        ax_sat.legend(handles=legend_handles, fontsize=7, loc="upper left")

        _set_equal_3d_axes(ax_sat, 1.35)
        ax_sat.view_init(elev=20, azim=40)
        ax_sat.set_xlabel("x_b")
        ax_sat.set_ylabel("y_b")
        ax_sat.set_zlabel("z_b")
        ax_sat.set_title("Satellite Frame vs Goal")

    def _draw_frame(frame_idx):
        snap = snapshots[frame_idx]
        X = snap["X"]
        U = snap["U"]
        q_goal = snap["q_goal"]
        boresight = snap.get("boresight", None)
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
        pe = _compute_pointing_error_deg_mixed(q, q_goal, boresight=boresight)

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
        if np.all(np.isnan(pe)):
            ax_pe.text(0.5, 0.5, "Pointing error unavailable", transform=ax_pe.transAxes, ha="center", va="center")
        else:
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

        state_idx = _state_index_for_frame(frame_idx, n_frames, X.shape[1])
        u_idx = _control_index_for_state(state_idx, U.shape[1])

        if num_mtq > 0 and U.shape[1] > 0:
            mtq_cmd = np.zeros(3, dtype=float)
            mtq_take = min(3, num_mtq)
            mtq_cmd[:mtq_take] = mtq_u[:mtq_take, u_idx]
        else:
            mtq_cmd = np.zeros(3, dtype=float)

        if num_rw > 0 and U.shape[1] > 0:
            rw_cmd = rw_u[:, u_idx]
        else:
            rw_cmd = np.zeros(0, dtype=float)

        B_snap = np.asarray(snap.get("B", np.zeros((3, X.shape[1]))), dtype=float)
        if B_snap.ndim != 2 or B_snap.shape[0] < 3 or B_snap.shape[1] == 0:
            b_inertial = np.zeros(3, dtype=float)
        else:
            b_inertial = B_snap[0:3, min(state_idx, B_snap.shape[1] - 1)]

        q_curr = q[:, state_idx]
        q_goal_curr = q_goal[:, state_idx]
        C_bi_curr = _quat_to_rotmat(q_curr).T
        b_body = C_bi_curr @ b_inertial
        tau_mtq = np.cross(mtq_cmd, b_body)
        tau_rw = _rw_to_body_vector(rw_cmd)
        tau_net = tau_mtq + tau_rw

        _draw_earth_animation(snap, state_idx)
        _draw_sat_animation(q_curr, q_goal_curr, b_inertial, tau_net)

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
