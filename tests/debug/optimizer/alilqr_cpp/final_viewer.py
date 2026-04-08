import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


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


def _compute_pointing_error_deg_mixed(q, q_goal, boresight_body=None):
    """Compute pointing error for mixed target encoding used in Generalized_ADCS.

    - Quaternion goal row: [q0, q1, q2, q3]
    - Vector goal row: [nan, tx, ty, tz]
    """
    n = q.shape[1]
    err_deg = np.full(n, np.nan, dtype=float)
    if q_goal is None:
        return err_deg

    for k in range(n):
        row = q_goal[:, k]
        if not np.isnan(row[0]):
            q_err = _quat_multiply(_quat_inverse(row), q[:, k])
            err_deg[k] = 2.0 * np.arctan2(np.linalg.norm(q_err[1:]), abs(q_err[0])) * 180.0 / np.pi
            continue

        if boresight_body is None:
            continue

        target_vec = np.asarray(row[1:4], dtype=float)
        if np.linalg.norm(target_vec) <= 0.0:
            continue

        if boresight_body.ndim == 1:
            bore_body = boresight_body
        else:
            bore_body = boresight_body[:, k]

        if np.linalg.norm(bore_body) <= 0.0:
            continue

        bore_unit = bore_body / np.linalg.norm(bore_body)
        target_unit = target_vec / np.linalg.norm(target_vec)
        bore_inertial = _boresight_eci(q[:, k], bore_unit)
        err_deg[k] = _vec_angle_deg(bore_inertial, target_unit)

    return err_deg


def _normalize_quat(q):
    nrm = np.linalg.norm(q)
    if nrm <= 0.0:
        return q
    return q / nrm


def _resample_zero_order_hold(jtime_coarse, q_goal_coarse, dt_seconds):
    """Resample goal quaternion using zero-order hold, matching optimizer behavior."""
    dt_cent = dt_seconds / (36525.0 * 86400.0)
    t0, tN = jtime_coarse[0], jtime_coarse[-1]
    jtime_fine = np.arange(t0, tN + dt_cent/2, dt_cent)
    if abs(jtime_fine[-1] - tN) > 1e-12:
        jtime_fine = np.append(jtime_fine, tN)
    
    # Match numpy.searchsorted(..., side='right') semantics
    idx = np.searchsorted(jtime_coarse[1:], jtime_fine, side='right')
    q_goal_fine = q_goal_coarse[:, idx]
    
    return q_goal_fine


def _expand_q_goal(q_goal, n, jtime=None, dt=None):
    """Expand compact goal quaternion inputs to a full N-step trajectory using zero-order hold.
    
    If jtime and dt are provided, uses proper resampling matching optimizer behavior.
    Otherwise falls back to index-based expansion.
    """
    if q_goal is None:
        return None
    if q_goal.ndim != 2 or q_goal.shape[0] != 4:
        return None
    # ADCS vector-goal encoding uses [nan, x, y, z]. Those rows are not
    # quaternions and must never be normalized.
    is_vector_goal = np.any(np.isnan(q_goal[0, :]))
    m = q_goal.shape[1]
    if n <= 0:
        return None
    if m == n:
        if is_vector_goal:
            return q_goal.copy()
        out = np.zeros_like(q_goal)
        for k in range(n):
            out[:, k] = _normalize_quat(q_goal[:, k])
        return out
    if n == 1 and m >= 1:
        if is_vector_goal:
            return q_goal[:, 0].reshape(4, 1)
        return _normalize_quat(q_goal[:, 0]).reshape(4, 1)
    if m == 1:
        if is_vector_goal:
            return np.repeat(q_goal[:, 0].reshape(4, 1), n, axis=1)
        q = _normalize_quat(q_goal[:, 0])
        return np.repeat(q.reshape(4, 1), n, axis=1)
    
    # Use proper time-based resampling if jtime provided
    if jtime is not None and dt is not None:
        return _resample_zero_order_hold(jtime, q_goal, dt)
    
    # Fallback to index-based expansion
    if m >= 2 and n >= 2:
        out = np.zeros((4, n))
        for k in range(n):
            s = k * (m - 1) / float(n - 1)
            seg = int(np.floor(s))
            if seg >= m - 1:
                seg = m - 1
            if is_vector_goal:
                out[:, k] = q_goal[:, seg]
            else:
                out[:, k] = _normalize_quat(q_goal[:, seg])
        return out
    return None


def _read_u_max(actuator):
    """Read actuator limit from either a pybind property or method."""
    limit = getattr(actuator, "u_max", None)
    if limit is None:
        return None
    return float(limit() if callable(limit) else limit)


def plot_final_trajectory(X: np.ndarray, U: np.ndarray, dt: float, satellite=None, q_goal=None, boresight_body=None, jtime=None, title: str = "AL-iLQR C++ Final Trajectory"):
    n = X.shape[1]
    t_state = np.arange(n) * dt
    n_u = min(U.shape[1], max(0, n - 1))
    t_control = np.arange(n_u) * dt
    U_use = U[:, :n_u]
    q_goal_full = _expand_q_goal(q_goal, n, jtime, dt)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    ax_q = axes[0, 0]
    ax_w = axes[0, 1]
    ax_h = axes[1, 0]
    ax_pe = axes[1, 1]
    ax_mtq = axes[2, 0]
    ax_rw = axes[2, 1]

    q = X[3:7, :]
    w = X[0:3, :]
    h = X[7:, :]
    has_rw_state = h.shape[0] > 0

    num_mtq = satellite.numMTQ if satellite is not None else 0
    num_rw = satellite.numRW if satellite is not None else 0

    # Quaternion with goal
    ax_q.plot(t_state, q[0, :], label="q0", linewidth=1.5)
    ax_q.plot(t_state, q[1, :], label="q1", linewidth=1.5)
    ax_q.plot(t_state, q[2, :], label="q2", linewidth=1.5)
    ax_q.plot(t_state, q[3, :], label="q3", linewidth=1.5)
    if q_goal_full is not None:
        if np.any(np.isnan(q_goal_full[0, :])):
            ax_q.plot(t_state, q_goal_full[1, :], "--", alpha=0.6, linewidth=1.2, label="target x")
            ax_q.plot(t_state, q_goal_full[2, :], "--", alpha=0.6, linewidth=1.2, label="target y")
            ax_q.plot(t_state, q_goal_full[3, :], "--", alpha=0.6, linewidth=1.2, label="target z")
        else:
            ax_q.plot(t_state, q_goal_full[0, :], "--", alpha=0.6, linewidth=1.2, label="q0 goal")
            ax_q.plot(t_state, q_goal_full[1, :], "--", alpha=0.6, linewidth=1.2, label="q1 goal")
            ax_q.plot(t_state, q_goal_full[2, :], "--", alpha=0.6, linewidth=1.2, label="q2 goal")
            ax_q.plot(t_state, q_goal_full[3, :], "--", alpha=0.6, linewidth=1.2, label="q3 goal")
    ax_q.set_title("Quaternion")
    ax_q.set_xlabel("Time [s]")
    ax_q.set_ylabel("q")
    ax_q.grid(True, alpha=0.3)
    ax_q.legend(fontsize=7, ncol=2)

    # Angular velocity with magnitude
    ax_w.plot(t_state, w[0, :], label="wx", linewidth=1.5)
    ax_w.plot(t_state, w[1, :], label="wy", linewidth=1.5)
    ax_w.plot(t_state, w[2, :], label="wz", linewidth=1.5)
    ax_w.plot(t_state, np.linalg.norm(w, axis=0), "k--", linewidth=2, label="||w||")
    ax_w.set_title("Angular Velocity")
    ax_w.set_xlabel("Time [s]")
    ax_w.set_ylabel("rad/s")
    ax_w.grid(True, alpha=0.3)
    ax_w.legend(fontsize=8)

    # Wheel momentum
    ax_h.clear()
    if has_rw_state and num_rw > 0:
        for j in range(min(h.shape[0], num_rw)):
            ax_h.plot(t_state, h[j, :], linewidth=1.5, label=f"h_rw{j}")
        ax_h.set_title("Wheel Momentum")
        ax_h.set_xlabel("Time [s]")
        ax_h.set_ylabel("N m s")
        ax_h.legend(fontsize=8)
    else:
        ax_h.text(0.5, 0.5, "No RW momentum states", ha="center", va="center", transform=ax_h.transAxes)
        ax_h.set_title("Wheel Momentum")
        ax_h.set_xlabel("Time [s]")
        ax_h.set_ylabel("N m s")
    ax_h.grid(True, alpha=0.3)

    # Pointing error
    ax_pe.clear()
    if q_goal_full is not None:
        pe = _compute_pointing_error_deg_mixed(q, q_goal_full, boresight_body=boresight_body)
        if np.all(np.isnan(pe)):
            ax_pe.text(0.5, 0.5, "Pointing error unavailable", ha="center", va="center", transform=ax_pe.transAxes)
        else:
            ax_pe.plot(t_state, pe, "o-", color="C3", markersize=3, linewidth=1.5)
        ax_pe.set_title("Pointing Error")
        ax_pe.set_xlabel("Time [s]")
        ax_pe.set_ylabel("deg")
    else:
        ax_pe.text(0.5, 0.5, "No goal quaternion provided", ha="center", va="center", transform=ax_pe.transAxes)
        ax_pe.set_title("Pointing Error")
        ax_pe.set_xlabel("Time [s]")
        ax_pe.set_ylabel("deg")
    ax_pe.grid(True, alpha=0.3)

    # MTQ control with limits
    ax_mtq.clear()
    if num_mtq > 0 and n_u > 0:
        mtq_u = U_use[0:num_mtq, :]
        for i in range(mtq_u.shape[0]):
            ax_mtq.plot(t_control, mtq_u[i, :], linewidth=1.5, label=f"m_mtq{i}")
        # Add limits if available
        if satellite is not None:
            for i in range(num_mtq):
                u_max = _read_u_max(satellite.getMTQ(i))
                if u_max is None:
                    continue
                u_max = abs(u_max)
                ax_mtq.axhline(u_max, color="r", linestyle="--", alpha=0.5, linewidth=1)
                ax_mtq.axhline(-u_max, color="r", linestyle="--", alpha=0.5, linewidth=1)
        ax_mtq.legend(fontsize=8)
    else:
        ax_mtq.text(0.5, 0.5, "No MTQ controls", ha="center", va="center", transform=ax_mtq.transAxes)
    ax_mtq.set_title("MTQ Control Inputs")
    ax_mtq.set_xlabel("Time [s]")
    ax_mtq.set_ylabel("A m²")
    ax_mtq.grid(True, alpha=0.3)

    # RW control with limits
    ax_rw.clear()
    if num_rw > 0 and n_u > 0:
        rw_u = U_use[num_mtq:num_mtq + num_rw, :]
        for i in range(rw_u.shape[0]):
            ax_rw.plot(t_control, rw_u[i, :], linewidth=1.5, label=f"tau_rw{i}")
        # Add limits if available
        if satellite is not None:
            for i in range(num_rw):
                u_max = _read_u_max(satellite.getRW(i))
                if u_max is None:
                    continue
                u_max = abs(u_max)
                ax_rw.axhline(u_max, color="r", linestyle="--", alpha=0.5, linewidth=1)
                ax_rw.axhline(-u_max, color="r", linestyle="--", alpha=0.5, linewidth=1)
        ax_rw.legend(fontsize=8)
    else:
        ax_rw.text(0.5, 0.5, "No RW controls", ha="center", va="center", transform=ax_rw.transAxes)
    ax_rw.set_title("RW Control Inputs")
    ax_rw.set_xlabel("Time [s]")
    ax_rw.set_ylabel("N m")
    ax_rw.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.show()
