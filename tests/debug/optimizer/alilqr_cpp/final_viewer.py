import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def plot_final_trajectory(X: np.ndarray, U: np.ndarray, dt: float, satellite=None, title: str = "AL-iLQR C++ Final Trajectory"):
    n = X.shape[1]
    t_state = np.arange(n) * dt
    n_u = min(U.shape[1], max(0, n - 1))
    t_control = np.arange(n_u) * dt
    U_use = U[:, :n_u]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    ax_q, ax_w, ax_mtq, ax_rw = axes.flatten()

    q = X[3:7, :]
    for i in range(q.shape[0]):
        ax_q.plot(t_state, q[i, :], label=f"q{i}")
    ax_q.set_title("Quaternion")
    ax_q.set_xlabel("Time [s]")
    ax_q.set_ylabel("q")
    ax_q.grid(True, alpha=0.3)
    ax_q.legend(fontsize=8)

    w = X[0:3, :]
    for i in range(w.shape[0]):
        ax_w.plot(t_state, w[i, :], label=f"w{i}")
    ax_w.plot(t_state, np.linalg.norm(w, axis=0), "k--", label="|w|")
    ax_w.set_title("Angular Rate")
    ax_w.set_xlabel("Time [s]")
    ax_w.set_ylabel("rad/s")
    ax_w.grid(True, alpha=0.3)
    ax_w.legend(fontsize=8)

    num_mtq = satellite.numMTQ if satellite is not None else 0
    num_rw = satellite.numRW if satellite is not None else 0

    if num_mtq > 0 and n_u > 0:
        mtq_u = U_use[0:num_mtq, :]
        for i in range(mtq_u.shape[0]):
            ax_mtq.plot(t_control, mtq_u[i, :], label=f"m_mtq{i}")
        ax_mtq.legend(fontsize=8)
    else:
        ax_mtq.text(0.5, 0.5, "No MTQ controls", ha="center", va="center", transform=ax_mtq.transAxes)
    ax_mtq.set_title("MTQ Control")
    ax_mtq.set_xlabel("Time [s]")
    ax_mtq.set_ylabel("A m^2")
    ax_mtq.grid(True, alpha=0.3)

    if num_rw > 0 and n_u > 0:
        rw_u = U_use[num_mtq:num_mtq + num_rw, :]
        for i in range(rw_u.shape[0]):
            ax_rw.plot(t_control, rw_u[i, :], label=f"tau_rw{i}")
        ax_rw.legend(fontsize=8)
    else:
        ax_rw.text(0.5, 0.5, "No RW controls", ha="center", va="center", transform=ax_rw.transAxes)
    ax_rw.set_title("RW Control")
    ax_rw.set_xlabel("Time [s]")
    ax_rw.set_ylabel("N m")
    ax_rw.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13)
    plt.show()
