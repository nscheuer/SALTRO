"""Visualization for warm-start trajectory results."""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def launch_viewer(X: np.ndarray, U: np.ndarray, dt: float, satellite=None, title: str = "Warm-start Trajectory"):
    """Plot quaternion, angular rates, and actuator controls.
    
    Parameters
    ----------
    X : ndarray (state_dim, N)
        State trajectory
    U : ndarray (control_dim, N)
        Control trajectory
    dt : float
        Timestep in seconds
    satellite : Satellite, optional
        Satellite object to extract actuator limits
    title : str
        Plot title
    """
    N = X.shape[1]
    t_state = np.arange(N) * dt
    t_control = np.arange(U.shape[1]) * dt

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    ax_q, ax_w, ax_mtq, ax_rw = axes.flatten()

    # Quaternion plot
    q = X[3:7, :]
    for i in range(q.shape[0]):
        ax_q.plot(t_state, q[i, :], label=f"q{i}")
    ax_q.set_title("Quaternion")
    ax_q.set_xlabel("Time [s]")
    ax_q.set_ylabel("q")
    ax_q.grid(True, alpha=0.3)
    ax_q.legend(fontsize=8)

    # Angular rate plot
    w = X[0:3, :]
    for i in range(w.shape[0]):
        ax_w.plot(t_state, w[i, :], label=f"w{i}")
    ax_w.plot(t_state, np.linalg.norm(w, axis=0), "k--", label="|w|")
    ax_w.set_title("Angular Rate")
    ax_w.set_xlabel("Time [s]")
    ax_w.set_ylabel("rad/s")
    ax_w.grid(True, alpha=0.3)
    ax_w.legend(fontsize=8)

    # MTQ control plot
    num_mtq = satellite.numMTQ if satellite is not None else 0
    if num_mtq > 0:
        mtq_u = U[0:num_mtq, :]
        for i in range(mtq_u.shape[0]):
            ax_mtq.plot(t_control, mtq_u[i, :], label=f"m_mtq{i}")
        
        # Plot MTQ limits
        for i in range(num_mtq):
            mtq = satellite.getMTQ(i)
            u_max = abs(mtq.u_max)
            ax_mtq.axhline(y=u_max, color='gray', linestyle=':', linewidth=1, alpha=0.7)
            ax_mtq.axhline(y=-u_max, color='gray', linestyle=':', linewidth=1, alpha=0.7)
        
        ax_mtq.legend(fontsize=8)
        ax_mtq.set_title("MTQ Control")
        ax_mtq.set_xlabel("Time [s]")
        ax_mtq.set_ylabel("A m^2")
    else:
        ax_mtq.text(0.5, 0.5, "No MTQ controls", ha="center", va="center", transform=ax_mtq.transAxes)
        ax_mtq.set_title("MTQ Control")
    ax_mtq.grid(True, alpha=0.3)

    # RW control plot
    num_rw = satellite.numRW if satellite is not None else 0
    if num_rw > 0:
        rw_u = U[num_mtq:num_mtq+num_rw, :]
        for i in range(rw_u.shape[0]):
            ax_rw.plot(t_control, rw_u[i, :], label=f"tau_rw{i}")
        
        # Plot RW limits
        for i in range(num_rw):
            rw = satellite.getRW(i)
            u_max = abs(rw.u_max)
            ax_rw.axhline(y=u_max, color='gray', linestyle=':', linewidth=1, alpha=0.7)
            ax_rw.axhline(y=-u_max, color='gray', linestyle=':', linewidth=1, alpha=0.7)
        
        ax_rw.legend(fontsize=8)
        ax_rw.set_title("RW Control")
        ax_rw.set_xlabel("Time [s]")
        ax_rw.set_ylabel("N m")
    else:
        ax_rw.text(0.5, 0.5, "No RW controls", ha="center", va="center", transform=ax_rw.transAxes)
        ax_rw.set_title("RW Control")
    ax_rw.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13)
    plt.show()
